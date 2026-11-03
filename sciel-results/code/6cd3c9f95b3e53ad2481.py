import numpy as np
import math
from scipy.optimize import minimize

N = 150

def construct():
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def random_on_sphere(n, rng):
        v = rng.normal(0, 1, size=(n, 3))
        norms = np.linalg.norm(v, axis=1)
        v /= norms[:, None]
        return v
        
    def compute_energy(P):
        # P is (N, 3)
        # d_ij = ||P_i - P_j||
        # Energy = sum_{i<j} log(1/d_ij) = - sum_{i<j} log(d_ij)
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        # d^2 = 2 - 2*dots
        d2 = 2.0 - 2.0 * dots
        # Ensure positive
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        # log(1/d) = -log(d)
        return -np.sum(np.log(d_iu))
        
    def compute_grad(P):
        # Gradient of E wrt P_i
        # dE/dP_i = sum_{j != i} dE/dd_ij * dd_ij/dP_i
        # E = - sum log(d)
        # dE/dd = -1/d
        # dd/dP_i = (P_i - P_j) / d
        # So term is -1/d * (P_i - P_j)/d = -(P_i - P_j)/d^2 = (P_j - P_i)/d^2
        # Sum over j.
        # G_i = sum_{j != i} (P_j - P_i) / ||P_j - P_i||^2
        # This is the Euclidean gradient.
        # Riemannian gradient is the projection of Euclidean gradient onto tangent space.
        # Proj_T_p(R) = R - <R, p> p
        
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        # Vectorized:
        # inv_d2 = 1/d2, set diagonal to 0
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        # Sum_j P_j / d2_ij
        G_euclidean = inv_d2 @ P
        
        # Sum_j 1/d2_ij
        sum_inv_d2 = np.sum(inv_d2, axis=1) # Shape (N,)
        
        G_euclidean -= sum_inv_d2[:, None] * P
        
        # Project to tangent space
        # <G, P>
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        
        return G_riemann

    def riemannian_exp(P, V):
        # V is tangent vector at P
        # exp_P(V) = P cos(||V||) + (V/||V||) sin(||V||)
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        # Renormalize to avoid drift
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def project_to_sphere(P):
        """Project points to unit sphere."""
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return P / norms

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    # Add more diverse random initializations
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P = P_init.copy()
        E = compute_energy(P)
        
        if E < best_energy:
            best_energy = E
            P_best = P.copy()
            
        # Use Riemannian L-BFGS style approach with better line search
        max_iters = 3000
        step_size = 1e-2
        min_step = 1e-12
        max_step = 0.5
        c1 = 1e-4  # Armijo
        beta = 0.9  # Backtracking factor
        
        # Momentum for accelerated convergence
        momentum = 0.0
        prev_G = None
        
        for it in range(max_iters):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-11:
                break
            
            # Simple momentum
            if prev_G is not None:
                momentum = 0.8 * momentum + 0.2 * (1.0 - momentum)
                d = -G - momentum * prev_G
            else:
                d = -G
                momentum = 0.0
            prev_G = G
            
            # Line Search with backtracking
            alpha = step_size
            E_cur = E
            G_cur = G
            
            # Armijo Condition: f(x + alpha d) <= f(x) + c1 * alpha * <G, d>
            slope = np.sum(G * d)  # This is negative
            
            found = False
            for ls_iter in range(40):
                P_trial = riemannian_exp(P, alpha * d)
                E_trial = compute_energy(P_trial)
                
                if E_trial <= E_cur + c1 * alpha * slope:
                    found = True
                    break
                alpha *= beta
                if alpha < min_step:
                    break
            
            if found:
                P = P_trial
                E = E_trial
                
                # Adaptive step size
                if alpha > min_step:
                    step_size = min(alpha * 1.05, max_step)
                else:
                    step_size = max(alpha * 0.5, min_step)
                
                if E < best_energy:
                    best_energy = E
                    P_best = P.copy()
            else:
                # If line search failed, reduce step and try again
                step_size *= 0.5
                if step_size < min_step:
                    break

    # Final polish with smaller steps
    P = P_best.copy()
    E = compute_energy(P)
    step_size = 1e-4
    for it in range(1000):
        G = compute_grad(P)
        grad_norm = np.linalg.norm(G)
        if grad_norm < 1e-12:
            break
        
        d = -G
        alpha = step_size
        E_cur = E
        slope = np.sum(G * d)
        
        found = False
        for ls_iter in range(30):
            P_trial = riemannian_exp(P, alpha * d)
            E_trial = compute_energy(P_trial)
            
            if E_trial <= E_cur + 1e-4 * alpha * slope:
                found = True
                break
            alpha *= 0.5
            if alpha < 1e-14:
                break
        
        if found:
            P = P_trial
            E = E_trial
            step_size = min(alpha * 1.1, 1e-3)
            if E < best_energy:
                best_energy = E
                P_best = P.copy()
        else:
            step_size *= 0.5
            if step_size < 1e-14:
                break

    return P_best
