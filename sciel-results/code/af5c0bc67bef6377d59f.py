import numpy as np
import math

N = 150

def construct():
    def fibonacci_spiral(n, rng=None, jitter=0.0):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        if rng is not None and jitter > 0:
            phi += rng.normal(0, jitter, size=n)
            z += rng.normal(0, jitter * 0.1, size=n)
            z = np.clip(z, -1.0, 1.0)
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
        
        # diff[i, j] = P_j - P_i ? 
        # Let's compute S_i = sum_j P_j / d2_ij
        # G_i = S_i - N * P_i / d2_ii ? No, d2_ii is 0.
        # G_i = sum_{j != i} P_j / d2_ij - P_i * sum_{j != i} 1/d2_ij
        
        # Vectorized:
        # inv_d2 = 1/d2, set diagonal to 0
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        # Sum_j P_j / d2_ij
        S = P.T @ inv_d2  # Shape (3, N) -> (N, 3) if we transpose?
        # inv_d2 is (N, N). P.T is (3, N). P.T @ inv_d2 is (3, N).
        # We want G_i = sum_j inv_d2[i, j] P_j.
        # So G = inv_d2 @ P.
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

    def riemannian_retract(P, V):
        # Simpler retraction: P_new = normalize(P + V)
        P_new = P + V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    initializations = []
    # Base Fibonacci spiral
    initializations.append(fibonacci_spiral(N))
    # Jittered Fibonacci spirals with increasing jitter
    for jitter in [0.05, 0.1, 0.2, 0.3, 0.5]:
        initializations.append(fibonacci_spiral(N, rng=rng, jitter=jitter))
    # Random initializations
    for _ in range(5):
        initializations.append(random_on_sphere(N, rng))
    # Hybrid: half Fibonacci, half random
    for _ in range(3):
        fib = fibonacci_spiral(N, rng=rng, jitter=0.1)
        rand = random_on_sphere(N, rng)
        hybrid = (fib + rand) / 2.0
        hybrid /= np.linalg.norm(hybrid, axis=1, keepdims=True)
        initializations.append(hybrid)
        
    # L-BFGS style memory
    m = 10  # Number of stored corrections
    
    for P_init in initializations:
        P = P_init.copy()
        E = compute_energy(P)
        
        if E < best_energy:
            best_energy = E
            P_best = P.copy()
            
        # Phase 1: Coarse optimization with momentum
        max_iters = 3000
        step_size = 0.01
        min_step = 1e-12
        max_step = 0.5
        c1 = 1e-4  # Armijo
        momentum = 0.9
        
        # Momentum buffer
        V_prev = np.zeros_like(P)
        
        # L-BFGS storage
        S_list = []  # Directions
        Y_list = []  # Gradient differences
        R_list = []  # Inner products
        W_list = []  # Weights
        rho_list = []  # 1/<s,y>
        
        for it in range(max_iters):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-11:
                break
            
            # Compute search direction using L-BFGS two-loop recursion
            # Start with q = G
            q = G.copy()
            alpha_lbfgs = np.zeros(m)
            
            # First loop: compute alpha
            for i in range(len(S_list) - 1, -1, -1):
                s = S_list[i]
                y = Y_list[i]
                rho = rho_list[i]
                alpha_lbfgs[i] = rho * np.sum(s * q)
                q -= alpha_lbfgs[i] * y
            
            # Two-loop recursion with initial Hessian approximation
            # Use gamma * I where gamma = <s_k, y_k> / <y_k, y_k>
            if len(S_list) > 0:
                s_last = S_list[-1]
                y_last = Y_list[-1]
                gamma = np.sum(s_last * y_last) / max(np.sum(y_last * y_last), 1e-15)
            else:
                gamma = 1.0 / max(np.sum(G * G), 1e-15)
            
            r = gamma * q
            
            # Second loop: compute d
            for i in range(len(S_list)):
                s = S_list[i]
                y = Y_list[i]
                rho = rho_list[i]
                beta = rho * np.sum(y * r)
                r += s * (alpha_lbfgs[i] - beta)
            
            d = -r
            
            # Add momentum
            d = momentum * V_prev + (1 - momentum) * d
            V_prev = d
            
            # Line Search
            alpha = step_size
            E_cur = E
            G_cur = G
            
            # Armijo Condition: f(x + alpha d) <= f(x) + c1 * alpha * <G, d>
            slope = np.sum(G * d)  # This is -||G||^2 (approximately)
            if slope >= 0:
                # Direction is not descent, use pure gradient
                d = -G
                slope = np.sum(G * d)
            
            found = False
            P_trial = None
            E_trial = None
            for ls_iter in range(40):
                P_trial = riemannian_retract(P, alpha * d)
                E_trial = compute_energy(P_trial)
                
                if E_trial <= E_cur + c1 * alpha * slope:
                    found = True
                    break
                alpha *= 0.5
                if alpha < min_step:
                    break
            
            if found:
                P_new = P_trial
                E_new = E_trial
                
                # Update L-BFGS memory
                G_new = compute_grad(P_new)
                s = P_new - P
                y = G_new - G
                sy = np.sum(s * y)
                
                if sy > 1e-12:  # Curvature condition
                    S_list.append(s)
                    Y_list.append(y)
                    rho_list.append(1.0 / sy)
                    if len(S_list) > m:
                        S_list.pop(0)
                        Y_list.pop(0)
                        rho_list.pop(0)
                
                P = P_new
                E = E_new
                
                # Update step size
                if alpha > min_step:
                    step_size = min(alpha * 1.05, max_step)
                else:
                    step_size = max(alpha * 0.5, min_step)
                
                if E < best_energy:
                    best_energy = E
                    P_best = P.copy()
            else:
                # If line search failed, break or shrink step
                step_size *= 0.5
                if step_size < min_step:
                    break

        # Phase 2: Fine-tuning with smaller steps
        step_size = 1e-4
        max_step = 1e-2
        min_step = 1e-14
        momentum = 0.5
        V_prev = np.zeros_like(P)
        
        for it in range(2000):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-12:
                break
            
            d = -G
            d = momentum * V_prev + (1 - momentum) * d
            V_prev = d
            
            alpha = step_size
            E_cur = E
            slope = np.sum(G * d)
            
            if slope >= 0:
                d = -G
                slope = np.sum(G * d)
            
            found = False
            P_trial = None
            E_trial = None
            for ls_iter in range(30):
                P_trial = riemannian_retract(P, alpha * d)
                E_trial = compute_energy(P_trial)
                
                if E_trial <= E_cur + c1 * alpha * slope:
                    found = True
                    break
                alpha *= 0.5
                if alpha < min_step:
                    break
            
            if found:
                P = P_trial
                E = E_trial
                step_size = min(alpha * 1.02, max_step)
                
                if E < best_energy:
                    best_energy = E
                    P_best = P.copy()
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break

    # Final verification: ensure all points are on unit sphere and valid
    P_best = np.asarray(P_best, dtype=np.float64)
    norms = np.linalg.norm(P_best, axis=1)
    P_best /= norms[:, None]
    
    return P_best
