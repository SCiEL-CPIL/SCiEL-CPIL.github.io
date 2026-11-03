import numpy as np
import math

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
        # G_i = sum_{j != i} (P_j - P_i) / ||P_j - P_i||^2
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
        
        # G_i = sum_j inv_d2[i, j] P_j - P_i * sum_j inv_d2[i, j]
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1) # Shape (N,)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        # Project to tangent space
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        
        return G_riemann

    def riemannian_exp(P, V):
        # V is tangent vector at P
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        # Renormalize to avoid drift
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    # Add Fibonacci
    initializations.append(fibonacci_spiral(N))
    # Add more random initializations to escape local minima
    for _ in range(20):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P = P_init.copy()
        E = compute_energy(P)
        
        if E < best_energy:
            best_energy = E
            P_best = P.copy()
            
        max_iters = 5000
        step_size = 0.1  # Start with a larger step
        min_step = 1e-10
        max_step = 1.0
        c1 = 1e-4 # Armijo
        
        for it in range(max_iters):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-12:
                break
            
            # Direction
            d = -G
            
            # Line Search
            alpha = step_size
            E_cur = E
            G_cur = G
            
            # Armijo Condition
            slope = np.sum(G * d) # This is -||G||^2
            
            found = False
            for ls_iter in range(30):
                P_trial = riemannian_exp(P, alpha * d)
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
                
                # Update step size
                # If we accepted the first trial (alpha == step_size), increase
                # If we had to backtrack, keep or decrease slightly
                if alpha >= step_size * 0.99:
                    step_size = min(step_size * 1.5, max_step)
                else:
                    step_size = max(step_size * 0.9, min_step)
                
                if E < best_energy:
                    best_energy = E
                    P_best = P.copy()
            else:
                # If line search failed, break or shrink step
                step_size *= 0.5
                if step_size < min_step:
                    break

    return P_best
