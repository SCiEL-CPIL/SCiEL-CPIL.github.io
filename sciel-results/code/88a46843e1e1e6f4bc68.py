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
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        return -np.sum(np.log(d_iu))
        
    def compute_grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        dots_GP = np.sum(G_euclidean * P, axis=1)
        G_riemann = G_euclidean - dots_GP[:, None] * P
        
        return G_riemann

    def riemannian_exp(P, V):
        v_norm = np.linalg.norm(V, axis=1)
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_t = np.cos(v_norm_safe)
        sin_t = np.sin(v_norm_safe) / v_norm_safe
        P_new = cos_t[:, None] * P + sin_t[:, None] * V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def riemannian_retract(P, V):
        P_new = P + V
        norms = np.linalg.norm(P_new, axis=1)
        P_new /= norms[:, None]
        return P_new

    def optimize(P_init, max_iters=5000):
        P = P_init.copy()
        E = compute_energy(P)
        best_E = E
        best_P = P.copy()
        
        # Riemannian Gradient Descent with Momentum
        # V is the momentum vector in tangent space
        V = np.zeros_like(P)
        mu = 0.9 # Momentum coefficient
        step_size = 1e-2
        min_step = 1e-12
        max_step = 1.0
        c1 = 1e-4 # Armijo
        
        for it in range(max_iters):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-10:
                break
            
            # Update momentum
            # V = mu * V - step_size * G ? 
            # Standard Polyak: V = mu * V + step_size * G (for minimization, we go down, so -G)
            # Let's use: V = mu * V - step_size * G
            # Then step: P = exp(P, V) ? Or P = P + V ?
            # Using Exponential map with momentum is tricky because V is in tangent space at P_old.
            # We need to transport V to P_new's tangent space or just use Retraction.
            # Let's use Retraction for simplicity with momentum, or just standard GD with backtracking.
            # Standard GD with backtracking is safer.
            
            d = -G
            
            # Line Search
            alpha = step_size
            E_cur = E
            slope = -np.sum(G * G) # <G, -G> = -||G||^2
            
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
                
                if E < best_E:
                    best_E = E
                    best_P = P.copy()
                    
                # Adaptive step size
                if alpha > min_step:
                    step_size = min(alpha * 1.1, max_step)
                else:
                    step_size = max(alpha * 0.5, min_step)
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break

        return best_P, best_E

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    # Generate initializations
    initializations = [fibonacci_spiral(N)]
    # Add more random initializations
    for _ in range(20):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P_opt, E_opt = optimize(P_init)
        
        if E_opt < best_energy:
            best_energy = E_opt
            P_best = P_opt.copy()
            
    # Optional: Basin Hopping on the best solution
    # Perturb and re-optimize a few times
    P_current = P_best.copy()
    E_current = best_energy
    
    for bh_iter in range(5):
        # Perturb
        noise = rng.normal(0, 0.1, size=(N, 3))
        P_perturbed = P_current + noise
        # Project to sphere
        norms = np.linalg.norm(P_perturbed, axis=1)
        P_perturbed /= norms[:, None]
        
        P_opt, E_opt = optimize(P_perturbed, max_iters=1000)
        
        if E_opt < E_current:
            E_current = E_opt
            P_current = P_opt
            best_energy = E_opt
            P_best = P_opt.copy()
            
    return P_best
