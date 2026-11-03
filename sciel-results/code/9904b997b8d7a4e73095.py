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
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        d_iu = np.sqrt(d2_iu)
        return -np.sum(np.log(d_iu))
        
    def compute_riemannian_grad(P):
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

    def adaptive_step_size(P, G, E_cur, alpha_init=0.01, max_iter=50):
        alpha = alpha_init
        slope = -np.sum(G * G)
        
        for _ in range(max_iter):
            P_trial = riemannian_exp(P, alpha * (-G))
            E_trial = compute_energy(P_trial)
            
            if E_trial <= E_cur + 0.1 * alpha * slope:
                return alpha, P_trial, E_trial
            
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        return alpha, P, E_cur

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = [fibonacci_spiral(N)]
    for _ in range(8):
        initializations.append(random_on_sphere(N, rng))
    
    fib = fibonacci_spiral(N)
    for perturb_scale in [0.05, 0.1, 0.2]:
        P_perturbed = fib + perturb_scale * rng.normal(0, 1, size=(N, 3))
        norms = np.linalg.norm(P_perturbed, axis=1)
        P_perturbed /= norms[:, None]
        initializations.append(P_perturbed)
        
    for init_idx, P_init in enumerate(initializations):
        P = P_init.copy()
        E = compute_energy(P)
        
        if E < best_energy:
            best_energy = E
            P_best = P.copy()
            
        momentum = np.zeros_like(P)
        beta1 = 0.9
        alpha = 0.01
        max_iters = 3000
        E_prev = E
        
        for it in range(max_iters):
            G = compute_riemannian_grad(P)
            grad_norm = np.linalg.norm(G)
            
            if grad_norm < 1e-12:
                break
            
            momentum = beta1 * momentum - G
            direction = momentum
            
            alpha, P_trial, E_trial = adaptive_step_size(P, direction, E, alpha)
            
            if np.allclose(P_trial, P, atol=1e-14):
                alpha *= 0.5
                if alpha < 1e-12:
                    break
                P_trial = riemannian_exp(P, alpha * direction)
                E_trial = compute_energy(P_trial)
                if E_trial >= E:
                    break
            
            P = P_trial
            E = E_trial
            
            if E_trial < E_prev - 1e-6:
                alpha = min(alpha * 1.05, 0.1)
            else:
                alpha = max(alpha * 0.9, 1e-6)
                
            if E < best_energy:
                best_energy = E
                P_best = P.copy()
                
            if it > 100 and abs(E - E_prev) < 1e-10:
                break
                
            E_prev = E

    if P_best is not None:
        P = P_best.copy()
        E = compute_energy(P)
        for _ in range(500):
            G = compute_riemannian_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-14:
                break
            alpha = 1e-4
            P_trial = riemannian_exp(P, -alpha * G)
            E_trial = compute_energy(P_trial)
            if E_trial < E:
                P = P_trial
                E = E_trial
                if E < best_energy:
                    best_energy = E
                    P_best = P.copy()
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break

    return P_best
