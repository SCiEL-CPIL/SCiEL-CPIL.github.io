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

    def flatten_P(P):
        return P.flatten()

    def unflatten_P(x):
        P = x.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        P /= norms[:, None]
        return P

    def objective(x):
        P = unflatten_P(x)
        return compute_energy(P)

    def gradient(x):
        P = unflatten_P(x)
        G = compute_grad(P)
        return G.flatten()

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    
    # More diverse random initializations
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))
    
    # Add perturbed fibonacci
    fib = fibonacci_spiral(N)
    for scale in [0.01, 0.05, 0.1]:
        P_pert = fib + rng.normal(0, scale, size=(N, 3))
        P_pert /= np.linalg.norm(P_pert, axis=1)[:, None]
        initializations.append(P_pert)
        
    for P_init in initializations:
        x_init = flatten_P(P_init)
        
        # Use L-BFGS-B with careful settings
        try:
            result = minimize(
                objective, 
                x_init, 
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-12, 'maxfun': 5000}
            )
            P_opt = unflatten_P(result.x)
            E_opt = compute_energy(P_opt)
            
            if E_opt < best_energy:
                best_energy = E_opt
                P_best = P_opt.copy()
        except Exception:
            pass
        
        # Also try a simple gradient descent refinement on the best found so far
        if P_best is not None:
            P = P_best.copy()
            E = compute_energy(P)
            
            # Short gradient descent polish
            for it in range(100):
                G = compute_grad(P)
                grad_norm = np.linalg.norm(G)
                if grad_norm < 1e-10:
                    break
                
                d = -G
                alpha = 0.1
                E_cur = E
                slope = np.sum(G * d)
                
                found = False
                for ls_iter in range(20):
                    P_trial = riemannian_exp(P, alpha * d)
                    E_trial = compute_energy(P_trial)
                    
                    if E_trial <= E_cur + 1e-4 * alpha * slope:
                        found = True
                        break
                    alpha *= 0.5
                    if alpha < 1e-10:
                        break
                
                if found:
                    P = P_trial
                    E = E_trial
                    if E < best_energy:
                        best_energy = E
                        P_best = P.copy()
                else:
                    break

    return P_best
