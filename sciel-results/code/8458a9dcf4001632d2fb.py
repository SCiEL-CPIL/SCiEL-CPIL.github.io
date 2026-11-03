import numpy as np
import math

N = 150

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Riemannian Gradient Descent
    # Energy: E = sum_{i<j} log(1/|pi-pj|) = - sum_{i<j} log|pi-pj|
    # Grad_E_k = sum_{j != k} (pj - pk) / |pk - pj|^2
    
    def energy(P):
        # Compute pairwise distances
        # |pi - pj|^2 = 2 - 2 pi.pj
        dots = P @ P.T
        # Avoid self-distances (diagonal)
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        # Clip to avoid log(0) or negative due to numerical errors
        d2 = np.maximum(d2, 1e-15)
        # Sum over i < j
        # The matrix d2 has 0 on diagonal.
        # Sum of log(1/sqrt(d2)) over i<j
        # = -0.5 * sum_{i!=j} log(sqrt(d2_ij)) ? No.
        # E = sum_{i<j} -log(sqrt(d2_ij)) = -0.5 sum_{i!=j} log(sqrt(d2_ij)) = -0.25 sum_{i!=j} log(d2_ij)
        # Let's just sum the upper triangle to be precise.
        iu = np.triu_indices(N, k=1)
        d2_iu = d2[iu]
        return -np.sum(np.log(np.sqrt(d2_iu)))

    def grad(P):
        dots = P @ P.T
        np.fill_diagonal(dots, 0)
        d2 = 2.0 - 2.0 * dots
        # d2 is |pi - pj|^2
        # Term for grad_k: sum_{j!=k} (pj - pk) / d2_kj
        # (pj - pk) / d2_kj
        # We can compute the sum vector for each k.
        # G_k = sum_j (P_j - P_k) / d2_kj
        # Let's compute matrix M where M_kj = (P_j - P_k) / d2_kj
        # This is a matrix operation.
        # d2 is (N, N).
        # We want sum over j for each k.
        # Vectorized:
        # diff = P[None, :, :] - P[:, None, :] -> (N, N, 3)
        # diff_kj = P_j - P_k
        diff = P[None, :, :] - P[:, None, :] # diff[k, j] = P_j - P_k
        # d2 is (N, N). We need to divide diff[k, j] by d2[k, j]
        # d2[:, :, None] is (N, N, 1)
        d2_safe = np.where(d2 > 1e-15, d2, 1e-15)
        terms = diff / d2_safe[:, :, None]
        # Sum over j (axis 1)
        G = np.sum(terms, axis=1)
        
        # Project to tangent space
        # g_k = G_k - (G_k . P_k) P_k
        dots_G_P = np.sum(G * P, axis=1)
        G_proj = G - dots_G_P[:, None] * P
        return G_proj

    eta = 1e-4
    best_energy = energy(P)
    P_best = P.copy()
    
    max_iters = 2000
    for it in range(max_iters):
        G = grad(P)
        
        # Update using exponential map
        # v = -eta * G
        v = -eta * G
        v_norm = np.linalg.norm(v, axis=1)
        
        # Handle zero norm
        v_norm_safe = np.where(v_norm > 1e-12, v_norm, 1e-12)
        cos_term = np.cos(v_norm_safe)
        sin_term = np.sin(v_norm_safe) / v_norm_safe
        
        # Exp_p(v) = cos(||v||) p + (sin(||v||)/||v||) v
        P_new = cos_term[:, None] * P + sin_term[:, None] * v
        
        # Normalize to ensure they stay on sphere (numerical stability)
        P_new /= np.linalg.norm(P_new, axis=1)[:, None]
        
        E_new = energy(P_new)
        
        if E_new < best_energy:
            best_energy = E_new
            P_best = P_new.copy()
            # Maybe increase eta slightly? Or keep it.
            # Adaptive step: if improved, keep eta. If not, reduce.
            # Since we track best, we can just try to improve from current P.
            # If E_new < energy(P), accept and maybe increase eta.
            # Let's do a simple adaptive step.
            eta *= 1.05
        else:
            # Rejected step
            eta *= 0.8
            if eta < 1e-7:
                break
            # P remains unchanged in this iteration
            continue
        P = P_new
        
    return P_best
