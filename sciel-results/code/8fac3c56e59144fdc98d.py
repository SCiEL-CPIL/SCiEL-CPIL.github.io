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
        # Projected to tangent space
        
        dots = P @ P.T
        np.fill_diagonal(dots, 0.0)
        d2 = 2.0 - 2.0 * dots
        d2 = np.maximum(d2, 1e-15)
        
        inv_d2 = np.zeros_like(d2)
        mask = d2 > 1e-15
        inv_d2[mask] = 1.0 / d2[mask]
        np.fill_diagonal(inv_d2, 0.0)
        
        # G_euclidean = sum_j inv_d2[i, j] P_j - P_i * sum_j inv_d2[i, j]
        G_euclidean = inv_d2 @ P
        sum_inv_d2 = np.sum(inv_d2, axis=1)
        G_euclidean -= sum_inv_d2[:, None] * P
        
        # Project to tangent space
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

    best_energy = float('inf')
    P_best = None
    
    rng = np.random.RandomState(42)
    
    initializations = []
    initializations.append(fibonacci_spiral(N))
    # Add more random initializations to escape local minima
    for _ in range(10):
        initializations.append(random_on_sphere(N, rng))
        
    for P_init in initializations:
        P = P_init.copy()
        E = compute_energy(P)
        
        if E < best_energy:
            best_energy = E
            P_best = P.copy()
            
        # Riemannian Adam
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        lr = 1e-2
        m = np.zeros_like(P)
        v = np.zeros_like(P)
        
        max_iters = 5000
        for t in range(1, max_iters + 1):
            G = compute_grad(P)
            grad_norm = np.linalg.norm(G)
            if grad_norm < 1e-12:
                break
            
            # Update biased first moment estimate
            m = beta1 * m + (1 - beta1) * G
            # Update biased second raw moment estimate
            v = beta2 * v + (1 - beta2) * (G * G)
            
            # Compute bias-corrected first moment estimate
            m_hat = m / (1 - beta1 ** t)
            # Compute bias-corrected raw moment estimate
            v_hat = v / (1 - beta2 ** t)
            
            # Direction
            d = -m_hat / (np.sqrt(v_hat) + epsilon)
            
            # Adaptive step size (simple decay or constant)
            # We use a constant lr but could adapt it
            alpha = lr
            
            # Riemannian update
            P_new = riemannian_exp(P, alpha * d)
            E_new = compute_energy(P_new)
            
            # Simple check: if energy increased significantly, reduce step?
            # For Adam, this is less common, but we can do a simple rejection
            if E_new > E + 1e-4:
                 # Try smaller step
                 alpha = alpha * 0.5
                 P_new = riemannian_exp(P, alpha * d)
                 E_new = compute_energy(P_new)
                 if E_new > E + 1e-4:
                     # Skip this step or break?
                     # Let's just break if we can't improve
                     break
            
            P = P_new
            E = E_new
            
            if E < best_energy:
                best_energy = E
                P_best = P.copy()

    return P_best
