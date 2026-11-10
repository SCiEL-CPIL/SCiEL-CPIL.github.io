# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). 
    Uses global optimization (Differential Evolution) on a Mixture of 2 Gaussians.
    """
    n = 10
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    def get_bin_indices(proj_func, min_val, max_val):
        indices = {}
        for val in range(min_val, max_val + 1):
            mask = (proj_func(X, Y) == val)
            indices[val] = np.argwhere(mask)
        return indices

    idx_neg1 = get_bin_indices(lambda X, Y: X - Y, -9, 9)
    idx_0 = get_bin_indices(lambda X, Y: X, 0, 9)
    idx_1 = get_bin_indices(lambda X, Y: X + Y, 0, 18)
    idx_2 = get_bin_indices(lambda X, Y: X + 2*Y, 0, 27)

    def compute_entropies(P):
        probs_neg1 = np.zeros(19)
        for val, indices in idx_neg1.items():
            if len(indices) > 0:
                probs_neg1[val + 9] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        probs_0 = np.zeros(10)
        for val, indices in idx_0.items():
            if len(indices) > 0:
                probs_0[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        probs_1 = np.zeros(19)
        for val, indices in idx_1.items():
            if len(indices) > 0:
                probs_1[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        probs_2 = np.zeros(28)
        for val, indices in idx_2.items():
            if len(indices) > 0:
                probs_2[val] = np.sum(P[indices[:, 0], indices[:, 1]])
        
        def ent(p):
            p = p[p > 1e-15]
            if len(p) == 0: return 0.0
            return -np.sum(p * np.log2(p))
            
        D = ent(probs_neg1)
        M = max(ent(probs_0), ent(probs_1), ent(probs_2))
        
        if M < 1e-9:
            return 0.0, 0.0
        return D, M

    def objective(params):
        # params: w1, mu1x, mu1y, s1x, s1y, r1, mu2x, mu2y, s2x, s2y, r2
        w1, mu1x, mu1y, s1x, s1y, r1, mu2x, mu2y, s2x, s2y, r2 = params
        
        # Constrain weights
        w1 = np.clip(w1, 0.01, 0.99)
        
        # Constrain correlations
        r1 = np.clip(r1, -0.99, 0.99)
        r2 = np.clip(r2, -0.99, 0.99)
        
        # Construct Covariances
        cov1 = np.array([
            [s1x**2, r1 * s1x * s1y],
            [r1 * s1x * s1y, s1y**2]
        ])
        cov2 = np.array([
            [s2x**2, r2 * s2x * s2y],
            [r2 * s2x * s2y, s2y**2]
        ])
        
        try:
            inv_cov1 = np.linalg.inv(cov1)
            inv_cov2 = np.linalg.inv(cov2)
        except np.linalg.LinAlgError:
            return 0.0
            
        dx1 = X - mu1x
        dy1 = Y - mu1y
        z1 = np.stack([dx1, dy1], axis=-1)
        d2_1 = np.einsum('...i,ij,...j->...', z1, inv_cov1, z1)
        
        dx2 = X - mu2x
        dy2 = Y - mu2y
        z2 = np.stack([dx2, dy2], axis=-1)
        d2_2 = np.einsum('...i,ij,...j->...', z2, inv_cov2, z2)
        
        # Gaussian PDFs (unnormalized)
        P1 = np.exp(-0.5 * d2_1)
        P2 = np.exp(-0.5 * d2_2)
        
        # Mixture
        P = w1 * P1 + (1 - w1) * P2
        
        total = P.sum()
        if total < 1e-15:
            return 0.0
        P = P / total
        
        D, M = compute_entropies(P)
        if M < 1e-9:
            return 0.0
        return D / M

    # Bounds for 11 parameters
    # w1: [0.01, 0.99]
    # mu: [0, 9]
    # sigma: [0.1, 10]
    # rho: [-1, 1]
    bounds = [
        (0.01, 0.99), # w1
        (0, 9), (0, 9), (0.1, 10), (0.1, 10), (-1, 1), # Component 1
        (0, 9), (0, 9), (0.1, 10), (0.1, 10), (-1, 1)  # Component 2
    ]
    
    # Run Differential Evolution
    # Maximize objective by minimizing negative
    res = differential_evolution(lambda p: -objective(p), bounds, maxiter=50, popsize=10, tol=1e-4, seed=42)
    
    best_params = res.x
    
    # Reconstruct P
    w1, mu1x, mu1y, s1x, s1y, r1, mu2x, mu2y, s2x, s2y, r2 = best_params
    w1 = np.clip(w1, 0.01, 0.99)
    r1 = np.clip(r1, -0.99, 0.99)
    r2 = np.clip(r2, -0.99, 0.99)
    
    cov1 = np.array([
        [s1x**2, r1 * s1x * s1y],
        [r1 * s1x * s1y, s1y**2]
    ])
    cov2 = np.array([
        [s2x**2, r2 * s2x * s2y],
        [r2 * s2x * s2y, s2y**2]
    ])
    
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)
    
    dx1 = X - mu1x
    dy1 = Y - mu1y
    z1 = np.stack([dx1, dy1], axis=-1)
    d2_1 = np.einsum('...i,ij,...j->...', z1, inv_cov1, z1)
    
    dx2 = X - mu2x
    dy2 = Y - mu2y
    z2 = np.stack([dx2, dy2], axis=-1)
    d2_2 = np.einsum('...i,ij,...j->...', z2, inv_cov2, z2)
    
    P1 = np.exp(-0.5 * d2_1)
    P2 = np.exp(-0.5 * d2_2)
    
    P = w1 * P1 + (1 - w1) * P2
    P = P / P.sum()
    
    return P

# EVOLVE-BLOCK-END
