"""Improved program: min-max game approach for autoconvolution optimization."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 50 nonnegative step heights to maximize the ratio:
        ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    
    Strategy: Formulate as a zero-sum game between attacker (choosing lag k)
    and defender (choosing h). Use iterative multiplicative updates (Sinkhorn-style)
    to find a saddle point.
    """
    deadline = time.time() + 135  # safety margin under 150s limit
    
    def compute_ratio(h):
        """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.asarray(h, dtype=np.float64)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        conv = np.convolve(h, h, mode='full')
        conv_max = np.max(conv)
        if conv_max < 1e-15:
            return 0.0
        return l2_sq / (l1 * conv_max)
    
    def sinkhorn_iterate(h, n_iters=200):
        """
        Iterative multiplicative updates for the defender:
        - Compute autoconvolution and find peak lag
        - For each position i, compute its exposure (contribution to peak lag)
        - Reduce h[i] proportionally to its exposure
        """
        h = h.copy()
        h = np.clip(h, 1e-15, None)
        h = h / np.sum(h)
        
        best_ratio = compute_ratio(h)
        best_h = h.copy()
        
        for it in range(n_iters):
            if time.time() > deadline - 5:
                break
            
            conv = np.convolve(h, h, mode='full')
            conv_max = np.max(conv)
            if conv_max < 1e-15:
                break
            
            k_star = np.argmax(conv)
            
            # Compute exposure of each position to the peak lag
            exposure = np.zeros(N)
            for i in range(N):
                j = k_star - i
                if 0 <= j < N:
                    exposure[i] = h[i] * h[j]
            
            exp_max = np.max(exposure)
            if exp_max < 1e-15:
                break
            
            # Multiplicative update: reduce h where exposure is high
            alpha = 0.5
            h_new = h / (1.0 + alpha * exposure / exp_max)
            h_new = np.clip(h_new, 1e-15, None)
            h_new = h_new / np.sum(h_new)
            
            r = compute_ratio(h_new)
            if r > best_ratio:
                best_ratio = r
                best_h = h_new.copy()
            
            h = h_new
        
        return best_h, best_ratio
    
    initializations = []
    rng = np.random.default_rng(42)
    
    # Uniform
    initializations.append(np.ones(N) / N)
    
    # Sparse random
    for support_size in [2, 3, 4, 5, 6, 8, 10]:
        for _ in range(5):
            idx = rng.choice(N, size=support_size, replace=False)
            h_init = np.zeros(N)
            h_init[idx] = 1.0 / support_size
            initializations.append(h_init)
    
    # Structured: evenly spaced
    for gap in [2, 3, 4, 5]:
        idx = list(range(0, N, gap))
        h_init = np.zeros(N)
        h_init[idx] = 1.0 / len(idx)
        initializations.append(h_init)
    
    # Sidon-like
    sidon_indices = [0, 1, 3, 7, 12, 20, 30, 42]
    h_sidon = np.zeros(N)
    h_sidon[sidon_indices] = 1.0 / len(sidon_indices)
    initializations.append(h_sidon)
    
    # Two-point with varying ratio and gap
    for ratio in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
        for gap in [1, 2, 3, 5, 10]:
            h_init = np.zeros(N)
            h_init[0] = ratio
            h_init[gap] = 1.0 - ratio
            initializations.append(h_init)
    
    best_h = np.ones(N) / N
    best_ratio = compute_ratio(best_h)
    
    for h_init in initializations:
        if time.time() > deadline - 10:
            break
        
        try:
            h_opt, r = sinkhorn_iterate(h_init, n_iters=100)
            if r > best_ratio:
                best_ratio = r
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Final local search
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + rng.standard_normal(N) * 0.02
            h_pert = np.clip(h_pert, 0, None)
            s = np.sum(h_pert)
            if s > 1e-15:
                h_pert = h_pert / s
                r = compute_ratio(h_pert)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h_pert.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
