"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    def compute_score(heights):
        """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
        h = np.array(heights)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-12:
            return 0.0
        # Autoconvolution (circular)
        conv = np.convolve(h, h, mode='full')
        # The problem states intervals of [-1/4, 1/4], so we take the central part
        # For N points on [-1/4, 1/4], the autoconvolution support is [-1/2, 1/2]
        # We need the supremum norm of the autoconvolution
        linf = np.max(conv)
        if linf < 1e-12:
            return 0.0
        return l2_sq / (l1 * linf)
    
    def neg_score(heights):
        """Negative score for minimization"""
        return -compute_score(heights)
    
    # Start with a sparse Sidon-like pattern
    # Use a quadratic residue or similar sparse set
    # For N=50, pick indices that form a Sidon set
    # Simple approach: use primes or quadratic residues mod p
    
    # Try several initial patterns and optimize from each
    best_heights = [1.0] * N
    best_score = compute_score(best_heights)
    
    # Pattern 1: Sparse - only every k-th position
    for k in [2, 3, 4, 5]:
        h = np.zeros(N)
        h[::k] = 1.0
        # Scale to have same L1 norm
        h = h / np.sum(h) * N
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_heights = h.copy()
    
    # Pattern 2: Sidon-like - quadratic residues
    # For a prime p, the quadratic residues form a Sidon set
    # Use p = 47 (prime near 50)
    p = 47
    residues = set()
    for x in range(p):
        residues.add((x*x) % p)
    h = np.zeros(N)
    for i in range(min(p, N)):
        if i in residues:
            h[i] = 1.0
    if np.sum(h) > 0:
        h = h / np.sum(h) * N
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_heights = h.copy()
    
    # Pattern 3: Random sparse patterns
    rng = np.random.RandomState(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        # Random sparse pattern with ~20-30% density
        density = 0.2 + 0.1 * trial / 20
        mask = rng.random(N) < density
        h = np.where(mask, 1.0, 0.0)
        if np.sum(h) < 3:
            continue
        h = h / np.sum(h) * N
        s = compute_score(h)
        if s > best_score:
            best_score = s
            best_heights = h.copy()
    
    # Local optimization from best starting point
    if time.time() < deadline - 20:
        h0 = best_heights.copy()
        # Ensure nonnegativity by optimizing in log space for positive entries
        # Simple approach: use Nelder-Mead with bounds
        try:
            result = minimize(
                neg_score, 
                h0, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < -best_score:
                # Project to nonnegative
                h_opt = np.maximum(result.x, 0)
                s = compute_score(h_opt)
                if s > best_score:
                    best_score = s
                    best_heights = h_opt.copy()
        except Exception:
            pass
    
    # Final cleanup: ensure nonnegative and valid
    best_heights = np.maximum(best_heights, 0)
    if np.sum(best_heights) < 1e-12:
        return [1.0] * N
    
    return list(best_heights)
# EVOLVE-BLOCK-END
