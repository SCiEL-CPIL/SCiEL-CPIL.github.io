"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 50 nonnegative heights on equal intervals of [-1/4, 1/4]
    to maximize the ratio of squared L2 norm to L1 norm times supremum norm of the autoconvolution.
    
    The score is the lower bound on C2, which is:
    score = ||h||_2^2 / (||h||_1 * ||h * h||_inf)
    
    where h * h is the discrete autoconvolution.
    
    Strategy: Use a sparse, exponentially decaying sequence supported on positions that
    minimize autoconvolution overlap. We use a greedy approach to place weights at positions
    that minimize the current autoconvolution peak, then optimize the weights.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_score = -np.inf
    best_h = np.ones(N)
    
    # Helper: compute the score for a given height vector
    def compute_score(h):
        h = np.asarray(h, dtype=float)
        if np.any(h < -1e-12):
            return -np.inf
        h = np.maximum(h, 0.0)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return -np.inf
        l2_sq = np.sum(h**2)
        # Discrete autoconvolution: (h * h)[k] = sum_i h[i] * h[k-i]
        # For a vector of length N, the convolution has length 2N-1
        conv = np.convolve(h, h, mode='full')
        l_inf = np.max(conv)
        if l_inf < 1e-15:
            return -np.inf
        return l2_sq / (l1 * l_inf)
    
    # Helper: compute autoconvolution
    def autoconv(h):
        return np.convolve(h, h, mode='full')
    
    # Strategy 1: Exponentially decaying sequence on a Sidon-like set
    # A Sidon set has all pairwise sums distinct, so autoconvolution has no collisions
    # For 50 points, we can use a geometric progression or similar
    
    # Try various exponential decay rates and starting positions
    for decay in [0.7, 0.8, 0.85, 0.9, 0.95]:
        if time.time() > deadline:
            break
        
        # Place weights at positions with exponential decay
        # Use a sparse pattern: place non-zero weights at selected positions
        # Sidon set for small N: use positions like i^2 mod something, or just spread out
        
        # Try a simple approach: put weights at evenly spaced positions with exponential decay
        # But to avoid convolution collisions, use a set where pairwise sums are distinct
        
        # For N=50, let's try a geometric spacing
        # Actually, let's just try exponential decay over all positions first
        for start_idx in range(0, N, 5):
            if time.time() > deadline:
                break
            
            h = np.zeros(N)
            # Place exponential decay starting from start_idx
            for i in range(N - start_idx):
                h[start_idx + i] = decay ** i
            
            score = compute_score(h)
            if score > best_score:
                best_score = score
                best_h = h.copy()
    
    # Strategy 2: Optimize weights on a fixed sparse support
    # Pick a support set that minimizes convolution overlap
    # Use a Sidon set: for indices 0..N-1, use i^2 mod M for large M
    # Or simply use a set with large gaps
    
    # Let's try optimizing the weights on a carefully chosen support
    # Support: every other position, or positions with large gaps
    support_options = [
        list(range(0, N, 1)),  # all positions
        list(range(0, N, 2)),  # every other
        list(range(0, N, 3)),  # every third
    ]
    
    for support in support_options:
        if time.time() > deadline:
            break
        
        m = len(support)
        if m < 5:
            continue
        
        # Initialize with exponential decay on the support
        h_full = np.zeros(N)
        for i, idx in enumerate(support):
            h_full[idx] = 0.9 ** i
        
        # Optimize the weights on the support
        def objective(x):
            x = np.maximum(x, 0.0)
            h = np.zeros(N)
            for i, idx in enumerate(support):
                h[idx] = x[i]
            return -compute_score(h)
        
        x0 = np.array([h_full[idx] for idx in support])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.success:
                x_opt = np.maximum(result.x, 0.0)
                h_opt = np.zeros(N)
                for i, idx in enumerate(support):
                    h_opt[idx] = x_opt[i]
                score = compute_score(h_opt)
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Strategy 3: Greedy placement
    # Start with one position, then greedily add positions that minimize the increase in ||h*h||_inf
    if time.time() < deadline - 30:
        h_greedy = np.zeros(N)
        h_greedy[0] = 1.0
        remaining = set(range(1, N))
        
        for step in range(N - 1):
            if time.time() > deadline:
                break
            
            best_pos = None
            best_score_candidate = -np.inf
            
            # Try adding a small weight at each remaining position
            for pos in remaining:
                h_test = h_greedy.copy()
                h_test[pos] = 0.1  # small test weight
                
                score = compute_score(h_test)
                if score > best_score_candidate:
                    best_score_candidate = score
                    best_pos = pos
            
            if best_pos is not None:
                h_greedy[best_pos] = 0.1
                remaining.remove(best_pos)
        
        # Now optimize the weights
        support_g = [i for i in range(N) if h_greedy[i] > 0]
        if len(support_g) > 0:
            m = len(support_g)
            x0 = np.array([h_greedy[idx] for idx in support_g])
            
            def objective_g(x):
                x = np.maximum(x, 0.0)
                h = np.zeros(N)
                for i, idx in enumerate(support_g):
                    h[idx] = x[i]
                return -compute_score(h)
            
            try:
                result = minimize(objective_g, x0, method='Nelder-Mead',
                                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
                if result.success:
                    x_opt = np.maximum(result.x, 0.0)
                    h_opt = np.zeros(N)
                    for i, idx in enumerate(support_g):
                        h_opt[idx] = x_opt[i]
                    score = compute_score(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final check: ensure nonnegative and return
    best_h = np.maximum(best_h, 0.0)
    
    # If best_score is still very bad, fall back to uniform
    if best_score < 0.5:
        best_h = np.ones(N)
    
    return list(best_h)
# EVOLVE-BLOCK-END
