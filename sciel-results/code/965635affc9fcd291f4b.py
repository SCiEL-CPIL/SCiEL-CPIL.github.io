import numpy as np
import time
from scipy.optimize import differential_evolution

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    def score(x):
        s = np.sum(x)
        if abs(s) < 1e-10:
            return float('inf')
        c = np.correlate(x, x, mode='full')
        max_c = np.max(np.abs(c))
        return max_c / (s * s)
    
    # Baseline: all ones
    x_baseline = np.ones(N, dtype=float)
    current_best = score(x_baseline)
    best_x = x_baseline.copy()
    
    # Sparse polynomial approach:
    # Represent x as sum of a few "blocks" (contiguous segments with constant sign)
    # This drastically reduces dimensionality from N to ~2*num_blocks
    
    def sparse_to_dense(params):
        """Convert sparse block representation to dense signal.
        params: [pos1, width1, sign1, pos2, width2, sign2, ...]
        Each block is a contiguous segment with constant sign (±1)
        """
        x = np.zeros(N, dtype=float)
        num_blocks = len(params) // 3
        for i in range(num_blocks):
            pos = int(params[3*i])
            width = int(params[3*i + 1])
            sign = params[3*i + 2]
            if pos >= 0 and pos < N:
                end = min(pos + width, N)
                x[pos:end] = sign
        return x
    
    def objective(params):
        """Evaluate the metric for a sparse parameter vector."""
        x = sparse_to_dense(params)
        s = np.sum(x)
        if abs(s) < 1e-10:
            return 1e10  # Large penalty for zero sum
        c = np.correlate(x, x, mode='full')
        max_c = np.max(np.abs(c))
        return max_c / (s * s)
    
    # Differential evolution over sparse block parameters
    # Try different numbers of blocks
    best_de_score = current_best
    best_de_x = best_x.copy()
    
    for num_blocks in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 20:  # Need time for finalization
            break
        
        # Parameters: for each block, (position, width, sign)
        # Position: 0 to N-1, Width: 1 to N, Sign: -1 to 1
        bounds = []
        for _ in range(num_blocks):
            bounds.append((0, N - 1))      # position
            bounds.append((1, N))           # width
            bounds.append((-1, 1))          # sign
        
        try:
            result = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=30,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=False,  # Skip polish to save time
                callback=None
            )
            
            if result.success or result.fun < best_de_score:
                x_de = sparse_to_dense(result.x)
                de_score = score(x_de)
                if de_score < best_de_score:
                    best_de_score = de_score
                    best_de_x = x_de
        except Exception:
            pass
    
    # Final local refinement: flip individual bits to improve
    if time.time() < deadline - 10:
        x_refine = best_de_x.copy()
        cur_score = score(x_refine)
        
        # Simple local search: try flipping each bit
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                x_trial = x_refine.copy()
                x_trial[i] *= -1
                trial_score = score(x_trial)
                if trial_score < cur_score:
                    x_refine = x_trial
                    cur_score = trial_score
                    improved = True
    
    # Return the best found
    if score(x_refine) < current_best:
        return x_refine
    elif score(best_de_x) < current_best:
        return best_de_x
    else:
        return x_baseline
