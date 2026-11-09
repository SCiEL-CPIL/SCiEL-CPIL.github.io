import numpy as np
import time

def _compute_value_fast(P, bins):
    # P is flattened 100x1
    p = P.flatten()
    
    # Projection distributions
    # diff: -9..9 -> 0..18
    p_diff = np.bincount(bins['diff'], weights=p, minlength=19)
    # x: 0..9 -> 0..9
    p_x = np.bincount(bins['x'], weights=p, minlength=10)
    # x+y: 0..18 -> 0..18
    p_xy = np.bincount(bins['xy'], weights=p, minlength=19)
    # x+2y: 0..27 -> 0..27
    p_x2y = np.bincount(bins['x2y'], weights=p, minlength=28)
    
    def entropy(prob):
        prob = prob[prob > 1e-15]
        if len(prob) == 0: return 0.0
        return -np.sum(prob * np.log2(prob))
        
    D = entropy(p_diff)
    M = max(entropy(p_x), entropy(p_xy), entropy(p_x2y))
    
    if M < 1e-9:
        return 0.0
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    # Pre-compute bin indices for each cell (y, x)
    # Index in P is y*n + x
    bin_diff = np.zeros((n, n), dtype=np.int32)
    bin_x = np.zeros((n, n), dtype=np.int32)
    bin_xy = np.zeros((n, n), dtype=np.int32)
    bin_x2y = np.zeros((n, n), dtype=np.int32)
    
    for y in range(n):
        for x in range(n):
            bin_diff[y, x] = x - y + (n - 1)
            bin_x[y, x] = x
            bin_xy[y, x] = x + y
            bin_x2y[y, x] = x + 2 * y
            
    bins = {
        'diff': bin_diff.flatten(),
        'x': bin_x.flatten(),
        'xy': bin_xy.flatten(),
        'x2y': bin_x2y.flatten()
    }
    
    # Initialize with uniform
    P = np.ones((n, n)) / (n * n)
    best_P = P.copy()
    best_val = _compute_value_fast(P, bins)
    
    start_time = time.time()
    deadline = start_time + 590  # 10s margin
    
    # Simulated Annealing / Random Search
    # We will perturb the distribution by moving mass between cells
    # or by scaling weights of a subset
    
    # Strategy: 
    # 1. Start with uniform.
    # 2. Iteratively pick 2 cells (i, j). Move mass from one to the other.
    # 3. Or, pick a subset of cells and optimize their weights via gradient ascent (numerical).
    
    # Given the small size, random search with restarts might work.
    # But "Knowledge" suggests greedy submodular.
    # Let's try a greedy construction of support first, then optimize weights.
    
    # Greedy Support Construction
    # Start with empty support. Add cell that maximizes ratio? 
    # Ratio is not submodular.
    # However, we can use a heuristic:
    # Try to find a set of cells where D is high and M is low.
    
    # Alternative: Random restarts with local optimization.
    # 1. Generate random sparse P (e.g. 10-30 non-zero cells, random weights).
    # 2. Local search: adjust weights of non-zero cells.
    # 3. Keep best.
    
    np.random.seed(42)
    
    for _ in range(1000): # Max iterations, but controlled by time
        if time.time() > deadline:
            break
            
        # Random initialization of sparse support
        k = np.random.randint(5, 30) # number of non-zero cells
        indices = np.random.choice(n*n, k, replace=False)
        weights = np.random.dirichlet(np.ones(k) * 2.0) # biased towards uniform-ish but random
        P_new = np.zeros(n*n)
        P_new[indices] = weights
        P_new = P_new.reshape(n, n)
        
        val = _compute_value_fast(P_new, bins)
        if val > best_val:
            best_val = val
            best_P = P_new.copy()
            
        # Local optimization of weights for this support
        # Use a simple gradient ascent or coordinate descent on the weights
        # Since k is small, we can do this.
        for step in range(20):
            if time.time() > deadline:
                break
            # Pick a random pair of indices in support to swap mass
            i1, i2 = np.random.choice(k, 2, replace=False)
            idx1, idx2 = indices[i1], indices[i2]
            
            w1 = P_new.flatten()[idx1]
            w2 = P_new.flatten()[idx2]
            
            if w1 < 1e-9 or w2 < 1e-9: continue
            
            # Try moving a small amount of mass
            delta = np.random.uniform(-0.1, 0.1) * min(w1, w2)
            if delta > w1 or -delta > w2: continue
            
            P_test = P_new.flatten().copy()
            P_test[idx1] += delta
            P_test[idx2] -= delta
            
            # Check validity
            if np.any(P_test < -1e-9): continue
            
            P_test = np.clip(P_test, 0, None)
            # Normalize? No, we are moving mass, sum is preserved.
            # But if we clip, sum changes. 
            # Better to ensure delta is small enough not to clip.
            
            val_test = _compute_value_fast(P_test, bins)
            if val_test > val:
                val = val_test
                P_new.flatten()[idx1] = P_test[idx1]
                P_new.flatten()[idx2] = P_test[idx2]
                if val > best_val:
                    best_val = val
                    best_P = P_new.copy()

    # Final check and cleanup
    best_P = np.clip(best_P, 0, None)
    s = np.sum(best_P)
    if s < 1e-9:
        return np.ones((n, n)) / (n * n)
    best_P /= s
    
    return best_P
