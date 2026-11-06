"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 55.0  # Safety margin for 60s limit

def _get_aps():
    """Generate all APs (length >= 2) as list of index lists."""
    aps = []
    for a in range(_N):
        for d in range(1, _N - a):
            L = (_N - 1 - a) // d + 1
            if L >= 2:
                idx = [a + d * t for t in range(L)]
                aps.append(idx)
    return aps

def _build_pos_to_aps(aps):
    """Map position to list of AP indices containing it."""
    pos_to_aps = {i: [] for i in range(_N)}
    for k, idx in enumerate(aps):
        for p in idx:
            pos_to_aps[p].append(k)
    return pos_to_aps

def _dft_proxy_optimize(x, steps=1000, lr=0.05):
    """
    Optimize continuous x to minimize spectral peaks.
    x: numpy array of shape (N,)
    """
    n = len(x)
    # We want to minimize the max magnitude of DFT coefficients.
    # Use a smooth approximation: sum of (|X_k|^2 - T)^2 for |X_k| > T?
    # Or simply minimize the sum of squares of the top M coefficients?
    # Let's minimize sum( X_k^4 ) to push energy down from peaks.
    
    for _ in range(steps):
        X = np.fft.fft(x)
        X_mag_sq = np.abs(X)**2
        # Gradient of sum(X_k^4) wrt x?
        # X = F x. Objective J = sum |X_k|^4 = sum X_k^2 X_k*^2.
        # dJ/dx = 2 * F^H ( 2 * X * (|X|^2) ) ? 
        # Let's use a simpler heuristic: gradient descent on sum(X_k^2) is constant.
        # We need to flatten the spectrum.
        # Minimize sum( (|X_k| - mean(|X|))^2 )?
        
        # Alternative: Use the fact that we want low discrepancy.
        # Let's just do a few steps of gradient descent on a proxy like
        # sum( max(0, |X_k| - threshold)^2 )
        
        # For speed and simplicity, let's use a fixed number of iterations
        # and a simple update rule.
        
        # Gradient of sum(|X|^4) is 4 * F^H ( X * |X|^2 )
        # We want to minimize this.
        grad = 4.0 * np.fft.ifft(X * (np.abs(X)**2)).real
        x = x - lr * grad
        
        # Project to [-1, 1]
        x = np.clip(x, -1.0, 1.0)
        
    return x

def construct():
    t0 = time.time()
    aps = _get_aps()
    K = len(aps)
    pos_to_aps = _build_pos_to_aps(aps)
    
    # 1. DFT Proxy Initialization
    rng = np.random.RandomState(42)
    x = rng.uniform(-1, 1, _N)
    
    # Run DFT optimization
    # Keep it short to leave time for local search
    time_budget_dft = 5.0
    steps = int(100 * (time_budget_dft / 0.01)) # Rough estimate, cap it
    steps = min(steps, 500)
    x = _dft_proxy_optimize(x, steps=steps, lr=0.02)
    
    # Project to signs
    s = np.where(x > 0, 1.0, -1.0).astype(np.float64)
    
    # 2. Calculate initial AP sums
    ap_sums = np.zeros(K, dtype=np.float64)
    for k, idx in enumerate(aps):
        ap_sums[k] = np.sum(s[idx])
    
    best_s = s.copy()
    best_val = np.max(np.abs(ap_sums))
    
    def remaining():
        return _DEADLINE - (time.time() - t0)

    # 3. Local Search with Incremental Updates
    # We maintain ap_sums.
    # To find best flip:
    # 1. Find worst AP index W.
    # 2. For each p in aps[W]:
    #    a. Flip s[p].
    #    b. Update ap_sums for all APs containing p.
    #    c. Compute new_max = max(|ap_sums|).
    #    d. Unflip s[p] and revert ap_sums.
    #    e. If new_max < current_max, record as best candidate.
    # 3. Apply best flip.
    
    stuck = 0
    while remaining() > 2.0:
        current_max = np.max(np.abs(ap_sums))
        worst_idx = int(np.argmax(np.abs(ap_sums)))
        worst_ap = aps[worst_idx]
        
        best_gain = 0.0
        best_pos = None
        best_new_max = current_max
        
        # Try flipping each position in the worst AP
        for p in worst_ap:
            # Flip
            old_sign = s[p]
            s[p] = -old_sign
            
            # Update sums for APs containing p
            affected = pos_to_aps[p]
            old_sums = []
            for k in affected:
                old_sums.append(ap_sums[k])
                ap_sums[k] += -2.0 * old_sign # Change is -2 * old_sign? 
                # sum_new = sum_old - old_sign + new_sign = sum_old - old_sign - old_sign = sum_old - 2*old_sign. Correct.
            
            # Check new max
            # Optimization: We only need to check if the max decreased.
            # The max can only change if one of the affected APs was the max or becomes the new max.
            # Simple check: np.max(np.abs(ap_sums))
            new_max = np.max(np.abs(ap_sums))
            
            # Revert
            for k, old_sum in zip(affected, old_sums):
                ap_sums[k] = old_sum
            s[p] = old_sign
            
            if new_max < best_new_max:
                best_new_max = new_max
                best_gain = current_max - new_max
                best_pos = p
        
        if best_pos is not None and best_gain > 0:
            # Apply best flip
            old_sign = s[best_pos]
            s[best_pos] = -old_sign
            affected = pos_to_aps[best_pos]
            for k in affected:
                ap_sums[k] += -2.0 * old_sign
            
            if best_new_max < best_val:
                best_val = best_new_max
                best_s = s.copy()
            stuck = 0
        else:
            stuck += 1
            # Random kick
            if stuck > 5:
                # Flip 3 random positions
                for _ in range(3):
                    p = rng.randint(_N)
                    old_sign = s[p]
                    s[p] = -old_sign
                    affected = pos_to_aps[p]
                    for k in affected:
                        ap_sums[k] += -2.0 * old_sign
                # Check if improved
                curr_max = np.max(np.abs(ap_sums))
                if curr_max < best_val:
                    best_val = curr_max
                    best_s = s.copy()
                    stuck = 0
                else:
                    # Revert kick? Or keep? Let's keep to explore.
                    # But if it's worse, maybe revert to avoid drift?
                    # Let's just increment stuck.
                    pass
            if stuck > 20:
                # Restart from best
                s = best_s.copy()
                ap_sums = np.zeros(K, dtype=np.float64)
                for k, idx in enumerate(aps):
                    ap_sums[k] = np.sum(s[idx])
                stuck = 0

    out = np.where(best_s > 0, 1, -1)
    return [int(x) for x in out]
# EVOLVE-BLOCK-END
