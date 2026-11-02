import numpy as np
import time
from itertools import combinations

N = 50

def score_sparse(h_indices, h_vals):
    """
    Calculate score for sparse vector defined by indices and values.
    h_vals are normalized such that sum is 1? No, keep general.
    """
    if not h_vals:
        return 0.0
    
    l1 = sum(h_vals)
    if l1 == 0:
        return 0.0
        
    l2_sq = sum(v*v for v in h_vals)
    
    # Compute convolution peak
    # Map index to value
    # Since K is small, we can just iterate over pairs
    # conv[k] = sum_{i+j=k} h_i h_j
    # We only care about the max.
    
    # Build a dict for fast lookup? Or just list of (idx, val)
    items = list(zip(h_indices, h_vals))
    
    max_conv = 0.0
    # The possible sums are 2*idx for each idx, and idx1+idx2 for pairs
    # We can just compute all pair sums
    n = len(items)
    for i in range(n):
        idx1, val1 = items[i]
        for j in range(i, n): # i<=j to handle symmetry, but need to double off-diagonal
            idx2, val2 = items[j]
            conv_val = val1 * val2
            if i != j:
                conv_val *= 2.0
            # This conv_val contributes to position idx1+idx2
            # We just need the global max
            if conv_val > max_conv:
                max_conv = conv_val
                
    # Note: This logic assumes that the max convolution value is simply the max of the individual pair products (weighted by 2 if i!=j).
    # Is it possible that two pairs contribute to the SAME index, summing up?
    # Yes! If i1+j1 == i2+j2, then the convolution at that index is the sum.
    # Example: indices 0, 1, 2.
    # Pairs: (0,0)->0, (0,1)->1, (1,1)->2, (0,2)->2, (1,2)->3, (2,2)->4.
    # At index 2, we have (1,1) and (0,2).
    # So we MUST accumulate sums per index.
    
    conv_dict = {}
    for i in range(n):
        idx1, val1 = items[i]
        for j in range(i, n):
            idx2, val2 = items[j]
            target_idx = idx1 + idx2
            conv_val = val1 * val2
            if i != j:
                conv_val *= 2.0
            if target_idx in conv_dict:
                conv_dict[target_idx] += conv_val
            else:
                conv_dict[target_idx] = conv_val
                
    max_conv = max(conv_dict.values()) if conv_dict else 0.0
    
    if max_conv == 0:
        return 0.0
        
    return l2_sq / (l1 * max_conv)

def optimize_amplitudes_fast(support_indices):
    K = len(support_indices)
    if K == 0:
        return 0.0, []
    if K == 1:
        return 1.0, [1.0]
    
    if K == 2:
        i, j = support_indices
        best_score = 0.0
        best_val = 0.5
        # Scan
        for a in np.linspace(0.01, 0.99, 100):
            b = 1.0 - a
            score = score_sparse([i, j], [a, b])
            if score > best_score:
                best_score = score
                best_val = a
        return best_score, [best_val, 1.0 - best_val]
        
    if K == 3:
        i, j, k = support_indices
        best_score = 0.0
        best_vals = [1/3, 1/3, 1/3]
        
        # Coarse grid
        # x+y+z=1
        for x in np.linspace(0.01, 0.98, 30):
            for y in np.linspace(0.01, 0.98 - x, max(2, int(30 * (0.98 - x)))):
                z = 1.0 - x - y
                if z < 0.001:
                    continue
                score = score_sparse([i, j, k], [x, y, z])
                if score > best_score:
                    best_score = score
                    best_vals = [x, y, z]
                    
        # Refine
        if best_score > 0:
            for _ in range(5):
                step = 0.01
                improved = False
                for idx in range(3):
                    for delta in [-step, step]:
                        new_vals = best_vals.copy()
                        new_vals[idx] += delta
                        # Renormalize
                        s = sum(new_vals)
                        if s <= 0 or min(new_vals) < -1e-6:
                            continue
                        new_vals = [max(v, 0) for v in new_vals]
                        s = sum(new_vals)
                        new_vals = [v/s for v in new_vals]
                        score = score_sparse([i, j, k], new_vals)
                        if score > best_score:
                            best_score = score
                            best_vals = new_vals
                            improved = True
                if not improved:
                    step /= 2
                if step < 1e-4:
                    break
                    
        return best_score, best_vals

    return 0.0, []

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_h = [0.0] * N
    
    # 1. 2-point supports
    for i in range(N):
        for j in range(i + 1, N):
            if time.time() > deadline:
                break
            score, vals = optimize_amplitudes_fast([i, j])
            if score > best_score:
                best_score = score
                h = [0.0] * N
                h[i] = vals[0]
                h[j] = vals[1]
                best_h = h
        if time.time() > deadline:
            break
            
    # 2. 3-point supports
    # To save time, we might skip some if 2-point is already very high?
    # But 3-point could be better.
    # We'll iterate.
    for combo in combinations(range(N), 3):
        if time.time() > deadline - 5.0:
            break
        score, vals = optimize_amplitudes_fast(list(combo))
        if score > best_score:
            best_score = score
            h = [0.0] * N
            h[combo[0]] = vals[0]
            h[combo[1]] = vals[1]
            h[combo[2]] = vals[2]
            best_h = h
            
    return best_h
