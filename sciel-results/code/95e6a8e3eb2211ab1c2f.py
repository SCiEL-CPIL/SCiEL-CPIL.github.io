import numpy as np
import time

def construct():
    N = 128
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    
    def is_bad(c0, c1, c2, c3):
        return len({c0, c1, c2, c3}) < 3

    def calculate_cost(colors):
        cost = 0
        for ap in aps:
            if is_bad(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                cost += 1
        return cost

    start_time = time.time()
    deadline = start_time + 85.0

    # Systematic search for smallest modular m
    best_colors = None
    best_m = 128
    
    for m in range(3, 128):
        if time.time() > deadline - 10: # Leave time for merging
            break
        
        # Check if i % m is valid
        colors = np.array([i % m for i in range(N)], dtype=np.int64)
        if calculate_cost(colors) == 0:
            best_colors = colors
            best_m = m
            break # Found smallest valid m, try to optimize it

    if best_colors is None:
        # Fallback to random if no modular solution found (should not happen for m=43)
        colors = np.random.randint(0, 128, N)
        return colors.tolist()

    # Try to reduce colors by merging
    # current colors are in 0..best_m-1
    # We want to map some colors to others
    # Greedy: try to merge color j into color i
    
    current_colors = best_colors.copy()
    num_used = best_m
    
    # Optimization: 
    # If we merge j into i, we update current_colors where current_colors==j to i.
    # Then check validity.
    # To speed up, we can just re-check cost.
    # Since N=128 and num_aps is ~1000, cost calculation is fast.
    
    merged = [i for i in range(best_m)]
    # merged[c] is the new color for c
    
    # Sort colors by frequency? Or just iterate.
    # We want to minimize the number of distinct labels in merged.
    
    for i in range(best_m):
        if time.time() > deadline - 2:
            break
        for j in range(i + 1, best_m):
            if time.time() > deadline - 2:
                break
            if merged[j] == i:
                continue
            if merged[i] != i: # i was merged into something else? 
                # This logic is tricky if we do it in place.
                # Let's keep a map: target_color -> list of original colors
                # Or simpler: 
                # Try to change all occurrences of color j to color i.
                # Save old state, apply, check, revert if bad.
                pass
            
            # Apply merge j -> i
            mask = (current_colors == j)
            if not np.any(mask):
                continue # j not used? (should be used if m <= N)
                
            old_vals = current_colors[mask].copy()
            current_colors[mask] = i
            
            # Check validity
            if calculate_cost(current_colors) == 0:
                # Keep merge
                # Update tracking: j is now i
                # We don't strictly need to track if we just look at current_colors
                # But we need to know which colors are "active" to try merging them.
                # Actually, if we just iterate i, j over 0..m-1, and modify current_colors,
                # we might try to merge a color that is already gone.
                # That's fine, mask will be empty.
                pass
            else:
                # Revert
                current_colors[mask] = old_vals

    # Final count of distinct colors
    unique_colors = np.unique(current_colors)
    
    # The task requires labels in 0..127.
    # Our labels are 0..best_m-1, which is subset of 0..127.
    # We return the list.
    
    return current_colors.tolist()
