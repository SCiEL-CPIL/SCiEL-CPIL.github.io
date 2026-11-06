"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for safety
    n = 128
    
    def is_valid_coloring(colors):
        """Check if the coloring satisfies the constraint for all 4-term APs."""
        for a in range(n):
            for d in range(1, (n - 1 - a) // 3 + 1):
                i0 = a
                i1 = a + d
                i2 = a + 2*d
                i3 = a + 3*d
                if i3 >= n:
                    break
                distinct = len({colors[i0], colors[i1], colors[i2], colors[i3]})
                if distinct < 3:
                    return False
        return True

    def greedy_coloring(k, seed=0):
        """Attempt to color [0, n) using colors 0..k-1 using a greedy strategy."""
        random.seed(seed)
        colors = [-1] * n
        # To make it more robust, we can shuffle the order of colors tried or positions
        # But simple left-to-right greedy is fast.
        
        for pos in range(n):
            # Try colors in a random order to avoid systematic bias, 
            # but we want to minimize label count, so we prefer smaller labels.
            # However, for finding *a* valid coloring with k colors, 
            # random order helps escape local minima of greedy failure.
            # Since we want to minimize the *count* of labels used, 
            # and we are fixing the max label to k-1, we just need any valid coloring.
            # The "minimize number of labels actually used" is satisfied if we use a subset of 0..k-1.
            # Actually, if we use k colors, the count is <= k.
            # We iterate k from 3 upwards. The first valid k is the minimum.
            
            valid_colors = []
            for c in range(k):
                # Check constraints with already assigned neighbors
                ok = True
                # Case 1: pos is 4th element (pos-3d, pos-2d, pos-d, pos)
                for d in range(1, (pos + 1) // 3 + 1):
                    i0 = pos - 3*d
                    if i0 < 0: break
                    i1 = pos - 2*d
                    i2 = pos - d
                    if colors[i0] == -1 or colors[i1] == -1 or colors[i2] == -1:
                        continue
                    if len({colors[i0], colors[i1], colors[i2], c}) < 3:
                        ok = False
                        break
                if not ok:
                    continue
                
                # Case 2: pos is 3rd element (pos-2d, pos-d, pos, pos+d)
                # pos+d is unassigned, so we can't check the full AP yet.
                # However, we can check if the prefix (pos-2d, pos-d, pos) is "safe".
                # Actually, the constraint is on the full AP. 
                # If pos+d is assigned later, it will check the AP then.
                # So we only need to check APs where ALL other 3 are assigned.
                
                # Case 3: pos is 2nd element (pos-d, pos, pos+d, pos+2d)
                # pos+d, pos+2d unassigned. Skip.
                
                # Case 4: pos is 1st element (pos, pos+d, pos+2d, pos+3d)
                # All others unassigned. Skip.
                
                valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Pick a color. 
            # To minimize the actual number of distinct colors used in the final array,
            # we might prefer smaller colors, but since we are testing for feasibility with k,
            # any choice works. 
            # However, choosing the smallest valid color might lead to a coloring 
            # that uses fewer than k colors? No, we just need *some* valid coloring.
            # The objective is to minimize the number of labels *actually used*.
            # If we find a valid coloring with max label k-1, the number of labels used is <= k.
            # We want the smallest k such that a valid coloring exists.
            # So we just need *any* valid coloring for this k.
            colors[pos] = valid_colors[0] # Deterministic choice for reproducibility
            
        return colors

    # Try increasing k
    for k in range(3, 15):
        if time.time() > deadline:
            break
        
        # Try a few random seeds to escape greedy traps
        found = None
        for attempt in range(10):
            if time.time() > deadline:
                break
            res = greedy_coloring(k, seed=42 + attempt)
            if res is not None:
                if is_valid_coloring(res):
                    # Count distinct colors used
                    # The problem says "Minimize the number of labels actually used"
                    # If we use colors from 0..k-1, the count is the number of distinct values in res.
                    # We can remap them to 0..count-1 to be safe, but the labels just need to be in 0..127.
                    # The evaluator counts distinct labels.
                    found = res
                    break
        
        if found is not None:
            # Remap colors to 0..m-1 where m is the number of distinct colors used
            # This ensures we minimize the count if we used a sparse subset of 0..k-1
            distinct_vals = sorted(list(set(found)))
            mapping = {v: i for i, v in enumerate(distinct_vals)}
            remapped = [mapping[c] for c in found]
            return remapped

    # Fallback: return a valid coloring with many colors (e.g., all distinct)
    # This is a high score but better than timeout.
    return list(range(n))

# EVOLVE-BLOCK-END
