import random
import math
import time

def construct():
    n = 128
    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, n // 3 + 1):
        for i in range(1, n - 3 * d + 1):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # Map position to list of AP indices it is part of
    pos_to_aps = [[] for _ in range(n + 1)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    def count_violations(colors):
        """colors is 1-indexed list of length n+1"""
        v = 0
        for i, j, k, l in aps:
            c_set = {colors[i], colors[j], colors[k], colors[l]}
            if len(c_set) < 3:
                v += 1
        return v

    def get_violation_count_for_ap(ap, colors):
        c_set = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
        return 1 if len(c_set) < 3 else 0

    def find_coloring(k, max_iters=20000, initial_temp=1.0, cooling=0.99):
        """
        Try to find a valid k-coloring using Simulated Annealing.
        Returns None if not found within max_iters.
        """
        # Initialize with random colors
        colors = [0] * (n + 1)
        for i in range(1, n + 1):
            colors[i] = random.randint(0, k - 1)
        
        current_v = count_violations(colors)
        best_v = current_v
        best_colors = colors[:]
        
        temp = initial_temp
        start_time = time.time()
        time_limit = 5.0 # Allow 5 seconds per k value
        
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random position to change
            pos = random.randint(1, n)
            old_color = colors[pos]
            
            # Pick a new color different from old
            if k > 1:
                new_color = random.randint(0, k - 1)
                while new_color == old_color:
                    new_color = random.randint(0, k - 1)
            else:
                continue # k=1 is trivially invalid for n>=4
            
            # Calculate delta energy
            # Only APs containing 'pos' are affected
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Energy before
                old_v = get_violation_count_for_ap(ap, colors)
                
                # Temporarily change
                colors[pos] = new_color
                new_v = get_violation_count_for_ap(ap, colors)
                
                delta += (new_v - old_v)
            
            # Accept or reject
            if delta <= 0 or random.random() < math.exp(-delta / temp):
                colors[pos] = new_color
                current_v += delta
                if current_v < best_v:
                    best_v = current_v
                    best_colors = colors[:]
                    if best_v == 0:
                        break
            else:
                colors[pos] = old_color
            
            temp *= cooling
            if temp < 1e-10:
                break

        if best_v == 0:
            return best_colors
        return None

    # Try k from 3 upwards
    # We expect k=3 or 4 to be sufficient
    for k in range(3, 11):
        # Run multiple restarts for each k
        found = None
        restarts = 5
        for r in range(restarts):
            random.seed(r) # Reproducibility per restart
            found = find_coloring(k)
            if found:
                break
        
        if found:
            # Compress colors to 0..m-1
            used_colors = sorted(set(found[1:]))
            mapping = {c: i for i, c in enumerate(used_colors)}
            result = [0] * (n + 1)
            for i in range(1, n + 1):
                result[i] = mapping[found[i]]
            
            # Return 0-indexed list for positions 1..128
            return result[1:]

    # Fallback: return a valid coloring from greedy if SA fails (unlikely)
    # Or just return a random valid one if we found one with higher k
    # The loop above should find one.
    # If not, we return the best we have, which might be invalid? 
    # No, the task requires valid. 
    # If SA fails, we might need a more robust solver.
    # But for n=128, SA should work.
    
    # Fallback to greedy to ensure validity if SA fails completely
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                # Check 4-APs ending at x
                # x, x-d, x-2d, x-3d
                if x - 3 * d >= 1:
                    c1 = colors[x - 1 - 3 * d]
                    c2 = colors[x - 1 - 2 * d]
                    c3 = colors[x - 1 - d]
                    c4 = color
                    if len({c1, c2, c3, c4}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
    return colors

