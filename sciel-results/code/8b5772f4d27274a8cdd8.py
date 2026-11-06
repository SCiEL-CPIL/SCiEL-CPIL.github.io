import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    n = 128
    
    # Precompute all 4-APs
    aps = []
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            p1 = a
            p2 = a + d
            p3 = a + 2 * d
            p4 = a + 3 * d
            if p4 <= n:
                aps.append((p1 - 1, p2 - 1, p3 - 1, p4 - 1)) # 0-indexed
    
    # For each position, list the indices of APs it belongs to
    pos_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(i)
            
    def is_valid_assignment(colors):
        # Check all APs
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            distinct = {c1, c2, c3, c4}
            if len(distinct) < 3:
                return False
        return True

    def solve_with_k(k):
        colors = [-1] * n
        
        # Heuristic: Try to fix first few positions to break symmetry
        # This is optional but can speed up search.
        # We'll just rely on randomization.
        
        sys.setrecursionlimit(2000)
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            
            if idx == n:
                return True
                
            # Try colors in random order
            possible_colors = list(range(k))
            random.shuffle(possible_colors)
            
            for c in possible_colors:
                colors[idx] = c
                
                # Check constraints involving idx where other 3 are assigned
                valid = True
                for ap_idx in pos_aps[idx]:
                    ap = aps[ap_idx]
                    # Check if other 3 are assigned
                    others = [p for p in ap if p != idx]
                    if all(colors[p] != -1 for p in others):
                        vals = [colors[p] for p in ap]
                        if len(set(vals)) < 3:
                            valid = False
                            break
                            
                if valid:
                    if backtrack(idx + 1):
                        return True
                else:
                    colors[idx] = -1
                    continue
            
            colors[idx] = -1
            return False
        
        success = backtrack(0)
        if success:
            return colors[:]
        return None

    # Try k=3, 4, 5
    for k in [3, 4, 5]:
        if time.time() > deadline - 5:
            break
        res = solve_with_k(k)
        if res is not None:
            if is_valid_assignment(res):
                return res
                
    # Fallback: Return a valid coloring with many colors
    # c_i = i % 128 ensures all 4 terms in any AP are distinct?
    # No, if d is multiple of 128, but d < n/3, so d is small.
    # If c_i = i, then c_a, c_{a+d}, c_{a+2d}, c_{a+3d} are distinct.
    # So c_i = i is valid.
    colors = list(range(128))
    return colors
