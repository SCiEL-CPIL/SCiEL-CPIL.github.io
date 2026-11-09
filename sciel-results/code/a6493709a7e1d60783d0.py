import time
import numpy as np

# Precompute all nonconstant 4-term progressions (a, a+d, a+2d, a+3d), 1-indexed a,d
# Convert to 0-indexed internally.
_ALL_APS = []
for a in range(1, 125):  # a+3d <= 128 => a <= 128-3
    max_d = (128 - a) // 3
    for d in range(1, max_d + 1):
        _ALL_APS.append((a-1, a+d-1, a+2*d-1, a+3*d-1))

def _progression_violated(colors, idx_set):
    # Determine if this progression is already IMPOSSIBLE to satisfy
    present = set()
    unassigned = 0
    for i in idx_set:
        c = colors[i]
        if c == -1:
            unassigned += 1
        else:
            present.add(c)
    if len(present) >= 3:
        return False  # already satisfied
    # Need to end with >=3 distinct colors
    if len(present) + unassigned < 3:
        return True  # impossible
    return False

def construct():
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s

    n = 128
    # Precompute for each position, which progressions it participates in (as a member)
    pos_in_aps = [[] for _ in range(n)]
    for ap in _ALL_APS:
        for i in ap:
            pos_in_aps[i].append(ap)

    best = None
    best_ncolors = n  # start with worst

    # Try increasing number of colors
    for k_colors in range(3, 6):
        if time.time() > DEADLINE:
            break
        
        # Backtracking search with MRV and LCV
        colors = [-1] * n
        unassigned = set(range(n))
        
        solution_found = False
        node_count = 0

        def get_allowed_colors(pos):
            al = []
            for c in range(k_colors):
                valid = True
                for ap in pos_in_aps[pos]:
                    # Check if assigning c to pos makes this AP impossible
                    # We need to check if the AP can still reach 3 colors
                    present = set()
                    unassigned_others = 0
                    for i in ap:
                        if i == pos:
                            present.add(c)
                        elif colors[i] != -1:
                            present.add(colors[i])
                        else:
                            unassigned_others += 1
                    
                    # If even with max possible new colors we can't reach 3, it's invalid
                    # Max new colors from unassigned_others is unassigned_others
                    # But they might duplicate existing colors.
                    # The condition "impossible" is len(present) + unassigned_others < 3
                    if len(present) + unassigned_others < 3:
                        valid = False
                        break
                if valid:
                    al.append(c)
            return al

        def backtrack():
            nonlocal solution_found, node_count
            if time.time() > DEADLINE:
                return False
            if solution_found:
                return True
            
            if not unassigned:
                # All assigned; verify all APs
                for ap in _ALL_APS:
                    if len({colors[i] for i in ap}) < 3:
                        return False
                solution_found = True
                return True
            
            node_count += 1
            if node_count % 1000 == 0 and time.time() > DEADLINE:
                return False

            # MRV: Select variable with fewest allowed colors
            min_len = k_colors + 1
            best_pos = -1
            best_allowed = []
            
            # To speed up MRV, we might not want to check all unassigned if we find a small one quickly
            # But checking all is O(N * Cost). N=128.
            for pos in unassigned:
                al = get_allowed_colors(pos)
                if len(al) < min_len:
                    min_len = len(al)
                    best_pos = pos
                    best_allowed = al
                    if min_len == 0:
                        break
                    if min_len == 1:
                        break

            if min_len == 0:
                return False

            # LCV: Order colors by least constraining value
            # For each color c in best_allowed, count how many options it leaves for neighbors
            # A simpler proxy: count how many APs become "tight" or "impossible" for neighbors
            # Or just random/shuffle if LCV is too slow. Let's try a simple LCV:
            # For each c, sum over neighbors of len(allowed_neighbor_given_c)
            
            def count_constraints(c):
                # Count how much this choice reduces options for neighbors
                # A high count means it constrains a lot (bad)
                # A low count means it constrains little (good)
                # We want to try low constraint first.
                # So we sort by count_constraints ASCENDING.
                
                # We only consider neighbors that are unassigned and in APs with pos
                neighbors = set()
                for ap in pos_in_aps[best_pos]:
                    for i in ap:
                        if i != best_pos and i in unassigned:
                            neighbors.add(i)
                
                total_constraint = 0
                for nb in neighbors:
                    # Calculate allowed colors for nb if best_pos takes c
                    # This is expensive.
                    # Approximation: just check if nb becomes 0 allowed?
                    # Or just use a static order?
                    # Given time limits, a simple heuristic might be better than full LCV if LCV is slow.
                    # Let's stick to a simpler LCV: prioritize colors that don't immediately block neighbors.
                    pass
                return 0 # Placeholder

            # Given the complexity of exact LCV, let's just use a random shuffle or static order for colors
            # to avoid overhead, as MRV is the most important part.
            # Or, we can just iterate best_allowed as is (0, 1, 2...).
            
            # Actually, let's try a very light LCV:
            # If a color c makes any neighbor have 0 options, it's bad.
            # We'll just iterate best_allowed.
            
            for c in best_allowed:
                colors[best_pos] = c
                unassigned.remove(best_pos)
                
                # Forward Checking: Check if any neighbor is now impossible
                # We don't need to check all unassigned, just neighbors of best_pos
                # But MRV in the next call will handle it.
                # However, if a neighbor has 0 options, we should backtrack immediately.
                # Let's let the next MRV call find it (it will find min_len=0).
                
                if backtrack():
                    return True
                
                unassigned.add(best_pos)
                colors[best_pos] = -1
            
            return False

        if backtrack():
            # Extract used colors
            used = sorted(set(c for c in colors if c != -1))
            best = colors[:]
            best_ncolors = len(used)
            break
        else:
            # Didn't find with k_colors within time; try more colors
            pass

    if best is not None:
        # Ensure all are valid non-negative and within 0..127
        return best
    else:
        # Fallback: use seed-like greedy but with more colors, or a simple pattern
        # Fallback: color by i mod 3 gives 3 colors but may not satisfy. 
        # Safer fallback: assign each position a unique color (128 colors) -> always satisfies.
        # But that's many colors. Better: try a quick greedy with many colors.
        colors = []
        for x in range(1, 129):
            for color in range(128):
                if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                        for d in range(1, (x-1)//3 + 1) if x-1-3*d >= 0)
                    and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                            for d in range(1, (x-1)//2 + 1) if x-1-2*d >= 0)):
                    colors.append(color)
                    break
            else:
                colors.append(0)
        return colors
