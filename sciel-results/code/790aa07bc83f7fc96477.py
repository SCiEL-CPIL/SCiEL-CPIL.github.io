import time
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) with 1-based indexing for logic
    # We will use 0-based indexing for the array `colors`
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position (1-based) to list of AP indices containing it
    # We will use 0-based indices in the `colors` array, so pos-1
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos - 1].append(idx)

    def check_ap(ap_idx, colors):
        """
        Check if the AP at ap_idx can still be satisfied given current colors.
        Returns True if satisfiable, False otherwise.
        """
        a, b, c, d = aps[ap_idx]
        ca = colors[a-1]
        cb = colors[b-1]
        cc = colors[c-1]
        cd = colors[d-1]
        
        assigned_colors = []
        unassigned_count = 0
        for val in (ca, cb, cc, cd):
            if val != -1:
                assigned_colors.append(val)
            else:
                unassigned_count += 1
        
        distinct_assigned = set(assigned_colors)
        num_distinct = len(distinct_assigned)
        
        # We need at least 3 distinct colors in total.
        # If we already have >= 3, it's fine.
        if num_distinct >= 3:
            return True
        
        # If we have < 3, we need the unassigned positions to provide the missing distinct colors.
        missing = 3 - num_distinct
        
        # We need at least 'missing' unassigned positions to be assigned distinct colors not in distinct_assigned.
        if unassigned_count < missing:
            return False
        
        return True

    def solve(k, colors, assigned_count):
        """
        Backtracking solver with MRV heuristic.
        k: max colors allowed (0..k-1)
        colors: list of length N, -1 for unassigned
        assigned_count: number of positions already assigned
        """
        if time.time() > deadline:
            return None
        
        if assigned_count == N:
            return colors[:]
        
        # Find the unassigned position with the most constraints (MRV-like)
        # We'll use a simple heuristic: pick the first unassigned position
        # that has the most APs to check
        best_pos = -1
        best_constraint_count = -1
        
        for pos in range(N):
            if colors[pos] == -1:
                constraint_count = len(ap_indices[pos])
                # Prefer positions with more constraints (more likely to fail early)
                if constraint_count > best_constraint_count:
                    best_constraint_count = constraint_count
                    best_pos = pos
        
        if best_pos == -1:
            # All assigned
            return colors[:]
        
        # Symmetry breaking: limit the color values we can assign
        # Count how many distinct colors have been used so far
        used_colors = set()
        for c in colors:
            if c != -1:
                used_colors.add(c)
        max_color_used = max(used_colors) if used_colors else -1
        
        # The new color can be at most max_color_used + 1 (to avoid symmetry)
        max_val = min(k - 1, max_color_used + 1)
        
        for v in range(max_val + 1):
            colors[best_pos] = v
            
            # Check constraints for all APs containing best_pos
            valid = True
            for ap_idx in ap_indices[best_pos]:
                if not check_ap(ap_idx, colors):
                    valid = False
                    break
            
            if valid:
                # Recurse
                result = solve(k, colors, assigned_count + 1)
                if result is not None:
                    return result
            
            colors[best_pos] = -1
        
        return None

    # Try k from 3 upwards
    # We expect k=3 or k=4 to work.
    for k in range(3, 129):
        colors = [-1] * N
        # Fix first position to 0 for symmetry breaking
        colors[0] = 0
        
        # Check time before starting search
        if time.time() > deadline:
            break
            
        result = solve(k, colors, 1) # Start with 1 position already assigned
        
        if result is not None:
            # Ensure we return a list of integers
            # The colors are 0..k-1
            # We should remap the colors to be consecutive 0..m-1 where m is the number of distinct colors used
            unique_colors = sorted(set(result))
            color_map = {c: i for i, c in enumerate(unique_colors)}
            remapped = [color_map[c] for c in result]
            return remapped
        
        # If k=3 failed, try k=4, etc.
        
    # Fallback: use the seed's greedy approach if we haven't found a solution
    # This is extremely unlikely to be reached if k=3 or 4 works, but good for safety.
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
