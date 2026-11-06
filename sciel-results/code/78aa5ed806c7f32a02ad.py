import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    N = 128
    
    # Precompute APs: for each position x (0-indexed), list of (i, j, k) 
    # where i, j, k, x form a 4-AP: x = i + 3d, j = i+d, k = i+2d
    # i.e., positions i, i+d, i+2d, i+3d = x
    aps_ending = [[] for _ in range(N)]
    for x in range(N):
        for d in range(1, x // 3 + 1):
            i = x - 3 * d
            if i >= 0:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))
    
    # Also precompute for each position, which APs it participates in (for forward checking)
    # aps_containing[x] = list of (other_positions_tuple) 
    # Actually, let's just use the ending-based approach with forward checking
    
    def solve_k(k, time_limit):
        """Try to find a valid coloring with k colors using backtracking."""
        colors = [-1] * N  # -1 means unassigned
        colors[0] = 0  # Symmetry breaking
        
        # For forward checking: for each unassigned position, track which colors are still possible
        # We'll compute this on the fly
        
        def get_valid_colors(pos, k):
            """Return set of valid colors for position pos given current assignment."""
            valid = set(range(k))
            for (i, j, kk) in aps_ending[pos]:
                ci = colors[i]
                cj = colors[j]
                ck = colors[kk]
                if ci == -1 or cj == -1 or ck == -1:
                    continue
                # For each candidate color c, check if {ci, cj, ck, c} has >= 3 distinct
                for c in list(valid):
                    if len({ci, cj, ck, c}) < 3:
                        valid.discard(c)
            return valid
        
        # MRV heuristic: pick the unassigned position with fewest valid colors
        def select_variable():
            min_count = k + 1
            min_pos = -1
            for pos in range(1, N):
                if colors[pos] == -1:
                    vc = get_valid_colors(pos, k)
                    if len(vc) < min_count:
                        min_count = len(vc)
                        min_pos = pos
                        if min_count == 0:
                            return min_pos, set()
                        if min_count == 1:
                            # Can break early if we find a forced assignment
                            pass
            if min_pos == -1:
                return -1, set(range(k))
            return min_pos, get_valid_colors(min_pos, k)
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            # Check if all assigned
            all_assigned = True
            for pos in range(N):
                if colors[pos] == -1:
                    all_assigned = False
                    break
            if all_assigned:
                return colors[:]
            
            pos, valid_colors = select_variable()
            if pos == -1:
                return colors[:]
            
            if len(valid_colors) == 0:
                return None
            
            # Order colors: try less-constraining ones first? Or just try in order
            for c in sorted(valid_colors):
                colors[pos] = c
                res = backtrack()
                if res is not None:
                    return res
                colors[pos] = -1
            
            return None
        
        return backtrack()
    
    # Try k=3 first
    sol = solve_k(3, 40.0)
    if sol is not None:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
    
    # Try k=4
    sol = solve_k(4, 80.0)
    if sol is not None:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]
    
    # Fallback: use the seed greedy approach
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 
                    for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) 
                        for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors
