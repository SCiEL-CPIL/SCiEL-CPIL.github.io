import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 128
    colors_limit = 3
    
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
            
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for pos in ap:
            pos_to_aps[pos].append(idx)
            
    colors = [-1] * n
    assigned = [False] * n
    
    def get_valid_colors(pos):
        """Return list of colors that do not cause immediate conflict."""
        valid = []
        for c in range(3):
            colors[pos] = c
            conflict = False
            for ap_idx in pos_to_aps[pos]:
                a, b, c_t, d_t = ap_list[ap_idx]
                vals = [colors[a], colors[b], colors[c_t], colors[d_t]]
                assigned_vals = [v for v in vals if v != -1]
                if len(assigned_vals) == 4:
                    if len(set(assigned_vals)) < 3:
                        conflict = True
                        break
            if not conflict:
                valid.append(c)
            colors[pos] = -1
        return valid

    def backtrack(assigned_count):
        if time.time() > deadline:
            return False
        
        if assigned_count == n:
            return True
        
        # Find unassigned variable with smallest domain (MRV)
        min_domain_size = 4
        best_pos = -1
        
        for pos in range(n):
            if assigned[pos]:
                continue
            
            # Estimate domain size
            # We can just count valid colors
            valid_colors = get_valid_colors(pos)
            if len(valid_colors) == 0:
                return False # Dead end
            
            if len(valid_colors) < min_domain_size:
                min_domain_size = len(valid_colors)
                best_pos = pos
                if min_domain_size == 1:
                    break
        
        if best_pos == -1:
            return False
            
        # Try colors in best_pos
        # Order: Least Constraining Value (LCV)
        # We can sort valid_colors by how many options they leave for neighbors.
        # For simplicity, just try them.
        
        valid_colors = get_valid_colors(best_pos)
        
        # Sort by constraint impact (optional optimization)
        # For now, just iterate
        for c in valid_colors:
            colors[best_pos] = c
            assigned[best_pos] = True
            
            if backtrack(assigned_count + 1):
                return True
            
            colors[best_pos] = -1
            assigned[best_pos] = False
            
        return False

    if backtrack(0):
        return colors
        
    return [0] * n
