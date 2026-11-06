import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    N = 256
    valid_vertices = []
    for v in range(1, N + 1):
        s = 2 * v
        r = int(math.isqrt(s))
        if r * r != s:
            valid_vertices.append(v)
            
    n = len(valid_vertices)
    if n == 0:
        return []
        
    val_to_idx = {v: i for i, v in enumerate(valid_vertices)}
    
    # Build conflict graph (edges if sum is square)
    # And compatibility graph (edges if sum is NOT square)
    conflict_adj = [0] * n
    compat_adj = [0] * n
    
    for i in range(n):
        vi = valid_vertices[i]
        for j in range(i + 1, n):
            vj = valid_vertices[j]
            s = vi + vj
            r = int(math.isqrt(s))
            is_sq = (r * r == s)
            if is_sq:
                # Conflict
                conflict_adj[i] |= (1 << j)
                conflict_adj[j] |= (1 << i)
            else:
                # Compatible
                compat_adj[i] |= (1 << j)
                compat_adj[j] |= (1 << i)
                
    # Initial Greedy Solution
    # Sort by conflict degree (ascending)
    degs = [conflict_adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: degs[x])
    
    current_set = []
    available = (1 << n) - 1
    for v in order:
        if available & (1 << v):
            # Check if v is compatible with all in current_set?
            # Since we picked low conflict degree, they might still conflict.
            # We need to ensure v is compatible with all existing.
            ok = True
            for u in current_set:
                if conflict_adj[v] & (1 << u):
                    ok = False
                    break
            if ok:
                current_set.append(v)
                # Remove v and its conflicts from available
                available &= ~(conflict_adj[v] | (1 << v))
                
    best_size = len(current_set)
    best_clique = current_set[:]
    
    def time_up():
        return time.time() > deadline

    def color_sort(cand):
        if cand == 0:
            return [], []
        
        verts = []
        c = cand
        while c:
            lsb = c & -c
            idx = lsb.bit_length() - 1
            verts.append(idx)
            c ^= lsb
            
        # Sort by degree in cand (compat degree) descending?
        # For coloring the compat graph, high degree vertices are constrained more.
        # Usually, coloring high degree first leads to fewer colors?
        verts.sort(key=lambda v: (compat_adj[v] & cand).bit_count(), reverse=True)
        
        color_of = {}
        max_color = 0
        for v in verts:
            neighbors_in_cand = compat_adj[v] & cand
            used_colors = set()
            nc = neighbors_in_cand
            while nc:
                lsb = nc & -nc
                nidx = lsb.bit_length() - 1
                if nidx in color_of:
                    used_colors.add(color_of[nidx])
                nc ^= lsb
                
            k = 1
            while k in used_colors:
                k += 1
            color_of[v] = k
            if k > max_color:
                max_color = k
                
        sorted_verts = sorted(verts, key=lambda v: color_of[v], reverse=True)
        sorted_colors = [color_of[v] for v in sorted_verts]
        
        return sorted_verts, sorted_colors

    def expand(cand, current_clique, current_size):
        nonlocal best_size, best_clique
        
        if time_up():
            return
            
        if cand == 0:
            if current_size > best_size:
                best_size = current_size
                best_clique = current_clique[:]
            return
            
        sorted_verts, sorted_colors = color_sort(cand)
        
        if not sorted_verts:
            return

        if current_size + sorted_colors[0] <= best_size:
            return
            
        for i, v in enumerate(sorted_verts):
            c_i = sorted_colors[i]
            
            if current_size + c_i <= best_size:
                break
                
            new_clique = current_clique + [v]
            
            remaining_mask = 0
            for k in range(i + 1, len(sorted_verts)):
                remaining_mask |= (1 << sorted_verts[k])
                
            new_cand = cand & compat_adj[v] & remaining_mask
            
            expand(new_cand, new_clique, current_size + 1)

    initial_cand = (1 << n) - 1
    expand(initial_cand, [], 0)
    
    if not best_clique:
        if valid_vertices:
            return [valid_vertices[0]]
        return []
        
    result = [valid_vertices[idx] for idx in best_clique]
    return sorted(result)
