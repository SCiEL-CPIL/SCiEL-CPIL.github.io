# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute N(i) & N(j) for all pairs to check triangle existence quickly
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
            
    def dfs(available, current_mask, current_size):
        nonlocal best
        # Pruning
        if current_size + available.bit_count() <= best:
            return
            
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Heuristic: pick a vertex with high degree in available for better pruning
        best_v = -1
        best_deg = -1
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj_bitmasks[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                best_v_bit = v_bit
            temp -= v_bit
        
        if best_v < 0:
            return
        
        v = best_v
        v_bit = best_v_bit
        
        # Option 1: Include v
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            temp_check = N_v_in_cur
            while temp_check:
                u_bit = temp_check & -temp_check
                u = u_bit.bit_length() - 1
                if adj_bitmasks[u] & N_v_in_cur:
                    safe_to_add = False
                    break
                temp_check -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            N_v_in_new = adj_bitmasks[v] & new_mask
            forbidden = 0
            temp_w = N_v_in_new
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                forbidden |= adj_bitmasks[w]
                temp_w -= w_bit
            
            forbidden &= adj_bitmasks[v]
            forbidden &= available
            
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
            
        # Option 2: Exclude v
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _adj_to_bitmasks(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    bm = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        bm[i] = mask
    return bm

def _check_k4_free_bitmask(bm, n):
    """Check if graph represented by bitmasks is K4-free."""
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                common = bm[i] & bm[j]
                while common:
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        return False
                    common -= k_bit
    return True

def _can_add_edge(bm, u, v, n):
    """Check if adding edge (u,v) maintains K4-freeness."""
    if bm[u] & (1 << v):
        return True # Already exists
    common = bm[u] & bm[v]
    # If there are two common neighbors that are adjacent, adding (u,v) creates K4
    # Actually, if u,v have common neighbors a,b and a-b is an edge, then {u,v,a,b} is K4
    # So we need N(u) & N(v) to be an independent set.
    temp = common
    while temp:
        a_bit = temp & -temp
        a = a_bit.bit_length() - 1
        # Check if a is connected to any other common neighbor
        other_common = common & ~a_bit
        if bm[a] & other_common:
            return False
        temp -= a_bit
    return True

def _random_k4_free_graph(n, seed, density=0.5):
    """Generate a random K4-free graph using bitmask representation."""
    rng = random.Random(seed)
    bm = [0] * n
    
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for i, j in edges_to_try:
        if rng.random() < density:
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def _find_max_tf_set(bm, n):
    """Find one maximum triangle-free induced subgraph."""
    best_set = 0
    best_size = 0
    full_mask = (1 << n) - 1
    
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = bm[i] & bm[j]
            common[j][i] = common[i][j]
            
    def dfs(available, current_mask, current_size):
        nonlocal best_set, best_size
        if current_size + available.bit_count() <= best_size:
            return
        if available == 0:
            if current_size > best_size:
                best_size = current_size
                best_set = current_mask
            return
            
        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (bm[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                best_v_bit = v_bit
            temp -= v_bit
        
        if best_v < 0:
            return
            
        v = best_v
        v_bit = best_v_bit
        
        # Include v
        N_v_in_cur = bm[v] & current_mask
        safe = True
        if N_v_in_cur:
            temp_check = N_v_in_cur
            while temp_check:
                u_bit = temp_check & -temp_check
                u = u_bit.bit_length() - 1
                if bm[u] & N_v_in_cur:
                    safe = False
                    break
                temp_check -= u_bit
        
        if safe:
            new_mask = current_mask | v_bit
            N_v_in_new = bm[v] & new_mask
            forbidden = 0
            temp_w = N_v_in_new
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                forbidden |= bm[w]
                temp_w -= w_bit
            
            forbidden &= bm[v]
            forbidden &= available
            
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
            
        # Exclude v
        dfs(available & ~v_bit, current_mask, current_size)

    dfs(full_mask, 0, 0)
    return best_set, best_size

def _targeted_local_search(bm, n, max_iter=200, seed=0, deadline=None):
    """
    Local search that specifically targets the largest triangle-free set.
    Tries to add edges within the max TF set to break it.
    """
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        tf_set, tf_size = _find_max_tf_set(bm, n)
        if tf_size <= 9:
            break
            
        # Get list of vertices in tf_set
        verts = []
        temp = tf_set
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            verts.append(v)
            temp -= v_bit
            
        # Try to add an edge (u,v) within verts
        # Prioritize pairs (u,v) that have a common neighbor in verts (creating a triangle)
        # and where adding the edge is K4-safe.
        
        candidates = []
        for i in range(len(verts)):
            for j in range(i+1, len(verts)):
                u = verts[i]
                v = verts[j]
                if not (bm[u] & (1 << v)):
                    # Check if adding (u,v) creates a triangle in the set
                    common_in_set = bm[u] & bm[v] & tf_set
                    if common_in_set:
                        # Check K4 safety
                        if _can_add_edge(bm, u, v, n):
                            # Score by number of triangles created
                            tri_count = common_in_set.bit_count()
                            candidates.append((tri_count, u, v))
                            
        if candidates:
            # Sort by triangle count descending, pick from top candidates
            candidates.sort(reverse=True)
            # Pick randomly from top 5 for diversity
            top_k = min(5, len(candidates))
            _, u, v = candidates[rng.randint(0, top_k - 1)]
            bm[u] |= (1 << v)
            bm[v] |= (1 << u)
            
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                # Undo if it didn't help
                bm[u] ^= (1 << v)
                bm[v] ^= (1 << u)
        else:
            # No targeted move possible, try random edge addition within the TF set
            # that doesn't create triangles but might help later
            possible_adds = []
            for i in range(len(verts)):
                for j in range(i+1, len(verts)):
                    u = verts[i]
                    v = verts[j]
                    if not (bm[u] & (1 << v)):
                        if _can_add_edge(bm, u, v, n):
                            possible_adds.append((u, v))
            
            if possible_adds:
                u, v = rng.choice(possible_adds)
                bm[u] |= (1 << v)
                bm[v] |= (1 << u)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[u] ^= (1 << v)
                    bm[v] ^= (1 << u)
            else:
                break
            
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _local_search_bitmask(bm, n, max_iter=500, seed=0, deadline=None):
    """Local search on bitmask representation."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        if not (bm[i] & (1 << j)):
            # Try adding edge
            if _can_add_edge(bm, i, j, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        else:
            # Try removing edge
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    
    # 1. Baseline: Turan graph T(18, 3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_bm[a] |= (1 << b)
                    baseline_bm[b] |= (1 << a)
    
    baseline_score = _max_tf_size_fast(baseline_bm, n)
    best_score = baseline_score
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if baseline_bm[i] & (1 << j):
                best_edges.append([i, j])
    
    # 2. Random restarts with varied densities and local search
    densities = [0.3, 0.4, 0.5, 0.6, 0.7]
    for seed in range(300):
        if time.time() > deadline - 5:
            break
        
        density = densities[seed % len(densities)]
        bm, edges = _random_k4_free_graph(n, seed, density=density)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
            
        # Targeted Local Search
        if time.time() > deadline - 10:
            break
            
        tl_edges, tl_score = _targeted_local_search(bm, n, max_iter=150, seed=seed, deadline=deadline)
        if tl_score < best_score:
            best_score = tl_score
            best_edges = tl_edges
            
        # Standard Local Search
        if time.time() > deadline - 15:
            break
            
        ls_edges, ls_score = _local_search_bitmask(bm, n, max_iter=300, seed=seed, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges

    # 3. Final Polish with multiple targeted passes
    if time.time() < deadline - 5:
        best_bm = [0] * n
        for e in best_edges:
            u, v = e
            best_bm[u] |= (1 << v)
            best_bm[v] |= (1 << u)
            
        # Multiple targeted local search passes with different seeds
        for polish_seed in [999, 2000, 3000]:
            if time.time() > deadline - 3:
                break
            polish_edges, polish_score = _targeted_local_search(best_bm, n, max_iter=300, seed=polish_seed, deadline=deadline)
            if polish_score < best_score:
                best_score = polish_score
                best_edges = polish_edges
                best_bm = [0] * n
                for e in best_edges:
                    u, v = e
                    best_bm[u] |= (1 << v)
                    best_bm[v] |= (1 << u)
            
            # Follow with standard local search
            if time.time() > deadline - 3:
                break
            msat_polish_edges, msat_polish_score = _local_search_bitmask(best_bm, n, max_iter=300, seed=polish_seed + 10000, deadline=deadline)
            if msat_polish_score < best_score:
                best_score = msat_polish_score
                best_edges = msat_polish_edges
                best_bm = [0] * n
                for e in best_edges:
                    u, v = e
                    best_bm[u] |= (1 << v)
                    best_bm[v] |= (1 << u)

    return best_edges
# EVOLVE-BLOCK-END
