import time
import random
from itertools import combinations

N = 16
ALL_BITS = (1 << N) - 1

def _groetzsch_edges():
    """
    Grötzsch graph (Mycielski of C5).
    11 vertices, 20 edges.
    Vertices 0..4 form C5.
    Vertices 5..9 are copies of 0..4.
    Vertex 10 is the center.
    """
    edges = []
    # Original C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i+1)%5])
        
    # Copies 5-9 correspond to 0-4.
    # Center 10.
    # Edges between original and copies:
    # If (u,v) is an edge in C5, then (v, u') and (u, v') are edges in Mycielski.
    # u' is 5+u, v' is 5+v.
    orig_edges = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in orig_edges:
        edges.append([v, 5+u])
        edges.append([u, 5+v])
        
    # Center 10 connected to all copies 5-9
    for i in range(5):
        edges.append([10, 5+i])
        
    return edges

def _adj_from_edges(edge_list):
    adj = [0] * N
    for u, v in edge_list:
        if u < N and v < N:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _edge_count(adj):
    return sum(bin(a).count("1") for a in adj) // 2

def _is_triangle_free_add(adj, u, v):
    # Check if adding (u,v) creates a triangle.
    # This happens if there is a common neighbor w.
    return (adj[u] & adj[v]) == 0

def _is_3_colorable(adj):
    """
    Fast backtracking to check if graph is 3-colorable.
    Optimized for speed.
    """
    colors = [-1] * N
    # Order vertices by degree descending
    order = sorted(range(N), key=lambda x: (-bin(adj[x]).count('1'), x))
    order_pos = [0] * N
    for i, v in enumerate(order):
        order_pos[v] = i
    
    # Precompute for each vertex in order: which neighbors come before it
    before_neighbors = [0] * N
    for idx, v in enumerate(order):
        m = adj[v]
        before = 0
        for u in range(N):
            if order_pos[u] < idx and (m >> u) & 1:
                before |= (1 << u)
        before_neighbors[v] = before
    
    def backtrack(idx):
        if idx == N:
            return True
        v = order[idx]
        
        # Find which colors are forbidden by colored (before) neighbors
        forbidden = 0
        m = before_neighbors[v]
        while m:
            lb = m & (-m)
            u = lb.bit_length() - 1
            m ^= lb
            forbidden |= (1 << colors[u])
        
        # Try colors 0, 1, 2 that are not forbidden
        # Heuristic: try the most constrained color first? Or just 0,1,2
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def _greedy_add_all(adj):
    """Greedily add all possible triangle-free edges."""
    changed = True
    while changed:
        changed = False
        for u in range(N):
            for v in range(u + 1, N):
                if not ((adj[u] >> v) & 1) and _is_triangle_free_add(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
    return adj

def _try_swap_remove_add2(adj, rem_u, rem_v, deadline):
    """Remove edge (rem_u, rem_v), then greedily add up to 2 edges."""
    adj_copy = [a for a in adj]
    adj_copy[rem_u] &= ~(1 << rem_v)
    adj_copy[rem_v] &= ~(1 << rem_u)
    
    # Find all edges that can now be added
    addable = []
    for u in range(N):
        for v in range(u + 1, N):
            if not ((adj_copy[u] >> v) & 1) and _is_triangle_free_add(adj_copy, u, v):
                addable.append((u, v))
    
    if not addable:
        return None, 0
    
    # Try adding 2 edges (or 1 if only 1 available)
    best_result = None
    best_cnt = -1
    
    if len(addable) >= 2:
        # Try all pairs of addable edges (limited for speed)
        rng_local = random.Random(hash((rem_u, rem_v)) % 10000)
        rng_local.shuffle(addable)
        max_pairs = min(len(addable) * (len(addable) - 1) // 2, 50)
        count = 0
        for i in range(min(len(addable), 20)):
            for j in range(i + 1, min(len(addable), 20)):
                if count >= max_pairs or time.time() > deadline:
                    break
                count += 1
                e1, e2 = addable[i], addable[j]
                # Check no conflict between the two new edges
                u1, v1 = e1
                u2, v2 = e2
                # They can't share a vertex in a way that creates issues
                # Just try adding both
                test_adj = [a for a in adj_copy]
                test_adj[u1] |= (1 << v1)
                test_adj[v1] |= (1 << u1)
                if _is_triangle_free_add(test_adj, u2, v2):
                    test_adj[u2] |= (1 << v2)
                    test_adj[v2] |= (1 << u2)
                    cnt = _edge_count(test_adj)
                    if cnt > best_cnt and not _is_3_colorable(test_adj):
                        best_cnt = cnt
                        best_result = test_adj
            if time.time() > deadline:
                break
    else:
        # Only 1 edge can be added
        u, v = addable[0]
        adj_copy[u] |= (1 << v)
        adj_copy[v] |= (1 << u)
        cnt = _edge_count(adj_copy)
        if cnt > best_cnt and not _is_3_colorable(adj_copy):
            best_cnt = cnt
            best_result = adj_copy
    
    return best_result, best_cnt

def _local_search(adj, deadline, rng, min_edges=0):
    """
    Aggressive local search to find maximum edge count triangle-free graph
    that is NOT 3-colorable, starting from adj.
    """
    best_cnt = min_edges
    best_adj = None
    
    current_adj = [a for a in adj]
    _greedy_add_all(current_adj)
    current_cnt = _edge_count(current_adj)
    
    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
        best_cnt = current_cnt
        best_adj = [a for a in current_adj]
    
    # Local search phase
    stall_count = 0
    max_stall = 10  # Reduced stall tolerance to allow more restarts within time budget
    
    while time.time() < deadline:
        improved = False
        stall_count += 1
        
        if stall_count > max_stall:
            # Restart with perturbation
            current_adj = [a for a in best_adj] if best_adj else [a for a in adj]
            # Randomly remove 1-3 edges
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                       if (current_adj[u] >> v) & 1]
            if existing:
                for _ in range(rng.randint(1, 3)):
                    if not existing:
                        break
                    idx = rng.randint(0, len(existing) - 1)
                    u, v = existing[idx]
                    current_adj[u] &= ~(1 << v)
                    current_adj[v] &= ~(1 << u)
                    existing = [(uu, vv) for uu in range(N) for vv in range(uu + 1, N) 
                               if (current_adj[uu] >> vv) & 1]
            _greedy_add_all(current_adj)
            current_cnt = _edge_count(current_adj)
            stall_count = 0
            continue
        
        # Phase 1: Try to add single edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if not ((current_adj[u] >> v) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline:
                break
            if _is_triangle_free_add(current_adj, u, v):
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                current_cnt += 1
                improved = True
                
                if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                    best_cnt = current_cnt
                    best_adj = [a for a in current_adj]
                break
        
        if improved:
            stall_count = 0
            continue
        
        # Phase 2: Try remove-1-add-2 swaps
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) 
                   if (current_adj[u] >> v) & 1]
        
        if existing:
            rng.shuffle(existing)
            found_swap = False
            
            for rem_u, rem_v in existing:
                if time.time() > deadline:
                    break
                if found_swap:
                    break
                
                result_adj, result_cnt = _try_swap_remove_add2(current_adj, rem_u, rem_v, deadline)
                
                if result_adj and result_cnt > current_cnt:
                    current_adj = result_adj
                    current_cnt = result_cnt
                    improved = True
                    found_swap = True
                    
                    if current_cnt > best_cnt and not _is_3_colorable(current_adj):
                        best_cnt = current_cnt
                        best_adj = [a for a in current_adj]
            
            if improved:
                stall_count = 0
                continue
        
        # Phase 3: Try remove-2-add-3 swaps (more aggressive)
        if len(existing) >= 2:
            rng.shuffle(existing)
            found_swap2 = False
            
            # Limit inner loop iterations to prevent long hangs
            limit = min(len(existing), 10)
            for i in range(limit):
                if time.time() > deadline or found_swap2:
                    break
                rem1_u, rem1_v = existing[i]
                for j in range(i + 1, limit):
                    if time.time() > deadline or found_swap2:
                        break
                    rem2_u, rem2_v = existing[j]
                    
                    adj_copy = [a for a in current_adj]
                    adj_copy[rem1_u] &= ~(1 << rem1_v)
                    adj_copy[rem1_v] &= ~(1 << rem1_u)
                    adj_copy[rem2_u] &= ~(1 << rem2_v)
                    adj_copy[rem2_v] &= ~(1 << rem2_u)
                    
                    _greedy_add_all(adj_copy)
                    cnt = _edge_count(adj_copy)
                    
                    # Only check 3-colorability if edge count improved
                    if cnt > current_cnt:
                        if not _is_3_colorable(adj_copy):
                            current_adj = adj_copy
                            current_cnt = cnt
                            improved = True
                            found_swap2 = True
                            
                            if current_cnt > best_cnt:
                                best_cnt = current_cnt
                                best_adj = [a for a in current_adj]
                            break
            
            if improved:
                stall_count = 0
                continue
        
        if not improved:
            stall_count += 1
    
    return best_adj, best_cnt

def construct():
    start = time.time()
    deadline = start + 115.0  # Safety margin to stay under 120s
    
    base_edges = _groetzsch_edges()
    base_adj = _adj_from_edges(base_edges)
    
    best_adj = None
    best_cnt = -1
    
    rng = random.Random(42)
    
    # First, run greedy fill on base
    greedy_adj = [a for a in base_adj]
    _greedy_add_all(greedy_adj)
    greedy_cnt = _edge_count(greedy_adj)
    
    if greedy_cnt > best_cnt and not _is_3_colorable(greedy_adj):
        best_cnt = greedy_cnt
        best_adj = [a for a in greedy_adj]
    
    # Reduced trials to prevent timeout. 200 was too many.
    max_trials = 50
    for trial in range(max_trials):
        if time.time() > deadline - 5.0:
            break
        
        # Start with Grötzsch graph
        adj = [a for a in base_adj]
        
        # Randomly perturb: add/remove some edges from the base
        if trial > 0:
            # Random perturbation
            base_edge_list = base_edges[:]
            num_remove = rng.randint(0, 3)
            for _ in range(num_remove):
                if base_edge_list:
                    idx = rng.randint(0, len(base_edge_list) - 1)
                    u, v = base_edge_list[idx]
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    base_edge_list.pop(idx)
        
        # Run local search
        result_adj, result_cnt = _local_search(adj, deadline, rng, best_cnt)
        
        if result_adj and result_cnt > best_cnt:
            best_cnt = result_cnt
            best_adj = result_adj
    
    # Final polish: try to add any remaining edges
    if best_adj is None:
        best_adj = base_adj
        best_cnt = _edge_count(best_adj)
    else:
        _greedy_add_all(best_adj)
        best_cnt = _edge_count(best_adj)
    
    # Final verification
    if _is_3_colorable(best_adj):
        # Fallback to Grötzsch
        best_adj = base_adj
        best_cnt = _edge_count(best_adj)
    
    # Convert to edge list
    out = []
    for u in range(N):
        m = best_adj[u]
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            m ^= lb
            if v > u:
                out.append([u, v])
                
    return out
