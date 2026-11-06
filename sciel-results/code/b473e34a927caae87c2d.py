import time

# ---------- Exact 3-colorability (triangle-free graph, 16 vertices) ----------
def _is_3_colorable(adj):
    """adj: list of 16 sets/lists. Returns True if graph is 3-colorable."""
    n = len(adj)
    # Precompute neighbor bitmasks for speed
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    color = [-1] * n
    color_mask = [0, 0, 0]
    degrees = [len(adj[i]) for i in range(n)]
    
    def backtrack(uncolored_count):
        if uncolored_count == 0:
            return True
        
        # Find the best vertex to color next: most constrained (fewest available colors)
        # Tie-break by higher degree
        best_v = -1
        best_num_avail = 4
        best_deg = -1
        best_mask = 0
        
        for v in range(n):
            if color[v] != -1:
                continue
            
            # Check available colors
            avail = 0
            if not (adj_mask[v] & color_mask[0]):
                avail |= 1
            if not (adj_mask[v] & color_mask[1]):
                avail |= 2
            if not (adj_mask[v] & color_mask[2]):
                avail |= 4
                
            if avail == 0:
                return False
            
            num_avail = bin(avail).count('1')
            
            # Most constrained first, then highest degree
            if num_avail < best_num_avail or (num_avail == best_num_avail and degrees[v] > best_deg):
                best_v = v
                best_num_avail = num_avail
                best_deg = degrees[v]
                best_mask = avail
        
        if best_v == -1:
            return True
            
        v = best_v
        for c in range(3):
            if not (best_mask & (1 << c)):
                continue
            
            color[v] = c
            color_mask[c] |= (1 << v)
            
            if backtrack(uncolored_count - 1):
                return True
            
            color_mask[c] &= ~(1 << v)
            color[v] = -1
            
        return False

    return backtrack(n)

def _build_adj(edges, n=16):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_triangle_free(adj):
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                for w in adj[u]:
                    if w != v and w in adj[v]:
                        return False
    return True

def _can_add_edge(u, v, adj):
    if u == v or v in adj[u]:
        return False
    # Check triangle-free: no common neighbor
    au = adj[u]
    for w in au:
        if w in adj[v]:
            return False
    return True

def _groetzsch_edges():
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    edges = [list(e) for e in cycle]
    for u, w in cycle:
        edges += [[u, 5 + w], [w, 5 + u]]
    edges += [[5 + u, 10] for u in range(5)]
    return edges

def _initial_4chromatic_graph():
    """
    Start with a known 4-chromatic triangle-free graph on 11 vertices (Mycielski of C5 = Grötzsch),
    then extend to 16 vertices with some structure to allow denser packing.
    """
    # Grötzsch graph on vertices 0-10
    edges = _groetzsch_edges()
    adj = _build_adj(edges, 16)
    
    # Add vertices 11-15 with some edges to make it harder to 3-color
    # Connect them in a way that preserves triangle-free but adds constraints
    # For now, just return the base and let greedy fill the rest
    return adj, edges

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s safety margin for 120s limit
    n = 16
    
    best_edges = []
    best_count = -1
    
    def elapsed_ok():
        return time.time() < deadline

    def try_add_edge(u, v, adj, edges):
        if not _can_add_edge(u, v, adj):
            return None
        new_adj = [s.copy() for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if _is_3_colorable(new_adj):
            return None
        return new_adj, edges + [[u, v]]

    def run_greedy(seed_order=None):
        # Start with Groetzsch
        base_edges = _groetzsch_edges()
        adj = _build_adj(base_edges, n)
        edges = [list(e) for e in base_edges]
        
        if not _is_triangle_free(adj) or _is_3_colorable(adj):
            return adj, edges, len(edges)

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        if seed_order is not None:
            import random
            rng = random.Random(seed_order)
            rng.shuffle(all_pairs)
        else:
            all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

        # Greedy addition phase
        changed = True
        while changed and elapsed_ok():
            changed = False
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    changed = True
        
        # Local Search with multiple strategies
        import random
        rng = random.Random(seed_order if seed_order is not None else 0)
        
        improved = True
        while improved and elapsed_ok():
            improved = False
            
            # Phase 1: Try to add any edge (systematic scan)
            for u, v in all_pairs:
                if not elapsed_ok():
                    break
                if v in adj[u]:
                    continue
                res = try_add_edge(u, v, adj, edges)
                if res is not None:
                    adj, edges = res
                    improved = True
                    break
            
            if improved or not elapsed_ok():
                continue

            # Phase 2: Try 1-1 swaps - try multiple random removals
            swapped = False
            if edges:
                # Try up to 20 random swap attempts
                for _ in range(20):
                    if not elapsed_ok():
                        break
                    idx = rng.randint(0, len(edges) - 1)
                    u_rem, v_rem = edges[idx]
                    
                    cand_adj = [s.copy() for s in adj]
                    cand_adj[u_rem].discard(v_rem)
                    cand_adj[v_rem].discard(u_rem)
                    cand_edges = edges[:idx] + edges[idx+1:]
                    
                    # Get all non-edges that are triangle-free compatible
                    pairs_to_try = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if v not in cand_adj[u] and _can_add_edge(u, v, cand_adj):
                                pairs_to_try.append((u, v))
                    
                    if not pairs_to_try:
                        continue
                    
                    rng.shuffle(pairs_to_try)
                    
                    # Try up to 30 candidate additions
                    for u_add, v_add in pairs_to_try[:30]:
                        if not elapsed_ok():
                            break
                        tmp_adj = [s.copy() for s in cand_adj]
                        tmp_adj[u_add].add(v_add)
                        tmp_adj[v_add].add(u_add)
                        if not _is_3_colorable(tmp_adj):
                            # Valid swap
                            adj = tmp_adj
                            edges = cand_edges + [[u_add, v_add]]
                            swapped = True
                            break
                    
                    if swapped:
                        break
            
            if swapped:
                improved = True
            
            # Phase 3: Shake - remove 2-3 edges randomly, then try to re-add
            if not improved and elapsed_ok():
                if len(edges) > 5:
                    shake_size = rng.randint(2, 3)
                    shake_indices = rng.sample(range(len(edges)), min(shake_size, len(edges)))
                    shake_set = set(shake_indices)
                    
                    shake_adj = [s.copy() for s in adj]
                    shake_edges = []
                    for i, (u, v) in enumerate(edges):
                        if i not in shake_set:
                            shake_edges.append([u, v])
                        else:
                            shake_adj[u].discard(v)
                            shake_adj[v].discard(u)
                    
                    # Try to add edges back
                    for u, v in all_pairs:
                        if not elapsed_ok():
                            break
                        if v in shake_adj[u]:
                            continue
                        if not _can_add_edge(u, v, shake_adj):
                            continue
                        tmp_adj = [s.copy() for s in shake_adj]
                        tmp_adj[u].add(v)
                        tmp_adj[v].add(u)
                        if not _is_3_colorable(tmp_adj):
                            shake_adj = tmp_adj
                            shake_edges.append([u, v])
                    
                    # Accept if we got at least as many edges
                    if len(shake_edges) >= len(edges):
                        adj = shake_adj
                        edges = shake_edges
                        improved = True
        
        return adj, edges, len(edges)

    # Run multiple greedy+local search instances with different random seeds
    max_restarts = 200  # More restarts with better time management
    for i in range(max_restarts):
        if not elapsed_ok():
            break
        
        # Use a different random seed for shuffling the pair order
        seed = i * 7919 + 13  # Prime-based seed for better distribution
        adj, edges, count = run_greedy(seed)
        
        if count > best_count:
            best_count = count
            best_edges = [list(e) for e in edges]
            
        # Early termination if we've found a very good solution and time is running out
        if best_count >= 45 and time.time() > deadline - 30:
            break

    # Final validation
    if best_edges:
        final_adj = _build_adj(best_edges, n)
        if not _is_triangle_free(final_adj) or _is_3_colorable(final_adj):
            # Fallback to Groetzsch if something went wrong
            base_edges = _groetzsch_edges()
            return [list(e) for e in base_edges]
        return best_edges
    else:
        # Fallback to Groetzsch if something went wrong
        base_edges = _groetzsch_edges()
        return [list(e) for e in base_edges]
