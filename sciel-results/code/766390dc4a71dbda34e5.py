import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148  # Safety margin before 150s kill

    n = 16
    # List of all possible edges
    all_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            all_edges.append((u, v))
    m_total = len(all_edges)  # 120

    # For each edge index, precompute its endpoints
    edge_u = [e[0] for e in all_edges]
    edge_v = [e[1] for e in all_edges]

    # Triangle constraint: for each edge i, which edges j form a triangle with i
    # Edge i=(u,v) and edge j=(a,b) form a triangle if they share a vertex
    # and the remaining two vertices are connected by a third edge.
    # For B&B: when we add edge i, we must forbid all edges that form a triangle with i
    # or with any previously added edge.
    
    # Build edge index lookup
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx

    # For each edge i, the set of edges that form a triangle with i
    # (i.e., edges j such that i and j share a vertex and the "closing" edge exists)
    # When edge i is ADDED, any edge j in tri_conflict[i] that is also present
    # forms a triangle. So tri_conflict[i] = edges that cannot coexist with i.
    tri_conflict = [set() for _ in range(m_total)]
    for i in range(m_total):
        u, v = edge_u[i], edge_v[i]
        for w in range(n):
            if w == u or w == v:
                continue
            # Triangle u-v-w: edges (u,v), (u,w), (v,w)
            ew1 = edge_index.get((u, w))
            ew2 = edge_index.get((v, w))
            if ew1 is not None and ew2 is not None and ew1 != i and ew2 != i:
                tri_conflict[i].add(ew1)
                tri_conflict[i].add(ew2)

    # Convert to sorted lists and bitmasks for fast checking
    # Use Python ints as bitmasks (120 bits fits in Python int)
    tri_conflict_mask = [0] * m_total
    for i in range(m_total):
        mask = 0
        for j in tri_conflict[i]:
            mask |= (1 << j)
        tri_conflict_mask[i] = mask

    # Precompute: for each vertex, bitmask of incident edges
    vert_incident = [0] * n
    for idx in range(m_total):
        u, v = edge_u[idx], edge_v[idx]
        vert_incident[u] |= (1 << idx)
        vert_incident[v] |= (1 << idx)

    # 3-colorability check using bitmasks
    # State: which vertices are colored with each color
    # We use a recursive backtracking with bitmask representation
    # color_mask[c] = bitmask of vertices with color c

    def is_3_colorable(frozen_mask):
        """Check if the graph with edges in frozen_mask is 3-colorable."""
        # Build adjacency from frozen_mask
        # adj[v] = bitmask of neighbors
        adj = [0] * n
        # Only consider edges in frozen_mask
        for idx in range(m_total):
            if (frozen_mask >> idx) & 1:
                u, v = edge_u[idx], edge_v[idx]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        # Greedy initial coloring to speed up
        color = [-1] * n
        # Order vertices by degree (desc) for better pruning
        order = sorted(range(n), key=lambda v: bin(adj[v]).count('1'), reverse=True)
        
        # Try to color greedily first
        for v in order:
            used = 0
            for c in range(3):
                if color[c]:  # not used
                    pass
            # Find available colors
            avail = [0, 1, 2]
            for c in range(3):
                # Check if any neighbor has color c
                conflict = False
                for c2 in range(3):
                    pass
                # Better: check adjacency
                for nb_bit in range(n):
                    if (adj[v] >> nb_bit) & 1:
                        if color[nb_bit] == c:
                            conflict = True
                            break
                if not conflict:
                    avail.append(c)
                    break
            if avail:
                color[v] = avail[0]
            else:
                avail = []
        
        # Full backtracking
        color = [-1] * n
        uncolored_mask = (1 << n) - 1
        
        def dfs(uncolored, color):
            if uncolored == 0:
                return True
            # Pick vertex with fewest available colors (MRV heuristic)
            # Find first uncolored vertex
            # For efficiency, pick vertex with min available colors
            best_v = -1
            best_avail_count = 4
            for v in range(n):
                if (uncolored >> v) & 1:
                    # Count available colors
                    avail_count = 0
                    for c in range(3):
                        conflict = False
                        # Check neighbors of v that are colored
                        nb_mask = adj[v] & (~uncolored) & ((1 << n) - 1)
                        # Actually check colored neighbors
                        for c in range(3):
                            conflict = False
                            nb = adj[v]
                            # Check if any neighbor has color c
                            # We need to track color assignment
                            break
                    # Simpler: just check each color
                    avail_colors = []
                    for c in range(3):
                        ok = True
                        # Check all colored neighbors
                        nb_mask = adj[v]
                        for w in range(n):
                            if (nb_mask >> w) & 1 and color[w] == c:
                                ok = False
                                break
                        if ok:
                            avail_colors.append(c)
                    if len(avail_colors) < best_avail_count:
                        best_avail_count = len(avail_colors)
                        best_v = v
                        if best_avail_count == 0:
                            break
                        if best_avail_count == 1:
                            break
            
            if best_v == -1:
                return True  # All colored
            
            if best_avail_count == 0:
                return False
            
            new_uncolored = uncolored & ~(1 << best_v)
            for c in range(3):
                # Check if color c is available for best_v
                ok = True
                nb_mask = adj[best_v]
                for w in range(n):
                    if (nb_mask >> w) & 1 and color[w] == c:
                        ok = False
                        break
                if ok:
                    color[best_v] = c
                    if dfs(new_uncolored, color):
                        return True
                    color[best_v] = -1
            return False
        
        return dfs((1 << n) - 1, color)

    # Better 3-colorability check - more optimized
    def is_3_colorable_fast(frozen_mask):
        """Fast 3-colorability check using bitsets."""
        adj = [0] * n
        for idx in range(m_total):
            if (frozen_mask >> idx) & 1:
                u, v = edge_u[idx], edge_v[idx]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        
        color = [-1] * n  # -1 = uncolored, 0,1,2 = colors
        
        def dfs(k):
            if k == n:
                return True
            v = k
            # Try each color
            for c in range(3):
                # Check conflict with colored neighbors
                conflict = False
                nb = adj[v]
                for w in range(n):
                    if (nb >> w) & 1 and color[w] == c:
                        conflict = True
                        break
                if not conflict:
                    color[v] = c
                    if dfs(k + 1):
                        return True
                    color[v] = -1
            return False
        
        return dfs(0)

    # Even better: use the fact that we're doing B&B, so frozen_mask grows
    # We can maintain state incrementally
    
    # Strategy: Branch-and-bound over edges
    # At each step, decide include/exclude next edge
    # Prune if:
    #   1. Triangle formed (check against frozen_mask)
    #   2. Even adding ALL remaining edges can't beat best_count
    #   3. The graph is 3-colorable (we need NOT 3-colorable)
    
    # For pruning condition 3, we check 3-colorability of the current graph
    # If current graph is already not 3-colorable, any superset is also not 3-colorable
    # If current graph IS 3-colorable, we might still find a non-3-colorable superset
    
    # Upper bound: current_count + remaining_edges (but need triangle-free check)
    # Simple upper bound: current_count + (m_total - pos)
    
    best_count = -1
    best_mask = 0
    search_calls = [0]
    time_exceeded = [False]

    def get_edge_count(mask):
        return bin(mask).count('1')

    def check_triangle_free_incremental(frozen_mask, edge_idx):
        """Check if adding edge_idx to frozen_mask keeps it triangle-free."""
        # Check if edge_idx conflicts with any edge in frozen_mask
        conflict = tri_conflict_mask[edge_idx] & frozen_mask
        return conflict == 0

    def upper_bound(pos, current_count):
        """Upper bound on edges we can still add."""
        # Simple: current_count + (m_total - pos)
        # But many of those will conflict
        # Better: for each remaining edge, check if it can be added
        # This is expensive, so use simple bound
        return current_count + (m_total - pos)

    # Initial: try to find a good starting point
    # Start with Grötzsch graph
    def groetzch_mask():
        edges = []
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, v in cycle:
            edges.append((u, v))
        for u, w in cycle:
            edges.append((u, 5 + w))
            edges.append((w, 5 + u))
        for u in range(5):
            edges.append((5 + u, 10))
        
        mask = 0
        for u, v in edges:
            idx = edge_index[(u, v)]
            mask |= (1 << idx)
        return mask

    g_mask = groetzch_mask()
    
    # Check if Grötzsch is triangle-free and not 3-colorable
    if not is_3_colorable_fast(g_mask):
        best_count = get_edge_count(g_mask)
        best_mask = g_mask
    
    # Greedy extension from Grötzsch
    current_mask = g_mask
    current_count = get_edge_count(current_mask)
    
    # Try to add edges greedily in different orders
    for trial in range(3):
        if time.time() > deadline - 5:
            break
        cm = g_mask
        cc = get_edge_count(cm)
        # Add edges in random order
        candidates = list(range(m_total))
        random.shuffle(candidates)
        for idx in candidates:
            if time.time() > deadline - 5:
                break
            if (cm >> idx) & 1:
                continue
            if check_triangle_free_incremental(cm, idx):
                new_mask = cm | (1 << idx)
                # Check if still not 3-colorable (it should be, since we're adding edges)
                # Actually adding edges preserves non-3-colorability
                cm = new_mask
                cc += 1
        
        if cc > best_count and not is_3_colorable_fast(cm):
            best_count = cc
            best_mask = cm
    
    # Now do B&B search
    # Order edges by some heuristic - edges that are in many triangles first?
    # Or just natural order
    
    def branch_and_bound(pos, frozen_mask, current_count):
        if time_exceeded[0]:
            return
        if time.time() > deadline - 2:
            time_exceeded[0] = True
            return
        
        search_calls[0] += 1
        if search_calls[0] % 10000 == 0:
            if time.time() > deadline - 2:
                time_exceeded[0] = True
                return
        
        # Pruning: upper bound
        if current_count + (m_total - pos) <= best_count:
            return
        
        # If we've processed all edges
        if pos == m_total:
            if current_count > best_count:
                nonlocal best_count, best_mask
                best_count = current_count
                best_mask = frozen_mask
            return
        
        idx = pos
        
        # Option 1: Exclude this edge
        branch_and_bound(pos + 1, frozen_mask, current_count)
        if time_exceeded[0]:
            return
        
        # Option 2: Include this edge (if triangle-free)
        if check_triangle_free_incremental(frozen_mask, idx):
            new_mask = frozen_mask | (1 << idx)
            # Quick check: is the new graph 3-colorable?
            # If it is, we can still continue (superset might not be)
            # But if current_count + 1 + (m_total - pos - 1) <= best_count, skip
            if current_count + 1 + (m_total - pos - 1) > best_count:
                branch_and_bound(pos + 1, new_mask, current_count + 1)

    # The B&B above is too slow for 120 edges. Need better approach.
    
    # Alternative: Local search / hill climbing
    # Start from a 4-chromatic triangle-free graph, try edge swaps
    
    # Or: use the fact that max edges for triangle-free on 16 vertices is 64 (Mantel)
    # We want the max edges among 4-chromatic triangle-free graphs
    
    # Let's try a different approach: 
    # 1. Generate random triangle-free graphs near the edge limit
    # 2. Check 3-colorability
    # 3. Try to add edges to make them denser
    
    # Random triangle-free graph generation
    def random_triangle_free():
        """Generate a random triangle-free graph."""
        adj = [set() for _ in range(n)]
        edges = set()
        
        # Randomly assign vertices to two independent sets (bipartite base)
        # Then add some edges within sets that don't create triangles
        # Actually, let's just build randomly with triangle check
        
        possible = list(range(m_total))
        random.shuffle(possible)
        
        for idx in possible:
            u, v = edge_u[idx], edge_v[idx]
            # Check triangle-free
            common = adj[u] & adj[v]
            if not common:
                adj[u].add(v)
                adj[v].add(u)
                edges.add(idx)
        
        return edges

    # Try many random graphs
    for trial in range(200):
        if time.time() > deadline - 5:
            break
        
        edges_set = random_triangle_free()
        mask = 0
        for idx in edges_set:
            mask |= (1 << idx)
        
        if not is_3_colorable_fast(mask):
            count = get_edge_count(mask)
            if count > best_count:
                best_count = count
                best_mask = mask
            # Try to add more edges
            for idx in range(m_total):
                if time.time() > deadline - 5:
                    break
                if (mask >> idx) & 1:
                    continue
                if check_triangle_free_incremental(mask, idx):
                    new_mask = mask | (1 << idx)
                    mask = new_mask
                    best_count = get_edge_count(mask)
                    best_mask = mask
    
    # Convert best_mask to edge list
    result = []
    for idx in range(m_total):
        if (best_mask >> idx) & 1:
            result.append([edge_u[idx], edge_v[idx]])
    
    if not result:
        # Fallback: Grötzsch graph
        result = []
        cycle = [(i, (i + 1) % 5) for i in range(5)]
        for u, v in cycle:
            result.append([u, v])
        for u, w in cycle:
            result.append([u, 5 + w])
            result.append([w, 5 + u])
        for u in range(5):
            result.append([5 + u, 10])
    
    return result
