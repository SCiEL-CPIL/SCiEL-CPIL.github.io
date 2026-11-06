import time
import numpy as np

def _is_not_3_colorable(adj, n):
    """Check if graph is NOT 3-colorable. adj is a 2D numpy bool array.
    Returns True if NOT 3-colorable (i.e., 4-chromatic or higher)."""
    # Fast vectorized check: enumerate 3-colorings with symmetry breaking
    # color[0] = 0, color[1] in {0,1}
    # For each coloring, check if any edge connects same-colored vertices
    #
    # We build: for each vertex v, the set of colors it's "forbidden" to use
    # given its neighbors' colors. A coloring is valid iff every vertex
    # gets a color not used by any neighbor.
    
    # adj is n x n bool. For a coloring c (vector of 0,1,2), 
    # valid iff for all edges (i,j): c[i] != c[j]
    # Equivalent: for all i, c[i] not in {c[j] : j neighbor of i}
    
    # Vectorized approach:
    # For each color c in {0,1,2}, compute mask_c = vertices with color c
    # A coloring is invalid iff exists edge within some mask_c
    # i.e., exists i,j with adj[i,j] and c[i]==c[j]==k for some k
    
    # We enumerate: color[0]=0, color[1] in {0,1}, color[2..15] in {0,1,2}
    # That's 2 * 3^14 = 9.56M colorings. Still a lot.
    
    # Better: use the fact that we can check via bitmask.
    # Precompute for each vertex v, its neighbor set as a bitmask.
    # For a coloring, vertex v is OK if color[v] is not in {color[u] : u in N(v)}.
    
    # Actually, let's just do a clean backtracking with numpy-accelerated checks.
    # But the simplest fast approach: use the 3^16 enumeration with numpy.
    
    # Let me use a different approach: 
    # A graph is 3-colorable iff for every subset S of vertices, 
    # ... no, that's not right either.
    
    # Cleanest fast approach: backtracking with MRV heuristic, 
    # but using bitmasks for speed.
    
    # Build neighbor bitmasks
    neigh = np.zeros(n, dtype=np.int64)
    for i in range(n):
        mask = 0
        row = adj[i]
        for j in range(n):
            if row[j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # Order vertices by degree descending
    deg = np.array([bin(neigh[i]).count('1') for i in range(n)])
    order = np.argsort(-deg)
    pos_of = np.zeros(n, dtype=np.int64)
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    # For each vertex v in order, compute which earlier-in-order vertices are neighbors
    earlier_neigh = np.zeros(n, dtype=np.int64)
    for idx in range(n):
        v = order[idx]
        mask = 0
        nv = neigh[v]
        for u in range(n):
            if (nv >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    # Backtracking
    color_masks = [0, 0, 0]  # color_masks[c] = bitmask of positions with color c
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        available = 7 & ~forbidden
        if available == 0:
            return False
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)


def _is_not_3_colorable_np(adj_np, n):
    """Fast 3-colorability check using numpy vectorization.
    adj_np: n x n numpy bool array (symmetric, no diagonal).
    Returns True if NOT 3-colorable."""
    
    # Vectorized: enumerate colorings
    # color[0] = 0 (WLOG), color[1] in {0,1}, rest in {0,1,2}
    # For a coloring c (length n, values 0-2), it's valid iff
    # for all i < j: not (adj[i,j] and c[i] == c[j])
    
    # Precompute: for each pair (i,j) with an edge, we need c[i] != c[j].
    # 
    # Efficient approach: for each vertex i, define bad_colors[i] = set of colors
    # used by neighbors. Coloring is valid iff c[i] not in bad_colors[i] for all i.
    #
    # Let's just do the 3^16 enumeration smartly with numpy.
    # Actually 3^16 = 43M is too slow even in numpy for repeated calls.
    # 
    # Better: use the backtracking above but make it faster with precomputation.
    
    # For n=16, the backtracking with MRV is usually very fast (< 1ms).
    # The bottleneck in the original was rebuilding things. Let me use a
    # cleaner version.
    
    # Convert to list of neighbor bitmasks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj_np[i, j]:
                mask |= (1 << j)
        neigh[i] = mask
    
    # MRV ordering: sort by degree descending
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    earlier_neigh = [0] * n
    for idx in range(n):
        v = order[idx]
        mask = 0
        nv = neigh[v]
        for u in range(n):
            if (nv >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        available = 7 & ~forbidden
        if available == 0:
            return False
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)


def _is_not_3_colorable_fast(adj_list, n):
    """Fast 3-colorability check. adj_list: list of sets or frozensets of neighbors."""
    # Convert to bitmasks
    neigh = [0] * n
    for i in range(n):
        mask = 0
        for j in adj_list[i]:
            mask |= (1 << j)
        neigh[i] = mask
    
    deg = [bin(neigh[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: -deg[x])
    pos_of = [0] * n
    for idx, v in enumerate(order):
        pos_of[v] = idx
    
    earlier_neigh = [0] * n
    for idx in range(n):
        v = order[idx]
        mask = 0
        nv = neigh[v]
        for u in range(n):
            if (nv >> u) & 1:
                if pos_of[u] < idx:
                    mask |= (1 << pos_of[u])
        earlier_neigh[v] = mask
    
    color_masks = [0, 0, 0]
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        em = earlier_neigh[v]
        forbidden = 0
        for c in range(3):
            if em & color_masks[c]:
                forbidden |= (1 << c)
        available = 7 & ~forbidden
        if available == 0:
            return False
        while available:
            c_bit = available & (-available)
            c = c_bit.bit_length() - 1
            color_masks[c] |= (1 << idx)
            if backtrack(idx + 1):
                return True
            color_masks[c] ^= (1 << idx)
            available ^= c_bit
        return False
    
    return not backtrack(0)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    
    # Start with the Grötzsch graph (11 vertices, triangle-free, not 3-colorable)
    # Then add 5 more vertices connected to independent sets.
    
    # Grötzsch graph on vertices 0-10
    adj = [[0]*n for _ in range(n)]
    # 5-cycle: 0-1-2-3-4-0
    for i in range(5):
        u, v = i, (i+1) % 5
        adj[u][v] = adj[v][u] = 1
    # Second 5-cycle: 5-6-7-8-9
    for i in range(5):
        u, v = 5+i, 5+(i+1)%5
        adj[u][v] = adj[v][u] = 1
    # Connect cycles: i connected to 5+(i+1)%5 and 5+(i+2)%5
    for i in range(5):
        adj[i][5+(i+1)%5] = adj[5+(i+1)%5][i] = 1
        adj[i][5+(i+2)%5] = adj[5+(i+2)%5][i] = 1
    # Vertex 10 connected to all of second cycle
    for i in range(5):
        adj[10][5+i] = adj[5+i][10] = 1
    
    # Find independent sets of size 5 in the Grötzsch part (vertices 0-10)
    # to connect new vertices 11-15
    from itertools import combinations
    
    def is_independent_set(S, adj, n):
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                if adj[S[i]][S[j]]:
                    return False
        return True
    
    iso_sets = []
    for comb in combinations(range(11), 5):
        if is_independent_set(comb, adj, n):
            iso_sets.append(list(comb))
        if len(iso_sets) >= 5:
            break
    
    # If we don't have 5, pad with duplicates (suboptimal but valid)
    while len(iso_sets) < 5:
        iso_sets.append(iso_sets[0] if iso_sets else list(range(5)))
    
    for i in range(5):
        v_new = 11 + i
        for u in iso_sets[i]:
            adj[v_new][u] = 1
            adj[u][v_new] = 1
    
    # Convert to numpy for faster operations
    adj_np = np.array(adj, dtype=bool)
    
    def count_edges(adj_np):
        return int(np.sum(adj_np)) // 2
    
    best_adj = [row[:] for row in adj]
    best_edges = count_edges(adj_np)
    
    # Compute common neighbor masks for triangle-freeness check
    # cn[i][j] = bitmask of common neighbors of i and j
    # An edge (i,j) can be added iff cn[i][j] == 0 (no common neighbor -> no triangle)
    
    def compute_cn(adj_np, n):
        cn = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(i+1, n):
                if adj_np[i, j]:
                    # common neighbors
                    common = adj_np[i] & adj_np[j]
                    mask = 0
                    for k in range(n):
                        if common[k]:
                            mask |= (1 << k)
                    cn[i, j] = mask
                    cn[j, i] = mask
        return cn
    
    cn = compute_cn(adj_np, n)
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    import random
    
    def try_add_edge(adj_np, cn, i, j, n):
        """Try to add edge (i,j). Returns True if successful (graph still not 3-colorable)."""
        if adj_np[i, j]:
            return False
        if cn[i, j] != 0:
            return False  # Would create triangle
        
        # Add edge
        adj_np[i, j] = True
        adj_np[j, i] = True
        
        # Update cn
        for k in range(n):
            if k != i and k != j:
                if adj_np[i, k]:
                    cn[i, k] |= (1 << j)
                    cn[k, i] |= (1 << j)
                if adj_np[j, k]:
                    cn[j, k] |= (1 << i)
                    cn[k, j] |= (1 << i)
        cn[i, j] = 0
        cn[j, i] = 0
        
        # Check 3-colorability
        if _is_not_3_colorable_np(adj_np, n):
            return True
        else:
            # Revert
            adj_np[i, j] = False
            adj_np[j, i] = False
            for k in range(n):
                if k != i and k != j:
                    if adj_np[i, k]:
                        cn[i, k] &= ~(1 << j)
                        cn[k, i] &= ~(1 << j)
                    if adj_np[j, k]:
                        cn[j, k] &= ~(1 << i)
                        cn[k, j] &= ~(1 << i)
            return False
    
    def try_remove_edge(adj_np, cn, i, j, n):
        """Remove edge (i,j) and update cn."""
        adj_np[i, j] = False
        adj_np[j, i] = False
        for k in range(n):
            if k != i and k != j:
                if adj_np[i, k]:
                    cn[i, k] &= ~(1 << j)
                    cn[k, i] &= ~(1 << j)
                if adj_np[j, k]:
                    cn[j, k] &= ~(1 << i)
                    cn[k, j] &= ~(1 << i)
        cn[i, j] = 0
        cn[j, i] = 0
    
    def try_readd_edge(adj_np, cn, i, j, n):
        """Re-add edge (i,j) and update cn."""
        adj_np[i, j] = True
        adj_np[j, i] = True
        for k in range(n):
            if k != i and k != j:
                if adj_np[i, k]:
                    cn[i, k] |= (1 << j)
                    cn[k, i] |= (1 << j)
                if adj_np[j, k]:
                    cn[j, k] |= (1 << i)
                    cn[k, j] |= (1 << i)
        cn[i, j] = 0
        cn[j, i] = 0
    
    # Main search loop
    for iteration in range(20000):
        if time.time() > deadline:
            break
        
        # Build list of addable non-edges
        non_edges = []
        for (i, j) in all_pairs:
            if not adj_np[i, j] and cn[i, j] == 0:
                non_edges.append((i, j))
        
        if not non_edges:
            # No more edges can be added; try remove+add
            edge_list = [(i, j) for (i, j) in all_pairs if adj_np[i, j]]
            random.shuffle(edge_list)
            improved = False
            for (ri, rj) in edge_list[:20]:
                if time.time() > deadline:
                    break
                try_remove_edge(adj_np, cn, ri, rj, n)
                # Now try to add multiple edges
                added_count = 0
                non_edges2 = []
                for (i, j) in all_pairs:
                    if not adj_np[i, j] and cn[i, j] == 0:
                        non_edges2.append((i, j))
                random.shuffle(non_edges2)
                for (ai, aj) in non_edges2:
                    if time.time() > deadline:
                        break
                    if try_add_edge(adj_np, cn, ai, aj, n):
                        added_count += 1
                try_readd_edge(adj_np, cn, ri, rj, n)
                if added_count > 0:
                    new_edges = count_edges(adj_np)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [[bool(adj_np[i][j]) for j in range(n)] for i in range(n)]
                    improved = True
                    break
            if not improved:
                continue
        else:
            random.shuffle(non_edges)
            improved = False
            for (i, j) in non_edges:
                if time.time() > deadline:
                    break
                if try_add_edge(adj_np, cn, i, j, n):
                    new_edges = count_edges(adj_np)
                    if new_edges > best_edges:
                        best_edges = new_edges
                        best_adj = [[bool(adj_np[i][j]) for j in range(n)] for i in range(n)]
                    improved = True
            if not improved:
                # Try remove+add
                edge_list = [(i, j) for (i, j) in all_pairs if adj_np[i, j]]
                random.shuffle(edge_list)
                for (ri, rj) in edge_list[:15]:
                    if time.time() > deadline:
                        break
                    try_remove_edge(adj_np, cn, ri, rj, n)
                    non_edges2 = []
                    for (i, j) in all_pairs:
                        if not adj_np[i, j] and cn[i, j] == 0:
                            non_edges2.append((i, j))
                    random.shuffle(non_edges2)
                    added = False
                    for (ai, aj) in non_edges2:
                        if time.time() > deadline:
                            break
                        if try_add_edge(adj_np, cn, ai, aj, n):
                            added = True
                            break
                    try_readd_edge(adj_np, cn, ri, rj, n)
                    if added:
                        new_edges = count_edges(adj_np)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_adj = [[bool(adj_np[i][j]) for j in range(n)] for i in range(n)]
                        improved = True
                        break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
