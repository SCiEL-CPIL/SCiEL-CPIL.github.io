import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Precompute all 60 distinct Hamiltonian cycles on 6 labeled vertices
    # for final verification
    _cycles_cache = None

    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            # Generate unique cycles up to rotation and reflection
            # Fix vertex 0 at start. Permute 1..5.
            # 5! = 120 permutations. / 2 for reflection = 60 cycles.
            for perm in __import__('itertools').permutations(range(1, 6)):
                cycle = (0,) + perm
                edges = frozenset(frozenset((cycle[k], cycle[(k + 1) % 6])) for k in range(6))
                if edges not in seen:
                    seen.add(edges)
                    cycles.append(cycle)
            _cycles_cache = cycles
        return _cycles_cache

    def _verify(colors, n):
        """Verify no color class contains a C6."""
        if n < 6:
            return True
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1

        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)

        cycles = _get_cycles()

        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if adj[u][w] != c:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    def _precompute_path4_bitmask(v, adj_bitmasks):
        """
        Compute path4_mask[c][i] for subgraph on vertices 0..v-1 using bitmasks.
        adj_bitmasks[c][i] is a bitmask of neighbors of i in color c.
        path4_mask[c][i] is a bitmask where bit d is set if there is a 
        simple path of length 4 from i to d in color c graph.
        Optimized: use iterative bit manipulation with early termination.
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            # Precompute for each vertex the set of vertices at distance exactly 2, 3, 4
            # dist2[c][i]: vertices reachable from i in exactly 2 steps
            # dist3[c][i]: vertices reachable from i in exactly 3 steps  
            # dist4[c][i]: vertices reachable from i in exactly 4 steps
            dist2 = [0] * v
            dist3 = [0] * v
            dist4 = [0] * v
            
            for i in range(v):
                mask_i = adj_bitmasks[c][i]
                if mask_i == 0:
                    continue
                
                # dist2: i -> a -> b, b != i
                d2 = 0
                a_mask = mask_i
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    # b neighbors of a, exclude i
                    d2 |= adj_bitmasks[c][a] & ~(1 << i)
                dist2[i] = d2
                
                # dist3: i -> a -> b -> e, e not in {i, a, b}
                d3 = 0
                a_mask = mask_i
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    b_mask = adj_bitmasks[c][a] & ~(1 << i)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        # e neighbors of b, exclude i, a
                        d3 |= adj_bitmasks[c][b] & ~(1 << i) & ~(1 << a)
                dist3[i] = d3
                
                # dist4: i -> a -> b -> e -> d, d not in {i, a, b, e}
                d4 = 0
                a_mask = mask_i
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    b_mask = adj_bitmasks[c][a] & ~(1 << i)
                    while b_mask:
                        b_bit = b_mask & (-b_mask)
                        b = b_bit.bit_length() - 1
                        b_mask ^= b_bit
                        e_mask = adj_bitmasks[c][b] & ~(1 << i) & ~(1 << a)
                        while e_mask:
                            e_bit = e_mask & (-e_mask)
                            e = e_bit.bit_length() - 1
                            e_mask ^= e_bit
                            # d neighbors of e, exclude i, a, b
                            d4 |= adj_bitmasks[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                dist4[i] = d4
                
                path4_mask[c][i] = d4
        
        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}.
        Optimized with:
        1. Better edge ordering (process vertices in order, but within each vertex,
           order edges by degree in current partial coloring)
        2. Color ordering: try colors with fewer conflicts first
        3. Forward checking: maintain constraint propagation
        """
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_bitmasks = [[0] * n_target for _ in range(4)]
        
        # Track assigned edges for constraint checking
        assigned_edges = [0] * (n_target * (n_target - 1) // 2)
        
        def _edge_idx(i, j):
            """Get index of edge (i,j) in lexicographic order."""
            if i > j:
                i, j = j, i
            return i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)

        def _update_bitmasks(i, j, c, add):
            """Update adj_bitmasks when adding/removing edge (i,j) with color c."""
            if add:
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
            else:
                adj_bitmasks[c][i] &= ~(1 << j)
                adj_bitmasks[c][j] &= ~(1 << i)

        # Incremental path4 tracking: for each vertex i and color c,
        # track which vertices are at distance 4 from i in color c
        # We'll recompute as needed but cache when possible
        
        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1).
            Use heuristic ordering: process edges in order of increasing
            constraint tightness.
            """
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            path4_mask = _precompute_path4_bitmask(v, adj_bitmasks)

            # Order edges: process vertex indices in order, but we can
            # use a heuristic to pick the most constrained edge first
            # For now, process in natural order (0, 1, ..., v-1)
            
            # masks[c] tracks which vertices have been assigned color c to edge (v, *)
            masks = [0, 0, 0, 0]
            
            # For forward checking, track for each unassigned vertex u < v,
            # which colors are still available given current assignments
            available_colors = [set([0,1,2,3]) for _ in range(v)]
            
            def _check_available(u, c, masks):
                """Check if color c is still available for edge (v, u)."""
                if c not in available_colors[u]:
                    return False
                if path4_mask[c][u] & masks[c]:
                    return False
                return True

            def _propagate(u, c, masks, add):
                """Update available_colors when edge (v,u) gets color c."""
                if add:
                    # For each other unassigned vertex w, check if c is still available
                    # This is expensive, so we do lazy checking instead
                    pass
                else:
                    pass

            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                u = idx  # vertex index for edge (v, u)
                
                # Try colors in order of increasing conflicts
                # Score each color by how many it blocks
                color_scores = []
                for c in range(4):
                    if not _check_available(u, c, masks):
                        continue
                    # Count how many other vertices this color would block
                    # (approximate: count bits in path4_mask[c][u] & masks[c] is 0,
                    #  so count future conflicts)
                    score = 0
                    for w in range(idx + 1, v):
                        if path4_mask[c][w] & masks[c]:
                            score += 1
                    color_scores.append((score, c))
                
                color_scores.sort()
                
                for score, c in color_scores:
                    # Assign
                    masks[c] |= (1 << u)
                    adj[u][v] = c
                    adj[v][u] = c
                    _update_bitmasks(u, v, c, True)

                    if _backtrack(idx + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << u)
                    adj[u][v] = -1
                    adj[v][u] = -1
                    _update_bitmasks(u, v, c, False)

                return False

            return _backtrack(0)

        # Try to build the coloring vertex by vertex
        for v in range(1, n_target):
            if time.time() > deadline:
                return None
            
            if not _assign_vertex(v):
                return None

        # Extract colors in lexicographic order
        colors = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                colors.append(adj[i][j])
        return colors

    def _local_search(colors, n, max_iters, deadline):
        """
        Local search polish: try to reduce C6 count by flipping edge colors.
        Uses incremental C6 counting for speed.
        Returns improved colors or None if no improvement found.
        """
        if time.time() > deadline:
            return None
        
        # Build adjacency from colors
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                adj[i][j] = colors[idx]
                adj[j][i] = colors[idx]
                idx += 1
        
        adj_list = [[set() for _ in range(n)] for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_list[c][i].add(j)
                adj_list[c][j].add(i)
        
        # Precompute edge index mapping for fast lookup
        edge_idx = {}
        idx2 = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx[(i, j)] = idx2
                idx2 += 1
        
        # Precompute for each edge (u,v) and color c, the list of (other 4 vertices, cycle)
        # that would form a C6 with edge (u,v) in color c
        # A C6 containing edge (u,v) in color c needs 4 more vertices a,b,d,e
        # such that u-a-b-e-d-v-u (or similar) are all color c edges
        # Actually, let's count C6s by checking all 6-subsets containing {u,v}
        
        def _count_c6s_involving_edge(u, v, c):
            """Count C6s in color c that use edge (u,v)."""
            count = 0
            # We need to find 4 more vertices from the remaining n-2
            # such that they form a path of length 4 from v to u (or u to v)
            # in color c, combined with edge (u,v) to form a C6
            # Path: v -> a -> b -> e -> d -> u, all edges in color c
            # And (u,v) is in color c
            if adj[u][v] != c:
                return 0
            
            # Find all simple paths of length 4 from v to u in color c
            # v -> a -> b -> e -> d -> u
            # a in adj_list[c][v], a != u
            # b in adj_list[c][a], b != v, a
            # e in adj_list[c][b], e != v, a, b
            # d in adj_list[c][e], d != v, a, b, e
            # u in adj_list[c][d], u != v, a, b, e, d
            
            for a in adj_list[c][v]:
                if a == u:
                    continue
                for b in adj_list[c][a]:
                    if b == v or b == a:
                        continue
                    for e in adj_list[c][b]:
                        if e == v or e == a or e == b:
                            continue
                        for d in adj_list[c][e]:
                            if d == v or d == a or d == b or d == e:
                                continue
                            if adj[d][u] == c:
                                count += 1
            return count
        
        def _count_all_c6s():
            """Count total C6s across all colors."""
            total = 0
            for c in range(4):
                for subset in combinations(range(n), 6):
                    for cycle in _get_cycles():
                        ok = True
                        for k in range(6):
                            u = subset[cycle[k]]
                            w = subset[cycle[(k + 1) % 6]]
                            if adj[u][w] != c:
                                ok = False
                                break
                        if ok:
                            total += 1
            return total
        
        current_c6s = _count_all_c6s()
        if current_c6s == 0:
            return colors
        
        best_colors = colors[:]
        best_c6s = current_c6s
        rng_ls = random.Random(42)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            e = rng_ls.randint(0, n*(n-1)//2 - 1)
            # Find (i,j) for index e
            found = False
            for i in range(n):
                for j in range(i+1, n):
                    if edge_idx[(i,j)] == e:
                        old_c = adj[i][j]
                        # Try all 3 other colors
                        improved = False
                        for new_c in range(4):
                            if new_c == old_c:
                                continue
                            # Count C6s involving this edge before and after
                            # Before: C6s in old_c using edge (i,j)
                            before_count = _count_c6s_involving_edge(i, j, old_c)
                            # After: C6s in new_c using edge (i,j)
                            after_count = _count_c6s_involving_edge(i, j, new_c)
                            
                            # Also need to account for C6s that don't use this edge
                            # but the flip might affect them? No, flipping one edge
                            # only affects C6s that use that edge.
                            
                            delta = after_count - before_count
                            if delta < 0:
                                # This flip reduces C6 count
                                # Apply the flip
                                adj[i][j] = new_c
                                adj[j][i] = new_c
                                adj_list[old_c][i].remove(j)
                                adj_list[old_c][j].remove(i)
                                adj_list[new_c][i].add(j)
                                adj_list[new_c][j].add(i)
                                
                                current_c6s += delta
                                if current_c6s < best_c6s:
                                    best_c6s = current_c6s
                                    best_colors = []
                                    for ii in range(n):
                                        for jj in range(ii+1, n):
                                            best_colors.append(adj[ii][jj])
                                improved = True
                                break
                            elif delta == 0 and current_c6s > 0:
                                # Accept with some probability (simulated annealing-like)
                                if rng_ls.random() < 0.1:
                                    adj[i][j] = new_c
                                    adj[j][i] = new_c
                                    adj_list[old_c][i].remove(j)
                                    adj_list[old_c][j].remove(i)
                                    adj_list[new_c][i].add(j)
                                    adj_list[new_c][j].add(i)
                                    improved = True
                                    break
                        if improved:
                            found = True
                            break
                if found:
                    break
            
            if current_c6s == 0:
                break
        
        if best_c6s == 0:
            return best_colors
        return best_colors if best_c6s < current_c6s else None

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    # Use adaptive seed count: spend more time on larger n
    time_budget = deadline - 25
    
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > time_budget:
            break

        found = False
        # Adaptive number of seeds based on remaining time
        # Start with fewer seeds for n=17, more for smaller n if needed
        if n >= 15:
            max_seeds = 30
        elif n >= 13:
            max_seeds = 60
        else:
            max_seeds = 120
            
        for seed in range(max_seeds):
            if time.time() > time_budget:
                break

            colors = _search(n, seed, time_budget)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    # Try local search to fix it
                    polished = _local_search(colors, n, 2000, time_budget)
                    if polished is not None and _verify(polished, n):
                        best_n = n
                        best_colors = polished
                        found = True
                        break

        if found:
            break

    if best_n == 0:
        # Fallback to a known small valid coloring
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        
        if _verify(colors, n):
            best_n = n
            best_colors = colors
        else:
            n = 5
            best_n = n
            best_colors = [0] * (n*(n-1)//2)

    return {'n': best_n, 'colors': best_colors}
