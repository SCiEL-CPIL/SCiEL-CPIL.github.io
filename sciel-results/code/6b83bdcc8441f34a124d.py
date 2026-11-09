import time
import random
from itertools import combinations, permutations
import numpy as np

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
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                mask_i = adj_bitmasks[c][i]
                if mask_i == 0:
                    continue
                
                mask = 0
                # Path: i -> a -> b -> e -> d
                # For each neighbor a of i:
                a_mask = mask_i
                while a_mask:
                    # Get lowest set bit
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    
                    # b must be neighbor of a, not i
                    b_mask = adj_bitmasks[c][a] & ~(1 << i)
                    if b_mask == 0:
                        continue
                    
                    # For each neighbor b of a (excluding i):
                    b_mask_iter = b_mask
                    while b_mask_iter:
                        b_bit = b_mask_iter & (-b_mask_iter)
                        b = b_bit.bit_length() - 1
                        b_mask_iter ^= b_bit
                        
                        # e must be neighbor of b, not i, a
                        e_mask = adj_bitmasks[c][b] & ~(1 << i) & ~(1 << a)
                        if e_mask == 0:
                            continue
                        
                        # For each neighbor e of b (excluding i, a):
                        e_mask_iter = e_mask
                        while e_mask_iter:
                            e_bit = e_mask_iter & (-e_mask_iter)
                            e = e_bit.bit_length() - 1
                            e_mask_iter ^= e_bit
                            
                            # d must be neighbor of e, not i, a, b
                            d_mask = adj_bitmasks[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                            mask |= d_mask
                
                path4_mask[c][i] = mask
        
        return path4_mask

    def _count_c6s_fast(adj, n, c):
        """Count C6s in color c using numpy matrix operations."""
        # Build adjacency matrix for color c
        A = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == c:
                    A[i][j] = 1
                    A[j][i] = 1
                idx += 1
        
        # A^4[i][j] counts walks of length 4 from i to j
        # But we need simple paths, so this is an upper bound
        # For small n, we can use the exact count via inclusion-exclusion
        # For now, use the walk count as a heuristic
        
        # Compute A^2, A^4
        A2 = A @ A
        A4 = A2 @ A2
        
        # Number of closed walks of length 6 starting from each vertex
        # This overcounts, but gives us a signal
        total = 0
        for i in range(n):
            # Walks of length 6 from i back to i
            # This includes non-simple walks, so it's an upper bound
            pass
        
        # For exact counting, use the cycle enumeration
        count = 0
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
                    count += 1
        return count

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]
        adj_bitmasks = [[0] * n_target for _ in range(4)]

        def _update_bitmasks(i, j, c, add):
            """Update adj_bitmasks when adding/removing edge (i,j) with color c."""
            if add:
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
            else:
                adj_bitmasks[c][i] &= ~(1 << j)
                adj_bitmasks[c][j] &= ~(1 << i)

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1) with smarter ordering."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            path4_mask = _precompute_path4_bitmask(v, adj_bitmasks)

            # Sort vertices by degree in the existing graph (most constrained first)
            # This helps fail-fast
            vertex_order = list(range(v))
            # Sort by number of assigned neighbors (descending) for better pruning
            vertex_order.sort(key=lambda x: sum(1 for y in range(v) if adj[x][y] >= 0), reverse=True)

            # Backtrack over edges in vertex_order
            masks = [0, 0, 0, 0]
            assigned = [False] * v
            
            def _backtrack(pos):
                if time.time() > deadline:
                    return False

                if pos == len(vertex_order):
                    return True

                idx = vertex_order[pos]
                
                # Try each color for edge (v, idx), ordered by constraint tightness
                color_order = sorted(range(4), key=lambda c: (path4_mask[c][idx] & masks[c]).bit_count(), reverse=True)
                
                for c in color_order:
                    # Check constraint: 
                    if path4_mask[c][idx] & masks[c]:
                        continue
                    
                    # Assign
                    masks[c] |= (1 << idx)
                    assigned[idx] = True
                    adj[idx][v] = c
                    adj[v][idx] = c
                    adj_list[c][idx].add(v)
                    adj_list[c][v].add(idx)
                    _update_bitmasks(idx, v, c, True)

                    if _backtrack(pos + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << idx)
                    assigned[idx] = False
                    adj[idx][v] = -1
                    adj[v][idx] = -1
                    adj_list[c][idx].remove(v)
                    adj_list[c][v].remove(idx)
                    _update_bitmasks(idx, v, c, False)

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
        Uses precomputed C6 contributions for speed.
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
        
        # Precompute C6 contribution for each edge and color
        # c6_contrib[c][(u,v)] = number of C6s in color c that use edge (u,v)
        c6_contrib = [[0] * (n * (n - 1) // 2) for _ in range(4)]
        
        def _count_c6s_involving_edge(u, v, c):
            """Count C6s in color c that use edge (u,v)."""
            if adj[u][v] != c:
                return 0
            
            count = 0
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
        
        # Precompute contributions
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j] == c:
                        c6_contrib[c][edge_idx[(i, j)]] = _count_c6s_involving_edge(i, j, c)
        
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
                            # Use precomputed contributions
                            before_count = c6_contrib[old_c][e]
                            after_count = c6_contrib[new_c][e] if adj[i][j] == new_c else 0
                            
                            # If new_c is not the current color, we need to compute
                            # how many C6s would be created in new_c
                            if adj[i][j] != new_c:
                                # Temporarily assume the flip and count
                                # This is expensive, so only do it if promising
                                if before_count > 0:
                                    # Quick check: would flipping create C6s?
                                    # Count paths of length 4 from i to j in new_c
                                    after_count = 0
                                    for a in adj_list[new_c][j]:
                                        if a == i:
                                            continue
                                        for b in adj_list[new_c][a]:
                                            if b == j or b == a:
                                                continue
                                            for e2 in adj_list[new_c][b]:
                                                if e2 == j or e2 == a or e2 == b:
                                                    continue
                                                for d in adj_list[new_c][e2]:
                                                    if d == j or d == a or d == b or d == e2:
                                                        continue
                                                    if adj[d][i] == new_c:
                                                        after_count += 1
                            
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
                                
                                # Update contributions
                                c6_contrib[old_c][e] = 0
                                c6_contrib[new_c][e] = after_count
                                
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
    # Use more seeds for larger n, fewer for smaller n
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 25:
            break

        found = False
        # More seeds for larger n
        max_seeds = 200 if n >= 15 else 100 if n >= 12 else 50
        for seed in range(max_seeds):
            if time.time() > deadline - 25:
                break

            colors = _search(n, seed, deadline - 25)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break
                else:
                    # Try local search to fix it
                    polished = _local_search(colors, n, 5000, deadline - 25)
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
