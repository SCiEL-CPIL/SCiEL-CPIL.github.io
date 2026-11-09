import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin

    _cycles_cache = None

    def _get_cycles():
        nonlocal _cycles_cache
        if _cycles_cache is None:
            cycles = []
            seen = set()
            for perm in permutations(range(1, 6)):
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

        # Build adjacency bitmasks for each color
        adj_masks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_masks[c][i] |= (1 << j)
                adj_masks[c][j] |= (1 << i)

        cycles = _get_cycles()

        for c in range(4):
            for subset in combinations(range(n), 6):
                for cycle in cycles:
                    ok = True
                    for k in range(6):
                        u = subset[cycle[k]]
                        w = subset[cycle[(k + 1) % 6]]
                        if not (adj_masks[c][u] >> w) & 1:
                            ok = False
                            break
                    if ok:
                        return False
        return True

    def _precompute_path4_bitmask(v, adj_masks):
        """
        Compute path4_mask[c][i] using bitmask operations.
        path4_mask[c][i] is a bitmask where bit d is set if there is a 
        simple path of length 4 from i to d in color c graph.
        
        Path: i -> a -> b -> e -> d
        We need: a in N(i), b in N(a)\\{i}, e in N(b)\\{i,a}, d in N(e)\\{i,a,b}
        
        Using bitmasks:
        For each i, a in N(i):
            candidates_b = N(a) & ~(1<<i)
            For each b in candidates_b:
                candidates_e = N(b) & ~(1<<i) & ~(1<<a)
                For each e in candidates_e:
                    candidates_d = N(e) & ~(1<<i) & ~(1<<a) & ~(1<<b)
                    mask |= candidates_d
        """
        path4_mask = [[0] * v for _ in range(4)]
        
        for c in range(4):
            for i in range(v):
                neighbors_i = adj_masks[c][i]
                if neighbors_i == 0:
                    continue
                
                mask = 0
                # Iterate over neighbors a of i
                a_bits = neighbors_i
                while a_bits:
                    a_bit = a_bits & (-a_bits)
                    a = a_bit.bit_length() - 1
                    a_bits ^= a_bit
                    
                    # b in N(a), b != i
                    b_bits = adj_masks[c][a] & ~(1 << i)
                    while b_bits:
                        b_bit = b_bits & (-b_bits)
                        b = b_bit.bit_length() - 1
                        b_bits ^= b_bit
                        
                        # e in N(b), e not in {i, a}
                        e_bits = adj_masks[c][b] & ~(1 << i) & ~(1 << a)
                        while e_bits:
                            e_bit = e_bits & (-e_bits)
                            e = e_bit.bit_length() - 1
                            e_bits ^= e_bit
                            
                            # d in N(e), d not in {i, a, b}
                            d_bits = adj_masks[c][e] & ~(1 << i) & ~(1 << a) & ~(1 << b)
                            mask |= d_bits
        
                path4_mask[c][i] = mask
        
        return path4_mask

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        # adj_masks[c][i] = bitmask of neighbors of i in color c
        adj_masks = [[0] * n_target for _ in range(4)]

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            path4_mask = _precompute_path4_bitmask(v, adj_masks)

            # Backtrack over edges (v, 0), (v, 1), ..., (v, v-1)
            masks = [0, 0, 0, 0]
            
            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                # Try each color for edge (v, idx)
                # Heuristic: try colors that allow more future options
                # Sort colors by how many vertices they conflict with
                color_order = []
                for c in range(4):
                    conflicts = path4_mask[c][idx] & masks[c]
                    color_order.append((conflicts.bit_count(), c))
                color_order.sort()
                
                for _, c in color_order:
                    # Check constraint: 
                    if path4_mask[c][idx] & masks[c]:
                        continue
                    
                    # Assign
                    masks[c] |= (1 << idx)
                    adj[idx][v] = c
                    adj[v][idx] = c
                    adj_masks[c][idx] |= (1 << v)
                    adj_masks[c][v] |= (1 << idx)

                    if _backtrack(idx + 1):
                        return True

                    # Unassign
                    masks[c] &= ~(1 << idx)
                    adj[idx][v] = -1
                    adj[v][idx] = -1
                    adj_masks[c][idx] &= ~(1 << v)
                    adj_masks[c][v] &= ~(1 << idx)

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

    def _local_search(colors, n, deadline):
        """
        Try to extend a valid coloring of K_n to K_{n+1} by local search.
        Returns (success, new_colors) or (False, None).
        """
        if time.time() > deadline:
            return False, None
        
        # Build adjacency from colors
        adj = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = c
                adj[j][i] = c
                idx += 1
        
        adj_masks = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                c = adj[i][j]
                adj_masks[c][i] |= (1 << j)
                adj_masks[c][j] |= (1 << i)
        
        # Try to add vertex n
        # Use randomized local search to find valid colors for edges (n, 0), ..., (n, n-1)
        best = None
        for attempt in range(100):
            if time.time() > deadline:
                return False, None
            
            # Random initial assignment
            assignment = [random.randint(0, 3) for _ in range(n)]
            
            # Iteratively fix conflicts
            for iteration in range(50):
                if time.time() > deadline:
                    return False, None
                
                # Check if current assignment is valid
                valid = True
                for c in range(4):
                    # Get vertices assigned color c
                    verts = [i for i in range(n) if assignment[i] == c]
                    if len(verts) < 5:
                        continue
                    # Check all C6 in these vertices
                    for subset in combinations(verts, 6):
                        for cycle in _get_cycles():
                            ok = True
                            for k in range(6):
                                u = subset[cycle[k]]
                                w = subset[cycle[(k + 1) % 6]]
                                if not (adj_masks[c][u] >> w) & 1:
                                    ok = False
                                    break
                            if ok:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Build new coloring
                    new_colors = colors[:]
                    for i in range(n):
                        new_colors.append(assignment[i])
                    return True, new_colors
                
                # Find a vertex to reassign
                # Pick a random vertex and try a different color
                i = random.randint(0, n - 1)
                old_c = assignment[i]
                new_c = (old_c + 1 + random.randint(0, 2)) % 4
                assignment[i] = new_c
        
        return False, None

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 20:
            break

        found = False
        # Try multiple seeds
        for seed in range(30):
            if time.time() > deadline - 20:
                break

            colors = _search(n, seed, deadline - 20)
            if colors is not None:
                # Verify
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    
                    # Try to extend to n+1 using local search
                    if n < 20:
                        success, extended = _local_search(colors, n, deadline - 10)
                        if success:
                            best_n = n + 1
                            best_colors = extended
                    
                    break
                else:
                    continue

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
