"""
Combined approach: vertex-by-vertex backtracking with Path4 precomputation
for C6-free 4-coloring of complete graphs.

Strategy:
- Build K_n vertex by vertex (0, 1, ..., n-1).
- When adding vertex v, assign colors to edges (v, 0), ..., (v, v-1).
- A C6 is formed iff, for some color c, two neighbors i < d of v in color c
  have a simple path of length 4 between them in the existing color-c graph.
- Precompute Path4[c][i][d] for the current subgraph to make the constraint
  check O(1) per (c, i, d) triple.
- Use backtracking with randomized restarts and a time budget.
- Improvement: Use bitmask-based path length 4 computation for speed.
- Improvement: Try multiple seeds and target sizes more aggressively.
"""
import time
import random
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s total, leave buffer)

    # Precompute all 60 distinct Hamiltonian cycles on 6 labeled vertices
    # for final verification
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

    def _precompute_path4_bitmask(v, adj_list):
        """
        For the subgraph on vertices 0..v-1, compute Path4[c][i][d]:
        True iff there is a simple path of exactly 4 edges from i to d.
        Uses bitmask representation for efficiency.
        """
        Path4 = [[[False] * v for _ in range(v)] for _ in range(4)]
        
        for c in range(4):
            # Build bitmask adjacency for each vertex in color c
            # adj_mask[i] is a bitmask where bit j is set if (i,j) is in color c
            adj_mask = [0] * v
            for i in range(v):
                mask = 0
                for j in adj_list[c][i]:
                    if j < v:
                        mask |= (1 << j)
                adj_mask[i] = mask
            
            # For each pair (i, d), check if there's a path of length 4
            for i in range(v):
                if adj_mask[i] == 0:
                    continue
                for d in range(v):
                    if i == d:
                        continue
                    found = False
                    # Path: i -> a -> b -> e -> d
                    # We need to check if there exists a, b, e such that:
                    # (i,a), (a,b), (b,e), (e,d) are all in color c
                    # and all vertices are distinct.
                    
                    # Enumerate a (neighbors of i)
                    a_mask = adj_mask[i]
                    while a_mask:
                        a_bit = a_mask & (-a_mask)
                        a = a_bit.bit_length() - 1
                        a_mask ^= a_bit
                        
                        if a == d:
                            continue
                        
                        # Enumerate b (neighbors of a, not i)
                        b_mask = adj_mask[a] & ~(1 << i)
                        while b_mask:
                            b_bit = b_mask & (-b_mask)
                            b = b_bit.bit_length() - 1
                            b_mask ^= b_bit
                            
                            if b == d or b == i:
                                continue
                            
                            # Enumerate e (neighbors of b, not i, a)
                            e_mask = adj_mask[b] & ~(1 << i) & ~(1 << a)
                            while e_mask:
                                e_bit = e_mask & (-e_mask)
                                e = e_bit.bit_length() - 1
                                e_mask ^= e_bit
                                
                                if e == d or e == i or e == a:
                                    continue
                                
                                # Check if e is adjacent to d
                                if adj_mask[e] & (1 << d):
                                    found = True
                                    break
                            
                            if found:
                                break
                        
                        if found:
                            break
                    
                    Path4[c][i][d] = found
        
        return Path4

    def _search(n_target, seed, deadline):
        """Backtracking search for a valid coloring of K_{n_target}."""
        rng = random.Random(seed)
        adj = [[-1] * n_target for _ in range(n_target)]
        adj_list = [[set() for _ in range(n_target)] for _ in range(4)]

        def _assign_vertex(v):
            """Assign colors to edges (v, 0), ..., (v, v-1)."""
            if time.time() > deadline:
                return False

            if v == 0:
                return True

            # Precompute Path4 for the current subgraph on vertices 0..v-1
            Path4 = _precompute_path4_bitmask(v, adj_list)

            # We need to assign colors to (v, 0), (v, 1), ..., (v, v-1)
            # The constraint is: for each color c, if (v, i) and (v, d) are both color c,
            # then Path4[c][i][d] must be False.
            # We backtrack over i = 0, 1, ..., v-1.

            # To improve pruning, order the vertices by how constrained they are.
            # A vertex i is more constrained if it has many neighbors d where Path4[c][i][d] is True.
            # But for simplicity, we'll use a randomized order based on the seed.
            
            # Actually, let's stick with sequential order but try colors in a randomized order.
            color_order = list(range(4))
            rng.shuffle(color_order)

            def _backtrack(idx):
                if time.time() > deadline:
                    return False

                if idx == v:
                    return True

                i = idx
                # Try each color in randomized order
                for c in color_order:
                    # Check constraint: for all d < i where adj[d][v] == c,
                    # Path4[c][i][d] must be False
                    valid = True
                    for d in range(i):
                        if adj[d][v] == c:
                            if Path4[c][i][d]:
                                valid = False
                                break
                    if not valid:
                        continue

                    adj[i][v] = c
                    adj[v][i] = c
                    adj_list[c][i].add(v)
                    adj_list[c][v].add(i)

                    if _backtrack(idx + 1):
                        return True

                    adj[i][v] = -1
                    adj[v][i] = -1
                    adj_list[c][i].remove(v)
                    adj_list[c][v].remove(i)

                return False

            return _backtrack(0)

        # Try to build the coloring
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

    best_n = 0
    best_colors = None

    # Try to find a coloring for n from 17 down to 10
    # Use more seeds for larger n to increase chance of finding a solution
    for n in [17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline - 10:
            break

        found = False
        # Use more seeds for n=17, fewer for smaller n
        num_seeds = 50 if n >= 16 else 30 if n >= 14 else 20
        for seed in range(num_seeds):
            if time.time() > deadline - 10:
                break

            colors = _search(n, seed, deadline - 10)
            if colors is not None:
                if _verify(colors, n):
                    best_n = n
                    best_colors = colors
                    found = True
                    break

        if found:
            break

    if best_n == 0:
        # Fallback: construct a small valid coloring
        # For n=9, use a simple pattern
        n = 9
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append((j - i) % 4)
        # Verify
        if not _verify(colors, n):
            # Try another pattern
            n = 8
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    colors.append((j - i) % 4)
            if not _verify(colors, n):
                n = 7
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append((j - i) % 4)
        best_n = n
        best_colors = colors

    return {'n': best_n, 'colors': best_colors}
