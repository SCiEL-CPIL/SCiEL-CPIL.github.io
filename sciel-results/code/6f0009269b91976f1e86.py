"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def solve_list_coloring(n, edges, lists):
        """
        Solve list coloring using Bitset-based Backtracking with MRV.
        This approach uses integer bitmasks for domains, which is significantly
        faster than Python sets for small domain sizes (3 colors).
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute degree for heuristic tie-breaking (higher degree first if MRV ties)
        degree = [len(adj[v]) for v in range(n)]
        
        # Initialize domains as bitmasks
        # Color c is represented by bit (1 << c)
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
            
        color = [-1] * n
        unassigned = [True] * n
        num_unassigned = n
        
        # Helper to get list of colors from mask
        def mask_to_colors(mask):
            colors = []
            while mask:
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                colors.append(c)
                mask ^= lsb
            return colors

        def backtrack():
            nonlocal num_unassigned
            
            # MRV: pick unassigned vertex with smallest domain size
            # Tie-break by highest degree
            min_size = 10
            min_v = -1
            max_deg = -1
            for v in range(n):
                if unassigned[v]:
                    size = domains[v].bit_count()
                    if size < min_size:
                        min_size = size
                        min_v = v
                        max_deg = degree[v]
                        if min_size == 1:
                            break
                    elif size == min_size and degree[v] > max_deg:
                        min_v = v
                        max_deg = degree[v]
            
            if min_v == -1:
                return True  # All assigned
            
            # If any unassigned vertex has empty domain, fail (should be caught by forward checking, but safe)
            if min_size == 0:
                return False
            
            # Try each color in domain of min_v
            colors = mask_to_colors(domains[min_v])
            
            for c in colors:
                # Assign
                color[min_v] = c
                unassigned[min_v] = False
                num_unassigned -= 1
                bit_c = 1 << c
                old_domains = []
                
                # Forward checking: remove c from neighbors' domains
                conflict = False
                for u in adj[min_v]:
                    if unassigned[u]:
                        if domains[u] & bit_c:
                            old_domains.append((u, domains[u]))
                            domains[u] ^= bit_c
                            if domains[u] == 0:
                                conflict = True
                                break
                
                if not conflict:
                    if backtrack():
                        return True
                
                # Undo forward checking
                for u, old_mask in old_domains:
                    domains[u] = old_mask
                    
                # Undo assignment
                color[min_v] = -1
                unassigned[min_v] = True
                num_unassigned += 1
            
            return False
        
        if backtrack():
            return [int(c) for c in color]
        else:
            # Fallback: should not happen, but return something
            return [lists[v][0] for v in range(n)]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_list_coloring(n, edges, lists)
        # Ensure all values are Python ints
        coloring = [int(c) for c in coloring]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
