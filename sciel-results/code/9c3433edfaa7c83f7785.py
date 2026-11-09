"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time

    deadline = time.time() + 140.0  # safety margin before 150s kill

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

    def backtrack_coloring(n, edges, lists):
        """
        Complete CSP backtracking with MRV + forward checking + bitmask domains.
        Guarantees finding a valid coloring for Alon-Tarsi-guaranteed instances.
        """
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for fast intersection
        # Bit i set means color i is available
        domains = []
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains.append(mask)
        
        # We'll use a recursive backtracking with MRV heuristic
        # To avoid deep recursion issues, we use an iterative approach with explicit stack
        # But for n up to ~400, recursion depth should be fine with sys.setrecursionlimit
        
        import sys
        sys.setrecursionlimit(10000)
        
        colour = [-1] * n
        
        def mrv_order():
            """Return vertices sorted by current domain size (MRV heuristic)."""
            order = list(range(n))
            # Sort by number of available colors (popcount of mask)
            # Only consider uncolored vertices
            order = [v for v in order if colour[v] == -1]
            order.sort(key=lambda v: domains[v].bit_count())
            return order
        
        def backtrack(depth):
            if time.time() > deadline:
                return False
            
            # Find uncolored vertex with minimum remaining values
            if depth == n:
                return True
            
            # MRV: pick uncolored vertex with fewest options
            best_v = -1
            best_count = 10  # max is 3, but use 10 as sentinel
            for v in range(n):
                if colour[v] == -1:
                    cnt = domains[v].bit_count()
                    if cnt == 0:
                        return False  # domain wipeout
                    if cnt < best_count:
                        best_count = cnt
                        best_v = v
                        if cnt == 1:
                            break
            
            if best_v == -1:
                return True  # all colored
            
            # Try each color in the domain of best_v
            mask = domains[best_v]
            for c in range(5):
                if not (mask & (1 << c)):
                    continue
                
                # Assign color c to best_v
                colour[best_v] = c
                domains[best_v] = 0  # mark as colored by clearing domain
                
                # Forward checking: propagate to neighbors
                saved_domains = []
                consistent = True
                for nb in adj[best_v]:
                    if colour[nb] == -1:
                        old = domains[nb]
                        new = old & ~(1 << c)
                        if new == 0:
                            consistent = False
                            break
                        domains[nb] = new
                        saved_domains.append((nb, old))
                
                if not consistent:
                    # Undo
                    for nb, old in saved_domains:
                        domains[nb] = old
                    colour[best_v] = -1
                    continue
                
                # Recurse
                if backtrack(depth + 1):
                    return True
                
                # Undo
                for nb, old in saved_domains:
                    domains[nb] = old
                colour[best_v] = -1
                domains[best_v] = mask
            
            return False
        
        # Initialize: set domains for all vertices
        # The domains array is already set from lists
        success = backtrack(0)
        
        if not success:
            # Fallback: should not happen for Alon-Tarsi instances, but just in case
            # Try a simple greedy as last resort
            for v in range(n):
                if colour[v] == -1:
                    taken = set()
                    for nb in adj[v]:
                        if colour[nb] != -1:
                            taken.add(colour[nb])
                    for c in lists[v]:
                        if c not in taken:
                            colour[v] = c
                            break
                    else:
                        colour[v] = lists[v][0]
        
        # Ensure all are Python ints
        return [int(c) for c in colour]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Return best effort: use greedy for remaining
            n, edges, lists = instance(splits, seed)
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            colour = [-1] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = set()
                for nb in adj[v]:
                    if colour[nb] != -1:
                        taken.add(colour[nb])
                for c in lists[v]:
                    if c not in taken:
                        colour[v] = c
                        break
                else:
                    colour[v] = lists[v][0]
            out.append([int(c) for c in colour])
            continue
        
        n, edges, lists = instance(splits, seed)
        colour = backtrack_coloring(n, edges, lists)
        out.append(colour)
    
    # Final type check: ensure all are Python ints in lists
    result = []
    for colouring in out:
        result.append([int(c) for c in colouring])
    
    return result
# EVOLVE-BLOCK-END
