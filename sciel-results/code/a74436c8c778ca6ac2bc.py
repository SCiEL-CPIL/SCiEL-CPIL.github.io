"""Optimized CSP solver using bitmask domains and aggressive arc consistency."""
# EVOLVE-BLOCK-START
import time

def construct():
    START_TIME = time.time()
    DEADLINE = START_TIME + 110.0

    def instance(splits, seed, palette=5):
        from random import Random
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

    def solve_instance(n, edges, lists):
        # Build adjacency list and convert to bitmask domains
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmask domains
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        # Precompute: for each vertex, the bitmask of colors that would conflict
        # with each neighbor (i.e., the neighbor's domain)
        # We'll use a queue-based arc consistency propagation
        
        def propagate_arc_consistency(domains, adj, n):
            """Perform arc consistency. Returns False if domain wipeout."""
            import collections
            queue = collections.deque()
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                # Check if domains[u] needs revision w.r.t. domains[v]
                # A color c in domains[u] is removed if domains[v] is a subset of {c}
                # i.e., domains[v] & ~(1 << c) == 0 and domains[v] != 0
                dv = domains[v]
                du = domains[u]
                if dv == 0:
                    return False
                
                # Colors in du that are the ONLY color in dv must be removed
                # This happens when dv is a singleton and that singleton is in du
                if dv & (dv - 1) == 0:  # dv is a power of 2 (singleton)
                    # Check if du contains this singleton color
                    if du & dv:
                        new_du = du & ~dv
                        if new_du == 0:
                            return False
                        if new_du != du:
                            domains[u] = new_du
                            for x in adj[u]:
                                if x != v:
                                    queue.append((x, u))
            
            return True
        
        # Initial arc consistency
        if not propagate_arc_consistency(domains, adj, n):
            return [0] * n
        
        # Check for empty domains
        for i in range(n):
            if domains[i] == 0:
                return [0] * n
        
        # Backtracking with MRV using bitmask popcount
        color = [-1] * n
        
        def popcount(x):
            return x.bit_count()
        
        def backtrack():
            if time.time() > DEADLINE:
                return False
            
            # MRV: find unassigned vertex with smallest domain
            min_size = n + 1
            min_v = -1
            # Optimization: Local variable access
            doms = domains
            cols = color
            for v in range(n):
                if cols[v] == -1:
                    size = doms[v].bit_count()
                    if size == 0:
                        return False
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size == 1:
                            break  # Can't do better than 1
            
            if min_v == -1:
                return True  # All assigned
            
            # Get colors in domain of min_v
            dv = domains[min_v]
            # Try each color in the domain
            while dv:
                c = (dv & -dv).bit_length() - 1  # Lowest set bit
                bit = 1 << c
                dv ^= bit  # Remove this bit from local copy
                
                # Assign color c to min_v
                color[min_v] = c
                
                # Forward checking: remove c from all unassigned neighbors
                backup = []
                valid = True
                doms = domains
                cols = color
                for nb in adj[min_v]:
                    if cols[nb] == -1:
                        if doms[nb] & bit:
                            backup.append((nb, doms[nb]))
                            doms[nb] ^= bit  # Remove color c
                            if doms[nb] == 0:
                                valid = False
                                break
                
                if valid:
                    # Propagate arc consistency
                    if propagate_arc_consistency(doms, adj, n):
                        if backtrack():
                            return True
                    else:
                        # Domain wipeout, undo and try next color
                        for nb, old_dom in backup:
                            doms[nb] = old_dom
                        cols[min_v] = -1
                        continue
                
                # Restore domains
                for nb, old_dom in backup:
                    doms[nb] = old_dom
                cols[min_v] = -1
            
            return False
        
        if backtrack():
            # Convert to list of ints
            result = []
            for c in color:
                if c == -1:
                    result.append(0)  # Should not happen
                else:
                    result.append(int(c))
            return result
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > DEADLINE:
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
        
        n, edges, lists = instance(splits, seed)
        
        if time.time() > DEADLINE:
            out.append([0] * n)
            continue
            
        sol = solve_instance(n, edges, lists)
        out.append(sol)
        
    return out
# EVOLVE-BLOCK-END
