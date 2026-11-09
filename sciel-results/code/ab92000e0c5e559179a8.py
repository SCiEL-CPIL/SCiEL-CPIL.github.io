"""Optimized CSP solver for List Coloring on Planar Bipartite Graphs."""
# EVOLVE-BLOCK-START
import time
import collections

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
        # Build adjacency list
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
        
        # Check for immediate failures
        for i in range(n):
            if domains[i] == 0:
                return [0] * n

        # Arc Consistency Propagation
        # We use a queue of arcs (u, v) where u's domain might be reduced by v's domain.
        # Optimization: Only singleton domains in binary CSP cause immediate removals in neighbors.
        # However, if v is not singleton, we don't remove anything from u immediately unless we do full AC.
        # Given the sparsity and small domain size (3 colors), full AC is cheap.
        
        def propagate(domains, adj, n, initial_arcs=None):
            queue = collections.deque()
            if initial_arcs:
                queue.extend(initial_arcs)
            else:
                for u in range(n):
                    for v in adj[u]:
                        queue.append((u, v))
            
            while queue:
                u, v = queue.popleft()
                dv = domains[v]
                if dv == 0:
                    return False
                
                # If v is singleton, remove that value from u
                if dv & (dv - 1) == 0:
                    du = domains[u]
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

        if not propagate(domains, adj, n):
            return [0] * n

        # Greedy pre-pass for singletons
        color = [-1] * n
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if color[i] == -1 and domains[i] & (domains[i] - 1) == 0:
                    c = (domains[i] & -domains[i]).bit_length() - 1
                    color[i] = c
                    changed = True
                    bit = 1 << c
                    for nb in adj[i]:
                        if color[nb] == -1:
                            if domains[nb] & bit:
                                domains[nb] ^= bit
                                if domains[nb] == 0:
                                    return [0] * n
            if changed:
                if not propagate(domains, adj, n):
                    return [0] * n
                for i in range(n):
                    if domains[i] == 0:
                        return [0] * n

        # Backtracking
        def backtrack():
            if time.time() > DEADLINE:
                return False
            
            # MRV + Degree Heuristic
            min_size = n + 1
            min_v = -1
            min_deg = -1
            for v in range(n):
                if color[v] == -1:
                    size = domains[v].bit_count()
                    if size == 0:
                        return False
                    if size < min_size or (size == min_size and len(adj[v]) > min_deg):
                        min_size = size
                        min_v = v
                        min_deg = len(adj[v])
                        if size == 1:
                            break
            
            if min_v == -1:
                return True
            
            dv = domains[min_v]
            # Sort colors by least constraining value (optional, but can help)
            # For now, just iterate through available bits
            while dv:
                c = (dv & -dv).bit_length() - 1
                bit = 1 << c
                dv ^= bit
                
                color[min_v] = c
                
                backup = []
                valid = True
                for nb in adj[min_v]:
                    if color[nb] == -1:
                        if domains[nb] & bit:
                            backup.append((nb, domains[nb]))
                            domains[nb] ^= bit
                            if domains[nb] == 0:
                                valid = False
                                break
                            if domains[nb] & (domains[nb] - 1) == 0:
                                # Propagate this singleton
                                # We could push to queue, but simple recursive AC is easier here
                                # For efficiency, we just check if this causes immediate wipeout in neighbors
                                pass # Full propagation is handled by the 'valid' check and subsequent backtrack calls
                                # Actually, we should propagate to ensure consistency
                                
                if valid:
                    # Run propagation to enforce AC
                    # We need to know which arcs to check. All neighbors of min_v are affected.
                    # And any neighbor of a neighbor that became singleton.
                    # Simpler: just run full AC on the affected subgraph or all.
                    # Given small n, running AC on all arcs is acceptable if optimized.
                    # But we can be smarter.
                    
                    # Let's just call propagate with the arcs affected by min_v
                    affected_arcs = []
                    for nb in adj[min_v]:
                        if color[nb] == -1:
                            for x in adj[nb]:
                                if x != min_v and color[x] == -1:
                                    affected_arcs.append((x, nb))
                    
                    if propagate(domains, adj, n, initial_arcs=affected_arcs):
                        if backtrack():
                            return True
                    else:
                        # Restore
                        for nb, old_dom in backup:
                            domains[nb] = old_dom
                        color[min_v] = -1
                        continue

                # Restore
                for nb, old_dom in backup:
                    domains[nb] = old_dom
                color[min_v] = -1
            
            return False

        if backtrack():
            # Ensure all are assigned and valid
            result = []
            for i in range(n):
                if color[i] == -1:
                    return [0] * n
                # Verify color is in list and neighbors differ
                if not (domains[i] & (1 << color[i])): # This check is redundant if domains were updated correctly, but safe
                    return [0] * n
                for nb in adj[i]:
                    if color[nb] == color[i]:
                        return [0] * n
                result.append(int(color[i]))
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
        
        # Final safety check: ensure types are int
        # The solver returns ints, but let's be explicit
        if len(sol) == n:
            out.append([int(c) for c in sol])
        else:
            out.append([0] * n)
            
    return out
# EVOLVE-BLOCK-END
