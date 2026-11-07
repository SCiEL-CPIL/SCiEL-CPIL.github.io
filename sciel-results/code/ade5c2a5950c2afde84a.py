# EVOLVE-BLOCK-START
def construct():
    import time
    import sys

    sys.setrecursionlimit(10000)

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

    def solve(n, edges, lists):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Use bitmask representation for domains (colors 0-4 fit in 5 bits)
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask

        color = [-1] * n

        # Arc Consistency (AC-3) preprocessing
        def enforce_arc_consistency():
            """Enforce arc consistency. Returns False if a domain becomes empty."""
            queue = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        queue.append((u, v))
                        queue.append((v, u))

            while queue:
                u, v = queue.pop()
                du = domains[u]
                dv = domains[v]
                new_du = du
                # For each color c in du, check if dv has some color != c
                # c has no support iff dv is subset of {c}, i.e., dv == 0 or dv == (1 << c)
                for c in range(5):
                    if du & (1 << c):
                        if (dv & ~(1 << c)) == 0:
                            new_du &= ~(1 << c)
                
                if new_du != du:
                    domains[u] = new_du
                    if new_du == 0:
                        return False
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))

            return True

        if not enforce_arc_consistency():
            return [0] * n

        # Precompute bit counts for fast MRV
        bit_count = [0] * 32  # max 5 bits, but use 32 for safety
        for i in range(32):
            bit_count[i] = bin(i).count('1')

        # MRV + Forward Checking Backtracking with bitmask domains
        # Enhanced with Unit Propagation (1-in-1)
        
        def propagate():
            """
            Perform unit propagation.
            Returns (success, changes_list) where changes_list contains (var, old_domain, old_color)
            for undoing.
            """
            changes = []
            queue = []
            
            # Initialize queue with all variables that have domain size 1
            for v in range(n):
                if color[v] == -1 and domains[v] != 0:
                    if bit_count[domains[v]] == 1:
                        queue.append(v)
            
            while queue:
                v = queue.pop()
                if color[v] != -1:
                    continue
                if domains[v] == 0:
                    return False, changes
                if bit_count[domains[v]] != 1:
                    continue
                    
                # v must take the single available color
                c = (domains[v] & -domains[v]).bit_length() - 1
                # Actually, simpler:
                c = 0
                while not (domains[v] & (1 << c)):
                    c += 1
                    
                # Record change for undo
                changes.append((v, domains[v], -1))
                
                color[v] = c
                domains[v] = 1 << c
                
                # Propagate to neighbors
                for u in adj[v]:
                    if color[u] == -1:
                        if domains[u] & (1 << c):
                            old_du = domains[u]
                            new_du = domains[u] & ~(1 << c)
                            if new_du == 0:
                                return False, changes
                            changes.append((u, old_du, color[u]))
                            domains[u] = new_du
                            if bit_count[new_du] == 1:
                                queue.append(u)
            
            return True, changes

        def backtrack():
            # Check for failure
            for v in range(n):
                if color[v] == -1 and domains[v] == 0:
                    return False
            
            # Try to propagate units
            ok, prop_changes = propagate()
            if not ok:
                return False
            
            # Find MRV variable among unassigned
            min_len = 1000
            best_v = -1
            for v in range(n):
                if color[v] == -1:
                    d_len = bit_count[domains[v]]
                    if d_len == 0:
                        return False
                    if d_len < min_len:
                        min_len = d_len
                        best_v = v
                        if min_len == 1:
                            break
            
            if best_v == -1:
                # All assigned
                # Undo propagation changes? No, if we return True, we keep them.
                # But if we return False (shouldn't happen if all assigned), we would need to undo.
                # Since we return True here, the caller will not undo if it returns True.
                return True

            v = best_v
            saved_domain = domains[v]
            
            # Get list of available colors
            available_colors = []
            c = 0
            while (1 << c) <= saved_domain:
                if saved_domain & (1 << c):
                    available_colors.append(c)
                c += 1
            
            # LCV: sort colors by number of values they remove from neighbors (ascending)
            def count_removed(c):
                count = 0
                for u in adj[v]:
                    if color[u] == -1:
                        if domains[u] & (1 << c):
                            count += 1
                return count
            
            available_colors.sort(key=count_removed)
            
            # Try each color in LCV order
            for c in available_colors:
                # Record v's state
                # We need to undo v's assignment and any propagation that happens in the recursive call
                # But wait, `propagate` was already done above.
                # We are now branching on v.
                
                color[v] = c
                domains[v] = 1 << c
                
                # We need to record changes to neighbors for this specific branch
                changes = []
                valid = True
                
                for u in adj[v]:
                    if color[u] == -1:
                        if domains[u] & (1 << c):
                            new_du = domains[u] & ~(1 << c)
                            changes.append((u, domains[u]))
                            domains[u] = new_du
                            if new_du == 0:
                                valid = False
                                break
                
                if valid:
                    # Recursive call
                    if backtrack():
                        return True
                
                # Undo neighbor changes
                for u, old_d in changes:
                    domains[u] = old_d
            
            # Undo v's assignment
            domains[v] = saved_domain
            color[v] = -1
            
            return False

        if backtrack():
            return [int(c) for c in color]
        else:
            return [0] * n

    out = []
    start_time = time.time()
    deadline = start_time + 110.0

    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Fallback to avoid crash, though unlikely
            n, edges, lists = instance(splits, seed)
            out.append([0] * n)
            continue
            
        n, edges, lists = instance(splits, seed)
        res = solve(n, edges, lists)
        out.append(res)
        
    return out
# EVOLVE-BLOCK-END
