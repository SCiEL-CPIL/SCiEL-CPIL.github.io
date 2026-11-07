# EVOLVE-BLOCK-START
import time
def construct():
    import time
    _deadline = time.time() + 140  # safety margin before 150s kill
    from random import Random
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

    def is_valid(colour, n, adj, lists):
        for v in range(n):
            c = colour[v]
            if c is None or c not in lists[v]:
                return False
            for u in adj[v]:
                if colour[u] == c:
                    return False
        return True

    def backtrack_fc(n, adj, lists, colour, domains, unassigned):
        if not unassigned:
            return True
        
        # MRV: Select variable with fewest remaining values
        min_len = float('inf')
        var = -1
        for v in unassigned:
            if len(domains[v]) < min_len:
                min_len = len(domains[v])
                var = v
                if min_len == 0:
                    return False
                if min_len == 1:
                    break
        
        # Forward Checking
        original_domains = {}
        for u in adj[var]:
            if u in unassigned:
                original_domains[u] = domains[u]
                new_domain = [c for c in domains[u] if c != colour[var]]
                domains[u] = new_domain
                if not new_domain:
                    # Restore and fail
                    for uu in original_domains:
                        domains[uu] = original_domains[uu]
                    return False
        
        for c in domains[var]:
            colour[var] = c
            new_unassigned = list(unassigned)
            new_unassigned.remove(var)
            
            if backtrack_fc(n, adj, lists, colour, domains, new_unassigned):
                return True
            
            # Restore domains
            for uu in original_domains:
                domains[uu] = original_domains[uu]
        
        colour[var] = None
        return False

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > _deadline - 5:
            # Fallback greedy
            n, edges, lists = instance(splits, seed)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in range(n):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append(colour)
            continue

        n, edges, lists = instance(splits, seed)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        # Initialize domains
        domains = [list(lists[v]) for v in range(n)]
        colour = [None] * n
        unassigned = list(range(n))
        
        solved = backtrack_fc(n, adj, lists, colour, domains, unassigned)
        
        if solved:
            out.append(colour)
        else:
            # If FC backtracking fails (shouldn't happen for these instances), try simple greedy as last resort
            colour_greedy = [None] * n
            order_greedy = sorted(range(n), key=lambda u: -len(adj[u]))
            for v in order_greedy:
                taken = {colour_greedy[u] for u in adj[v] if colour_greedy[u] is not None}
                free = [c for c in lists[v] if c not in taken]
                colour_greedy[v] = free[0] if free else lists[v][0]
            out.append(colour_greedy)

    # Ensure all are Python ints
    for i in range(len(out)):
        out[i] = [int(c) for c in out[i]]
    return out
# EVOLVE-BLOCK-END
