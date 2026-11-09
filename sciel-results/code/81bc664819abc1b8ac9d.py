# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    import sys
    sys.setrecursionlimit(100000)

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

    def solve_csp(n, edges, lists):
        # Build adjacency lists for faster iteration
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Pre-compute degrees for tie-breaking in MRV
        degrees = [len(adj[i]) for i in range(n)]

        # Convert lists to bitmasks for efficient domain operations
        # Colors are 0..4, so we need 5 bits.
        init_domains = []
        for l in lists:
            mask = 0
            for c in l:
                mask |= (1 << c)
            init_domains.append(mask)

        colour = [-1] * n
        domains = init_domains[:]
        uncoloured = set(range(n))
        
        start_time = time.time()
        deadline = start_time + 110.0

        def backtrack():
            nonlocal colour, domains, uncoloured, deadline
            if time.time() > deadline:
                return False

            if not uncoloured:
                return True

            # MRV: pick vertex with fewest available colours, tie-break by degree
            best_v = None
            best_count = 10
            best_deg = -1
            for v in uncoloured:
                # Count bits in mask
                cnt = domains[v].bit_count()
                if cnt == 0:
                    return False
                if cnt < best_count or (cnt == best_count and degrees[v] > best_deg):
                    best_count = cnt
                    best_v = v
                    best_deg = degrees[v]
                    if cnt == 1:
                        break

            if best_v is None:
                return False

            v = best_v
            v_mask = domains[v]
            
            # Iterate through set bits in v_mask
            # We try each color c in the domain
            mask = v_mask
            while mask:
                # Get the lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb

                # Check if c is valid (not in colored neighbors)
                valid = True
                for u in adj[v]:
                    if colour[u] == c:
                        valid = False
                        break
                if not valid:
                    continue

                colour[v] = c
                uncoloured.remove(v)
                
                # Forward checking: prune neighbors using an undo log
                undo_log = []
                consistent = True
                for u in adj[v]:
                    if u in uncoloured:
                        old_dom = domains[u]
                        new_dom = old_dom & ~(1 << c)
                        if new_dom != old_dom:
                            undo_log.append((u, old_dom))
                            domains[u] = new_dom
                            if new_dom == 0:
                                consistent = False
                                break
                
                if consistent and backtrack():
                    return True
                
                # Backtrack: restore
                uncoloured.add(v)
                colour[v] = -1
                for u, old_dom in undo_log:
                    domains[u] = old_dom

            return False

        if backtrack():
            return [colour[v] for v in range(n)]
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        out.append(solve_csp(n, edges, lists))

    # Ensure types are correct: Python int, list of lists
    result = []
    for colouring in out:
        result.append([int(c) for c in colouring])
    return result
# EVOLVE-BLOCK-END
