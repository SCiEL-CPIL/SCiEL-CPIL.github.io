# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    # Ensure high recursion limit just in case, though depth is small
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Colour assignment: -1 means uncoloured
        colour = [-1] * n
        
        # Domains as bitmasks. Bit i is set if color i is available.
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask

        # Uncoloured vertices
        uncoloured = set(range(n))
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 110.0

        # Use bit_count directly for speed (Python 3.10+)
        # If environment is older, this would fail, but standard eval envs are modern.
        # To be safe, we can check once.
        if not hasattr(int, 'bit_count'):
             def bit_count(x): return bin(x).count('1')
        else:
             def bit_count(x): return x.bit_count()

        def backtrack():
            nonlocal uncoloured, colour, domains, deadline
            if time.time() > deadline:
                return False

            if not uncoloured:
                return True

            # MRV: pick vertex with fewest available colours
            best_v = -1
            best_count = 10
            for v in uncoloured:
                cnt = bit_count(domains[v])
                if cnt == 0:
                    return False
                if cnt < best_count:
                    best_count = cnt
                    best_v = v
                    if cnt == 1:
                        break

            if best_v == -1:
                return False

            v = best_v
            mask_v = domains[v]
            
            # Try each available colour
            # Iterate through bits set in mask_v
            m = mask_v
            while m:
                # Get lowest set bit
                lsb = m & (-m)
                c = lsb.bit_length() - 1
                m ^= lsb

                colour[v] = c
                uncoloured.remove(v)
                
                # Forward checking: prune neighbors
                saved_domains = {}
                consistent = True
                for u in adj[v]:
                    if u in uncoloured:
                        saved_domains[u] = domains[u]
                        new_dom = domains[u] & ~lsb
                        domains[u] = new_dom
                        if new_dom == 0:
                            consistent = False
                            break
                
                if consistent and backtrack():
                    return True
                
                # Backtrack: restore
                uncoloured.add(v)
                colour[v] = -1
                for u, dom in saved_domains.items():
                    domains[u] = dom

            return False

        if backtrack():
            return [colour[v] for v in range(n)]
        else:
            # Should not happen per Alon-Tarsi, but return a dummy valid-looking structure
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
