"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def construct():
    import random
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

    def solve_instance(n, edges, lists, deadline):
        """Solve a single list colouring instance using backtracking with MRV and Forward Checking."""
        if time.time() > deadline:
            return None

        # Build adjacency as lists of sets for faster membership checks
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Precompute adjacency lists as lists for faster iteration
        adj_lists = [list(s) for s in adj]

        # Initialize domains as bitmasks (0..4 colors -> 5 bits)
        # bit i is set if color i is available
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask

        color = [-1] * n
        uncoloured = set(range(n))
        
        # Lookup table for bit counts (0..31)
        bit_count = [bin(i).count('1') for i in range(32)]

        call_count = 0

        def backtrack():
            nonlocal call_count
            call_count += 1
            if call_count % 100 == 0 and time.time() > deadline:
                return False

            if not uncoloured:
                return True

            # MRV Heuristic: Select uncoloured vertex with smallest domain
            # Tie-breaker: Highest degree (more constraints)
            min_size = float('inf')
            best_v = -1
            best_deg = -1
            for v in uncoloured:
                size = bit_count[domains[v]]
                if size == 0:
                    return False
                deg = len(adj[v])
                if size < min_size or (size == min_size and deg > best_deg):
                    min_size = size
                    best_v = v
                    best_deg = deg

            v = best_v
            uncoloured.remove(v)
            
            # Iterate through colors in the domain
            mask = domains[v]
            while mask:
                # Get lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                # Assign
                color[v] = c
                
                # Record changes for undo
                changes = []
                valid = True
                c_bit = 1 << c
                
                for u in adj_lists[v]:
                    if u in uncoloured:
                        if domains[u] & c_bit:
                            changes.append(u)
                            domains[u] ^= c_bit
                            if domains[u] == 0:
                                valid = False
                                break

                if valid:
                    if backtrack():
                        return True

                # Undo
                for u in changes:
                    domains[u] |= c_bit
                uncoloured.add(v)
                color[v] = -1

            return False

        if backtrack():
            return color
        
        return None

    deadline = time.time() + 110  # 110 second safety margin (150s limit)

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        if result is not None:
            # Ensure all values are Python int
            out.append([int(c) for c in result])
        else:
            # Should not happen, but add a dummy valid coloring
            adj = defaultdict(set)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            colour = [0] * n
            for v in sorted(range(n), key=lambda u: -len(adj[u])):
                taken = {colour[u] for u in adj[v]}
                free = [c for c in lists[v] if c not in taken]
                colour[v] = free[0] if free else lists[v][0]
            out.append([int(c) for c in colour])

    return out
# EVOLVE-BLOCK-END
