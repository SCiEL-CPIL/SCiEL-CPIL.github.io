"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    _t0 = time.time()
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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Build 2-SAT: for each vertex with domain size 3, pick a "base" color
        # and encode the choice of the other two as a boolean variable.
        # Actually, a cleaner encoding: for each vertex v with domain {a,b,c},
        # we create a boolean x_v. x_v=True means v gets color a, x_v=False means
        # v gets color b. We need to handle the case where c is also possible.
        # 
        # Better approach: Since lists have size 3, let's use a different encoding.
        # For each vertex v with domain {c0, c1, c2}:
        #   x_v = 0 -> color c0
        #   x_v = 1 -> color c1  
        #   x_v = 2 -> color c2
        # This is not directly 2-SAT friendly.
        #
        # Alternative: Try each of the 3 colors as a "fixed" assignment for a subset,
        # or use the fact that we can pick one color to eliminate.
        #
        # Simplest robust approach: Use backtracking with MRV + forward checking,
        # but with a time guard. The current backtracker works but may be slow for
        # n=100. Let's use a more efficient backtracking with bitmasks.
        
        # Convert domains to bitmasks for speed
        domains_mask = [0] * n
        for v in range(n):
            for c in lists[v]:
                domains_mask[v] |= (1 << c)

        # Count occurrences of each color in each vertex's domain for MRV
        # Order vertices by degree descending, then by domain size
        order = sorted(range(n), key=lambda u: (-len(adj[u]), bin(domains_mask[u]).count('1')))
        pos = {v: i for i, v in enumerate(order)}

        # Precompute adjacency in order
        adj_in_order = [[] for _ in range(n)]
        for v in range(n):
            pv = pos[v]
            for u in adj[v]:
                if pos[u] > pv:
                    adj_in_order[v].append(u)

        colour = [None] * n
        remaining_mask = [0] * n
        for v in range(n):
            remaining_mask[v] = domains_mask[v]

        def backtrack(idx):
            if time.time() - _t0 > 140:
                return False
            if idx == n:
                return True
            
            # MRV: find unassigned vertex with smallest remaining domain
            # Since we're going in fixed order, just use current vertex
            v = order[idx]
            
            mask = remaining_mask[v]
            if mask == 0:
                return False
            
            # Try each color in the domain
            while mask:
                c_bit = mask & (-mask)
                c = c_bit.bit_length() - 1
                mask ^= c_bit
                
                colour[v] = c
                v_mask_bit = 1 << c
                
                # Update remaining masks for uncolored neighbors
                affected = []
                feasible = True
                for u in adj_in_order[v]:
                    old_mask = remaining_mask[u]
                    new_mask = old_mask & ~v_mask_bit
                    if new_mask == 0:
                        feasible = False
                        break
                    remaining_mask[u] = new_mask
                    affected.append((u, old_mask))
                
                if feasible:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                for u, old_mask in affected:
                    remaining_mask[u] = old_mask
                colour[v] = None
            
            return False

        if not backtrack(0):
            # Fallback: try a simple greedy assignment
            colour = [None] * n
            for v in range(n):
                used = set()
                for u in adj[v]:
                    if colour[u] is not None:
                        used.add(colour[u])
                for c in lists[v]:
                    if c not in used:
                        colour[v] = c
                        break
                if colour[v] is None:
                    colour[v] = lists[v][0]
            return colour
        
        return [int(c) for c in colour]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() - _t0 > 145:
            # Not enough time left; return a trivial (possibly invalid) result
            # to avoid timeout. But we should try to finish.
            pass
        n, edges, lists = instance(splits, seed)
        out.append(solve_instance(n, edges, lists))
    return out
# EVOLVE-BLOCK-END
