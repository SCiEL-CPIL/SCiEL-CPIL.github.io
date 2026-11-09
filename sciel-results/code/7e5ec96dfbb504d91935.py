"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit

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
        # Build adjacency list and bitmask adjacency
        adj = [[] for _ in range(n)]
        adj_mask = [0] * n
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        
        # Convert lists to bitmasks
        # Palette is 0..4, so bits 0..4 are relevant
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        colour = [-1] * n
        uncolored = set(range(n))
        
        def backtrack():
            if not uncolored:
                return True
            
            # MRV Heuristic: Pick vertex with fewest available colors in its domain
            # that are not used by colored neighbors
            min_count = 5
            target = -1
            best_domain = 0
            for v in uncolored:
                # Effective domain: original domain minus colors used by colored neighbors
                # Since we do forward checking, domains[v] already reflects constraints from colored neighbors?
                # Actually, standard forward checking updates domains of uncolored neighbors when a node is colored.
                # So domains[v] should be the current valid domain.
                count = domains[v].bit_count()
                if count < min_count:
                    min_count = count
                    target = v
                    best_domain = domains[v]
                    if count == 0:
                        return False
                    if count == 1:
                        break
            
            if target == -1:
                return False

            v = target
            uncolored.remove(v)
            
            # Try each color in the domain
            d = best_domain
            while d:
                # Get lowest set bit (color)
                c_mask = d & (-d)
                c = c_mask.bit_length() - 1
                
                # Assign color
                colour[v] = c
                
                # Forward checking: update neighbors
                # For each uncolored neighbor u of v:
                #   remove c from domains[u]
                #   if domains[u] becomes 0, backtrack
                failed = False
                for u in adj[v]:
                    if u in uncolored:
                        if domains[u] & c_mask:
                            domains[u] ^= c_mask
                            if domains[u] == 0:
                                failed = True
                                break
                
                if not failed and backtrack():
                    return True
                
                # Undo
                colour[v] = -1
                for u in adj[v]:
                    if u in uncolored:
                        if not (domains[u] & c_mask):
                            domains[u] |= c_mask
                
                # Move to next color
                d ^= c_mask
            
            # Restore vertex to uncolored set if all colors failed
            uncolored.add(v)
            return False

        if backtrack():
            return colour
        else:
            return [0] * n

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # If we are running out of time, fill remaining with dummy solutions
            # to ensure we return a valid structure (list of 4 lists of ints)
            # though they will score 0.
            while len(out) < 4:
                 out.append([0])
            break
        
        n, edges, lists = instance(splits, seed)
        # Ensure lists are lists of ints
        sol = solve_instance(n, edges, lists)
        # Convert to Python ints just in case
        out.append([int(c) for c in sol])
        
    # If we exited early due to deadline, ensure we have 4 entries
    while len(out) < 4:
        out.append([0])
        
    return out
# EVOLVE-BLOCK-END
