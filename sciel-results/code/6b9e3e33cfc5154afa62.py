"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
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

    # Precompute popcount for 5-bit integers (colors 0..4)
    POP_COUNT = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]

    def solve_one(n, edges, lists, seed_offset=0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        masks = []
        for lst in lists:
            m = 0
            for c in lst:
                m |= (1 << c)
            masks.append(m)
        
        # Greedy attempt
        rng = Random(seed_offset)
        for attempt in range(100):
            if time.time() > deadline:
                return None
            order = list(range(n))
            rng.shuffle(order)
            colour = [None] * n
            valid = True
            for v in order:
                taken = 0
                for u in adj[v]:
                    if colour[u] is not None:
                        taken |= (1 << colour[u])
                free_mask = masks[v] & ~taken
                if free_mask:
                    # Pick a random color from free_mask
                    free_colors = [c for c in range(5) if (free_mask >> c) & 1]
                    colour[v] = rng.choice(free_colors)
                else:
                    valid = False
                    break
            
            if valid:
                # Verify
                for v in range(n):
                    if colour[v] is None: continue
                    if not (masks[v] >> colour[v]) & 1:
                        valid = False
                        break
                    for u in adj[v]:
                        if u > v and colour[u] == colour[v]:
                            valid = False
                            break
                    if not valid: break
                if valid:
                    return colour

        # Backtracking with bitmasks
        domains = masks[:]
        assigned = [None] * n
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with min domain size
            min_v = -1
            min_size = 5
            for v in range(n):
                if assigned[v] is None:
                    size = POP_COUNT[domains[v]]
                    if size < min_size:
                        min_size = size
                        min_v = v
                        if size <= 1:
                            break
            
            if min_v == -1:
                return True
            
            if min_size == 0:
                return False
            
            # Try each color in domain
            d = domains[min_v]
            while d:
                # Extract lowest set bit
                lsb = d & -d
                c = lsb.bit_length() - 1
                d -= lsb
                
                assigned[min_v] = c
                mask_bit = 1 << c
                
                saved = {}
                consistent = True
                for u in adj[min_v]:
                    if assigned[u] is None:
                        if domains[u] & mask_bit:
                            saved[u] = domains[u]
                            domains[u] &= ~mask_bit
                            if domains[u] == 0:
                                consistent = False
                                break
                
                if consistent and backtrack():
                    return True
                
                # Undo
                for u, val in saved.items():
                    domains[u] = val
                assigned[min_v] = None
            
            return False
        
        if backtrack():
            return assigned
        
        return None

    out = []
    for idx, (splits, seed) in enumerate(((20, 701), (40, 702), (60, 703), (100, 704))):
        if time.time() > deadline - 5:
            # Not enough time, return what we have
            break
        
        n, edges, lists = instance(splits, seed)
        result = solve_one(n, edges, lists, seed_offset=idx * 1000)
        
        if result is None:
            # Fallback: just assign first colour from list (may be invalid)
            result = [lst[0] for lst in lists]
        
        out.append([int(c) for c in result])
    
    # If we ran out of time, fill remaining with simple assignments
    while len(out) < 4:
        idx = len(out)
        splits, seed = ((20, 701), (40, 702), (60, 703), (100, 704))[idx]
        n, edges, lists = instance(splits, seed)
        out.append([int(lst[0]) for lst in lists])
    
    return out
# EVOLVE-BLOCK-END
