"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    # Set a safety margin. The limit is 150s, so we aim to finish by 140s.
    deadline = time.time() + 140.0

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
    # This allows O(1) domain size checks
    POP_COUNT = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]

    def solve_one(n, edges, lists, seed_offset=0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks for fast intersection and checking
        masks = []
        for lst in lists:
            m = 0
            for c in lst:
                m |= (1 << c)
            masks.append(m)
        
        # 1. Greedy Attempt with Random Restarts
        # We use a deterministic seed per instance to ensure consistent behavior
        # but multiple restarts to handle potential local minima in greedy ordering.
        rng = Random(seed_offset)
        
        # Try a few fast greedy passes first
        for attempt in range(50):
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
                    # Pick the first available color to be deterministic within the shuffle
                    # or random. Random is usually better for escaping bad local states.
                    # Using rng.choice on the bits is efficient.
                    free_colors = []
                    fm = free_mask
                    while fm:
                        lsb = fm & -fm
                        c = lsb.bit_length() - 1
                        free_colors.append(c)
                        fm -= lsb
                    colour[v] = rng.choice(free_colors)
                else:
                    valid = False
                    break
            
            if valid:
                # Verification step
                for v in range(n):
                    c = colour[v]
                    if c is None:
                        valid = False
                        break
                    if not (masks[v] >> c) & 1:
                        valid = False
                        break
                    for u in adj[v]:
                        if u > v and colour[u] == c:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return colour

        # 2. Backtracking with MRV (Minimum Remaining Values) and Forward Checking
        # This is the robust fallback that guarantees a solution if one exists.
        domains = masks[:]
        assigned = [None] * n
        
        # To optimize MRV selection, we keep track of domain sizes implicitly via popcount
        # We scan for the unassigned vertex with the smallest domain.
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find unassigned vertex with min domain size
            min_v = -1
            min_size = 5 # Max possible is 3, but 5 is safe upper bound
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
            
            # Try each color in the domain of min_v
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
    instances = ((20, 701), (40, 702), (60, 703), (100, 704))
    
    for idx, (splits, seed) in enumerate(instances):
        if time.time() > deadline - 2:
            break
        
        n, edges, lists = instance(splits, seed)
        result = solve_one(n, edges, lists, seed_offset=idx * 1000)
        
        if result is None:
            # Fallback: This should never happen for these specific instances
            # but we provide a valid structure to avoid crashes.
            # Assigning the first color in the list is a valid *type* but likely invalid *solution*.
            # However, the task says "wrong length or type makes the whole result invalid".
            # So we must return a list of ints.
            result = [int(lst[0]) for lst in lists]
        
        out.append(result)
    
    # If we ran out of time, fill remaining with simple assignments to ensure length 4
    while len(out) < 4:
        idx = len(out)
        splits, seed = instances[idx]
        n, edges, lists = instance(splits, seed)
        out.append([int(lst[0]) for lst in lists])
        
    return out
# EVOLVE-BLOCK-END
