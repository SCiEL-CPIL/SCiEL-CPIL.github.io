"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
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

    # Precompute popcount for 0..31 (max domain size is 5 colors)
    POPCOUNT = [bin(i).count('1') for i in range(32)]

    def solve_exact(n, edges, lists):
        """Exact backtracking with MRV and forward checking using bitmasks for domains."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmask domains
        # Bit i is set if color i is available
        domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            domains[i] = mask
        
        color = [-1] * n
        uncolored_count = n
        
        def backtrack():
            nonlocal uncolored_count
            if uncolored_count == 0:
                return True
            
            # MRV: find variable with smallest non-zero domain
            min_var = -1
            min_size = 16 # max possible popcount is 5
            
            for v in range(n):
                if color[v] == -1:
                    d = domains[v]
                    if d == 0:
                        return False
                    cnt = POPCOUNT[d]
                    if cnt < min_size:
                        min_size = cnt
                        min_var = v
                        if min_size == 1:
                            break
            
            if min_var == -1:
                return False

            # Get colors for min_var
            d = domains[min_var]
            colors = []
            while d:
                lsb = d & -d
                c = lsb.bit_length() - 1
                # Count how many neighbors have this color in their domain
                constraint_count = 0
                for nb in adj[min_var]:
                    if color[nb] == -1 and (domains[nb] & lsb):
                        constraint_count += 1
                colors.append((constraint_count, c))
                d ^= lsb
                
            # Sort by constraint count (ascending)
            colors.sort(key=lambda x: x[0])
            
            for _, c in colors:
                mask_c = 1 << c
                color[min_var] = c
                uncolored_count -= 1
                
                # Forward checking
                saved_domains = []
                valid = True
                for nb in adj[min_var]:
                    if color[nb] == -1:
                        if domains[nb] & mask_c:
                            saved_domains.append((nb, domains[nb]))
                            domains[nb] &= ~mask_c
                            if domains[nb] == 0:
                                valid = False
                                break
                
                if valid and backtrack():
                    return True
                
                # Undo
                for nb, old_d in saved_domains:
                    domains[nb] = old_d
                color[min_var] = -1
                uncolored_count += 1
            
            return False
        
        if backtrack():
            return color
        return None

    def solve_greedy(n, edges, lists):
        """Greedy coloring as fallback."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        colour = [None] * n
        for v in sorted(range(n), key=lambda u: -len(adj[u])):
            taken = {colour[u] for u in adj[v] if colour[u] is not None}
            free = [c for c in lists[v] if c not in taken]
            colour[v] = free[0] if free else lists[v][0]
        return colour

    deadline = time.time() + 120.0
    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        
        # Try exact backtracking first
        result = solve_exact(n, edges, lists)
        
        if result is None or time.time() > deadline - 10.0:
            # Fallback to greedy
            result = solve_greedy(n, edges, lists)
        
        # Ensure all values are Python int
        out.append([int(c) for c in result])
    
    return out
# EVOLVE-BLOCK-END
