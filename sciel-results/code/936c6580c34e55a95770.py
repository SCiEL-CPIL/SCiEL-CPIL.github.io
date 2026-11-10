import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

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
        # Build adjacency as lists for fast iteration
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute adjacency as tuples for faster iteration
        adj = [tuple(lst) for lst in adj]
        
        # Convert lists to bitmasks
        init_domains = [0] * n
        for i, lst in enumerate(lists):
            mask = 0
            for c in lst:
                mask |= (1 << c)
            init_domains[i] = mask

        # Fast singleton color extraction using bit_length
        def get_single_color(mask):
            # mask is a power of 2, return the bit index
            return (mask.bit_length() - 1)

        # Propagation function with trail for undo
        # Returns (success, trail) where trail is list of (vertex, old_domain)
        def propagate(domains, start_vertices):
            trail = []
            stack = list(start_vertices)
            while stack:
                v = stack.pop()
                dv = domains[v]
                if dv == 0:
                    return False, trail
                if dv & (dv - 1) == 0:  # Singleton
                    k = get_single_color(dv)
                    bit = 1 << k
                    for u in adj[v]:
                        du = domains[u]
                        if du & bit:
                            old_du = du
                            domains[u] = du & ~bit
                            trail.append((u, old_du))
                            if domains[u] == 0:
                                return False, trail
                            if domains[u] & (domains[u] - 1) == 0:
                                stack.append(u)
            return True, trail

        # Undo trail
        def undo(domains, trail):
            for u, old_du in reversed(trail):
                domains[u] = old_du

        # Backtracking search with trail-based undo
        def backtrack(domains, depth):
            if time.time() > deadline:
                return None
            
            # Find MRV variable with degree as tiebreaker
            min_size = 5
            var = -1
            min_deg = -1
            for v in range(n):
                dv = domains[v]
                if dv == 0:
                    return None
                # Count bits
                size = bin(dv).count('1')
                if size > 1:
                    deg = len(adj[v])
                    if size < min_size or (size == min_size and deg > min_deg):
                        min_size = size
                        var = v
                        min_deg = deg
            
            if var == -1:
                # All domains are singletons - extract solution
                sol = [0] * n
                for v in range(n):
                    sol[v] = get_single_color(domains[v])
                return sol
            
            # Try each color in domains[var]
            dv = domains[var]
            # Iterate through set bits
            temp = dv
            while temp:
                # Get lowest set bit
                bit = temp & (-temp)
                idx = get_single_color(bit)
                
                # Save old domain
                old_var_domain = domains[var]
                domains[var] = bit  # Assign single color
                
                # Propagate
                success, trail = propagate(domains, [var])
                
                if success:
                    result = backtrack(domains, depth + 1)
                    if result is not None:
                        # Undo before returning (shouldn't need to, but safe)
                        undo(domains, trail)
                        domains[var] = old_var_domain
                        return result
                
                # Undo
                undo(domains, trail)
                domains[var] = old_var_domain
                
                temp &= temp - 1  # Clear lowest set bit
            
            return None

        # Start search with initial propagation
        success, trail = propagate(init_domains, range(n))
        if not success:
            return None
        
        result = backtrack(init_domains, 0)
        return result

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            while len(out) < 4:
                out.append([0] * 8)
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        
        if colour is None:
            # Fallback: Randomized Greedy with Restarts
            colour = None
            adj_set = [set() for _ in range(n)]
            for u, v in edges:
                adj_set[u].add(v)
                adj_set[v].add(u)
            
            for attempt in range(200):
                if time.time() > deadline:
                    break
                
                colour_try = [None] * n
                order = list(range(n))
                Random(seed + attempt).shuffle(order)
                
                success = True
                for v in order:
                    taken = set()
                    for u in adj_set[v]:
                        if colour_try[u] is not None:
                            taken.add(colour_try[u])
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        colour_try[v] = Random(seed + attempt + v).choice(free)
                    else:
                        success = False
                        break
                
                if success:
                    # Verify
                    valid = True
                    for v in range(n):
                        if colour_try[v] not in lists[v]:
                            valid = False
                            break
                        for u in adj_set[v]:
                            if u < v and colour_try[u] == colour_try[v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        colour = [int(c) for c in colour_try]
                        break
            
            if colour is None:
                colour = [0] * n
        
        out.append(colour)
    
    while len(out) < 4:
        out.append([0] * 8)
    
    return out
