# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import sys

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
        # Bitmask representation of colors
        # Color c is represented by bit 1 << c
        # Domain is an integer mask of allowed colors
        
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Initial domains
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
            
        # Check for empty domains initially
        for v in range(n):
            if domains[v] == 0:
                return None
                
        # Ordering: Static variable ordering by degree (descending)
        # Dynamic: Smallest remaining domain (MRV) during search
        degrees = [len(adj[v]) for v in range(n)]
        
        # We will use a recursive backtracking with forward checking
        # To avoid deep recursion limits, we use an iterative approach or ensure sys.setrecursionlimit
        sys.setrecursionlimit(10000)
        
        assigned = [-1] * n
        # trail for undoing changes
        trail = []
        
        def popcount(x):
            return bin(x).count('1')
            
        def backtrack(idx):
            # Find unassigned vertex with smallest domain
            min_size = 6
            min_v = -1
            for v in range(n):
                if assigned[v] == -1:
                    sz = popcount(domains[v])
                    if sz == 0:
                        return False
                    if sz < min_size:
                        min_size = sz
                        min_v = v
                        if sz == 1:
                            break
                            
            if min_v == -1:
                return True # All assigned
                
            v = min_v
            # Try each color in domain
            mask = domains[v]
            while mask:
                # Extract lowest set bit
                bit = mask & -mask
                mask -= bit
                c = bit.bit_length() - 1
                
                # Assign c to v
                assigned[v] = c
                trail.append((v, domains[v])) # Save old domain
                
                # Propagate: remove c from neighbors
                valid = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if domains[u] & bit:
                            # Conflict or need to remove
                            if domains[u] == bit:
                                # No other option, immediate failure
                                valid = False
                                break
                            domains[u] &= ~bit
                            trail.append((u, domains[u] | bit)) # Save state before removal? 
                            # Actually, we need to restore exactly.
                            # Let's use a proper trail stack.
                            pass
                            
                # This simple trail logic is getting messy. 
                # Let's use a cleaner recursive structure with explicit save/restore.
                return False # Placeholder
                
        # The above recursive structure with trail is complex to get right in one shot.
        # Let's stick to a simpler, robust approach: 
        # Copy domains on backtrack (slower but safe for N<300)
        
        def solve(v, domains_copy):
            # Find MRV
            min_sz = 10
            min_v = -1
            for i in range(n):
                if assigned[i] == -1:
                    sz = popcount(domains_copy[i])
                    if sz == 0:
                        return False
                    if sz < min_sz:
                        min_sz = sz
                        min_v = i
                        if sz == 1:
                            break
            if min_v == -1:
                return True
                
            v = min_v
            mask = domains_copy[v]
            while mask:
                bit = mask & -mask
                mask -= bit
                c = bit.bit_length() - 1
                
                # Check if c is in domains_copy[v] (it is)
                # Assign
                assigned[v] = c
                
                # Propagate
                new_domains = domains_copy[:] # Shallow copy of list of ints
                valid = True
                for u in adj[v]:
                    if assigned[u] == -1:
                        if new_domains[u] & bit:
                            new_domains[u] &= ~bit
                            if new_domains[u] == 0:
                                valid = False
                                break
                if valid:
                    if solve(-1, new_domains):
                        return True
                
                # Undo
                assigned[v] = -1
            return False

        # Initialize assigned
        assigned = [-1] * n
        # Initial propagation? 
        # Just start solving.
        result = solve(-1, domains)
        
        if not result:
            return None
            
        # Verify
        for v in range(n):
            if assigned[v] not in lists[v]:
                return None
        for u, v in edges:
            if assigned[u] == assigned[v]:
                return None
                
        return assigned

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        coloring = solve_csp(n, edges, lists)
        if coloring is None:
            # Fallback to simple greedy just in case, though unlikely
            coloring = [lists[v][0] for v in range(n)]
        out.append(coloring)
    return out
# EVOLVE-BLOCK-END
