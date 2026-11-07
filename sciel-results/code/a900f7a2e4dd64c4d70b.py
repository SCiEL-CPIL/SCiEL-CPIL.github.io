"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    from random import Random
    from collections import deque

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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert lists to bitmasks
        # Bit i is set if color i is in the list
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask
        
        # Check for initial empty domains
        for v in range(n):
            if domains[v] == 0:
                return None
        
        assigned = [-1] * n
        
        # Helper to get list of set bits
        def get_bits(mask):
            bits = []
            while mask:
                lsb = mask & -mask
                bits.append(lsb.bit_length() - 1)
                mask ^= lsb
            return bits

        # Helper to count bits
        def popcount(mask):
            count = 0
            while mask:
                mask &= mask - 1
                count += 1
            return count

        # AC-3 style propagation using bitmasks
        # Queue of (u, v) meaning "check if domain[u] is reduced by constraint (u,v)"
        # For != constraint, if domain[v] is singleton {c}, then c must be removed from domain[u]
        
        def propagate(queue):
            while queue:
                u, v = queue.popleft()
                if assigned[u] != -1 or assigned[v] != -1:
                    continue
                
                # If domain[v] is singleton, remove that color from domain[u]
                if domains[v] and (domains[v] & (domains[v] - 1)) == 0:
                    only_color_mask = domains[v]
                    if domains[u] & only_color_mask:
                        domains[u] &= ~only_color_mask
                        if domains[u] == 0:
                            return False
                        # Add neighbors of u to queue
                        for z in adj[u]:
                            if z != v:
                                queue.append((z, u))
            return True

        # Initial propagation
        init_queue = deque()
        for u in range(n):
            for v in adj[u]:
                init_queue.append((u, v))
        
        if not propagate(init_queue):
            return None

        def backtrack():
            if time.time() > deadline:
                return False
            
            # MRV: Find unassigned variable with smallest domain
            min_size = 5 # Max list size is 3, so 5 is infinity
            var = -1
            max_degree = -1
            
            for v in range(n):
                if assigned[v] == -1:
                    size = popcount(domains[v])
                    if size < min_size or (size == min_size and len(adj[v]) > max_degree):
                        min_size = size
                        var = v
                        max_degree = len(adj[v])
                        if size <= 1:
                            break
                            
            if var == -1:
                return True # All assigned
            
            # LVC: Order values by how few constraints they impose
            values = get_bits(domains[var])
            
            def value_score(c):
                count = 0
                c_mask = 1 << c
                for u in adj[var]:
                    if assigned[u] == -1 and (domains[u] & c_mask):
                        count += 1
                return count
            
            values.sort(key=value_score)
            
            for c in values:
                c_mask = 1 << c
                if not (domains[var] & c_mask):
                    continue
                
                # Assign
                assigned[var] = c
                old_domain_var = domains[var]
                domains[var] = c_mask # Effectively removing other options for this var in propagation context? 
                # Actually, we just mark it assigned. We need to remove c from neighbors.
                
                # Forward Checking
                saved_neighbors = []
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1:
                        if domains[u] & c_mask:
                            saved_neighbors.append((u, domains[u]))
                            domains[u] &= ~c_mask
                            if domains[u] == 0:
                                ok = False
                                break
                
                if ok:
                    # Run local propagation? 
                    # For efficiency, we just recurse. The MRV step in the next call will handle constraints.
                    # However, AC-3 is safer. Let's use a local queue for propagation.
                    local_queue = deque()
                    for u, _ in saved_neighbors:
                        for z in adj[u]:
                            if z != var:
                                local_queue.append((z, u))
                    
                    if propagate(local_queue):
                        if backtrack():
                            return True
                    
                    # Undo propagation? 
                    # Propagation modifies domains in place. We need to restore them.
                    # This is tricky with the current propagate function.
                    # Let's simplify: Don't run full AC-3 in every step, just forward checking.
                    # The MRV heuristic is strong enough for these small instances.
                    pass
                else:
                    # Undo forward checking
                    pass

                # Undo
                assigned[var] = -1
                domains[var] = old_domain_var
                for u, old_dom in saved_neighbors:
                    domains[u] = old_dom
                    
                if time.time() > deadline:
                    return False
            
            return False

        # Note: The above backtrack function has a flaw: it doesn't properly undo the 
        # modifications made by `propagate` inside the recursion if we were to call it.
        # Given the small size (max 408 vertices), simple backtracking with forward checking 
        # and strong heuristics (MRV/LVC) is extremely fast. 
        # Let's refine the backtrack to be clean.

        def clean_backtrack():
            if time.time() > deadline:
                return False
            
            min_size = 5
            var = -1
            max_degree = -1
            
            for v in range(n):
                if assigned[v] == -1:
                    size = popcount(domains[v])
                    if size < min_size or (size == min_size and len(adj[v]) > max_degree):
                        min_size = size
                        var = v
                        max_degree = len(adj[v])
                        if size <= 1:
                            break
                            
            if var == -1:
                return True
            
            values = get_bits(domains[var])
            values.sort(key=lambda c: sum(1 for u in adj[var] if assigned[u] == -1 and (domains[u] & (1 << c))))
            
            for c in values:
                c_mask = 1 << c
                if not (domains[var] & c_mask):
                    continue
                
                assigned[var] = c
                saved_neighbors = []
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1:
                        if domains[u] & c_mask:
                            saved_neighbors.append((u, domains[u]))
                            domains[u] &= ~c_mask
                            if domains[u] == 0:
                                ok = False
                                break
                
                if ok:
                    if clean_backtrack():
                        return True
                
                # Undo
                assigned[var] = -1
                for u, old_dom in saved_neighbors:
                    domains[u] = old_dom
                
                if time.time() > deadline:
                    return False
            
            return False

        if clean_backtrack():
            # Convert assigned back to list of ints
            # assigned contains the color index
            return [int(assigned[v]) for v in range(n)]
        else:
            return None

    start = time.time()
    deadline = start + 115.0

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline - 5:
            n, edges, lists = instance(splits, seed)
            out.append([int(lists[v][0]) for v in range(n)])
            continue
        
        n, edges, lists = instance(splits, seed)
        result = solve_instance(n, edges, lists, deadline)
        
        if result is None or len(result) != n:
            result = [int(lists[v][0]) for v in range(n)]
        
        # Final validation
        valid = True
        for i, val in enumerate(result):
            if not isinstance(val, int) or isinstance(val, bool):
                valid = False
                break
            if val < 0 or val > 4:
                valid = False
                break
            if val not in lists[i]:
                valid = False
                break
        
        if not valid:
            result = [int(lists[v][0]) for v in range(n)]
            
        out.append(result)
    
    return out
# EVOLVE-BLOCK-END
