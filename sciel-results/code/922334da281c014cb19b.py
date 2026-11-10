import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin from 120s limit

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

        # Convert lists to bitmasks
        # Palette is 0..4, so we need 5 bits.
        # Bit i represents color i.
        initial_domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            initial_domains[i] = mask

        # Check for empty domains initially
        for i in range(n):
            if initial_domains[i] == 0:
                return None

        # AC-3 arc consistency preprocessing
        # For each directed arc (u,v), remove values from dom(u) that have no support in dom(v)
        # Since colors are from a list of 3 and we need different colors on edges,
        # a color c in dom(u) is supported by dom(v) if there exists d in dom(v) with d != c.
        # This means c is NOT supported only if dom(v) == {c} (single element equal to c).
        from collections import deque
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))

        domains = initial_domains[:]  # copy for AC-3

        while queue:
            u, v = queue.popleft()
            # Check if any value in dom(u) has no support in dom(v)
            # A value c in dom(u) has no support in dom(v) iff dom(v) is a subset of {c}
            # i.e., dom(v) & ~(1<<c) == 0, meaning dom(v) contains only c (or is empty)
            du = domains[u]
            dv = domains[v]
            if dv == 0:
                # v's domain is empty, so u cannot use any color that v has... 
                # Actually if dv==0, no color in du is supported. But this means unsat.
                # In practice, if dv==0 we should have caught it. Let's handle:
                # Remove all values from du that equal the only possible... 
                # Actually if dv==0, then for any c in du, there's no d in dv with d!=c (since dv is empty).
                # So all of du should be removed.
                if du != 0:
                    domains[u] = 0
                    if domains[u] == 0:
                        return None
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))
                continue

            # For each color c in du, check if dv has a value different from c
            # c is unsupported iff dv & ~(1<<c) == 0, i.e., dv is a subset of {c}
            # Since dv != 0, this means dv == (1<<c)
            unsupported = 0
            for c in range(5):
                if du & (1 << c):
                    if dv == (1 << c):
                        unsupported |= (1 << c)
            if unsupported:
                new_du = du & ~unsupported
                if new_du != du:
                    domains[u] = new_du
                    if new_du == 0:
                        return None
                    for w in adj[u]:
                        if w != v:
                            queue.append((w, u))

        # Use AC-3 pruned domains for backtracking
        initial_domains = domains

        # MRV selection helper
        # We maintain a list of unassigned nodes.
        # For speed, we can just scan all nodes each time if n is small (~400).
        # Or use a priority queue. Scanning is O(n), which is fine for n=400.
        
        # Popcount lookup table for 5-bit masks
        _popcount = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                     1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]

        def count_bits(x):
            return _popcount[x]

        def backtrack(assigned_count, domains, assigned):
            if assigned_count == n:
                return True

            # MRV: Find unassigned node with smallest domain size
            min_size = 6 # Max is 5, so 6 is infinity
            mrv_node = -1
            for i in range(n):
                if i not in assigned:
                    size = count_bits(domains[i])
                    if size < min_size:
                        min_size = size
                        mrv_node = i
                        if size == 1:
                            break
                            
            if mrv_node == -1:
                # All assigned?
                if assigned_count == n:
                    return True
                return False
            
            if min_size == 0:
                return False
            
            # LCV: Order values in domain by how many options they leave for neighbors
            # Get values in domain
            vals = []
            mask = domains[mrv_node]
            for c in range(5):
                if mask & (1 << c):
                    vals.append(c)
            
            # Calculate constraint count for each value
            # A value c constrains neighbor v if c is in v's domain
            # We want to try values that remove the *fewest* options from neighbors
            def constraining_power(c):
                count = 0
                for v in adj[mrv_node]:
                    if v not in assigned:
                        if domains[v] & (1 << c):
                            count += 1
                return count
                
            vals.sort(key=constraining_power)
            
            for val in vals:
                # Assign val to mrv_node
                assigned[mrv_node] = val
                new_domains = domains[:] # Copy domains (list of ints is cheap)
                new_domains[mrv_node] = 1 << val
                
                valid = True
                # Forward Checking
                for v in adj[mrv_node]:
                    if v not in assigned:
                        if new_domains[v] & (1 << val):
                            new_domains[v] &= ~(1 << val)
                            if new_domains[v] == 0:
                                valid = False
                                break
                            
                if valid:
                    if backtrack(assigned_count + 1, new_domains, assigned):
                        return True
                
                # Undo assignment (not strictly needed for domains since we copied,
                # but we need to remove from assigned dict for backtracking)
                del assigned[mrv_node]

            return False

        assigned = {}
        if backtrack(0, initial_domains, assigned):
            return [assigned[i] for i in range(n)]
        return None

    targets = [(20, 701), (40, 702), (60, 703), (100, 704)]
    results = []
    
    for i, (splits, seed) in enumerate(targets):
        if time.time() > deadline:
            # Out of time. Generate dummy for this and remaining.
            n, edges, lists = instance(splits, seed)
            results.append([0] * n)
            # Fill rest
            for j in range(i + 1, 4):
                s_j, seed_j = targets[j]
                n_j, _, _ = instance(s_j, seed_j)
                results.append([0] * n_j)
            break
        else:
            n, edges, lists = instance(splits, seed)
            solution = solve_csp(n, edges, lists)
            
            if solution is not None:
                # Verify
                valid = True
                for k in range(n):
                    if solution[k] not in lists[k]:
                        valid = False
                        break
                if valid:
                    for u, v in edges:
                        if solution[u] == solution[v]:
                            valid = False
                            break
                if valid:
                    results.append([int(x) for x in solution])
                else:
                    results.append([0] * n)
            else:
                results.append([0] * n)
                
    return results
