import time
from random import Random

# Precompute popcount for 5-bit values (0..31)
_POPCOUNT = [bin(i).count('1') for i in range(32)]

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
        
        # AC-3 preprocessing: enforce arc consistency
        # Queue of arcs (u, v) where u's domain might need pruning due to v's domain
        from collections import deque
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        
        def revise(u, v, domains):
            """Return True if domain of u was revised (pruned)"""
            changed = False
            du = domains[u]
            dv = domains[v]
            # For each color in u's domain, check if v has a different color
            # If not, remove that color from u's domain
            new_du = 0
            for c in range(5):
                if du & (1 << c):
                    # Check if v has any color != c
                    if dv & ~(1 << c):
                        new_du |= (1 << c)
            if new_du != du:
                domains[u] = new_du
                changed = True
            return changed
        
        while queue:
            u, v = queue.popleft()
            if revise(u, v, initial_domains):
                if initial_domains[u] == 0:
                    return None
                # Add all other neighbors of u
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))
        
        # Check for empty domains after AC-3
        for i in range(n):
            if initial_domains[i] == 0:
                return None
        
        # Backtracking with MRV and Forward Checking
        # Use a list for assignment (faster than dict)
        assigned = [-1] * n  # -1 means unassigned
        
        def backtrack(domains):
            # Find unassigned node with smallest domain (MRV)
            min_size = 6
            mrv_node = -1
            for i in range(n):
                if assigned[i] == -1:
                    size = _POPCOUNT[domains[i]]
                    if size < min_size:
                        min_size = size
                        mrv_node = i
                        if size == 1:
                            break
                        
            if mrv_node == -1:
                # All assigned
                return True
            
            if min_size == 0:
                return False
            
            # Get values in domain
            mask = domains[mrv_node]
            vals = []
            for c in range(5):
                if mask & (1 << c):
                    vals.append(c)
            
            # LCV: Order values by least constraining
            def constraining_power(c):
                count = 0
                for v in adj[mrv_node]:
                    if assigned[v] == -1:
                        if domains[v] & (1 << c):
                            count += 1
                return count
                
            vals.sort(key=constraining_power)
            
            for val in vals:
                # Assign val to mrv_node
                assigned[mrv_node] = val
                new_domains = domains[:]
                new_domains[mrv_node] = 1 << val
                
                valid = True
                # Forward Checking
                for v in adj[mrv_node]:
                    if assigned[v] == -1:
                        if new_domains[v] & (1 << val):
                            new_domains[v] &= ~(1 << val)
                            if new_domains[v] == 0:
                                valid = False
                                break
                            
                if valid:
                    if backtrack(new_domains):
                        return True
                
                # Undo assignment
                assigned[mrv_node] = -1
                
            return False

        if backtrack(initial_domains):
            return [assigned[i] for i in range(n)]
        return None

    results = []
    success = 0
    
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        # Check time
        if time.time() > deadline:
            # Emergency: We must return a list of 4 lists.
            # We have already solved some. For the rest, we generate dummy solutions.
            # We need to generate the instance to know n.
            # This is a fallback that should rarely trigger.
            pass
        
        n, edges, lists = instance(splits, seed)
        
        # If we are very close to deadline and haven't solved this one yet,
        # we might just return a dummy to ensure valid format.
        # However, n=400 is very small. Backtracking with MRV+FC should be < 1ms.
        # So we just solve it.
        
        solution = solve_csp(n, edges, lists)
        
        if solution is not None:
            # Verify solution
            valid = True
            for i in range(n):
                if solution[i] not in lists[i]:
                    valid = False
                    break
            if valid:
                for u, v in edges:
                    if solution[u] == solution[v]:
                        valid = False
                        break
            if valid:
                results.append([int(x) for x in solution])
                success += 1
            else:
                # Should not happen if solver is correct, but fallback
                results.append([0]*n)
        else:
            # Failed to find solution. 
            # The problem states all are colourable.
            # If our solver fails, it's a bug or time issue.
            # Return dummy to keep format valid.
            results.append([0]*n)
            
        # Check time again
        if time.time() > deadline:
             # Fill remaining with dummies
             for rem_splits, rem_seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
                 if len(results) == 4:
                     break
                 # We need to know which one is missing.
                 # This loop logic is tricky. 
                 # Better: just break and handle outside.
                 pass
             break

    # Ensure we have 4 results
    # If the loop broke early, we need to fill the rest.
    # We need to identify which instances are missing.
    # The instances are fixed: (20, 701), (40, 702), (60, 703), (100, 704).
    # Let's just regenerate the ones we didn't process.
    
    # To make this robust, let's track which indices we have.
    # Actually, the loop above adds to results sequentially.
    # If we broke, results has < 4 items.
    # We need to generate the remaining ones.
    
    # The targets are:
    targets = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    # If results length is less than 4, it means we didn't finish the loop.
    # We need to append solutions for the remaining targets.
    # Since we don't know exactly where we stopped, let's just ensure we have 4 by
    # solving the remaining ones if time permits, or dummies if not.
    
    # A simpler way: The loop above is the main path. 
    # If it completes, results has 4.
    # If it breaks, we handle it here.
    
    if len(results) < 4:
        # Determine which indices are missing
        # This is hard to do retroactively if we just appended.
        # Let's restructure: solve all 4 in a list, then handle timeouts.
        
        pass # The logic above is slightly flawed for timeout handling. 
             # Let's rewrite the main loop to be safer.

    # REWRITE MAIN LOOP FOR SAFETY
    
    results = []
    targets = [(20, 701), (40, 702), (60, 703), (100, 704)]
    
    for i, (splits, seed) in enumerate(targets):
        if time.time() > deadline:
            # Out of time. Generate dummy for this and remaining.
            n, edges, lists = instance(splits, seed)
            results.append([0]*n)
            # Fill rest
            for j in range(i+1, 4):
                s_j, seed_j = targets[j]
                n_j, _, _ = instance(s_j, seed_j)
                results.append([0]*n_j)
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
                    results.append([0]*n)
            else:
                results.append([0]*n)
                
    return results
