"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Initial domains
        domains = [set(lst) for lst in lists]
        
        # AC-3 arc consistency
        def ac3():
            queue = []
            for u in range(n):
                for v in adj[u]:
                    queue.append((u, v))
            
            while queue:
                i, j = queue.pop()
                removed = False
                for ci in list(domains[i]):
                    if len(domains[j]) == 1 and ci in domains[j]:
                        domains[i].remove(ci)
                        removed = True
                        if not domains[i]:
                            return False
                if removed:
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking search with forward checking
        # Use in-place domain modification to avoid copying
        assigned = [-1] * n
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # MRV Heuristic: Find unassigned variable with smallest domain
            var = -1
            min_size = n + 1
            for v in range(n):
                if assigned[v] == -1:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
            
            if var == -1:
                return True # All assigned
            
            # Try colors in sorted order
            for c in sorted(domains[var]):
                # Check if color is valid (not used by any assigned neighbor)
                valid = True
                for u in adj[var]:
                    if assigned[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                assigned[var] = c
                domains[var].clear()
                domains[var].add(c)
                
                # Forward checking: reduce domains of unassigned neighbors
                removed_values = []
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed_values.append((u, c))
                            if not domains[u]:
                                ok = False
                                break
                
                if ok:
                    if backtrack():
                        return True
                else:
                    # Restore domains on failure/dead end
                    for u, val in removed_values:
                        domains[u].add(val)
                
                # Restore state
                assigned[var] = -1
                domains[var].clear()
                # Note: We need to restore the original domain for var?
                # No, we cleared it and added c. But we need the original values for backtracking other branches.
                # Actually, standard FC keeps the domain of var as it was before assignment?
                # No, usually we just check neighbors. The domain of var itself doesn't need to be restored for neighbors,
                # but for the MRV calculation in the parent call, it needs to be correct.
                # However, since we are iterating over `sorted(domains[var])`, we have already iterated over the values.
                # We should restore the domain of var to its state before this assignment attempt.
                # To do this efficiently, we should have saved the domain of var.
                # Let's adjust: Save domain of var before modifying.
                
                # Wait, the logic above clears domains[var]. 
                # If we backtrack, we need domains[var] to contain the other options.
                # So we must save and restore domains[var].
                
                # Let's rewrite the inner loop logic to be safe.
                
                pass # Placeholder for re-implementation below
            
            return False
        
        # Re-implementing backtrack with proper domain saving for the variable itself
        def backtrack_v2():
            if time.time() > deadline:
                return False
            
            var = -1
            min_size = n + 1
            for v in range(n):
                if assigned[v] == -1:
                    size = len(domains[v])
                    if size < min_size:
                        min_size = size
                        var = v
            
            if var == -1:
                return True
            
            original_domain = set(domains[var])
            
            for c in sorted(original_domain):
                # Check validity
                valid = True
                for u in adj[var]:
                    if assigned[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                assigned[var] = c
                
                # Forward checking
                removed_values = []
                ok = True
                for u in adj[var]:
                    if assigned[u] == -1:
                        if c in domains[u]:
                            domains[u].remove(c)
                            removed_values.append((u, c))
                            if not domains[u]:
                                ok = False
                                break
                
                if ok:
                    if backtrack_v2():
                        return True
                
                # Restore
                assigned[var] = -1
                for u, val in removed_values:
                    domains[u].add(val)
                domains[var] = original_domain # Restore var's domain to original state for next iteration?
                # Actually, we just need to ensure domains[var] is correct for the next `c` iteration.
                # Since we are iterating over `original_domain`, and we haven't modified `domains[var]` in the loop above
                # (we only modified neighbors), `domains[var]` is still `original_domain`.
                # Wait, did we modify domains[var]? No.
                # So we don't need to restore domains[var] inside the loop.
                # We just need to ensure `assigned[var]` is -1.
            
            return False

        if not backtrack_v2():
            return None
        
        # Convert to list of ints
        colour = [0] * n
        for v in range(n):
            colour[v] = int(assigned[v])
        return colour

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if time.time() > deadline:
            # Return what we have so far with placeholder for remaining
            while len(out) < 4:
                out.append([0] * 8)
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        
        if colour is None:
            # Fallback: try a simpler greedy approach with random restarts
            # This is unlikely to be needed given the CSP solver, but ensures robustness
            colour = None
            for attempt in range(50):
                if time.time() > deadline:
                    break
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                colour_try = [None] * n
                # Random order
                order = list(range(n))
                Random(seed + attempt).shuffle(order)
                
                success = True
                for v in order:
                    taken = {colour_try[u] for u in adj[v] if colour_try[u] is not None}
                    free = [c for c in lists[v] if c not in taken]
                    if free:
                        colour_try[v] = free[0]
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
                        for u in adj[v]:
                            if u < v and colour_try[u] == colour_try[v]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        colour = [int(c) for c in colour_try]
                        break
            
            if colour is None:
                # Last resort: return zeros (will score 0 for this instance if invalid, 
                # but structure is valid)
                colour = [0] * n
        
        out.append(colour)
    
    # Ensure we have exactly 4 results
    while len(out) < 4:
        out.append([0] * 8)
    
    return out
# EVOLVE-BLOCK-END
