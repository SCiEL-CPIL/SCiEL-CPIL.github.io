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
        
        # Convert to sets for faster membership tests
        adj_sets = [set(neigh) for neigh in adj]
        
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
                # Optimization: check if any value in domains[i] is NOT in domains[j]
                # If all values in domains[i] are in domains[j], then for every ci in domains[i],
                # there exists a cj in domains[j] such that ci != cj (since domains are sets of distinct colors, 
                # and if domains[i] == domains[j], then for any ci, we can pick cj != ci if len(domain)>1.
                # Actually, the constraint is !=. 
                # A value ci is supported if there exists cj in domains[j] such that ci != cj.
                # This is false ONLY if domains[j] == {ci}.
                # So ci is removed if domains[j] is a subset of {ci}, i.e., domains[j] == {ci}.
                
                # Let's stick to the standard check for clarity and correctness, 
                # but we can optimize the inner loop.
                for ci in list(domains[i]):
                    if not any(ci != cj for cj in domains[j]):
                        domains[i].remove(ci)
                        removed = True
                        if not domains[i]:
                            return False
                if removed:
                    for k in adj_sets[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking search with forward checking
        # Order variables by most constrained first (smallest domain), then highest degree
        order = sorted(range(n), key=lambda v: (len(domains[v]), -len(adj[v])))
        
        def backtrack(assigned):
            if time.time() > deadline:
                return None
            
            # Find next unassigned variable in the pre-sorted order
            var = None
            for v in order:
                if v not in assigned:
                    var = v
                    break
            
            if var is None:
                return assigned
            
            # Try colors in sorted order for determinism, though any order works
            for c in sorted(domains[var]):
                # Check if color is valid (not used by any assigned neighbor)
                valid = True
                for u in adj[var]:
                    if u in assigned and assigned[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign and check forward checking
                new_assigned = dict(assigned)
                new_assigned[var] = c
                new_domains = [set(d) for d in domains]
                
                # Forward checking: reduce domains of unassigned neighbors
                ok = True
                for u in adj[var]:
                    if u not in new_assigned:
                        if c in new_domains[u]:
                            new_domains[u].remove(c)
                            if not new_domains[u]:
                                ok = False
                                break
                
                if not ok:
                    continue
                
                # Recurse
                result = backtrack(new_assigned)
                if result is not None:
                    return result
            
            return None
        
        result = backtrack({})
        if result is None:
            return None
        
        # Convert to list of ints
        colour = [0] * n
        for v in range(n):
            colour[v] = int(result[v])
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
