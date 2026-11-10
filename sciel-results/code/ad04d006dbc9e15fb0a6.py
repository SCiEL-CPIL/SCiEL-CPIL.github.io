"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    def instance(splits, seed, palette=5):
        rng = Random(seed)
        # Initial cube faces
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
        
        # Generate random lists of 3 colors from 0..4
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
                # Optimization for != constraint:
                # ci is unsupported if domains[j] is exactly {ci}
                if len(domains[j]) == 1:
                    cj = next(iter(domains[j]))
                    if cj in domains[i]:
                        domains[i].discard(cj)
                        removed = True
                        if not domains[i]:
                            return False
                else:
                    # If domains[j] has >1 elements, every ci in domains[i] has support
                    # because we can pick a cj != ci.
                    # So no removal needed in this branch for != constraint.
                    pass

                if removed:
                    for k in adj_sets[i]:
                        if k != j:
                            queue.append((k, i))
            return True
        
        if not ac3():
            return None
        
        # Backtracking search with forward checking
        # Order variables by most constrained first (smallest domain), then highest degree
        # We calculate this once initially. Dynamic MRV is more robust but slower per step.
        # Given the graph structure, static MRV is usually sufficient.
        order = sorted(range(n), key=lambda v: (len(domains[v]), -len(adj[v])))
        
        # We maintain a dict of assigned values
        # To speed up forward checking, we track which colors are used by neighbors?
        # Or just check assigned neighbors.
        
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
            
            # Try colors in sorted order
            for c in sorted(domains[var]):
                # Check if color is valid (not used by any assigned neighbor)
                valid = True
                for u in adj[var]:
                    if u in assigned and assigned[u] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Assign
                assigned[var] = c
                
                # Forward checking: reduce domains of unassigned neighbors
                # We need to check if removing c from any neighbor's domain makes it empty.
                # Note: We are modifying the global 'domains' structure in a destructive way?
                # The current seed program copies domains: new_domains = [set(d) for d in domains]
                # This is expensive. 
                # Better approach: 
                # 1. Check if any unassigned neighbor has domain size 1 and that value is c.
                # 2. If so, backtrack.
                # 3. If not, we don't actually need to update the domains for the *next* step 
                #    if we use the "assigned" dict to filter valid colors in the next step.
                #    However, standard FC updates domains to prune future choices.
                #    Let's stick to the efficient check:
                #    A neighbor u is blocked if its current domain (intersected with unassigned constraints) 
                #    loses c and becomes empty.
                
                fc_ok = True
                for u in adj[var]:
                    if u not in assigned:
                        # If c is the only option left for u, we have a conflict
                        if len(domains[u]) == 1 and c in domains[u]:
                            fc_ok = False
                            break
                        # Note: We are NOT removing c from domains[u] here.
                        # This is a slight deviation from strict Forward Checking which modifies domains.
                        # However, since we check `assigned` neighbors in the `valid` check above,
                        # we effectively handle the constraint. 
                        # The only benefit of FC is early detection of empty domains.
                        # If we don't remove c from domains[u], we might try c again for u later?
                        # No, because u is unassigned, and when we get to u, we check against assigned neighbors.
                        # So if var is assigned c, and u is a neighbor, when we process u, 
                        # we will see var in assigned and c == assigned[var], so we skip c.
                        # So we don't strictly need to modify domains for correctness, 
                        # only for pruning the search space of *other* neighbors?
                        # Actually, the "valid" check in the next step only looks at *assigned* neighbors.
                        # So if we don't update domains, we might waste time checking colors that are 
                        # effectively removed by *unassigned* neighbors? No, unassigned neighbors don't constrain yet.
                        # The constraint is only between assigned and unassigned, or assigned-assigned.
                        # Wait, CSP constraints are binary. 
                        # When we assign var=c, we must ensure that for every unassigned neighbor u,
                        # u still has at least one value in its domain that is != c.
                        # If domains[u] == {c}, then u has no valid value.
                        # So the check `if len(domains[u]) == 1 and c in domains[u]` is sufficient 
                        # to detect immediate failure.
                        # We do not need to remove c from domains[u] because when we eventually assign u,
                        # we will check against assigned neighbors (which includes var) and skip c.
                        # This "lazy" forward checking is faster than copying sets.
                
                if not fc_ok:
                    del assigned[var]
                    continue
                
                # Recurse
                result = backtrack(assigned)
                if result is not None:
                    return result
                
                # Unassign
                del assigned[var]
            
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
                # Placeholder that is technically a valid list of ints, 
                # but likely to score 0 for that instance if not correct.
                # However, the task says "wrong length or type makes the whole result invalid".
                # So we must provide a list of correct length.
                # We don't know n for the unsolved ones easily without calling instance again.
                # But we can just call instance to get n, then fill with 0.
                n_inst, _, _ = instance(splits, seed)
                out.append([0] * n_inst)
            break
        
        n, edges, lists = instance(splits, seed)
        colour = solve_csp(n, edges, lists)
        
        if colour is None:
            # Fallback: try a simpler greedy approach with random restarts
            # This is unlikely to be needed given the CSP solver, but ensures robustness
            colour = None
            for attempt in range(100):
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
                # Last resort: return zeros (will score 0 for this instance)
                colour = [0] * n
        
        out.append([int(c) for c in colour])
    
    # Ensure we have exactly 4 results
    while len(out) < 4:
        # If we ran out of time, we need to generate placeholders for the remaining instances
        # We need to know n for these.
        # The loop above breaks and fills out. 
        # If we didn't break, we have 4.
        pass

    return out
# EVOLVE-BLOCK-END
