"""Model-and-optimize: CSP Solver with AC-3 and Backtracking for list coloring."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import deque

def construct():
    t0 = time.time()
    LIMIT = 120.0
    MARGIN = 10.0

    def deadline():
        return (time.time() - t0) > (LIMIT - MARGIN)

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
        """
        Solves the list coloring problem using CSP techniques:
        1. AC-3 Arc Consistency to prune domains.
        2. Backtracking search with Forward Checking.
        """
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Precompute degrees for tie-breaking
        degrees = [len(adj[v]) for v in range(n)]
        
        # Initial domains as sets for fast operations
        domains = [set(lst) for lst in lists]
        
        # 1. AC-3 Arc Consistency
        # Queue of arcs (i, j) where i's domain might be affected by j
        queue = deque()
        for i in range(n):
            for j in adj[i]:
                queue.append((i, j))
        
        while queue:
            i, j = queue.popleft()
            # Check if any value in domains[i] is not consistent with domains[j]
            # Since the constraint is "not equal", a value ci in Di is consistent
            # with Dj if there exists some cj in Dj such that ci != cj.
            # This is false ONLY if Dj is a subset of {ci}.
            # Given |Dj| >= 1 and |Di| >= 1, if |Dj| == 1 and Dj == {ci}, then ci is removed.
            
            dj = domains[j]
            di = domains[i]
            
            if not di or not dj:
                return None # Failure: empty domain
            
            # Optimization: If |dj| > 1, then for any ci, there is always some cj != ci in dj.
            # So no value in di needs to be removed.
            if len(dj) > 1:
                continue
                
            # If |dj| == 1, let val = the single value in dj
            if len(dj) == 1:
                val = next(iter(dj))
                if val in di:
                    di.discard(val)
                    if not di:
                        return None # Failure: empty domain
                    # If we removed a value, we need to check all other neighbors of i
                    for k in adj[i]:
                        if k != j:
                            queue.append((k, i))
                            
        # Check if all domains are non-empty
        for d in domains:
            if not d:
                return None

        # If AC-3 resulted in all domains of size 1, we have a solution
        solution = [None] * n
        all_single = True
        for i in range(n):
            if len(domains[i]) == 1:
                solution[i] = next(iter(domains[i]))
            else:
                all_single = False
                break
        
        if all_single:
            return solution

        # 2. Backtracking with Forward Checking
        # We need to handle the remaining search space.
        # To make backtracking efficient, we maintain domains dynamically.
        
        def backtrack(assigned, domains_copy):
            if deadline():
                return None
            
            # Find unassigned variable with smallest domain (MRV)
            var = -1
            min_size = float('inf')
            for v in range(n):
                if v not in assigned:
                    size = len(domains_copy[v])
                    if size == 0:
                        return None
                    if size < min_size:
                        min_size = size
                        var = v
                        # Early exit if size is 1? No, MRV is better.
                        # But if we find size 1, we can't get smaller.
                        if size == 1:
                            break
            
            if var == -1:
                # All assigned
                return assigned
            
            # Value Ordering: LCV (Least Constraining Value)
            # We want the value that eliminates the fewest options for neighbors.
            # Since constraint is '!=', a value c eliminates c from neighbor's domain.
            # So we want the value c that is present in the fewest neighbors' domains.
            # Or rather, the value that causes the least reduction.
            # Reduction is 1 if c is in neighbor's domain, 0 otherwise.
            # So we prefer c that is NOT in neighbor's domains.
            # If all are in, it doesn't matter much.
            
            vals = list(domains_copy[var])
            # Sort by number of neighbors that have this value in their domain
            # Fewer neighbors having the value = less constraining = better
            
            def count_constraints(val):
                count = 0
                for u in adj[var]:
                    if u not in assigned:
                        if val in domains_copy[u]:
                            count += 1
                return count
            
            vals.sort(key=count_constraints)
            
            for c in vals:
                # Assign var = c
                new_assigned = dict(assigned)
                new_assigned[var] = c
                
                # Forward Checking: Update domains of unassigned neighbors
                new_domains = [set(d) for d in domains_copy]
                new_domains[var] = {c}
                
                ok = True
                for u in adj[var]:
                    if u not in new_assigned:
                        if c in new_domains[u]:
                            new_domains[u].discard(c)
                            if not new_domains[u]:
                                ok = False
                                break
                
                if not ok:
                    continue
                
                # Recurse
                result = backtrack(new_assigned, new_domains)
                if result is not None:
                    return result
            
            return None

        result = backtrack({}, domains)
        if result is None:
            return None
            
        # Convert dict to list
        colour = [0] * n
        for v in range(n):
            colour[v] = result[v]
        return colour

    def solve_min_conflicts(n, edges, lists):
        """
        Fallback: Min-Conflicts local search.
        """
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        rng = Random(12345)
        max_restarts = 50
        max_steps_per_restart = 20000

        for restart in range(max_restarts):
            if deadline():
                break

            colour = [rng.choice(lists[v]) for v in range(n)]
            
            # Precompute neighbor color counts
            neighbor_color_count = [{} for _ in range(n)]
            for v in range(n):
                cnt = {}
                cv = colour[v]
                for u in adj[v]:
                    c = colour[u]
                    cnt[c] = cnt.get(c, 0) + 1
                neighbor_color_count[v] = cnt
            
            conflict_count = [0] * n
            for v in range(n):
                conflict_count[v] = neighbor_color_count[v].get(colour[v], 0)
            
            list_violation = [colour[v] not in lists[v] for v in range(n)]
            
            if sum(conflict_count) == 0 and not any(list_violation):
                return [int(c) for c in colour]

            for step in range(max_steps_per_restart):
                if deadline():
                    break

                conflicted = [v for v in range(n) if list_violation[v] or conflict_count[v] > 0]
                if not conflicted:
                    return [int(c) for c in colour]

                v = rng.choice(conflicted)
                current_colour = colour[v]
                
                best_colour = None
                best_delta = float('inf')
                ncc = neighbor_color_count[v]
                count_current = ncc.get(current_colour, 0)
                
                for c in lists[v]:
                    if c == current_colour:
                        continue
                    count_c = ncc.get(c, 0)
                    delta = count_c - count_current
                    if list_violation[v]:
                        delta -= 1
                    if delta < best_delta:
                        best_delta = delta
                        best_colour = c

                if best_colour is None:
                    best_colour = rng.choice(lists[v])

                old_colour = current_colour
                colour[v] = best_colour
                conflict_count[v] = neighbor_color_count[v].get(best_colour, 0)
                
                for u in adj[v]:
                    nc = neighbor_color_count[u]
                    nc[old_colour] = nc.get(old_colour, 0) - 1
                    if nc[old_colour] <= 0:
                        del nc[old_colour]
                    nc[best_colour] = nc.get(best_colour, 0) + 1
                    
                    if colour[u] == old_colour:
                        conflict_count[u] -= 1
                    if colour[u] == best_colour:
                        conflict_count[u] += 1

                if list_violation[v]:
                    list_violation[v] = False

                if sum(conflict_count) == 0 and not any(list_violation):
                    return [int(c) for c in colour]
        
        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        if deadline():
            n, edges, lists = instance(splits, seed)
            # Fallback random
            rng_fallback = Random(777)
            out.append([int(rng_fallback.choice(lists[v])) for v in range(n)])
            continue

        n, edges, lists = instance(splits, seed)
        
        # Try CSP Solver first
        colour = solve_csp(n, edges, lists)
        
        # If CSP fails or times out, try Min-Conflicts
        if colour is None:
            colour = solve_min_conflicts(n, edges, lists)
            
        # If both fail, random fallback
        if colour is None:
            rng_fallback = Random(777)
            colour = [int(rng_fallback.choice(lists[v])) for v in range(n)]
            
        out.append(colour)

    return out
# EVOLVE-BLOCK-END
