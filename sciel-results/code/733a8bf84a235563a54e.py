import time
import sys
from random import Random
from collections import deque

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]

    def fallback():
        out = []
        for spec in specs:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
        return out

    def compress(color):
        m = len(color)
        if m == 0:
            return []
        mapping = {}
        new = [0] * m
        nxt = 0
        for i, c in enumerate(color):
            if c not in mapping:
                mapping[c] = nxt
                nxt += 1
            new[i] = mapping[c]
        return new

    def solve_instance(n, d, seed, time_budget, deadline):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        # Build conflict graph
        # Vertex i in conflict graph corresponds to edge i in original graph
        # Two vertices are connected if the corresponding edges share a vertex or are joined by an edge
        adj_orig = [set() for _ in range(n)]
        for u, v in edges:
            adj_orig[u].add(v)
            adj_orig[v].add(u)
            
        neigh = [[] for _ in range(m)]
        for i in range(m):
            a, b = edges[i]
            for j in range(i + 1, m):
                c, dd = edges[j]
                # Check if they share a vertex
                if a == c or a == dd or b == c or b == dd:
                    neigh[i].append(j)
                    neigh[j].append(i)
                else:
                    # Check if they are joined by an edge
                    # i.e., c is neighbor of a or b, or dd is neighbor of a or b
                    if c in adj_orig[a] or dd in adj_orig[a] or c in adj_orig[b] or dd in adj_orig[b]:
                        neigh[i].append(j)
                        neigh[j].append(i)
                        
        # Degrees in conflict graph
        deg = [len(neigh[i]) for i in range(m)]
        
        # Lower bound: max degree + 1
        LB = max(deg) + 1 if m > 0 else 0
        
        # Upper bound: simple greedy coloring
        # We'll use a greedy approach to get an initial UB
        color_greedy = [-1] * m
        satcols = [set() for _ in range(m)]
        # Order by degree descending
        order = sorted(range(m), key=lambda x: deg[x], reverse=True)
        for v in order:
            used = satcols[v]
            c = 0
            while c in used:
                c += 1
            color_greedy[v] = c
            for u in neigh[v]:
                satcols[u].add(c)
        UB = max(color_greedy) + 1
        
        # If LB == UB, we are done
        if LB == UB:
            return compress(color_greedy)
            
        # Backtracking solver with Forward Checking
        # We want to find if a K-coloring exists.
        # We will try K from LB to UB.
        
        # Pre-compute neighbors for faster access
        # domains[v] will be a list of available colors
        # We use a simple backtracking with MRV and Forward Checking
        
        def find_coloring(K, rng, time_limit):
            # K is the number of colors allowed
            # domains: list of sets
            domains = [set(range(K)) for _ in range(m)]
            color = [-1] * m
            
            # MRV heuristic: pick variable with smallest domain
            # We need to efficiently find the unassigned variable with smallest domain size
            
            # To speed up, we can pre-sort variables by degree and use that as a tie-breaker
            # But MRV is dynamic.
            
            # We'll implement a simple recursive backtracking.
            # Given the small size (m ~ 100-200), recursion depth is manageable.
            
            # Optimization: Use bitmasks for domains?
            # K is small (<= 50), so bitmask is fine.
            
            dom_masks = [(1 << K) - 1 for _ in range(m)]
            assigned = [False] * m
            
            # Count unassigned
            unassigned_count = m
            
            # We'll use a stack for iterative backtracking to avoid recursion limits and allow easy time checks
            # State: (variable, color, next_variable_to_try, ...)
            # Actually, simple recursion is easier to write correctly. 
            # Python recursion limit can be increased.
            sys.setrecursionlimit(10000)
            
            start_time = time.time()
            deadline_local = start_time + time_limit
            
            # Order variables by degree descending for initial exploration
            # But MRV will override.
            
            def backtrack():
                nonlocal dom_masks, assigned, color, unassigned_count, deadline_local
                
                if time.time() > deadline_local:
                    return False
                    
                if unassigned_count == 0:
                    return True
                    
                # Find variable with MRV (smallest domain)
                # Among those, highest degree
                min_size = K + 1
                best_v = -1
                best_deg = -1
                
                for v in range(m):
                    if not assigned[v]:
                        size = dom_masks[v].bit_count()
                        if size == 0:
                            return False # Pruned
                        if size < min_size or (size == min_size and deg[v] > best_deg):
                            min_size = size
                            best_v = v
                            best_deg = deg[v]
                            
                if best_v == -1:
                    return False # Should not happen if unassigned_count > 0
                    
                v = best_v
                # Try colors in domain
                # Shuffle colors for randomness? Or just iterate.
                # Iterating 0..K-1 is fine.
                
                # Get list of available colors
                # Instead of creating a list, iterate through bits
                mask = dom_masks[v]
                
                # We need to try each color c in mask
                # For each c, we assign v=c, and update neighbors' domains.
                
                # To allow backtracking, we need to save state.
                # We'll save the masks of neighbors that change.
                
                # Actually, Forward Checking modifies domains of neighbors.
                # We need to restore them on backtrack.
                
                # Let's collect changes
                changes = [] # list of (neighbor, old_mask)
                
                # Iterate through possible colors for v
                # We can extract colors from mask
                colors_to_try = []
                mm = mask
                while mm:
                    lsb = mm & -mm
                    c = lsb.bit_length() - 1
                    colors_to_try.append(c)
                    mm ^= lsb
                    
                # Randomize order?
                rng.shuffle(colors_to_try)
                
                for c in colors_to_try:
                    if time.time() > deadline_local:
                        return False
                        
                    # Assign v = c
                    assigned[v] = True
                    color[v] = c
                    unassigned_count -= 1
                    dom_masks[v] = 0 # Mark as assigned
                    
                    # Update neighbors
                    feasible = True
                    for u in neigh[v]:
                        if not assigned[u]:
                            old_mask = dom_masks[u]
                            new_mask = old_mask & ~(1 << c)
                            if new_mask == 0:
                                # Infeasible
                                feasible = False
                                break
                            changes.append((u, old_mask))
                            dom_masks[u] = new_mask
                            
                    if feasible:
                        if backtrack():
                            return True
                            
                    # Undo
                    for u, old_mask in changes:
                        dom_masks[u] = old_mask
                    changes.clear()
                    
                    assigned[v] = False
                    color[v] = -1
                    unassigned_count += 1
                    dom_masks[v] = mask # Restore original mask for v? 
                    # Wait, dom_masks[v] was set to 0. We should restore it to its state before assignment.
                    # But v is unassigned now. Its domain should be what it was before we tried this color.
                    # Actually, v's domain doesn't change when we assign it, only when we unassign it?
                    # No, v's domain is fixed until assigned. When we backtrack, we restore v's domain to what it was.
                    # But v's domain wasn't changed by assigning it. It was just marked assigned.
                    # So restoring dom_masks[v] = mask is correct if mask is the domain before assignment.
                    # Yes, mask is the domain of v before we picked a color.
                    
                return False
                
            # Initialize
            # We can also do Arc Consistency (AC-3) initially to prune domains.
            # Let's add AC-3 for better performance.
            
            def ac3(domains, K):
                # domains: list of sets or bitmasks
                # We use bitmasks
                queue = deque()
                for i in range(m):
                    for j in neigh[i]:
                        if j > i:
                            queue.append((i, j))
                            queue.append((j, i))
                            
                while queue:
                    i, j = queue.popleft()
                    if assigned[i] or assigned[j]:
                        continue
                    # Check if domain of i is consistent with j
                    # For each value in domain[i], there must be a value in domain[j]
                    # Since it's graph coloring, values must be different.
                    # So for each c in domain[i], domain[j] must have some value != c.
                    # This is equivalent to: domain[j] is not a subset of {c}
                    # Actually, simpler: if domain[i] == {c}, then c must not be in domain[j].
                    # If domain[j] == {c}, then c must not be in domain[i].
                    
                    # General AC:
                    # For each c in domain[i], if there is no d in domain[j] such that c != d, remove c from domain[i].
                    # This means if domain[j] is empty, or domain[j] is a subset of {c} (i.e., domain[j] == {c}), then c is removed from domain[i].
                    
                    mask_i = dom_masks[i]
                    mask_j = dom_masks[j]
                    
                    # If mask_j is 0, it's already pruned.
                    # If mask_j has only 1 bit, say c, then c must be removed from mask_i.
                    if mask_j != 0:
                        # Check if mask_j is a power of 2 (single value)
                        if mask_j & (mask_j - 1) == 0:
                            # Single value c
                            c_bit = mask_j
                            if mask_i & c_bit:
                                new_mask_i = mask_i & ~c_bit
                                if new_mask_i != mask_i:
                                    if new_mask_i == 0:
                                        return False
                                    dom_masks[i] = new_mask_i
                                    # Enqueue neighbors of i (except j)
                                    for k in neigh[i]:
                                        if k != j and not assigned[k]:
                                            queue.append((k, i))
                        else:
                            # Multiple values in j
                            # Is there any c in i such that c is the ONLY value in j? No, j has multiple.
                            # So for any c in i, there is some d in j such that d != c (since j has at least 2 values, or even if 1, handled above).
                            # Wait, if j has values {1, 2}, and i has {1}, then 1 is consistent with 2 in j.
                            # So no pruning needed if j has > 1 value.
                            pass
                            
                return True
            
            # Run AC-3 initially
            if not ac3(dom_masks, K):
                return False
                
            # Check if any domain is empty
            for i in range(m):
                if dom_masks[i] == 0:
                    return False
            
            # Start backtracking
            if backtrack():
                return color
            return None

        best_color = None
        best_K = UB
        
        rng = Random(seed * 7919 + 13)
        
        # Try K from LB to UB
        # We have a time budget.
        # We can try increasing K.
        # If we find a solution for K, that's the answer for this K.
        # We want the minimum K.
        
        # Strategy:
        # 1. Try K=LB. If success, done.
        # 2. If fail, try K=LB+1, etc.
        # To avoid spending too much time on a hard K that fails, we limit time per K.
        
        end_time = min(time.time() + time_budget, deadline - 2.0)
        
        for K in range(LB, UB + 1):
            if time.time() > end_time:
                break
                
            # Allocate time for this K
            # If K is close to LB, give it more time?
            # Or just give it a fixed slice.
            slice_time = min(2.0, max(0.1, end_time - time.time()))
            
            # Try to find coloring with K
            # We can run the solver multiple times with different rng seeds?
            # The solver is deterministic given the order, but we shuffle colors.
            # Let's run it once with a time limit.
            
            res = find_coloring(K, rng, slice_time)
            if res is not None:
                best_color = res
                best_K = K
                # If K == LB, we found optimal, break
                if K == LB:
                    break
            else:
                # If it failed, it might be due to time limit or infeasibility.
                # If it's infeasible, we should move to K+1.
                # But we don't know if it was infeasible or just timed out.
                # Given the small size, if it times out, it's likely hard.
                # We'll just move to K+1.
                pass
                
        if best_color is None:
            # Fallback to greedy
            return compress(color_greedy)
        else:
            return compress(best_color)

    try:
        start = time.time()
        deadline = start + 140.0
        out = []
        for idx, (n, d, seed) in enumerate(specs):
            remaining = deadline - time.time()
            share = min(remaining / (len(specs) - idx), 45.0)
            share = max(share, 0.5)
            out.append(solve_instance(n, d, seed, share, deadline))
        return out
    except Exception:
        return fallback()
