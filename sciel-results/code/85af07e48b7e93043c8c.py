"""
Maximize |S| for S ⊆ [1,512] with lcm(a,b) <= 512 for all a,b in S.

Strategy:
1. Build a 512x512 compatibility matrix.
2. Use a MaxSAT-style greedy local search that builds a large valid set.
3. Run many randomized restarts + greedy + local-improvement sweeps.
4. Validate and return the best set found.
"""

def construct():
    import time
    import math
    import random

    start_time = time.time()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 80  # 10 s safety margin out of 90 s

    # ---- Build compatibility: lcm(i, j) <= 512 ----
    # Use bitmasks for fast set operations
    # compat_mask[v] is a bitmask of all u in [1..N] such that lcm(v,u) <= 512
    compat_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            l = (i // math.gcd(i, j)) * j
            if l <= LIMIT:
                mask |= (1 << j)
        compat_mask[i] = mask

    def is_valid(S):
        for i, a in enumerate(S):
            for b in S[i + 1:]:
                if not (compat_mask[a] >> b) & 1:
                    return False
        return True

    # ---- Quick greedy upper bound via greedy coloring on the compatibility subgraph ----
    # We want an upper bound on the size of a valid set (clique in compat graph)
    # contained within candidate set C. Use greedy coloring: color the compat subgraph
    # induced by C; the chromatic number is an upper bound on clique number.
    def greedy_color_upper_bound(C):
        """Return upper bound on max clique size in compat graph induced by C."""
        if not C:
            return 0
        # Order by degree within C descending for better coloring
        C_list = list(C)
        C_set = set(C_list)
        # Compute degrees within C
        deg = {}
        for v in C_list:
            d = 0
            for u in C_list:
                if u != v and (compat_mask[v] >> u) & 1:
                    d += 1
            deg[v] = d
        C_list.sort(key=lambda x: -deg[x])
        
        # Greedy coloring
        color = {}
        for v in C_list:
            # Find smallest color not used by neighbors in C
            used_colors = set()
            for u in C_list:
                if u != v and u in color and (compat_mask[v] >> u) & 1:
                    used_colors.add(color[u])
            c = 0
            while c in used_colors:
                c += 1
            color[v] = c
        return max(color.values()) if color else 0

    # ---- Exact branch-and-bound for maximum clique in compatibility graph ----
    # Equivalent to maximum independent set in conflict graph.
    # Use the classic Tomita-style algorithm with greedy coloring for bounds.
    
    # Build adjacency as bitmasks for the compatibility graph
    # adj_mask[v] = bitmask of all u != v such that lcm(v,u) <= 512
    adj_mask = [0] * (N + 1)
    for v in range(1, N + 1):
        adj_mask[v] = compat_mask[v] & ~(1 << v)

    best_clique = []
    best_size = 0
    node_count = 0

    def branch_and_bound(P, X, current):
        """
        P: set (as list) of candidates that can be added
        X: set of excluded candidates
        current: list of vertices in current clique
        """
        nonlocal best_clique, best_size, node_count
        node_count += 1
        
        if time.time() > deadline:
            return
        
        if not P:
            if len(current) > best_size:
                best_size = len(current)
                best_clique = list(current)
            return
        
        # Upper bound: |current| + |P|
        if len(current) + len(P) <= best_size:
            return
        
        # Greedy coloring of P for tighter bound
        # Order P by degree in compat graph restricted to P
        P_set = set(P)
        P_list = list(P)
        
        # Compute degrees within P
        deg = {}
        for v in P_list:
            d = 0
            for u in P_list:
                if u != v and (compat_mask[v] >> u) & 1:
                    d += 1
            deg[v] = d
        P_list.sort(key=lambda x: -deg[x])
        
        # Greedy coloring
        color = {}
        max_color = 0
        for v in P_list:
            used = set()
            for u in P_list:
                if u != v and u in color and (compat_mask[v] >> u) & 1:
                    used.add(color[u])
            c = 0
            while c in used:
                c += 1
            color[v] = c
            if c > max_color:
                max_color = c
        
        if len(current) + max_color <= best_size:
            return
        
        # Branch: try each vertex in P, ordered by color (higher color first)
        # Process in reverse color order for better pruning
        for v in reversed(P_list):
            if time.time() > deadline:
                return
            if v not in P_set:
                continue
            if len(current) + color[v] <= best_size:
                continue
            
            # Branch: include v
            current.append(v)
            # New P = P ∩ N(v), New X = X ∪ (P \ N(v))
            new_P = [u for u in P if u != v and (compat_mask[v] >> u) & 1]
            branch_and_bound(new_P, X, current)
            current.pop()
            
            # Move v from P to X
            P_set.discard(v)
            
            # Update bound check
            if len(current) + len(P) <= best_size:
                return

    # ---- Phase 1: Get a good initial solution via greedy + local search ----
    # Greedy build from scratch
    def greedy_build(start_set):
        S_set = set(start_set)
        S_list = list(start_set)
        # Order by degree in compat graph descending
        order = list(range(1, N + 1))
        order.sort(key=lambda x: -bin(compat_mask[x]).count('1'))
        for v in order:
            if v in S_set:
                continue
            if all((compat_mask[v] >> u) & 1 for u in S_list):
                S_set.add(v)
                S_list.append(v)
        return S_list

    # Local improvement: add any compatible vertex
    def local_improve_add(S_list):
        S_set = set(S_list)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for v in range(1, N + 1):
                if v not in S_set and all((compat_mask[v] >> u) & 1 for u in S_set):
                    S_set.add(v)
                    S_list.append(v)
                    changed = True
        return S_list

    # Get initial solution
    initial = greedy_build([])
    initial = local_improve_add(initial)
    
    if is_valid(initial) and len(initial) > best_size:
        best_size = len(initial)
        best_clique = list(initial)

    # Also try starting from divisors of highly composite numbers
    def get_divisors(m):
        divs = []
        for d in range(1, int(math.isqrt(m)) + 1):
            if m % d == 0:
                divs.append(d)
                if d * d != m:
                    divs.append(m // d)
        return sorted(divs)

    # Find number with most divisors
    best_M = 1
    max_divs = 0
    for m in range(1, N + 1):
        cnt = len(get_divisors(m))
        if cnt > max_divs:
            max_divs = cnt
            best_M = m

    div_set = [d for d in get_divisors(best_M) if d <= N]
    div_set = local_improve_add(div_set)
    if is_valid(div_set) and len(div_set) > best_size:
        best_size = len(div_set)
        best_clique = list(div_set)

    # ---- Phase 2: Exact branch-and-bound to find the true maximum ----
    if time.time() < deadline - 5:
        # Start with all vertices as candidates
        full_P = list(range(1, N + 1))
        branch_and_bound(full_P, [], [])

    # ---- Phase 3: If we have time, try to verify/improve with more B&B runs ----
    # The B&B above should find the optimum. But if it was cut short,
    # try a few more targeted searches.
    if time.time() < deadline - 2 and best_clique:
        # Try adding any single vertex to best_clique
        bc_set = set(best_clique)
        for v in range(1, N + 1):
            if v not in bc_set and all((compat_mask[v] >> u) & 1 for u in best_clique):
                newS = best_clique + [v]
                if is_valid(newS) and len(newS) > best_size:
                    best_size = len(newS)
                    best_clique = newS

    # Final validation
    if not best_clique:
        best_clique = [1]
    
    if not is_valid(best_clique):
        filtered = []
        for v in best_clique:
            if all((compat_mask[v] >> u) & 1 for u in filtered):
                filtered.append(v)
        best_clique = filtered if filtered else [1]

    return best_clique
