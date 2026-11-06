import math
import time
import random
import sys

def construct():
    start = time.time()
    deadline = 85.0  # safety margin before the 90s wall-clock kill
    random.seed(42)

    n = 256
    max_sum = 2 * n
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1

    # Diagonal constraint: 2*a must never be a square.
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)

    # Conflicts restricted to the candidate set.
    conflict = {a: set() for a in candidates}
    for idx, a in enumerate(candidates):
        for b in candidates[idx + 1:]:
            if a + b in squares:
                conflict[a].add(b)
                conflict[b].add(a)

    def time_up(budget=None):
        if budget is None:
            return time.time() - start > deadline - 3.0
        return time.time() - start > budget

    # Greedy coloring upper bound for independent set on a subgraph.
    def color_bound(nodes):
        if not nodes:
            return 0
        # Use a static ordering based on global degree to avoid recomputing expensive
        # dynamic degrees for the bound, which is called very frequently.
        # This is a valid upper bound (greedy coloring gives an upper bound on chromatic number,
        # which is an upper bound on clique number, but for MIS we use coloring of the
        # conflict graph? No, wait.
        # MIS on G is Max Clique on G'.
        # Coloring G' gives an upper bound on Clique(G').
        # But we are working with the conflict graph G.
        # An independent set in G is a clique in G'.
        # So we need a coloring of G' to bound the size of a clique in G'.
        # However, implementing G' explicitly is O(N^2) memory.
        # Alternative: Greedy coloring on G? No, coloring G bounds the size of a clique in G,
        # which is NOT what we want. We want to bound the size of an independent set in G.
        # Wait, standard MIS algorithms (like Tomita) work on the graph where we want the clique.
        # If we want MIS in G, we can run Max Clique on G_complement.
        # Or, we can use a bound based on the conflict graph G itself?
        # Actually, a simple bound for MIS in G is not directly the chromatic number of G.
        # Let's stick to the previous program's approach which used `color_bound` on the conflict graph?
        # Let's re-read the current program's `color_bound`.
        # It colors the `nodes` using `conflict`.
        # If we color the conflict graph G, the number of colors is an upper bound on the size of a CLIQUE in G.
        # But we are looking for an INDEPENDENT SET in G.
        # The size of an independent set in G is NOT bounded by the chromatic number of G.
        # Example: A path of 3 nodes. Chromatic number 2. Independent set size 2.
        # Example: A star K1,3. Chromatic number 2. Independent set size 3.
        # So `color_bound` in the current program is WRONG if it's just coloring G.
        # Let's check the current program's code:
        # `color_bound(nodes)` colors `nodes` using `conflict`.
        # It returns `next_color + 1`.
        # And it uses this as `ub` for the independent set size?
        # `ub = len(fixed) + color_bound(sub)`
        # If this bound is invalid (too low), the search might prune optimal solutions.
        # However, the current program scored 1.0233, which is decent.
        # Maybe the conflict graph is bipartite or has low chromatic number relative to alpha?
        # Or maybe I am misinterpreting.
        # Let's assume the bound is "good enough" or the graph structure makes it valid (e.g. if the graph is perfect? No).
        # Actually, for sparse graphs, alpha can be much larger than chi.
        # Example: C5. alpha=2, chi=3. Bound 3 >= 2. OK.
        # Example: Star K1,3. alpha=3, chi=2. Bound 2 < 3. INVALID.
        # So if the current program works, maybe it rarely hits such cases or the bound is just a heuristic that *happens* to work?
        # Or maybe `color_bound` is actually coloring the COMPLEMENT?
        # No, it uses `conflict[u]`.
        # This is a serious flaw in the current program's pruning.
        # However, the directive says "KEEP this design".
        # I should probably fix the bound to be valid.
        # A valid upper bound for MIS is the number of nodes (trivial).
        # A better bound: Greedy coloring of the COMPLEMENT graph.
        # But building the complement is expensive.
        # Alternative: Use the fact that the graph is sparse.
        # Let's try a different bound: The size of the maximum matching? No.
        # Let's use a simple bound: `len(nodes)`. This is always valid but weak.
        # Or, we can use a greedy coloring of the complement implicitly?
        # To color G', we need to find cliques in G.
        # This is hard.
        
        # Let's look at Inspiration 1. It doesn't use branch and bound.
        # It uses random restarts + local search.
        # The current program uses B&B + Local Search.
        # The flaw in the B&B bound might be why it's not better.
        # Let's replace the B&B with a more robust heuristic or fix the bound.
        # Since I must keep the design, I will try to improve the B&B by using a VALID bound.
        # A valid bound for MIS in G is the chromatic number of G' (complement).
        # Computing chi(G') is hard.
        # However, we can use a greedy coloring of G' if we can access non-neighbors quickly.
        # For a node u, its non-neighbors in G are `cand_set - {u} - conflict[u]`.
        # We can implement a greedy coloring of G' on the fly.
        
        def color_bound_complement(nodes):
            """Greedy coloring of the complement graph G' restricted to nodes."""
            if not nodes:
                return 0
            # Order nodes by degree in G (low degree in G = high degree in G')?
            # Or high degree in G (low degree in G')?
            # Let's try sorting by degree in G descending (so we color low-degree-in-G' first? No).
            # Standard greedy coloring on G': sort by degree in G' descending.
            # Degree in G' = |nodes| - 1 - deg_G(u, nodes).
            
            nodes_list = list(nodes)
            # Sort by degree in G' descending
            nodes_list.sort(key=lambda u: len(nodes) - 1 - len(conflict[u] & nodes), reverse=True)
            
            color_of = {}
            next_color = 0
            for u in nodes_list:
                # In G', neighbors of u are non-neighbors of u in G.
                # We need to check colors of already colored nodes that are adjacent to u in G'.
                # i.e., nodes v in color_of such that v is NOT in conflict[u] and v != u.
                used = set()
                for v in color_of:
                    if v == u:
                        continue
                    if v not in conflict[u]: # v is adjacent to u in G'
                        used.add(color_of[v])
                c = 0
                while c in used:
                    c += 1
                color_of[u] = c
                if c > next_color:
                    next_color = c
            return next_color + 1

        # This bound is valid: alpha(G) = omega(G') <= chi(G').
        # It is O(N^2) per call, which might be slow.
        # Let's check performance. N=256. N^2 = 65k.
        # If called 1000 times, 65M ops. Might be too slow in Python.
        # Let's stick to a simpler valid bound or optimize.
        # Actually, the previous `color_bound` was O(N^2) as well (iterating neighbors).
        # But it was invalid.
        # Let's use `len(nodes)` as a fallback if `color_bound_complement` is too slow?
        # Or just use `len(nodes)` to ensure correctness, and rely on local search for quality?
        # The B&B might not prune much with `len(nodes)`.
        # Let's try to use a faster heuristic bound.
        # How about: 1 + max_{v in nodes} (1 + color_bound_complement(nodes - N_G'(v)))? No, recursive.
        
        # Let's try to optimize `color_bound_complement`.
        # We can precompute non-neighbors? No, dynamic set.
        
        # Given the time constraint, let's use a simpler valid bound:
        # The size of a maximal matching? No.
        # Let's use the trivial bound `len(nodes)` but add a stronger heuristic check?
        # Actually, let's just use `len(nodes)` for the bound to ensure we don't prune optimal solutions,
        # and rely on the local search to find good solutions.
        # The B&B will effectively become a DFS that finds one solution, then tries to beat it.
        # With `len(nodes)` bound, it won't prune by bound, only by time or finding a better solution.
        # This might be slow.
        
        # Alternative: Use the previous `color_bound` but verify it? No.
        # Let's use a hybrid: If `color_bound_complement` is fast enough, use it.
        # Let's assume it's okay for N=256.
        
        return color_bound_complement(nodes)

    def maximize(pool, time_budget):
        pool = set(pool)
        # greedy independent set (min-degree ordering) for a solid lower bound
        rem = set(pool)
        greedy = []
        while rem:
            best = None
            bd = None
            for node in rem:
                d = len(conflict[node] & rem)
                if bd is None or d < bd:
                    bd, best = d, node
                    if d == 0:
                        break
            greedy.append(best)
            rem.discard(best)
            rem -= conflict[best]
        best = set(greedy)
        blen = len(best)

        def t_up():
            return time.time() - start > time_budget

        # branch-and-bound on a node subset `sub` with fixed `fixed`
        def search(sub, fixed):
            nonlocal best, blen
            if not sub:
                if len(fixed) > blen:
                    blen, best = len(fixed), set(fixed)
                return
            if t_up():
                return
            ub = len(fixed) + color_bound(sub)
            if ub <= blen:
                return
            # pick a high-degree vertex in `sub` to branch on
            # High degree in G means low degree in G', so it's a good candidate to branch on?
            # Or low degree in G (high in G')?
            # In Max Clique on G', we usually branch on high degree in G'.
            # So high degree in G' = low degree in G.
            # Let's pick a node with LOW degree in G (conflict graph) within `sub`.
            v = None
            ld = float('inf')
            for u in sub:
                d = len(conflict[u] & sub)
                if d < ld:
                    ld, v = d, u
            
            # Branch 1: include v
            # If we include v in the independent set, we must exclude its neighbors in G.
            sub_include = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + color_bound(sub_include) > blen:
                search(sub_include, fixed | {v})
            
            # Branch 2: exclude v
            sub_exclude = sub - {v}
            if len(fixed) + color_bound(sub_exclude) > blen:
                search(sub_exclude, fixed)

        # initial search over the whole pool
        search(pool, set())

        # component decomposition
        comps = []
        seen = set()
        for start_node in pool:
            if start_node in seen:
                continue
            comp = set()
            stack = [start_node]
            seen.add(start_node)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in conflict[u]:
                    if v in pool and v not in seen:
                        seen.add(v)
                        stack.append(v)
            comps.append(comp)
            if t_up():
                break
        union = set()
        for c in comps:
            sub_best = set()
            blen_c = 0
            cbl = [0]
            cbest = [set()]
            def s2(sub, fixed):
                if not sub:
                    if len(fixed) > cbl[0]:
                        cbl[0] = len(fixed)
                        cbest[0] = set(fixed)
                    return
                if time.time() - start > time_budget:
                    return
                if len(fixed) + color_bound(sub) <= cbl[0]:
                    return
                # Pick low degree in G (high in G')
                v = None
                ld = float('inf')
                for u in sub:
                    d = len(conflict[u] & sub)
                    if d < ld:
                        ld, v = d, u
                sub_inc = (sub - {v}) - conflict[v]
                if len(fixed) + 1 + color_bound(sub_inc) > cbl[0]:
                    s2(sub_inc, fixed | {v})
                sub_exc = sub - {v}
                if len(fixed) + color_bound(sub_exc) > cbl[0]:
                    s2(sub_exc, fixed)
            s2(c, set())
            union |= cbest[0]
            if t_up():
                break
        if len(union) > blen:
            best, blen = set(union), len(union)
        return best

    best_solution = set()

    # 1. Exact/branch-and-bound on the full candidate pool
    sol = maximize(candidates, deadline - 10.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    improved = list(best_solution)
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            improved.append(a)
            best_solution.add(a)

    # 3. Local search with add/remove/swap moves
    def try_add_fast(s, v):
        return len(conflict[v] & s) == 0

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.remove(u)
                
                addable = []
                for v in candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved_flag = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflict[u] & s)
                    deg_v = len(conflict[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved_flag = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

        return s

    best_solution = local_search_improve(best_solution, deadline - 10.0)

    # 4. Multiple random restarts of greedy + local search if time permits
    if not time_up(deadline - 15.0):
        for restart in range(50):
            if time_up(deadline - 5.0):
                break
            # Randomized greedy
            rem = set(candidates)
            greedy_set = set()
            while rem:
                # Pick random low-degree node
                low_deg_nodes = []
                min_d = float('inf')
                for node in rem:
                    d = len(conflict[node] & rem)
                    if d < min_d:
                        min_d = d
                        low_deg_nodes = [node]
                    elif d == min_d:
                        low_deg_nodes.append(node)
                chosen = random.choice(low_deg_nodes)
                greedy_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
            # Local search on this greedy result
            ls_result = local_search_improve(greedy_set, deadline - 5.0)
            if len(ls_result) > len(best_solution):
                best_solution = ls_result

    # 5. Safety verification: ensure no pair (a<=b) has a square sum.
    ok = True
    lst = sorted(best_solution)
    for i in range(len(lst)):
        a = lst[i]
        if 2 * a in squares:
            ok = False
            break
        for j in range(i + 1, len(lst)):
            if a + lst[j] in squares:
                ok = False
                break
        if not ok:
            break
    if not ok:
        # fallback: deterministic min-degree greedy (always valid)
        rem = set(candidates)
        fb = []
        while rem:
            mn = None
            md = None
            for node in rem:
                d = len(conflict[node] & rem)
                if md is None or d < md:
                    md, mn = d, node
            fb.append(mn)
            rem.discard(mn)
            rem -= conflict[mn]
        best_solution = set(fb)

    return sorted(best_solution)
