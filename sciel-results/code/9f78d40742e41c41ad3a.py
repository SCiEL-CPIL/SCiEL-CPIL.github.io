import math
import time
import random

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

    # Greedy coloring upper bound for independent set on a subgraph.
    _color_cache = {}
    def color_bound(nodes):
        if not nodes:
            return 0
        key = frozenset(nodes)
        if key in _color_cache:
            return _color_cache[key]
        degs = [(len(conflict[u] & nodes), u) for u in nodes]
        degs.sort(reverse=True)
        ordered = [u for _, u in degs]
        
        color_of = {}
        next_color = 0
        for u in ordered:
            used = set()
            for v in conflict[u]:
                if v in nodes and v in color_of:
                    used.add(color_of[v])
            c = 0
            while c in used:
                c += 1
            color_of[u] = c
            if c > next_color:
                next_color = c
        result = next_color + 1
        _color_cache[key] = result
        return result

    def time_up(budget=None):
        if budget is None:
            return time.time() - start > deadline - 3.0
        return time.time() - start > budget

    # Independent-set maximizer over the candidate pool.
    # Uses a greedy lower bound + exact branch-and-bound with coloring upper bound.
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
            v = None
            hv = -1
            for u in sub:
                d = len(conflict[u] & sub)
                if d > hv:
                    hv, v = d, u
            # branch 1: include v
            sub_include = (sub - {v}) - conflict[v]
            if len(fixed) + 1 + color_bound(sub_include) > blen:
                search(sub_include, fixed | {v})
            # branch 2: exclude v
            sub_exclude = sub - {v}
            if len(fixed) + color_bound(sub_exclude) > blen:
                search(sub_exclude, fixed)

        # initial search over the whole pool
        search(pool, set())

        # component decomposition: solve each connected component exactly,
        # union the solutions (cross-component vertices never conflict)
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
                v = None
                hv = -1
                for u in sub:
                    d = len(conflict[u] & sub)
                    if d > hv:
                        hv, v = d, u
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

    # 1. Exact/branch-and-bound on the full candidate pool (fast due to matching structure).
    sol = maximize(candidates, deadline - 10.0)
    best_solution = set(sol)

    # 2. Local augmentation: add any candidate that conflicts with nothing in the set.
    improved = list(best_solution)
    for a in candidates:
        if a not in best_solution and not (conflict[a] & best_solution):
            improved.append(a)
            best_solution.add(a)

    # 3. Local search with add/remove/swap moves (inspiration 1 style, more aggressive)
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

            # 3. Try 2-swap: remove 2 nodes, add 3+ nodes for net gain
            if not improved_flag and not time_up(time_budget):
                s_list = list(s)
                for i in range(len(s_list)):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, len(s_list)):
                        if time_up(time_budget):
                            break
                        u1, u2 = s_list[i], s_list[j]
                        s.remove(u1)
                        s.remove(u2)
                        
                        addable = []
                        for v in candidates:
                            if v not in s and v != u1 and v != u2:
                                if try_add_fast(s, v):
                                    addable.append(v)
                        
                        if len(addable) >= 3:
                            for v in addable:
                                s.add(v)
                            improved_flag = True
                            break
                        else:
                            s.add(u1)
                            s.add(u2)
                    if improved_flag:
                        break

        return s

    best_solution = local_search_improve(best_solution, deadline - 10.0)

    # 4. Multiple random restarts of greedy + local search if time permits
    if not time_up(deadline - 15.0):
        for restart in range(80):
            if time_up(deadline - 5.0):
                break
            # Randomized greedy with temperature-based selection
            rem = set(candidates)
            greedy_set = set()
            while rem:
                # Compute degrees
                degs = {}
                for node in rem:
                    degs[node] = len(conflict[node] & rem)
                min_d = min(degs.values())
                # Soft-min: prefer low-degree but allow some randomness
                # Use exponential weighting: weight = exp(-d/tau)
                tau = max(1.0, min_d * 0.5)
                weights = []
                total_w = 0.0
                for node in rem:
                    w = math.exp(-degs[node] / tau)
                    weights.append(w)
                    total_w += w
                # Weighted random choice
                r = random.random() * total_w
                cum = 0.0
                chosen = list(rem)[0]
                for node, w in zip(rem, weights):
                    cum += w
                    if r <= cum:
                        chosen = node
                        break
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

    # 6. Final polish: try one more round of local search with tighter budget
    if not time_up(deadline - 2.0):
        best_solution = local_search_improve(best_solution, deadline - 1.0)
        # Re-verify after polish
        ok2 = True
        lst2 = sorted(best_solution)
        for i in range(len(lst2)):
            a = lst2[i]
            if 2 * a in squares:
                ok2 = False
                break
            for j in range(i + 1, len(lst2)):
                if a + lst2[j] in squares:
                    ok2 = False
                    break
            if not ok2:
                break
        if not ok2:
            # revert to pre-polish solution
            # (we didn't save it, so just use the fallback)
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
