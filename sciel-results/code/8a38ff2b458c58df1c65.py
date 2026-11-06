import math
import time
import random

def construct():
    start = time.time()
    deadline = 85.0
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
    num_cands = len(candidates)
    cand_index = {a: i for i, a in enumerate(candidates)}

    # Build conflict adjacency as lists for each candidate
    conflict = [set() for _ in range(num_cands)]
    for idx, a in enumerate(candidates):
        for jdx in range(idx + 1, num_cands):
            b = candidates[jdx]
            if a + b in squares:
                conflict[idx].add(jdx)
                conflict[jdx].add(idx)

    def time_up(budget=None):
        if budget is None:
            return time.time() - start > deadline - 3.0
        return time.time() - start > budget

    def try_add_fast(s_mask, v_idx):
        # s_mask is a frozenset or set of indices
        return len(conflict[v_idx] & s_mask) == 0

    def greedy_min_degree():
        remaining = set(range(num_cands))
        independent_set = set()
        while remaining:
            min_deg = float('inf')
            chosen = None
            for v in remaining:
                deg = len(conflict[v] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    chosen = v
            independent_set.add(chosen)
            remaining.discard(chosen)
            remaining -= conflict[chosen]
        return independent_set

    def local_search_improve(s, time_budget):
        s = set(s)
        improved_flag = True
        while improved_flag and not time_up(time_budget):
            improved_flag = False

            # 1. Try to add any single node
            for v in range(num_cands):
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved_flag = True

            # 2. Try to remove one node to add >= 2 nodes
            removal_improved = False
            for u in list(s):
                if time_up(time_budget):
                    break
                s.discard(u)
                
                addable = []
                for v in range(num_cands):
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

            # 3. Try 2-swap: remove 2 nodes, add 3+ nodes
            if not improved_flag and not time_up(time_budget):
                s_list = list(s)
                for i in range(len(s_list)):
                    if time_up(time_budget):
                        break
                    for j in range(i + 1, len(s_list)):
                        if time_up(time_budget):
                            break
                        u1, u2 = s_list[i], s_list[j]
                        s.discard(u1)
                        s.discard(u2)
                        
                        addable = []
                        for v in range(num_cands):
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

            # 4. Random kick: remove 2-3 random nodes, then greedily re-add
            if not improved_flag and not time_up(time_budget):
                s_list = list(s)
                if len(s_list) > 5:
                    kick_size = min(3, len(s_list) - 3)
                    kicked = random.sample(s_list, kick_size)
                    for k in kicked:
                        s.discard(k)
                    # Greedily re-add
                    for v in range(num_cands):
                        if v not in s and try_add_fast(s, v):
                            s.add(v)
                    if len(s) > len(s_list) - kick_size:
                        improved_flag = True

        return s

    # 1. Start with greedy min-degree solution
    best_solution = greedy_min_degree()
    best_size = len(best_solution)

    # 2. Local search on the greedy solution
    best_solution = local_search_improve(best_solution, deadline - 20.0)
    best_size = len(best_solution)

    # 3. Multiple random restarts with different orderings
    for restart in range(200):
        if time_up(deadline - 10.0):
            break
        
        strategy = restart % 3
        if strategy == 0:
            shuffled = list(range(num_cands))
            random.shuffle(shuffled)
            current_set = set()
            for v in shuffled:
                if try_add_fast(current_set, v):
                    current_set.add(v)
        elif strategy == 1:
            rem = set(range(num_cands))
            current_set = set()
            while rem:
                degs = {}
                for node in rem:
                    degs[node] = len(conflict[node] & rem)
                min_d = min(degs.values()) if degs else 0
                tau = max(1.0, min_d * 0.5)
                weights = []
                total_w = 0.0
                for node in rem:
                    w = math.exp(-degs[node] / tau)
                    weights.append(w)
                    total_w += w
                r = random.random() * total_w
                cum = 0.0
                chosen = list(rem)[0]
                for node, w in zip(rem, weights):
                    cum += w
                    if r <= cum:
                        chosen = node
                        break
                current_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
        else:
            rem = set(range(num_cands))
            current_set = set()
            while rem:
                min_deg = float('inf')
                candidates_min = []
                for v in rem:
                    deg = len(conflict[v] & rem)
                    if deg < min_deg:
                        min_deg = deg
                        candidates_min = [v]
                    elif deg == min_deg:
                        candidates_min.append(v)
                chosen = random.choice(candidates_min)
                current_set.add(chosen)
                rem.discard(chosen)
                rem -= conflict[chosen]
        
        ls_result = local_search_improve(current_set, deadline - 10.0)
        if len(ls_result) > best_size:
            best_size = len(ls_result)
            best_solution = ls_result

    # 4. Final polish
    if not time_up(deadline - 2.0):
        best_solution = local_search_improve(best_solution, deadline - 1.0)

    # 5. Safety verification
    ok = True
    result_vals = [candidates[i] for i in best_solution]
    result_set = set(result_vals)
    lst = sorted(result_vals)
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
        rem = set(range(num_cands))
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
        result_vals = [candidates[i] for i in fb]
        best_solution = set(fb)

    return sorted(result_vals)
