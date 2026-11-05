import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 145

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V_set = set(elements)
        n = len(V_set)

        # Build adjacency: for each v, list of (u, w) such that {v,u,w} is a hyperedge
        adj = {v: [] for v in V_set}
        V_list = list(V_set)
        for i, a in enumerate(V_list):
            # a + a = 2a
            if 2 * a in V_set:
                adj[a].append((a, 2 * a))
                adj[2 * a].append((a, a))
            for j in range(i + 1, n):
                b = V_list[j]
                c = a + b
                if c in V_set:
                    adj[a].append((b, c))
                    adj[b].append((a, c))
                    adj[c].append((a, b))

        # Degree: number of hyperedges containing v
        deg = {v: 0 for v in V_set}
        for v in V_set:
            deg[v] = len(adj[v])

        def check_conflict(v, S):
            """Return True if v can be added to S without conflict."""
            for (u, w) in adj[v]:
                if u == v:
                    if w in S:
                        return False
                else:
                    if u in S and w in S:
                        return False
            return True

        def greedy_fill(order, S=None):
            """Greedy fill following given order."""
            if S is None:
                S = set()
            for v in order:
                if v not in S and check_conflict(v, S):
                    S.add(v)
            return S

        rng = random.Random(seed_off)

        # --- Multiple greedy initializations ---
        candidates = []

        # 1. Odds
        S_odds = set(x for x in V_set if x % 2 != 0)
        candidates.append(S_odds)

        # 2. Greedy by degree (ascending)
        order_deg = sorted(V_set, key=lambda x: (deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg))

        # 3. Greedy by degree (descending)
        order_deg_desc = sorted(V_set, key=lambda x: (-deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg_desc))

        # 4. Greedy by absolute value (ascending)
        order_abs = sorted(V_set, key=lambda x: abs(x))
        candidates.append(greedy_fill(order_abs))

        # 5. Greedy by absolute value (descending)
        order_abs_desc = sorted(V_set, key=lambda x: -abs(x))
        candidates.append(greedy_fill(order_abs_desc))

        # 6-12. Random orderings
        for _ in range(7):
            order = list(V_set)
            rng.shuffle(order)
            candidates.append(greedy_fill(order))

        # Pick best candidate
        best_S = max(candidates, key=len)
        best_size = len(best_S)

        # --- Local search with 1-swap and 2-swap moves ---
        def local_search(S, budget, rng):
            t_ls_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            no_improve = 0
            max_no_improve = 80000

            while time.time() - t_ls_start < budget and no_improve < max_no_improve:
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break

                # Choose move type: 1-swap (add or replace) or 2-swap
                use_2swap = len(S) > 5 and rng.random() < 0.3

                if not use_2swap:
                    # 1-swap: try to add a random vertex
                    v = rng.choice(remaining)

                    # Find conflicts
                    conflicts = []
                    for (u, w) in adj[v]:
                        if u == v:
                            if w in S:
                                conflicts.append((w, w))
                        else:
                            if u in S and w in S:
                                conflicts.append((u, w))

                    if not conflicts:
                        S.add(v)
                        no_improve = 0
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        continue

                    # Try to resolve by removing one vertex
                    freq = {}
                    for (u, w) in conflicts:
                        if u == w:
                            freq[u] = freq.get(u, 0) + 1
                        else:
                            freq[u] = freq.get(u, 0) + 1
                            freq[w] = freq.get(w, 0) + 1

                    if not freq:
                        no_improve += 1
                        continue

                    max_f = max(freq.values())
                    candidates_rem = [k for k, c in freq.items() if c == max_f]
                    rem = rng.choice(candidates_rem)

                    trial_S = set(S)
                    trial_S.discard(rem)
                    trial_S.add(v)

                    # Verify trial_S is valid
                    valid = True
                    for (u, w) in conflicts:
                        if u == w:
                            if u in trial_S:
                                valid = False
                                break
                        else:
                            if u in trial_S and w in trial_S:
                                valid = False
                                break

                    if valid:
                        # 1-swap: size unchanged; accept with exploration probability
                        if rng.random() < 0.12:
                            S = trial_S
                        no_improve += 1
                    else:
                        no_improve += 1
                else:
                    # 2-swap: remove two vertices, add two others
                    # Pick two candidates to remove from S
                    S_list = list(S)
                    if len(S_list) < 2:
                        no_improve += 1
                        continue
                    rem1, rem2 = rng.sample(S_list, 2)

                    # Find all conflicts for each removed vertex
                    conf1 = []
                    for (u, w) in adj[rem1]:
                        if u == rem1:
                            if w in S and w != rem2:
                                conf1.append((w, w))
                        else:
                            if u in S and w in S and u != rem2 and w != rem2:
                                conf1.append((u, w))

                    conf2 = []
                    for (u, w) in adj[rem2]:
                        if u == rem2:
                            if w in S and w != rem1:
                                conf2.append((w, w))
                        else:
                            if u in S and w in S and u != rem1 and w != rem2:
                                conf2.append((u, w))

                    # After removing rem1 and rem2, find vertices that could be added
                    trial_without = set(S)
                    trial_without.discard(rem1)
                    trial_without.discard(rem2)

                    add_candidates = []
                    for v in remaining:
                        # Check if v can be added to trial_without
                        ok = True
                        for (u, w) in adj[v]:
                            if u == v:
                                if w in trial_without:
                                    ok = False
                                    break
                            else:
                                if u in trial_without and w in trial_without:
                                    ok = False
                                    break
                        if ok:
                            add_candidates.append(v)

                    if len(add_candidates) >= 2:
                        add1, add2 = rng.sample(add_candidates, 2)
                        trial_S = trial_without | {add1, add2}

                        # Verify full validity
                        valid = True
                        for v in trial_S:
                            for (u, w) in adj[v]:
                                if u == v:
                                    if w in trial_S:
                                        valid = False
                                        break
                                else:
                                    if u in trial_S and w in trial_S:
                                        valid = False
                                        break
                            if not valid:
                                break

                        if valid and len(trial_S) > len(S):
                            S = trial_S
                            no_improve = 0
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = set(S)
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1

            return best_S

        # Split budget: 70% local search, 30% restarts
        ls_budget = budget * 0.7
        restart_budget = budget * 0.3

        # First local search from best initial
        best_S = local_search(best_S, ls_budget, rng)
        best_size = len(best_S)

        # Try a few restarts if time remains
        t_restart_start = time.time()
        for restart in range(15):
            if time.time() - t_restart_start > restart_budget:
                break
            # New random greedy
            order = list(V_set)
            rng.shuffle(order)
            S_init = greedy_fill(order)
            S_ls = local_search(S_init, budget * 0.04, rng)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_S = S_ls

        # --- ILP polish step if time remains ---
        remaining_time = DEADLINE - time.time() - 5
        if remaining_time > 8 and n <= 240:
            try:
                from scipy.optimize import milp, LinearConstraint, Bounds
                import numpy as np

                # Variables: x_v in {0,1} for each v in V_set
                V_idx = {v: i for i, v in enumerate(V_list)}
                m = n

                # Objective: maximize sum x_v
                c = -np.ones(m)  # minimize negative sum

                # Constraints: for each hyperedge {a,b,c} with a+b=c, x_a + x_b + x_c <= 2
                # (cannot have all three)
                A_rows = []
                b_ub = []
                b_lb = []
                for a in V_list:
                    for (b, c) in adj[a]:
                        # Hyperedge {a, b, c} where a+b=c or similar
                        # We need: not (x_a=1 and x_b=1 and x_c=1)
                        # i.e., x_a + x_b + x_c <= 2
                        ia = V_idx[a]
                        ib = V_idx[b]
                        ic = V_idx[c]
                        row = np.zeros(m)
                        row[ia] = 1
                        row[ib] = 1
                        row[ic] = 1
                        A_rows.append(row)
                        b_lb.append(0)
                        b_ub.append(2)

                if A_rows:
                    A = np.array(A_rows)
                    lb = np.array(b_lb)
                    ub = np.array(b_ub)
                    constraints = LinearConstraint(A, lb, ub)
                else:
                    constraints = LinearConstraint(np.zeros((0, m)), np.zeros(0), np.zeros(0))

                bounds = Bounds(np.zeros(m), np.ones(m))
                integrality = np.ones(m)  # all binary

                result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality,
                              options={"time_limit": remaining_time - 2})

                if result.status == 0 and result.x is not None:
                    ilp_S = set()
                    for i, v in enumerate(V_list):
                        if result.x[i] > 0.5:
                            ilp_S.add(v)
                    if len(ilp_S) > best_size:
                        best_S = ilp_S
                        best_size = len(best_S)
            except Exception:
                pass

        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [10, 25, 50]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
