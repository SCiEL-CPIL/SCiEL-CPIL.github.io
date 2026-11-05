import time
import random

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
        V_list = list(V_set)

        # Build hyperedge structure: for each v, list of pairs (u,w) where {v,u,w} is a sum triple
        # adj[v] = list of (u, w) meaning v+u=w or v+w=u or u+w=v
        adj = {v: [] for v in V_set}
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

        deg = {v: len(adj[v]) for v in V_set}

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

        # 2. Greedy by degree ascending
        order_deg = sorted(V_set, key=lambda x: (deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg))

        # 3. Greedy by degree descending
        order_deg_desc = sorted(V_set, key=lambda x: (-deg[x], abs(x)))
        candidates.append(greedy_fill(order_deg_desc))

        # 4. Greedy by abs value ascending
        order_abs = sorted(V_set, key=lambda x: abs(x))
        candidates.append(greedy_fill(order_abs))

        # 5. Greedy by abs value descending
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

        # --- Enhanced local search ---
        def local_search(S, budget, rng):
            t_ls = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            no_improve = 0
            max_no_improve = 80000

            while time.time() - t_ls < budget and no_improve < max_no_improve:
                remaining = [x for x in V_set if x not in S]
                if not remaining:
                    break

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

                # Build frequency map of conflicting vertices
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

                # Score each candidate removal: prefer removing vertices with high degree
                # (they're "costly" to keep) and high frequency in conflicts
                def removal_score(x):
                    return (freq.get(x, 0), deg.get(x, 0))

                scored = sorted(freq.keys(), key=removal_score, reverse=True)

                # Try top few removal candidates
                improved = False
                for rem in scored[:min(3, len(scored))]:
                    trial_S = set(S)
                    trial_S.discard(rem)
                    trial_S.add(v)

                    # Verify validity
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
                        S = trial_S
                        no_improve = 0
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = set(S)
                        improved = True
                        break

                if not improved:
                    # Try 2-swap: remove 2, add 1 (net -1, but may enable future gains)
                    if len(scored) >= 2 and rng.random() < 0.3:
                        rem1 = scored[0]
                        rem2 = scored[1]
                        trial_S = set(S)
                        trial_S.discard(rem1)
                        trial_S.discard(rem2)
                        trial_S.add(v)

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

                        if valid and rng.random() < 0.1:
                            S = trial_S

                    no_improve += 1

            # Final polish: try adding all remaining in degree order
            S_final = set(best_S)
            remaining = [x for x in V_set if x not in S_final]
            remaining.sort(key=lambda x: deg[x])
            for v in remaining:
                if check_conflict(v, S_final):
                    S_final.add(v)
            if len(S_final) > best_size:
                best_S = S_final
                best_size = len(S_final)

            return best_S

        # Budget allocation: 60% main LS, 25% restarts, 15% final polish
        ls_budget = budget * 0.60
        restart_budget = budget * 0.25
        polish_budget = budget * 0.15

        # First local search
        best_S = local_search(best_S, ls_budget, rng)
        best_size = len(best_S)

        # Restarts
        t_restart = time.time()
        restart_count = 0
        while time.time() - t_restart < restart_budget:
            order = list(V_set)
            rng.shuffle(order)
            S_init = greedy_fill(order)
            if len(S_init) >= best_size:
                S_ls = local_search(S_init, budget * 0.04, rng)
                if len(S_ls) > best_size:
                    best_size = len(S_ls)
                    best_S = S_ls
            restart_count += 1
            if restart_count > 50:
                break

        # Final polish with dedicated budget
        if time.time() - t_start < DEADLINE - 8:
            S_polish = set(best_S)
            # Try adding remaining vertices in multiple orders
            for _ in range(3):
                remaining = [x for x in V_set if x not in S_polish]
                rng.shuffle(remaining)
                for v in remaining:
                    if check_conflict(v, S_polish):
                        S_polish.add(v)
            if len(S_polish) > best_size:
                best_S = S_polish
                best_size = len(S_polish)

        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [15, 35, 70]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 6
        if remaining < 5:
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(5, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        results.append(res)

    return results
