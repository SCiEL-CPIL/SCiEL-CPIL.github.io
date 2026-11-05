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

        # 6-10. Random orderings
        for _ in range(5):
            order = list(V_set)
            rng.shuffle(order)
            candidates.append(greedy_fill(order))

        # Pick best candidate
        best_S = max(candidates, key=len)
        best_size = len(best_S)

        # --- Local search ---
        def local_search(S, budget, rng):
            t_ls_start = time.time()
            S = set(S)
            best_S = set(S)
            best_size = len(S)
            no_improve = 0
            max_no_improve = 50000

            while time.time() - t_ls_start < budget and no_improve < max_no_improve:
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
                    # 1-swap: size unchanged
                    # Accept with probability based on exploration
                    if rng.random() < 0.15:
                        S = trial_S
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
        for restart in range(10):
            if time.time() - t_restart_start > restart_budget:
                break
            # New random greedy
            order = list(V_set)
            rng.shuffle(order)
            S_init = greedy_fill(order)
            S_ls = local_search(S_init, budget * 0.05, rng)
            if len(S_ls) > best_size:
                best_size = len(S_ls)
                best_S = S_ls

        return list(best_S)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    budgets = [12, 30, 60]
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
