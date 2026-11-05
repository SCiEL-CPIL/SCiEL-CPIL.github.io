import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 135  # safety margin before 150s kill

    def gen_instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    def can_add(S, x):
        """Check if x can be added to sum-free set S without violation."""
        for a in S:
            if (a + x) in S or (x - a) in S:
                return False
        if (2 * x) in S:
            return False
        return True

    def conflict_count(S, x):
        """Count how many elements in S would conflict with x."""
        cnt = 0
        for a in S:
            if (a + x) in S or (x - a) in S:
                cnt += 1
        if (2 * x) in S:
            cnt += 1
        return cnt

    def greedy_from_order_priority(order, V_set, S0=None):
        """Greedy with conflict-aware priority: try elements with fewer conflicts first."""
        S = set(S0) if S0 is not None else set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def greedy_with_priority(V_set, S0=None, max_passes=3):
        """Greedy that repeatedly picks the element with fewest conflicts."""
        S = set(S0) if S0 is not None else set()
        candidates = [x for x in V_set if x not in S]
        for _ in range(max_passes):
            if not candidates:
                break
            # Sort by conflict count (fewer conflicts first), then by value
            candidates.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            added_any = False
            new_candidates = []
            for x in candidates:
                if x not in S and can_add(S, x):
                    S.add(x)
                    added_any = True
                elif x not in S:
                    new_candidates.append(x)
            candidates = new_candidates
            if not added_any:
                break
        return S

    def is_sum_free(S):
        s = set(S)
        L = list(S)
        for i in range(len(L)):
            a = L[i]
            for j in range(len(L)):
                b = L[j]
                if (a + b) in s:
                    return False
        return True

    def greedy_from_order(order, V_set, S0=None):
        S = set(S0) if S0 is not None else set()
        for x in order:
            if x not in S and can_add(S, x):
                S.add(x)
        return S

    def try_add_remaining(S, V_set, deadline):
        added = True
        while added and time.time() < deadline:
            added = False
            # Sort by conflict count for better ordering
            remaining = [x for x in V_set if x not in S]
            remaining.sort(key=lambda x: (conflict_count(S, x), abs(x), x))
            for x in remaining:
                if time.time() >= deadline:
                    break
                if x not in S and can_add(S, x):
                    S.add(x)
                    added = True
        return S

    def solve_instance(elements, seed_off, budget):
        t_start = time.time()
        V = list(elements)
        V_set = set(V)
        n = len(V)
        best = set()
        best_size = 0
        rng = random.Random(seed_off)

        # Candidate structural orders
        orders = []

        # A. By value descending (largest first) — upper interval
        orders.append(sorted(V, key=lambda x: x, reverse=True))
        # B. By value ascending
        orders.append(sorted(V, key=lambda x: x))
        # C. By |x| descending
        orders.append(sorted(V, key=lambda x: abs(x), reverse=True))
        # D. Odds first, then evens by |x|
        odds = [x for x in V if x % 2 != 0]
        evens = [x for x in V if x % 2 == 0]
        orders.append(sorted(odds, key=lambda x: abs(x), reverse=True) + sorted(evens, key=lambda x: abs(x), reverse=True))
        # E. Positives by value desc, then negatives by value asc (most neg)
        pos = sorted([x for x in V if x > 0], reverse=True)
        neg = sorted([x for x in V if x < 0])
        orders.append(pos + neg)
        # F. Negatives first (most negative), then positives
        neg2 = sorted([x for x in V if x < 0], reverse=True)
        pos2 = sorted([x for x in V if x > 0])
        orders.append(neg2 + pos2)
        # G. Upper value interval only: top 60% by value
        sv = sorted(V)
        k = int(n * 0.6)
        orders.append(sv[n-k:])
        # H. Top 60% by |x|
        sm = sorted(V, key=lambda x: abs(x), reverse=True)
        orders.append(sm[:k])
        # I. Random orders
        for _ in range(5):
            rand_order = V[:]
            rng.shuffle(rand_order)
            orders.append(rand_order)

        # Run greedy on each order
        for order in orders:
            if time.time() - t_start > budget:
                break
            S = greedy_from_order(order, V_set)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Also try conflict-aware greedy from scratch
        if time.time() - t_start < budget:
            S = greedy_with_priority(V_set, max_passes=5)
            if len(S) > best_size:
                best_size = len(S)
                best = set(S)

        # Local improvement: try adding any remaining element
        best = try_add_remaining(best, V_set, t_start + budget)
        best_size = len(best)

        # 1-opt swaps: remove one, try to add two
        t_swap = time.time()
        improved = True
        while improved and time.time() - t_start < budget * 0.9:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for a in S_list:
                if time.time() - t_start > budget * 0.9:
                    break
                trial = set(best)
                trial.discard(a)
                # try to fill with conflict-aware ordering
                remaining = [x for x in V_set if x not in trial]
                remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                for x in remaining:
                    if x not in trial and can_add(trial, x):
                        trial.add(x)
                if len(trial) > best_size:
                    best = trial
                    best_size = len(best)
                    improved = True
                    break
        best = try_add_remaining(best, V_set, t_start + budget)

        # 2-opt swaps: remove two, try to add three
        improved = True
        while improved and time.time() - t_start < budget:
            improved = False
            S_list = list(best)
            rng.shuffle(S_list)
            for i in range(len(S_list)):
                if time.time() - t_start > budget:
                    break
                a = S_list[i]
                for j in range(i + 1, len(S_list)):
                    if time.time() - t_start > budget:
                        break
                    b = S_list[j]
                    trial = set(best)
                    trial.discard(a)
                    trial.discard(b)
                    # try to fill with conflict-aware ordering
                    remaining = [x for x in V_set if x not in trial]
                    remaining.sort(key=lambda x: (conflict_count(trial, x), abs(x), x))
                    for x in remaining:
                        if x not in trial and can_add(trial, x):
                            trial.add(x)
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        improved = True
                        break
                if improved:
                    break
        best = try_add_remaining(best, V_set, t_start + budget)

        return list(best)

    instances = [
        gen_instance(60, 801),
        gen_instance(120, 802),
        gen_instance(240, 803),
    ]

    results = []
    # time budgets, leave margin
    budgets = [12, 30, 60]
    for i, elems in enumerate(instances):
        elapsed = time.time() - t0
        remaining = DEADLINE - elapsed - 5
        if remaining < 4:
            # fallback: odds
            res = [x for x in elems if x % 2 != 0]
            results.append(res)
            continue
        b = min(budgets[i], max(4, remaining))
        res = solve_instance(elems, seed_off=801 + i * 7, budget=b)
        # verify
        if not is_sum_free(res):
            res = [x for x in elems if x % 2 != 0]
        results.append(res)

    return results
