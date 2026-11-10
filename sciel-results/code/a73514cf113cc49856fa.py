import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    # ---- fast cardinality-indexed subset-sum DP machinery (from current program) ----
    def check_and_add(chosen, S_sets, x, deadline):
        """Tries to add x to chosen. Returns (success, new_chosen, new_S_sets)."""
        if time.time() > deadline:
            return False, chosen, S_sets
        m = len(chosen)
        N = [set() for _ in range(m + 2)]
        for k in range(m + 1, 0, -1):
            if k - 1 < len(S_sets):
                N[k] = {s + x for s in S_sets[k - 1]}
        # validity: N[k] disjoint from all S_sets[j] (j!=k) and all N[j] (j!=k)
        for k in range(1, m + 2):
            if not N[k]:
                continue
            for j in range(0, m + 1):
                if j == k:
                    continue
                if j < len(S_sets) and S_sets[j]:
                    if N[k] & S_sets[j]:
                        return False, chosen, S_sets
            for j in range(1, m + 2):
                if j == k:
                    continue
                if N[j]:
                    if N[k] & N[j]:
                        return False, chosen, S_sets
        new_S_sets = [set(s) for s in S_sets]
        for k in range(1, m + 2):
            if N[k]:
                if k >= len(new_S_sets):
                    new_S_sets.append(set())
                new_S_sets[k].update(N[k])
        return True, chosen + [x], new_S_sets

    def rebuild(chosen):
        """Recompute S_sets for a chosen list (for removal / swap)."""
        m = len(chosen)
        S = [set() for _ in range(m + 1)]
        S[0].add(0)
        for val in chosen:
            for k in range(len(S) - 2, -1, -1):
                if S[k]:
                    for s in S[k]:
                        S[k + 1].add(s + val)
        return S

    def validate(elements):
        """Exact check of the constraint."""
        if not elements:
            return True
        if len(set(elements)) != len(elements):
            return False
        for x in elements:
            if not isinstance(x, int) or x < 1 or x > 256:
                return False
        S = rebuild(elements)
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if S[i] & S[j]:
                    return False
        return True

    # ---- greedy construction (fast) over a given order ----
    def greedy_construct(order, deadline):
        chosen = []
        S_sets = [set([0])]
        for x in order:
            if time.time() > deadline:
                break
            ok, chosen, S_sets = check_and_add(chosen, S_sets, x, deadline)
            if ok:
                pass
        return chosen

    # ---- global conflict-driven removal (from cell findings) ----
    # For a candidate element x, measure how many cross-cardinality collisions
    # would be *caused* by x. Removing the most-conflicting element lets greedy
    # re-add others, escaping local optima.
    def conflict_score(chosen, x, S_sets):
        """Count cross-cardinality sum collisions attributable to x."""
        m = len(chosen)
        # new sums if x were present: N[k] from old S[k-1]
        N = [set() for _ in range(m + 2)]
        for k in range(m + 1, 0, -1):
            if k - 1 < len(S_sets):
                N[k] = {s + x for s in S_sets[k - 1]}
        score = 0
        for k in range(1, m + 2):
            if not N[k]:
                continue
            for j in range(0, m + 1):
                if j == k:
                    continue
                if j < len(S_sets) and S_sets[j]:
                    score += len(N[k] & S_sets[j])
            for j in range(1, m + 2):
                if j == k:
                    continue
                if N[j]:
                    score += len(N[k] & N[j])
        return score

    def conflict_driven_search(order, deadline):
        """Greedy + iterative global removal of most-conflicting element."""
        chosen = []
        S_sets = [set([0])]
        # first greedy pass
        for x in order:
            if time.time() > deadline:
                break
            ok, chosen, S_sets = check_and_add(chosen, S_sets, x, deadline)
            if ok:
                pass
        # now try to grow by removing the most-conflicting element and re-adding
        # blocked candidates repeatedly.
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # find most-conflicting element in chosen
            best_idx = -1
            best_conf = -1
            for idx, x in enumerate(chosen):
                # remove x, recompute, then measure how many remaining candidates conflict
                # cheap proxy: conflict of x against current S_sets
                conf = conflict_score(chosen, x, S_sets)
                if conf > best_conf:
                    best_conf = conf
                    best_idx = idx
            if best_idx < 0:
                break
            removed = chosen[best_idx]
            # rebuild without removed
            new_chosen = [x for i, x in enumerate(chosen) if i != best_idx]
            new_S = rebuild(new_chosen)
            # try to add back removed AND other candidates
            for x in order:
                if time.time() > deadline:
                    break
                if x in new_chosen:
                    continue
                ok, new_chosen, new_S = check_and_add(new_chosen, new_S, x, deadline)
                if ok:
                    pass
            if len(new_chosen) > len(chosen):
                chosen, S_sets = new_chosen, new_S
                improved = True
            else:
                # no improvement from removing best_idx; try removing next-most
                # (simple: stop to avoid thrashing)
                break
        return chosen

    # ---- local search: swap phase ----
    def swap_local_search(chosen, S_sets, deadline):
        """Try swapping out an element for a better one to grow the set."""
        chosen = list(chosen)
        S_sets = rebuild(chosen)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(chosen)):
                if time.time() > deadline:
                    break
                x = chosen[idx]
                base = [v for i, v in enumerate(chosen) if i != idx]
                base_S = rebuild(base)
                # try each candidate not in base
                for cand in range(1, 257):
                    if cand in base:
                        continue
                    ok, trial, trial_S = check_and_add(base, base_S, cand, deadline)
                    if ok:
                        # trial has same size as base (1 less than chosen)
                        # try to extend trial with more candidates to exceed chosen size
                        for extra in range(1, 257):
                            if time.time() > deadline:
                                break
                            if extra in trial:
                                continue
                            ok2, trial2, trial_S2 = check_and_add(trial, trial_S, extra, deadline)
                            if ok2 and len(trial2) > len(chosen):
                                chosen, S_sets = trial2, trial_S2
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return chosen

    # ---- main search over diverse orders ----
    best_set = list(range(128, 136))  # safe baseline (consecutive 8)
    best_size = len(best_set)

    orders = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(129, 257)) + list(range(1, 129)),
    ]
    random.seed(1234)
    for _ in range(12):
        if time.time() > deadline - 8:
            break
        perm = list(range(1, 257))
        random.shuffle(perm)
        orders.append(perm)

    for order in orders:
        if time.time() > deadline:
            break
        # fast greedy
        res = greedy_construct(order, deadline)
        if len(res) > best_size:
            best_size = len(res)
            best_set = list(res)
        # global conflict-driven removal on top
        res2 = conflict_driven_search(order, deadline)
        if len(res2) > best_size:
            best_size = len(res2)
            best_set = list(res2)
        # local swap polish on current best
        if time.time() < deadline - 3:
            res3 = swap_local_search(best_set, rebuild(best_set), deadline)
            if len(res3) > best_size:
                best_size = len(res3)
                best_set = list(res3)

    # final validation + fallback
    if not validate(best_set):
        best_set = list(range(128, 136))

    best_set = [int(x) for x in best_set]
    return best_set
