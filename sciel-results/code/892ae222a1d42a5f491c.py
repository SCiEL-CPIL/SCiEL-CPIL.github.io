import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # a = bc/(b+c), so (b+c) | bc
    triples = []
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            s = b + c
            prod = b * c
            if prod % s == 0:
                a = prod // s
                if 1 <= a < b <= N:
                    triples.append((a, b, c))

    # Build lookup maps
    triple_map_a = {}  # a -> set of (b,c)
    triple_map_b = {}  # b -> set of (a,c)
    triple_map_c = {}  # c -> set of (a,b)

    for a, b, c in triples:
        triple_map_a.setdefault(a, set()).add((b, c))
        triple_map_b.setdefault(b, set()).add((a, c))
        triple_map_c.setdefault(c, set()).add((a, b))

    # Precompute, for each element, the total number of triples it participates in
    # (a proxy for "conflict level" / how hard it is to keep in a large set).
    conflict_count = {}
    for a, b, c in triples:
        conflict_count[a] = conflict_count.get(a, 0) + 1
        conflict_count[b] = conflict_count.get(b, 0) + 1
        conflict_count[c] = conflict_count.get(c, 0) + 1

    def has_triple(S):
        for a in S:
            if a in triple_map_a:
                for b, c in triple_map_a[a]:
                    if b in S and c in S:
                        return True
        return False

    def can_add(S, x):
        if x in triple_map_a:
            for b, c in triple_map_a[x]:
                if b in S and c in S:
                    return False
        if x in triple_map_b:
            for a, c in triple_map_b[x]:
                if a in S and c in S:
                    return False
        if x in triple_map_c:
            for a, b in triple_map_c[x]:
                if a in S and b in S:
                    return False
        return True

    def freedom(S):
        # Surrogate "smooth" score: prefer larger sets, and among equal size,
        # sets whose remaining (outside) elements have fewer conflicts with S
        # (i.e. more room to grow). Computed cheaply.
        size = len(S)
        room = 0
        for x in range(1, N + 1):
            if x not in S:
                # count how many of x's partners are already in S (fewer = more room)
                cnt = 0
                if x in triple_map_a:
                    for b, c in triple_map_a[x]:
                        if b in S and c in S:
                            cnt += 1
                if x in triple_map_b:
                    for a, c in triple_map_b[x]:
                        if a in S and c in S:
                            cnt += 1
                if x in triple_map_c:
                    for a, b in triple_map_c[x]:
                        if a in S and b in S:
                            cnt += 1
                room += (1.0 / (1.0 + cnt))
        return size * 1000.0 + room

    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search_add(S, max_iters=2000):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 6.0:
                break
            improved = False
            iters += 1
            for x in range(1, N + 1):
                if x not in S:
                    if can_add(S, x):
                        S.add(x)
                        improved = True
                        break
        return S

    rng = random.Random(12345)

    # ---- Phase 1: diverse initial constructions ----
    candidates = []

    # 1a: descending (high numbers first, safest)
    candidates.append(greedy_build(list(range(N, 0, -1))))
    # 1b: ascending
    candidates.append(greedy_build(list(range(1, N + 1))))
    # 1c: conflict-weighted order (fewest conflicts first -> more room)
    order_conf = sorted(range(1, N + 1), key=lambda x: (conflict_count.get(x, 0), x))
    candidates.append(greedy_build(order_conf))
    # 1d: reverse conflict order
    order_conf_rev = sorted(range(1, N + 1), key=lambda x: (-conflict_count.get(x, 0), x))
    candidates.append(greedy_build(order_conf_rev))

    # 1e: several random orders
    for _ in range(60):
        if time.time() >= deadline - 40.0:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidates.append(greedy_build(order))

    for c in candidates:
        c = local_search_add(c, max_iters=300)
    candidates = [c for c in candidates if not has_triple(c)]

    best = max(candidates, key=len)
    best_size = len(best)

    # ---- Phase 2: rearrangement / simulated-annealing search ----
    # Moves: remove a small blocker set R, add a small candidate set A.
    # Accept if size improves, or (with annealing probability) if freedom improves.
    def rearrange_search(S, rng, max_iters=20000):
        S = set(S)
        cur_freedom = freedom(S)
        temp = 3.0
        cool = 0.99985
        iters = 0
        while iters < max_iters:
            if time.time() >= deadline - 8.0:
                break
            iters += 1

            # pick move type
            r = rng.random()
            if r < 0.55:
                # 1-for-1 swap (rearrange, keep size)
                if not S:
                    continue
                y = rng.choice(list(S))
                x = rng.randint(1, N)
                if x == y or x in S:
                    continue
                R = {y}
                A = {x}
            elif r < 0.80:
                # 1-for-2 (try to grow by 1)
                if not S:
                    continue
                y = rng.choice(list(S))
                cands = [x for x in range(1, N + 1) if x not in S]
                if len(cands) < 2:
                    continue
                a1, a2 = rng.sample(cands, 2)
                R = {y}
                A = {a1, a2}
            else:
                # 2-for-1 (shrink by 1 to free space, then local-add may regrow)
                if len(S) < 2:
                    continue
                y1, y2 = rng.sample(list(S), 2)
                cands = [x for x in range(1, N + 1) if x not in S]
                if not cands:
                    continue
                x = rng.choice(cands)
                R = {y1, y2}
                A = {x}

            S_trial = S - R
            ok = True
            for x in A:
                if x in S_trial:
                    ok = False
                    break
                if not can_add(S_trial, x):
                    ok = False
                    break
            if not ok:
                continue
            S_trial = S_trial | A
            if has_triple(S_trial):
                continue

            new_size = len(S_trial)
            if new_size > len(S):
                S = S_trial
                cur_freedom = freedom(S)
                continue
            if new_size == len(S):
                new_freedom = freedom(S_trial)
                diff = new_freedom - cur_freedom
                if diff >= 0:
                    S = S_trial
                    cur_freedom = new_freedom
                else:
                    # accept with annealing probability
                    if rng.random() < (1.0 - 1.0 / (1.0 + 0.5 * abs(diff) / (temp + 1e-9))):
                        S = S_trial
                        cur_freedom = new_freedom
            # new_size < len(S): reject (we never want to shrink net)
            temp *= cool
        return S

    # Apply rearrangement search from the best and a few good candidates
    for seed in [best] + [c for c in candidates if len(c) >= best_size - 1][:4]:
        if time.time() >= deadline - 15.0:
            break
        S_try = rearrange_search(seed, rng, max_iters=8000)
        S_try = local_search_add(S_try, max_iters=500)
        if not has_triple(S_try) and len(S_try) > best_size:
            best = S_try
            best_size = len(S_try)

    # ---- Phase 3: final polish ----
    best = local_search_add(best, max_iters=5000)
    # one more short rearrange pass to exploit the enlarged set
    S_try = rearrange_search(best, rng, max_iters=6000)
    S_try = local_search_add(S_try, max_iters=2000)
    if not has_triple(S_try) and len(S_try) > best_size:
        best = S_try
        best_size = len(S_try)
    best = local_search_add(best, max_iters=2000)

    # Verify
    if has_triple(best):
        # Fallback: high numbers only
        best = set(range(91, N + 1))

    return list(best)
