import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    N = 192
    MAX = 48
    all_candidates = list(range(1, N + 1))

    def is_valid(S):
        """Full validity check for a set S."""
        n = len(S)
        for i in range(n):
            x = S[i]
            # x must not divide sum of any 3 distinct OTHER elements
            for (a, b, c) in combinations(S, 3):
                if x in (a, b, c):
                    continue
                if (a + b + c) % x == 0:
                    return False
        return True

    def can_add(x, S):
        """Check if x can be added to S (S is a list/set of distinct ints)."""
        n = len(S)
        if n >= 3:
            # x dividing sum of 3 others in S
            for (a, b, c) in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        # each a in S must not divide x + u + v for distinct u,v in S\{a}
        for a in S:
            others = [v for v in S if v != a]
            if len(others) < 2:
                continue
            # quick bound: if x + max two < a, impossible
            m1 = -1
            m2 = -1
            for v in others:
                if v > m1:
                    m2 = m1
                    m1 = v
                elif v > m2:
                    m2 = v
            if x + m1 + m2 < a:
                continue
            for u in others:
                for v in others:
                    if v == u:
                        continue
                    if (x + u + v) % a == 0:
                        return False
        return True

    def run_greedy(order):
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        return S

    def score_candidate(x, S, rng):
        """Randomized heuristic score: prefer large, low-divisibility, low-collisions."""
        # base: prefer large
        s = x
        # penalty: number of a in S that divide x+u+v for some pair u,v in S\{a}
        # approximate with a sample to keep it cheap
        if S:
            # count how many 'a' would be violated if x added (sample pairs)
            viol = 0
            others_all = list(S)
            # sample a few a
            if len(others_all) > 12:
                sample_a = rng.sample(others_all, 12)
            else:
                sample_a = others_all
            for a in sample_a:
                rest = [v for v in S if v != a]
                if len(rest) < 2:
                    continue
                # sample pairs
                pairs = list(combinations(rest, 2))
                if len(pairs) > 20:
                    pairs = rng.sample(pairs, 20)
                for (u, v) in pairs:
                    if (x + u + v) % a == 0:
                        viol += 1
                        break
            s -= 8 * viol
        return s

    def greedy_with_scores(cands, rng, seed_bias):
        S = []
        Sset = set()
        remaining = list(cands)
        while remaining:
            # score all remaining
            best = None
            best_sc = -1
            for x in remaining:
                if not can_add(x, S):
                    continue
                sc = score_candidate(x, S, rng) + seed_bias * (N - x)
                if sc > best_sc:
                    best_sc = sc
                    best = x
            if best is None:
                break
            S.append(best)
            Sset.add(best)
            remaining.remove(best)
        return S

    best_set = []
    best_size = 0

    # 1. quick deterministic greedy
    for order in (sorted(all_candidates, reverse=True), sorted(all_candidates)):
        S = run_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 2. randomized scoring greedy + local search
    rng = random.Random(12345)

    def local_search(S, rng, rounds):
        S = list(S)
        Sset = set(S)
        improved = True
        while improved and rounds > 0:
            improved = False
            rounds -= 1
            # try additions
            for x in all_candidates:
                if x in Sset:
                    continue
                if not can_add(x, S):
                    continue
                S.append(x)
                Sset.add(x)
                improved = True
                break
            if improved:
                continue
            # try swaps: remove a, add x
            # pick a random element to remove
            for _ in range(6):
                a = rng.choice(S)
                S2 = [v for v in S if v != a]
                S2set = set(S2)
                # find a good x to add
                cands = [x for x in all_candidates if x not in S2set]
                rng.shuffle(cands)
                found = False
                for x in cands:
                    if not can_add(x, S2):
                        continue
                    # only accept if it doesn't reduce size (it won't, we removed 1 added 1)
                    # accept swap to diversify
                    S2.append(x)
                    S2set.add(x)
                    S = S2
                    Sset = S2set
                    improved = True
                    found = True
                    break
                if found:
                    break
        return S

    # multiple randomized greedy starts
    for it in range(200):
        if time.time() > deadline - 8.0:
            break
        # build a randomized scoring order
        S = greedy_with_scores(all_candidates, rng, seed_bias=rng.uniform(0, 0.5))
        S = local_search(S, rng, rounds=40)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # 3. final polish: try to add any missing elements to best_set
    S = list(best_set)
    Sset = set(S)
    changed = True
    while changed:
        changed = False
        for x in all_candidates:
            if x in Sset:
                continue
            if can_add(x, S):
                S.append(x)
                Sset.add(x)
                changed = True
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    return best_set
