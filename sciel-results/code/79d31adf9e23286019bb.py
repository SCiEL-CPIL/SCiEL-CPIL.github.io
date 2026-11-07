import time
import random

# EVOLVE-BLOCK-START
N = 128
DEADLINE_FRAC = 0.90  # safety margin: stop at 90% of 90s wall-clock

def _build_conflicts():
    # conflict[a] = list of (b, c) such that {a,b,c,a+b,a+c,b+c} is a forbidden 6-config
    # a < b < c, distinct, all six values in 1..N (a+b,a+c,b+c <= N)
    conflict = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            ab = a + b
            if ab > N:
                break
            for c in range(b + 1, N + 1):
                ac = a + c
                bc = b + c
                if ac > N or bc > N:
                    break
                # all six values valid range; a,b,c distinct by construction
                conflict[a].append((b, c))
                conflict[b].append((a, c))
                conflict[c].append((a, b))
    return conflict

def _active_conflicts(S, conflict):
    # count forbidden 6-configs fully contained in S
    total = 0
    for a in S:
        for (b, c) in conflict[a]:
            if b in S and c in S:
                total += 1
    return total

def _remove_one(S, conflict, max_iter=20000):
    # greedily remove elements to break all conflicts; returns new set
    S = set(S)
    for _ in range(max_iter):
        bad = None
        for a in S:
            for (b, c) in conflict[a]:
                if b in S and c in S:
                    bad = (a, b, c)
                    break
            if bad:
                break
        if bad is None:
            break
        # remove the element in the conflict with the largest degree (fewest future conflicts)
        a, b, c = bad
        best = min((a, b, c), key=lambda x: len(conflict[x]))
        S.discard(best)
    return S

def _add_remove_local(S, conflict, max_iter, deadline):
    S = set(S)
    prev_size = -1
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        # try adding each candidate
        for x in range(1, N + 1):
            if x in S:
                continue
            if _active_conflicts(S | {x}, conflict) == 0:
                S.add(x)
                improved = True
                break
        if improved:
            continue
        # try removing one element then re-adding to grow
        for x in list(S):
            S2 = S - {x}
            # see if we can add >=2 new elements
            added = 0
            S3 = set(S2)
            for y in range(1, N + 1):
                if y in S3:
                    continue
                if _active_conflicts(S3 | {y}, conflict) == 0:
                    S3.add(y)
                    added += 1
            if len(S3) > len(S):
                S = S3
                improved = True
                break
        if not improved:
            break
    return S

def _soft_guided_search(conflict, deadline, seed):
    rng = random.Random(seed)
    # start from top-half seed set (known valid)
    S = set(range(N // 2 + 1, N + 1))
    best = set(S)
    best_size = len(S)
    # local search rounds
    it = 0
    while time.time() < deadline:
        S = _add_remove_local(S, conflict, max_iter=5000, deadline=deadline)
        S = _remove_one(S, conflict)  # safety: ensure valid
        if len(S) > best_size:
            best = set(S)
            best_size = len(S)
        # perturbation: random remove + re-add
        if len(best) < N:
            S = set(best)
            for _ in range(rng.randint(1, 4)):
                if S:
                    S.discard(rng.choice(sorted(S)))
            S = _add_remove_local(S, conflict, max_iter=3000, deadline=deadline)
            S = _remove_one(S, conflict)
            if len(S) > best_size:
                best = set(S)
                best_size = len(S)
        it += 1
        if it > 200:
            break
    return best

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_FRAC * 90.0
    conflict = _build_conflicts()

    # baseline valid set
    baseline = set(range(N // 2 + 1, N + 1))
    best = baseline

    # run a few guided searches with different seeds
    for seed in (12345, 999, 2024, 777, 42):
        if time.time() > deadline:
            break
        cand = _soft_guided_search(conflict, deadline, seed)
        if len(cand) > len(best):
            best = cand
        if len(best) >= N:  # perfect
            break

    # final safety: ensure validity
    best = _remove_one(best, conflict)

    result = sorted(best)
    # type check: ensure Python ints in a list
    out = [int(x) for x in result]
    return out
# EVOLVE-BLOCK-END
