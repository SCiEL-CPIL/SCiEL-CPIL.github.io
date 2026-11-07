"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import gcd

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))

# ---------------------------------------------------------------------------
# Precomputation: req[x][b] = x // gcd(x, b)  so  x | b*c  <=>  req[x][b] | c
# ---------------------------------------------------------------------------
REQ = {}
for _x in range(2, NMAX + 1):
    _row = [0] * (NMAX + 1)
    for _b in range(2, NMAX + 1):
        if _x == _b:
            continue
        _row[_b] = _x // gcd(_x, _b)
    REQ[_x] = _row


def _valid_addition(x, S_list):
    """Check whether x can join S_list without breaking the constraint.
    For every ordered pair (b, c) of DISTINCT existing members we must have:
        x does not divide b*c   <=>  c % req[x][b] != 0
        b does not divide x*c   <=>  c % req[b][x] != 0
        c does not divide x*b   <=>  b % req[c][x] != 0
    """
    rx = REQ[x]
    n = len(S_list)
    for i in range(n):
        b = S_list[i]
        rb = REQ[b]
        dbx = rx[b]          # x | b*c  -> c % dbx == 0
        rbx = rb[x]          # b | x*c  -> c % rbx == 0
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if c % dbx == 0:
                return False
            if c % rbx == 0:
                return False
            if b % REQ[c][x] == 0:
                return False
    return True


def _conflict_degree(x, S_list):
    """Number of ordered distinct pairs (b, c) in S_list for which x creates a
    violation: x|b*c, b|x*c, or c|x*b.  Higher degree = more 'blocking'."""
    rx = REQ[x]
    n = len(S_list)
    deg = 0
    for i in range(n):
        b = S_list[i]
        rb = REQ[b]
        dbx = rx[b]
        rbx = rb[x]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            if c % dbx == 0:
                deg += 1
            if c % rbx == 0:
                deg += 1
            if b % REQ[c][x] == 0:
                deg += 1
    return deg


# ---------------------------------------------------------------------------
# Greedy construction variants
# ---------------------------------------------------------------------------
def _greedy_order(order, deadline):
    chosen = []
    for x in order:
        if time.time() > deadline:
            break
        if _valid_addition(x, chosen):
            chosen.append(x)
    return chosen


def _greedy_random(req_seed, deadline):
    rng = random.Random(req_seed)
    cands = UNIVERSE[:]
    rng.shuffle(cands)
    return _greedy_order(cands, deadline)


def _greedy_weighted(seed, alpha, deadline):
    """Weighted sampling favouring larger (or smaller) numbers, no replacement."""
    rng = random.Random(seed)
    cands = UNIVERSE[:]
    weights = [(x / NMAX) ** alpha for x in cands]
    chosen = []
    while cands:
        if time.time() > deadline:
            break
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        idx = 0
        for k in range(len(cands)):
            cum += weights[k]
            if r <= cum:
                idx = k
                break
        x = cands.pop(idx)
        weights.pop(idx)
        if _valid_addition(x, chosen):
            chosen.append(x)
    return chosen


# ---------------------------------------------------------------------------
# Removal heuristic: iteratively drop the highest-conflict element
# ---------------------------------------------------------------------------
def _greedy_remove(deadline):
    S = UNIVERSE[:]
    S_set = set(S)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        worst = None
        worst_deg = 0
        for x in S:
            others = [y for y in S if y != x]
            if len(others) < 2:
                continue
            d = _conflict_degree(x, others)
            if d > worst_deg:
                worst_deg = d
                worst = x
        if worst is not None and worst_deg > 0:
            S.remove(worst)
            S_set.discard(worst)
            changed = True
    return S


# ---------------------------------------------------------------------------
# Local search (add / swap / remove)
# ---------------------------------------------------------------------------
def _local_search(S, deadline, max_iters=3000, seed=99999):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)

    for _it in range(max_iters):
        if time.time() > deadline:
            break
        op = rng.random()

        if op < 0.45:
            # try to add
            cands = [x for x in UNIVERSE if x not in S_set]
            if cands:
                rng.shuffle(cands)
                added = False
                for x in cands[:25]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        elif op < 0.85 and S:
            # swap: remove one, try to add a better one
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i + 1:]
            S_temp_set = S_set - {removed}
            cands = [x for x in UNIVERSE if x not in S_temp_set]
            if cands:
                rng.shuffle(cands)
                improved = False
                for x in cands[:35]:
                    if _valid_addition(x, S_temp):
                        S = S_temp + [x]
                        S_set = S_temp_set | {x}
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = S_temp_set | {removed}
            else:
                S = S_temp + [removed]
                S_set = S_temp_set | {removed}
        elif S:
            # remove one with low conflict (frees room)
            # pick a random element, but prefer low-degree ones
            sample = rng.sample(range(len(S)), min(6, len(S)))
            pick = min(sample, key=lambda idx: _conflict_degree(S[idx],
                                                                [y for y in S if y != S[idx]]))
            removed = S[pick]
            S.pop(pick)
            S_set.discard(removed)
            # try to refill
            cands = [x for x in UNIVERSE if x not in S_set]
            if cands:
                rng.shuffle(cands)
                for x in cands[:25]:
                    if _valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        break

    return best


# ---------------------------------------------------------------------------
# Two-opt: remove a pair, add a pair
# ---------------------------------------------------------------------------
def _two_opt_search(S, deadline, max_iters=400, seed=77777):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)

    for _it in range(max_iters):
        if time.time() > deadline:
            break
        if len(S) < 2:
            break
        i, j = rng.sample(range(len(S)), 2)
        i, j = min(i, j), max(i, j)
        S_temp = S[:i] + S[i + 1:j] + S[j + 1:]
        S_temp_set = set(S_temp)
        cands = [x for x in UNIVERSE if x not in S_temp_set]
        if not cands:
            continue
        rng.shuffle(cands)
        for x in cands[:25]:
            if not _valid_addition(x, S_temp):
                continue
            S_temp2 = S_temp + [x]
            S_temp2_set = S_temp_set | {x}
            for y in cands[:25]:
                if y == x or y in S_temp2_set:
                    continue
                if _valid_addition(y, S_temp2):
                    S = S_temp2 + [y]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    break
            else:
                continue
            break
    return best


# ---------------------------------------------------------------------------
# Simulated annealing
# ---------------------------------------------------------------------------
def _simulated_annealing(S, deadline, max_iters=4000, seed=55555):
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    T = 30.0
    cooling = (0.5 / 30.0) ** (1.0 / max(1, max_iters))

    for _it in range(max_iters):
        if time.time() > deadline:
            break
        T *= cooling
        op = rng.random()
        if op < 0.5:
            cands = [x for x in UNIVERSE if x not in S_set]
            if cands:
                x = rng.choice(cands)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        elif S:
            i = rng.randrange(len(S))
            removed = S[i]
            S.pop(i)
            S_set.discard(removed)
            cands = [x for x in UNIVERSE if x not in S_set]
            if cands:
                x = rng.choice(cands)
                if _valid_addition(x, S):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                else:
                    if T > 0.05 and rng.random() < 0.05:
                        S.append(x)
                        S_set.add(x)
                    else:
                        S.append(removed)
                        S_set.add(removed)
    return best


def construct():
    start = time.time()
    deadline = start + 80.0

    best = None
    best_size = 0

    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)

    # ---- Phase 1: diverse greedy initialisations ----
    consider(_greedy_order(UNIVERSE[::-1], deadline - 70))          # largest first
    consider(_greedy_order(UNIVERSE, deadline - 70))                # smallest first
    consider(_greedy_remove(deadline - 65))                         # removal heuristic

    for seed in [1, 7, 13, 42, 99, 128, 200, 256, 313, 424]:
        if time.time() > deadline - 55:
            break
        consider(_greedy_random(seed, deadline - 55))

    for seed in [5, 15, 25]:
        for alpha in [0.3, 0.7, 1.0]:
            if time.time() > deadline - 45:
                break
            consider(_greedy_weighted(seed, alpha, deadline - 45))

    if best is None:
        best = list(range(129, 257))
        best_size = len(best)

    # ---- Phase 2: local search on the best seed ----
    consider(_local_search(best, deadline - 30, max_iters=4000, seed=99999))

    # ---- Phase 3: two-opt ----
    consider(_two_opt_search(best, deadline - 20, max_iters=500, seed=77777))

    # ---- Phase 4: simulated annealing ----
    consider(_simulated_annealing(best, deadline - 12, max_iters=5000, seed=55555))

    # ---- Phase 5: perturbed restarts (remove a few, re-solve locally) ----
    rng = random.Random(31337)
    for _restart in range(12):
        if time.time() > deadline - 5:
            break
        S_p = best[:]
        n_remove = rng.randint(1, min(4, len(S_p)))
        for idx in sorted(rng.sample(range(len(S_p)), n_remove), reverse=True):
            S_p.pop(idx)
        consider(_local_search(S_p, deadline - 5, max_iters=800, seed=rng.randrange(10**6)))

    # ---- Phase 6: final polish on the incumbent ----
    consider(_local_search(best, deadline - 2, max_iters=3000, seed=88888))

    return [int(x) for x in best]
