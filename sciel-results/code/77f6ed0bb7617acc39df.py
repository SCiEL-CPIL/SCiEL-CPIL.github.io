"""Combined program: fixed candidate + exact Sidon evaluator + structured/random search.
Minimize the largest subset (with all unordered pair sums incl. diagonals unique)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time


def _pair_sums(x, S):
    # sums of x + S[j] for j >= idx(x)  (diagonal included once)
    idx = {v: i for i, v in enumerate(S)}
    i = idx[x]
    return [x + S[j] for j in range(i, len(S))]


def max_sidon_size(S):
    """Exact largest k such that some k-subset of S has all unordered pair sums
    (including diagonals) unique. Returns that k."""
    S = list(S)
    n = len(S)
    if n <= 1:
        return n
    idx = {v: i for i, v in enumerate(S)}
    # precompute pair-sums per element (only j >= i)
    sums_by_i = [
        [S[i] + S[j] for j in range(i, n)]
        for i in range(n)
    ]
    # all distinct sums that appear
    allsums = set()
    for lst in sums_by_i:
        allsums.update(lst)
    sum_to_id = {s: k for k, s in enumerate(allsums)}
    m = len(allsums)
    MAXS = 256  # any sum > 2*128=255, so 256 is a safe sentinel

    best = 1
    # conflict-count array: how many times each sum is already used by chosen elements
    # (a sum used by two different chosen elements => conflict)
    cnt = [0] * m

    def dfs(start, cur, size):
        nonlocal best
        # prune: even taking all remaining can't beat best
        if size + (n - start) <= best:
            return
        # prune: best already covers all we can possibly reach
        if best >= n:
            return
        if size > best:
            best = size
        for i in range(start, n):
            # try adding S[i]
            ok = True
            touched = []
            for s in sums_by_i[i]:
                sid = sum_to_id[s]
                if cnt[sid] >= 1:
                    ok = False
                    break
                cnt[sid] += 1
                touched.append(sid)
            if ok:
                dfs(i + 1, S[i], size + 1)
            for sid in touched:
                cnt[sid] -= 1
        # note: we also allow skipping i (handled by loop advancing)

    dfs(0, None, 0)
    return best


def _valid(c):
    if len(c) != 12 or len(set(c)) != 12:
        return False
    return all(0 <= x <= 128 for x in c)


def _candidates():
    cands = []
    # --- block families: {a*i + b_i} ---
    for stride in (10, 11, 12, 13, 14, 16, 18, 20, 21, 22):
        for off in (1, 2, 3, 4, 5):
            c = sorted(set([stride * i for i in range(6)] +
                           [stride * i + off for i in range(6)]))
            if _valid(c):
                cands.append(c)
    # --- two-AP unions of length 6 ---
    for s1 in (4, 5, 6, 7, 8, 9, 10):
        for s2 in (4, 5, 6, 7, 8, 9, 10):
            for o1 in (0, 1, 2, 3):
                for o2 in (0, 1, 2, 3):
                    a = [o1 + s1 * i for i in range(6)]
                    b = [o2 + s2 * i for i in range(6)]
                    c = sorted(set(a + b))
                    if _valid(c):
                        cands.append(c)
    # --- grid: rows x cols ---
    for rows in (2, 3, 4, 6):
        for cols in (2, 3, 4, 6):
            if rows * cols != 12:
                continue
            for rs in (8, 9, 10, 11, 12, 14, 16):
                for cs in (2, 3, 4, 5):
                    c = sorted(set([rs * r + cs * col for r in range(rows) for col in range(cols)]))
                    if _valid(c):
                        cands.append(c)
    # --- near-prime / coprime strides to reduce accidental collisions ---
    for p in (7, 11, 13, 17, 19):
        for q in (7, 11, 13, 17, 19):
            if p == q:
                continue
            c = sorted(set([p * i for i in range(6)] + [q * i + 1 for i in range(6)]))
            if _valid(c):
                cands.append(c)
    # --- random ---
    rng = random.Random(12345)
    for _ in range(400):
        c = sorted(rng.sample(range(0, 129), 12))
        cands.append(c)
    # --- local perturbations of a few bases (added later in search) ---
    return cands


def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin under 60/90s
    base_cands = _candidates()

    best_set = None
    best_val = 99
    seen = set()

    def consider(c):
        nonlocal best_set, best_val
        key = tuple(sorted(c))
        if key in seen:
            return
        seen.add(key)
        if not _valid(c):
            return
        v = max_sidon_size(c)
        if v < best_val:
            best_val = v
            best_set = list(c)
        return v

    # evaluate structured candidates
    for c in base_cands:
        if time.time() > deadline:
            break
        consider(c)

    if best_set is None:
        best_set = list(range(12))
        best_val = max_sidon_size(best_set)

    # local search: perturb the current best and its neighbors
    rng = random.Random(777)
    # seed a few good starting points
    seeds = [best_set]
    for c in base_cands:
        if _valid(c):
            seeds.append(c)
        if len(seeds) >= 12:
            break

    for seed in seeds:
        if time.time() > deadline:
            break
        cur = list(seed)
        cur_val = max_sidon_size(cur)
        for _ in range(40):
            if time.time() > deadline:
                break
            # random single-coordinate perturbation
            c2 = list(cur)
            pos = rng.randrange(12)
            delta = rng.choice([-2, -1, 1, 2, 3, -3])
            nv = c2[pos] + delta
            if nv < 0 or nv > 128:
                continue
            if nv in c2:
                continue
            c2[pos] = nv
            c2.sort()
            v = max_sidon_size(c2)
            if v < cur_val:
                cur = c2
                cur_val = v
            elif v == cur_val and rng.random() < 0.5:
                # accept lateral move sometimes to escape plateaus
                cur = c2
        consider(cur)

    if time.time() > deadline:
        return best_set
    return best_set
# EVOLVE-BLOCK-END
