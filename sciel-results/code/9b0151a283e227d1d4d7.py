import math
import time

_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _score_set(U):
    """Compute the lower-bound score for a finite set U containing 0."""
    if not U or 0 not in U:
        return -1.0
    U = list(U)
    n = len(U)
    M = max(U)
    if M <= 0:
        return -1.0
    # Sumset and difference set
    s = set()
    d = set()
    for a in U:
        for b in U:
            s.add(a + b)
            d.add(a - b)
    denom = 2 * M + 1
    if denom <= 0:
        return -1.0
    val = 1.0 + math.log(len(d) / len(s)) / math.log(denom)
    return val


def _build_greedy(base, target_n, time_budget):
    """Greedy: extend base by adding elements that maximize new-diff / new-sum growth."""
    U = list(base)
    U_set = set(U)
    # Precompute current sum/diff sets
    S = set()
    D = set()
    for a in U:
        for b in U:
            S.add(a + b)
            D.add(a - b)
    cur_score = _score_set(U)
    best_score = cur_score
    best_U = list(U)
    deadline = _START + time_budget

    # Candidate pool: numbers up to a reasonable bound
    maxU = max(U)
    pool_max = max(2 * maxU + 50, 4 * len(U) * (len(U) + 1))
    pool_max = min(pool_max, 200000)

    while len(U) < target_n and _elapsed() < deadline:
        # Try a batch of candidates
        improved = False
        # Candidate generation: spread out numbers
        cands = []
        step = max(1, (pool_max - maxU) // 200)
        for c in range(maxU + 1, pool_max + 1, step):
            cands.append(c)
        # Also try some random-ish candidates
        import random
        for _ in range(50):
            cands.append(random.randint(maxU + 1, pool_max))
        cands = list(set(cands))
        cands.sort()

        best_c = None
        best_c_score = cur_score
        for c in cands:
            if c in U_set:
                continue
            # Compute new sums and diffs
            new_sums = 0
            new_diffs = 0
            for a in U:
                if (a + c) not in S:
                    new_sums += 1
                if (a - c) not in D:
                    new_diffs += 1
                if (c - a) not in D:
                    new_diffs += 1
            # Also c+c and c-c
            if (c + c) not in S:
                new_sums += 1
            # c-c = 0 already in D
            new_n = len(U) + 1
            new_M = max(maxU, c)
            new_denom = 2 * new_M + 1
            if new_denom <= 0:
                continue
            new_s_size = len(S) + new_sums
            new_d_size = len(D) + new_diffs
            if new_s_size <= 0 or new_d_size <= 0:
                continue
            val = 1.0 + math.log(new_d_size / new_s_size) / math.log(new_denom)
            if val > best_c_score:
                best_c_score = val
                best_c = c
        if best_c is None:
            break
        # Commit best_c
        U.append(best_c)
        U_set.add(best_c)
        maxU = max(maxU, best_c)
        for a in U:
            S.add(a + best_c)
            D.add(a - best_c)
            D.add(best_c - a)
        S.add(2 * best_c)
        cur_score = best_c_score
        if cur_score > best_score:
            best_score = cur_score
            best_U = list(U)
        improved = True
        if not improved:
            break

    return best_U, best_score


def _build_structured(n_layers, layer_sizes):
    """Build a structured set: layer i has elements at positions ~ (2^i) * j for j in range."""
    U = [0]
    U_set = {0}
    maxU = 0
    for i, sz in enumerate(layer_sizes):
        base = 1 << i  # 2^i
        for j in range(1, sz + 1):
            v = base * j
            if v not in U_set:
                U.append(v)
                U_set.add(v)
                maxU = max(maxU, v)
    return U, maxU


def construct():
    # Deadline guard
    if _elapsed() > _DEADLINE - 5:
        return [0, 1, 3]

    # Seed set
    seed = [0, 1, 3]
    seed_score = _score_set(seed)

    # Try several structured configurations
    candidates = []
    candidate_scores = []

    # Configuration 1: layers with sizes [1, 2, 4, 8, 16, 32]
    for cfg in [
        [1, 2, 4, 8, 16, 32],
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 3, 5, 8, 13, 21],
        [1, 2, 4, 7, 12, 20, 33],
        [1, 3, 5, 9, 16, 28],
        [2, 4, 8, 16, 32, 64, 128],
    ]:
        if _elapsed() > _DEADLINE - 10:
            break
        U, maxU = _build_structured(1, cfg)
        if 0 in U:
            sc = _score_set(U)
            candidates.append(U)
            candidate_scores.append(sc)

    # Greedy extension from the best structured set
    best_idx = max(range(len(candidate_scores)), key=lambda i: candidate_scores[i]) if candidate_scores else None
    if best_idx is not None:
        base = candidates[best_idx]
        # Try greedy extension with a time budget
        time_left = _DEADLINE - _elapsed() - 15
        if time_left > 5:
            gU, gSc = _build_greedy(base, len(base) + 50, time_left)
            candidates.append(gU)
            candidate_scores.append(gSc)

    # Also try greedy from seed
    if _elapsed() < _DEADLINE - 20:
        time_left = _DEADLINE - _elapsed() - 15
        if time_left > 5:
            gU2, gSc2 = _build_greedy(seed, 100, time_left)
            candidates.append(gU2)
            candidate_scores.append(gSc2)

    # Pick the best
    if not candidates:
        return seed
    best_idx = max(range(len(candidates)), key=lambda i: candidate_scores[i])
    best_U = candidates[best_idx]
    # Ensure 0 is in the set and it's sorted
    best_U = sorted(set(best_U))
    if 0 not in best_U:
        best_U.insert(0, 0)
    return best_U


# Fallback: if nothing works, return the seed
def _safe_construct():
    try:
        result = construct()
        if not isinstance(result, list) or not result or 0 not in result:
            return [0, 1, 3]
        # Verify all nonnegative integers
        for x in result:
            if not isinstance(x, int) or x < 0:
                return [0, 1, 3]
        return result
    except Exception:
        return [0, 1, 3]


# Override construct to be safe
_orig_construct = construct
def construct():
    return _safe_construct()
