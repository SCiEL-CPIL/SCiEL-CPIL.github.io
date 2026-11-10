import random
import time

# EVOLVE-BLOCK-START
def _square(c):
    """Exact square of coefficient list (ascending degree). Returns list."""
    n = len(c)
    out = [0] * (2 * n - 1)
    for i in range(n):
        ci = c[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = c[j]
            if cj == 0:
                continue
            out[i + j] += ci * cj
    return out

def _nonzero_count(c):
    return sum(1 for x in c if x != 0)

def _trim(c):
    # remove trailing zeros
    while c and c[-1] == 0:
        c = c[:-1]
    return c

def _count_nonzero_terms(c):
    return _nonzero_count(_trim(c))

def _build_P(base, shift):
    # P = base + x^shift * base
    n = len(base)
    P = [0] * (n + shift)
    for i, v in enumerate(base):
        P[i] += v
        P[i + shift] += v
    return _trim(P)

def _validate(P):
    if not P:
        return False
    if len(P) > 41:
        return False
    if P[-1] == 0:
        return False
    for v in P:
        if v < -64 or v > 64:
            return False
    if _nonzero_count(P) != 12:
        return False
    return True

def _score_of(P):
    if not _validate(P):
        return None
    sq = _square(P)
    return _count_nonzero_terms(sq)

def _random_base(rng, max_deg=20, max_c=8, nnz=6):
    # choose nnz distinct degrees, assign small nonzero coefficients
    degs = sorted(rng.sample(range(0, max_deg + 1), nnz))
    c = [0] * (max_deg + 1)
    for d in degs:
        c[d] = rng.choice([-1, 1]) * rng.randint(1, max_c)
    return _trim(c)

def _try_shifts(base, shifts):
    best = None
    for s in shifts:
        P = _build_P(base, s)
        if not _validate(P):
            continue
        sc = _score_of(P)
        if sc is None:
            continue
        if best is None or sc < best[0]:
            best = (sc, P)
    return best

def _local_refine(P, rng, max_steps=200):
    # Try flipping signs of individual terms, or small coefficient nudges,
    # to reduce nonzero terms in square. Keep best.
    best = _score_of(P)
    bestP = list(P)
    if best is None:
        return P
    steps = 0
    improved = True
    while improved and steps < max_steps:
        improved = False
        steps += 1
        n = len(bestP)
        # candidate mutations: flip sign of one nonzero term
        for i in range(n):
            if bestP[i] == 0:
                continue
            trial = list(bestP)
            trial[i] = -trial[i]
            if not _validate(trial):
                continue
            sc = _score_of(trial)
            if sc is not None and sc < best:
                best = sc
                bestP = trial
                improved = True
                break
        if not improved:
            # try nudging a coefficient by +-1 within bounds
            for i in range(n):
                for d in (-1, 1):
                    nv = bestP[i] + d
                    if nv == 0 or nv < -64 or nv > 64:
                        continue
                    trial = list(bestP)
                    trial[i] = nv
                    if not _validate(trial):
                        continue
                    sc = _score_of(trial)
                    if sc is not None and sc < best:
                        best = sc
                        bestP = trial
                        improved = True
                        break
                if improved:
                    break
    return bestP

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # safety margin under 90s kill
    rng = random.Random(12345)

    # known good baseline: P = [1]*12 -> score 12
    baseline = [1] * 12
    best_score = _score_of(baseline)
    best_P = list(baseline)
    if best_score is None:
        best_score = 12

    # candidate shifts: must keep P length <=41 and 12 nonzero terms.
    # For base of length L (<=21), P length = L + shift. Need <=41.
    # Also the two halves must not overlap in a way that reduces nonzero count below 12
    # (we validate anyway). Try a spread of shifts.
    shifts = list(range(3, 22))

    # Phase 1: random search over 6-term bases
    while time.time() < deadline:
        base = _random_base(rng, max_deg=20, max_c=8, nnz=6)
        res = _try_shifts(base, shifts)
        if res is not None:
            sc, P = res
            if sc < best_score:
                best_score = sc
                best_P = list(P)
                # local refine to push further
                refined = _local_refine(P, rng, max_steps=60)
                rsc = _score_of(refined)
                if rsc is not None and rsc < best_score:
                    best_score = rsc
                    best_P = refined
            if best_score <= 6:
                break  # cannot do better than 6 with this P=A+x^k A structure

    # Phase 2: if we have a good P, do a more thorough local search on it
    if time.time() < deadline - 5.0 and best_score > 6:
        refined = _local_refine(best_P, rng, max_steps=400)
        rsc = _score_of(refined)
        if rsc is not None and rsc < best_score:
            best_score = rsc
            best_P = refined

    # Final safety: ensure we return a valid candidate
    if not _validate(best_P):
        best_P = [1] * 12
    return best_P
# EVOLVE-BLOCK-END
