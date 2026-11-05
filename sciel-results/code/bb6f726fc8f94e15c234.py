"""Fresh design: randomized search over sparse 12-term integer polynomials to minimize
the nonzero term count of the exact integer square P^2."""
import random
import time
import itertools

# Deadline guard: evaluator kills at 90s; keep a safety margin.
_DEADLINE = time.time() + 75.0  # 75s of 90s budget


def _count_nonzero_square(coeffs):
    """Return (nonzero_count, square_coeffs_list) for coeffs in ascending degree.
    coeffs is a list of (exponent, coefficient) pairs OR a dense list; here we use dense list."""
    n = len(coeffs)
    # Square degree up to 2*(n-1)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        ci = coeffs[i]
        if ci == 0:
            continue
        for j in range(n):
            cj = coeffs[j]
            if cj == 0:
                continue
            sq[i + j] += ci * cj
    cnt = 0
    for v in sq:
        if v != 0:
            cnt += 1
    return cnt, sq


def _valid(coeffs):
    """Check constraints: length<=41, nonzero final, entries in [-64,64], exactly 12 nonzero."""
    if len(coeffs) > 41:
        return False
    if coeffs[-1] == 0:
        return False
    nz = 0
    for c in coeffs:
        if c != 0:
            nz += 1
            if c < -64 or c > 64:
                return False
    if nz != 12:
        return False
    return True


def _random_candidate(rng, max_deg=40):
    """Generate a random 12-term polynomial with exponents in [0,max_deg]."""
    # Choose 12 distinct exponents
    exps = sorted(rng.sample(range(0, max_deg + 1), 12))
    coeffs = [0] * (exps[-1] + 1)
    for e in exps:
        # Small integer coefficients; include negatives for cancellation potential
        coeffs[e] = rng.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32])
    # Ensure last coefficient nonzero (it is, by construction)
    return coeffs


def _flip_signs(coeffs, rng, n_flips=1):
    """Randomly flip signs of some nonzero coefficients to test cancellation changes."""
    nz_positions = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz_positions:
        return coeffs
    new = list(coeffs)
    for _ in range(n_flips):
        i = rng.choice(nz_positions)
        new[i] = -new[i]
    return new


def _perturb_exponent(coeffs, rng, max_deg=40):
    """Move one nonzero term to a nearby exponent."""
    nz_positions = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz_positions:
        return coeffs
    i = rng.choice(nz_positions)
    delta = rng.choice([-2, -1, 1, 2, 3])
    j = i + delta
    if j < 0 or j > max_deg:
        return coeffs
    # If target already occupied, skip (keep 12 nonzero constraint)
    if coeffs[j] != 0:
        return coeffs
    new = list(coeffs)
    new[j] = new[i]
    new[i] = 0
    # Ensure last coefficient still nonzero
    if new[-1] == 0:
        return coeffs
    return new


def _local_search(coeffs, rng, max_iters=2000):
    """Hill-climb on nonzero count of square via sign flips and exponent moves."""
    best = coeffs
    best_cnt, _ = _count_nonzero_square(best)
    for _ in range(max_iters):
        if time.time() > _DEADLINE:
            break
        # Try a random move
        if rng.random() < 0.5:
            cand = _flip_signs(best, rng, n_flips=1)
        else:
            cand = _perturb_exponent(best, rng)
        if not _valid(cand):
            continue
        cnt, _ = _count_nonzero_square(cand)
        if cnt < best_cnt:
            best = cand
            best_cnt = cnt
            if best_cnt <= 12:
                break
    return best, best_cnt


def construct():
    global _DEADLINE
    rng = random.Random(12345)

    # Known-good baseline: [1]*12 (12 terms, square has 25 nonzeros)
    baseline = [1] * 12
    base_cnt, _ = _count_nonzero_square(baseline)
    best = baseline
    best_cnt = base_cnt

    start = time.time()
    # Randomized restarts + local search
    max_restarts = 400
    for r in range(max_restarts):
        if time.time() > _DEADLINE:
            break
        cand = _random_candidate(rng)
        if not _valid(cand):
            continue
        cnt, _ = _count_nonzero_square(cand)
        if cnt < best_cnt:
            best = cand
            best_cnt = cnt
            if best_cnt <= 13:
                break
        # Local search from this candidate
        cand2, cnt2 = _local_search(cand, rng, max_iters=300)
        if cnt2 < best_cnt:
            best = cand2
            best_cnt = cnt2
            if best_cnt <= 13:
                break

    # Also do local search from baseline
    if time.time() < _DEADLINE:
        cand3, cnt3 = _local_search(baseline, rng, max_iters=500)
        if cnt3 < best_cnt:
            best = cand3
            best_cnt = cnt3

    # Final validity check
    if not _valid(best):
        best = baseline

    return best


if __name__ == "__main__":
    res = construct()
    print("length:", len(res), "nonzero:", sum(1 for c in res if c != 0))
    cnt, sq = _count_nonzero_square(res)
    print("square nonzero:", cnt)
    print("score (12/cnt):", 12 / cnt if cnt else 0)
