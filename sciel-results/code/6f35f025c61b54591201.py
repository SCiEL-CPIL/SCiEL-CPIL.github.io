"""Fresh design: search for a 12-term integer polynomial (deg<=40, coeffs in [-64,64])
whose EXACT integer square has the fewest nonzero terms."""

import random
import time

_DEADLINE = 80.0  # safety margin below 90s kill


def _square_nonzero(coeffs):
    """Return number of nonzero coefficients of the exact square of coeffs (ascending)."""
    n = len(coeffs)
    sq = [0] * (2 * n - 1)
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        for j in range(i, n):
            v = a * coeffs[j]
            sq[i + j] += v
            if i != j:
                sq[i + j] += v  # 2*a*coeffs[j]
    # count nonzero
    c = 0
    for v in sq:
        if v != 0:
            c += 1
    return c


def _is_valid(coeffs):
    # length<=41, nonzero final coefficient, entries in [-64,64], exactly 12 nonzero
    if len(coeffs) > 41:
        return False
    if len(coeffs) == 0 or coeffs[-1] == 0:
        return False
    nz = 0
    for v in coeffs:
        if v != 0:
            nz += 1
            if v < -64 or v > 64:
                return False
        else:
            pass
    if nz != 12:
        return False
    return True


def construct():
    start = time.time()
    rng = random.Random(12345)

    best_coeffs = None
    best_score = None

    # Baseline: a known valid 12-term polynomial with a reasonably sparse square.
    # Use powers of 2 exponents so squares don't collide -> square has 12 nonzero terms.
    base_exps = [0, 1, 2, 4, 8, 16, 20, 24, 28, 32, 36, 40]
    base_coeffs = [0] * 41
    for e in base_exps:
        base_coeffs[e] = 1
    assert _is_valid(base_coeffs)
    best_coeffs = base_coeffs[:]
    best_score = _square_nonzero(best_coeffs)

    def try_update(cand):
        nonlocal best_coeffs, best_score
        if not _is_valid(cand):
            return
        s = _square_nonzero(cand)
        if s < best_score:
            best_score = s
            best_coeffs = cand[:]

    # Local search: start from base, mutate exponents/coeffs, hill-climb on square sparsity.
    cur = best_coeffs[:]
    cur_score = best_score

    # Precompute nonzero positions for quick mutation
    def nz_positions(c):
        return [i for i, v in enumerate(c) if v != 0]

    iters = 0
    while time.time() < _DEADLINE:
        iters += 1
        pos = nz_positions(cur)
        # pick a random nonzero position to mutate
        p = rng.choice(pos)
        new = cur[:]
        r = rng.random()
        if r < 0.5:
            # change coefficient
            new[p] = rng.randint(-64, 64)
            if new[p] == 0:
                # removing a term; maybe add a new term elsewhere
                # find a zero slot
                zeros = [i for i in range(len(new)) if new[i] == 0]
                if zeros:
                    nz2 = rng.choice(zeros)
                    new[nz2] = rng.randint(1, 64)
                    if rng.random() < 0.5:
                        new[nz2] = -new[nz2]
        else:
            # move the term to a new exponent (keep same coeff)
            cval = new[p]
            new[p] = 0
            zeros = [i for i in range(len(new)) if new[i] == 0]
            if zeros:
                nz2 = rng.choice(zeros)
                new[nz2] = cval

        try_update(new)
        if _is_valid(new):
            ns = _square_nonzero(new)
            if ns <= cur_score:
                cur = new[:]
                cur_score = ns

    # Ensure we return a valid list
    if not _is_valid(best_coeffs):
        return base_coeffs
    return best_coeffs
