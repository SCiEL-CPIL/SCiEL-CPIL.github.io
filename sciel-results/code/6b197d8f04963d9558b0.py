"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

_DEADLINE = time.time() + 82.0
_RNG = random.Random(12345)

def _square(coeffs):
    """Return coefficient list of coeffs*coeffs (ascending degree)."""
    n = len(coeffs)
    out = [0] * (2 * n - 1)
    for i in range(n):
        a = coeffs[i]
        if a == 0:
            continue
        for j in range(n):
            b = coeffs[j]
            if b == 0:
                continue
            out[i + j] += a * b
    return out

def _nnz_square(coeffs):
    return sum(1 for v in _square(coeffs) if v != 0)

def _validate(coeffs):
    if len(coeffs) > 41:
        return False
    if not coeffs or coeffs[-1] == 0:
        return False
    if any(c < -64 or c > 64 for c in coeffs):
        return False
    if sum(1 for c in coeffs if c != 0) != 12:
        return False
    return True

def _random_candidate():
    # choose 12 distinct positions in [0,40]
    pos = _RNG.sample(range(41), 12)
    pos.sort()
    coeffs = [0] * 41
    for p in pos:
        # bias toward small coefficients for better cancellation
        coeffs[p] = _RNG.choice([1, 1, 1, -1, -1, 1, -1, 2, -2, 3, -3])
    # ensure final coefficient nonzero (pos includes up to 40? not necessarily)
    # trim trailing zeros but keep length<=41 and nonzero final
    while coeffs and coeffs[-1] == 0:
        coeffs.pop()
    if not coeffs:
        return None
    return coeffs

def _try_flip(coeffs):
    """Try flipping one coefficient to a small value; return improved or None."""
    best = None
    best_score = _nnz_square(coeffs)
    n = len(coeffs)
    # candidate values to try
    vals = [1, -1, 2, -2, 3, -3, 4, -4, 5, -5]
    # try each position, each value
    for i in range(n):
        orig = coeffs[i]
        for v in vals:
            if v == orig:
                continue
            coeffs[i] = v
            s = _nnz_square(coeffs)
            if s < best_score:
                best_score = s
                best = (i, v, s)
        coeffs[i] = orig
    if best is not None:
        i, v, s = best
        coeffs[i] = v
    return best_score

def _local_improve(coeffs, max_rounds=5):
    # Very limited local search to avoid timeout
    for _ in range(max_rounds):
        if time.time() > _DEADLINE:
            break
        before = _nnz_square(coeffs)
        improved = False
        n = len(coeffs)
        # Only try small values to keep it fast
        vals = [1, -1, 2, -2]
        for i in range(n):
            if time.time() > _DEADLINE:
                break
            orig = coeffs[i]
            for v in vals:
                if v == orig:
                    continue
                coeffs[i] = v
                s = _nnz_square(coeffs)
                if s < before:
                    before = s
                    improved = True
                    break
            coeffs[i] = orig
        if not improved:
            break
    return coeffs

def construct():
    start = time.time()
    best_coeffs = None
    best_score = 10**9

    # Strategy: Generate candidates based on Arithmetic Progressions
    # and specific coefficient patterns to encourage cancellation.
    
    # Helper to create a candidate from a start, step, and coeff pattern
    def make_candidate(start, step, coeffs_pattern):
        if start + 11 * step > 40:
            return None
        coeffs = [0] * 41
        for i in range(12):
            pos = start + i * step
            coeffs[pos] = coeffs_pattern[i]
        # Trim trailing zeros
        while coeffs and coeffs[-1] == 0:
            coeffs.pop()
        if not coeffs or coeffs[-1] == 0:
            return None
        if len(coeffs) > 41:
            return None
        if sum(1 for c in coeffs if c != 0) != 12:
            return None
        return coeffs

    # Patterns to try
    patterns = [
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        [1, -2, 1, -2, 1, -2, 1, -2, 1, -2, 1, -2],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    ]

    # Try different steps and starts
    # Step 1 is dense, Step 2 is sparse, etc.
    for step in range(1, 4):
        max_start = 40 - 11 * step
        for start in range(0, max_start + 1):
            if time.time() > _DEADLINE - 2.0:
                break
            for pat in patterns:
                if time.time() > _DEADLINE - 2.0:
                    break
                coeffs = make_candidate(start, step, pat)
                if coeffs is None:
                    continue
                # Quick local improvement
                coeffs = _local_improve(coeffs, max_rounds=3)
                s = _nnz_square(coeffs)
                if s < best_score:
                    best_score = s
                    best_coeffs = coeffs[:]
                    if best_score <= 13:
                        # Found a very good one, maybe stop early or do a few more
                        pass

    # If we haven't found anything good, do a few random ones as fallback
    if best_score > 20:
        for _ in range(50):
            if time.time() > _DEADLINE - 2.0:
                break
            coeffs = _random_candidate()
            if coeffs is None:
                continue
            coeffs = _local_improve(coeffs, max_rounds=2)
            s = _nnz_square(coeffs)
            if s < best_score:
                best_score = s
                best_coeffs = coeffs[:]

    if best_coeffs is None:
        return [1] * 12

    if not _validate(best_coeffs):
        return [1] * 12
    
    return best_coeffs
# EVOLVE-BLOCK-END
