"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache
from itertools import combinations

def _is_smooth(x):
    y = x
    for p in (2, 3, 5):
        while y % p == 0:
            y //= p
    return y == 1

def _build_smooth_list():
    return [x for x in range(1, 129) if _is_smooth(x)]

def _antichain_ok(terms):
    """Check no term divides another in the list."""
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return False
    return True

def _precompute_answers():
    smooth = _build_smooth_list()
    answers = [None] * 129  # index 1..128

    # For each target, find min-cardinality antichain representation
    # Try 1-term, then 2-term, then 3-term, etc.
    for n in range(1, 129):
        found = None
        # 1 term: n itself must be smooth
        if _is_smooth(n):
            found = [n]
        else:
            # 2 terms
            found = None
            for i in range(len(smooth)):
                a = smooth[i]
                if a >= n:
                    break
                b = n - a
                if b <= a:
                    continue
                if b > 128 or not _is_smooth(b):
                    continue
                if b % a == 0 or a % b == 0:
                    continue
                found = [a, b]
                break
            if found is None:
                # 3 terms
                for combo in combinations([x for x in smooth if x < n], 3):
                    if sum(combo) == n and _antichain_ok(combo):
                        found = list(combo)
                        break
            if found is None:
                # 4 terms
                for combo in combinations([x for x in smooth if x < n], 4):
                    if sum(combo) == n and _antichain_ok(combo):
                        found = list(combo)
                        break
            if found is None:
                # 5 terms
                for combo in combinations([x for x in smooth if x < n], 5):
                    if sum(combo) == n and _antichain_ok(combo):
                        found = list(combo)
                        break
        answers[n] = found

    return answers

# Precompute at module load to ensure construct() is fast
_ANSWERS = _precompute_answers()

def construct():
    return _ANSWERS[1:129]
# EVOLVE-BLOCK-END
