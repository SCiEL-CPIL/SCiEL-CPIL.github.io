"""Rewritten program: ILP set-cover + local-search polish for the ternary covering code."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog


def _digits_of_int(x, q, n):
    """Convert integer x (base q, n digits) to a list of n digits (most significant first)."""
    return [(x // (q ** (n - 1 - i))) % q for i in range(n)]


def _int_of_digits(d, q, n):
    """Convert a list of n digits to its base-q integer (most significant first)."""
    val = 0
    for i in range(n):
        val = val * q + d[i]
    return val


def _all_cover_lists(q, n, radius):
    """For every word w in {0..q-1}^n, return the list of codeword candidates within distance <= radius.

    Returns an array `covers` of shape (total, total) where covers[w, c] is True iff
    word w is within distance <= radius of candidate c. Both indexed by base-q integer.
    """
    total = q ** n
    # digits of all words
    digits = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        digits[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    # Hamming distance matrix (word w vs candidate c)
    dist = np.zeros((total, total), dtype=np.int8)
    for i in range(n):
        dist += (digits[:, i][:, None] != digits[:, i][None, :]).astype(np.int8)
    return dist <= radius  # shape (total, total)


def construct(q, n, radius):
    deadline = time.time() + 135  # safety margin under the 150s kill

    # Fallback direct-sum construction (valid for the ternary tetracode case)
    def _fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(3), repeat=n - 4)]

    if not (q == 3 and n == 7 and radius == 1):
        # Generic greedy fallback for other parameters
        return _generic_greedy(q, n, radius, deadline)

    # ---- Specialized path for q=3, n=7, radius=1 ----
    total = 3 ** 7  # 2187

    # Precompute the full cover matrix: covers[w, c] True iff w within dist<=1 of c
    # This is 2187x2187 bool, ~4.8MB, fast to build.
    covers = _all_cover_lists(q, n, radius)  # (total, total)

    # Build a curated candidate pool of codewords to consider in the ILP.
    # Include the tetracode direct-sum seed words + a set of high-utility random words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_ints = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            seed_ints.add(_int_of_digits(h + list(tail), q, n))

    rng = random.Random(12345)
    candidate_ints = list(seed_ints)
    # Add random high-utility candidates to give the ILP more freedom
    extra_target = 700
    attempts = 0
    while len(candidate_ints) < extra_target and attempts < 4000:
        attempts += 1
        c = rng.randrange(total)
        if c in seed_ints and len(candidate_ints) >= extra_target - 200:
            continue
        candidate_ints.append(c)

    # De-duplicate
    candidate_ints = list(dict.fromkeys(candidate_ints))
    m = len(candidate_ints)  # number of candidate codewords

    # Utility of each candidate = number of words it covers (row sum over covers[:, c])
    # covers is (word, candidate), so column c's sum = how many words c covers
    col_utility = covers.sum(axis=0)  # shape (total,)
    cand_utility = col_utility[np.array(candidate_ints)]

    # Sort candidates by descending utility, keep top ones to bound ILP size
    order = np.argsort(-cand_utility)
    top_k = min(m, 900)  # bound ILP variables
    candidate_ints = [candidate_ints[i] for i in order[:top_k]]
    m = len(candidate_ints)

    # covers_sub: (total, m) sub-matrix for our candidate pool
    cand_arr = np.array(candidate_ints)
    covers_sub = covers[:, cand_arr]  # (total, m) bool

    # ---- ILP set cover: minimize sum x_c s.t. for every word w, sum_c covers_sub[w,c] x_c >= 1 ----
    # A = covers_sub (each row = one word constraint, each col = one candidate)
    # min c^T x  s.t. A x >= 1,  x in [0,1]
    c_obj = np.ones(m, dtype=float)
    A_ub = -covers_sub.astype(float)  # -A x <= -1  <=>  A x >= 1
    b_ub = -np.ones(total, dtype=float)
    bounds = [(0.0, 1.0)] * m

    code_ints = None
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'time_limit': 60, 'presolve': True})
        if res.status in (0, 2) and res.x is not None:
            # Round: take candidates with x > 0.5
            sel = np.where(res.x > 0.5)[0]
            code_ints = [candidate_ints[i] for i in sel]
            # Verify coverage; if any word uncovered, add a covering candidate
            if code_ints:
                sel_set = set(sel.tolist())
                covered_mask = np.zeros(total, dtype=bool)
                for i in sel_set:
                    covered_mask |= covers_sub[:, i]
                if not covered_mask.all():
                    # Greedily add candidates to cover the rest
                    uncovered = ~covered_mask
                    while uncovered.any() and time.time() < deadline - 5:
                        # candidate covering most uncovered
                        gains = covers_sub[:, cand_arr[uncovered]].sum(axis=1) if False else None
                        # compute gain = number of uncovered words each candidate covers
                        gain = (covers_sub & uncovered[:, None]).sum(axis=0)
                        best = int(np.argmax(gain))
                        if gain[best] == 0:
                            break
                        code_ints.append(candidate_ints[best])
                        covered_mask |= covers_sub[:, best]
                        uncovered = ~covered_mask
    except Exception:
        code_ints = None

    # If ILP failed, fall back to greedy over the candidate pool
    if not code_ints:
        code_ints = _greedy_pool(covers_sub, candidate_ints, deadline)

    # Convert to list-of-digits
    code_digits = [_digits_of_int(c, q, n) for c in code_ints]

    # ---- Local search polish: remove redundancies, then swaps ----
    code_digits = _polish(code_digits, covers, deadline)

    # Final validation
    if not _is_valid(code_digits, q, n, radius, covers):
        # fallback to seed
        return _fallback()
    return code_digits


def _greedy_pool(covers_sub, candidate_ints, deadline):
    """Greedy set cover over the candidate pool using the precomputed covers_sub."""
    total = covers_sub.shape[0]
    m = covers_sub.shape[1]
    covered = np.zeros(total, dtype=bool)
    chosen = []
    remaining = np.ones(m, dtype=bool)
    while True:
        if time.time() > deadline - 5:
            break
        if covered.all():
            break
        # gain = number of currently-uncovered words each remaining candidate covers
        gain = (covers_sub & (~covered)[:, None]).sum(axis=0)
        gain[~remaining] = -1
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        chosen.append(candidate_ints[best])
        covered |= covers_sub[:, best]
        remaining[best] = False
    return chosen


def _is_valid(code_digits, q, n, radius, covers):
    """Check every word is within radius of some codeword."""
    total = q ** n
    if not code_digits:
        return total == 0
    covered = np.zeros(total, dtype=bool)
    for d in code_digits:
        c = _int_of_digits(d, q, n)
        covered |= covers[:, c]
    return covered.all()


def _polish(code_digits, covers, deadline):
    """Remove redundant codewords and try 1-for-1 swaps to reduce size.

    Returns a (possibly smaller) valid code.
    """
    total = covers.shape[0]
    q = 3
    n = 7

    def coverage_mask(code):
        mask = np.zeros(total, dtype=bool)
        for d in code:
            c = _int_of_digits(d, q, n)
            mask |= covers[:, c]
        return mask

    code = list(code_digits)
    best = list(code)
    best_size = len(code)

    # Phase 1: remove redundant codewords (a codeword is redundant if removing it
    # still leaves full coverage)
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        for i in range(len(code)):
            if time.time() > deadline - 10:
                break
            trial = code[:i] + code[i + 1:]
            if coverage_mask(trial).all():
                code = trial
                changed = True
                break

    if len(code) < best_size:
        best = list(code)
        best_size = len(code)

    # Phase 2: 1-for-1 swaps to reduce size is impossible (same size); instead try
    # removing one and adding a better one is net zero. So focus on further removals
    # via re-evaluation after each removal (already looped). Do a final dedup.
    # Remove exact duplicate codewords.
    seen = set()
    dedup = []
    for d in code:
        key = tuple(d)
        if key not in seen:
            seen.add(key)
            dedup.append(d)
    if len(dedup) < best_size:
        code = dedup
        best = list(code)
        best_size = len(code)

    return best
# EVOLVE-BLOCK-END


def _generic_greedy(q, n, radius, deadline):
    """Generic vectorized greedy covering code for arbitrary q, n, radius."""
    total = q ** n
    digits = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        digits[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    dist = np.zeros((total, total), dtype=np.int8)
    for i in range(n):
        dist += (digits[:, i][:, None] != digits[:, i][None, :]).astype(np.int8)
    covers = dist <= radius
    covered = np.zeros(total, dtype=bool)
    code = []
    while True:
        if time.time() > deadline - 5:
            break
        if covered.all():
            break
        gain = (covers & (~covered)[:, None]).sum(axis=0)
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        code.append(list(digits[best]))
        covered |= covers[:, best]
    return code
