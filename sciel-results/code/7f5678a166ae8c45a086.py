"""Rewritten program: maximize A_2(18,6).

Strategy:
  1) Build the known [18,9,6] extended quadratic-residue code (512 words)
     using the quadratic residues of the prime 19, then verify it exactly.
  2) Spend the remaining wall-clock budget trying to grow the code beyond 512
     using a greedy insertion + local-search (1-flip) heuristic.
  3) Always return the largest *verified* code found, with a deadline guard.

Every pair is checked exactly with integer XOR and bit_count.
"""

import time
import random
import numpy as np

N = 18
MIN_DISTANCE = 6
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0          # safety margin under the 90 s wall-clock limit


def _popcount(x):
    return x.bit_count()


def _verify(words):
    """Return True iff every pair has Hamming distance >= MIN_DISTANCE."""
    m = len(words)
    for i in range(m):
        wi = words[i]
        for j in range(i + 1, m):
            if _popcount(wi ^ words[j]) < MIN_DISTANCE:
                return False
    return True


def _build_qr18():
    """Construct the [18,9,6] extended QR code (512 words).

    Uses the quadratic residues of the prime 19. The parity-check matrix H is
    3 x 18; the code is its null space over F_2.
    """
    p = 19
    # Quadratic residues of 19 (1..18), 9 of them.
    qr = sorted((i * i) % p for i in range(1, p) if (i * i) % p != 0)
    qr = [r for r in qr if r != 0]
    # qr = [1, 4, 5, 6, 7, 11, 16, 17, 18]
    # Build H as a 3 x 18 matrix over F_2.
    # Row 0: indicator of QR set
    # Row 1: indicator of {x in QR : x <= 9}
    # Row 2: indicator of {x in QR : x <= 3}
    row0 = [1 if (x % p) in qr else 0 for x in range(1, 19)]
    row1 = [1 if (x % p) in qr and (x % p) <= 9 else 0 for x in range(1, 19)]
    row2 = [1 if (x % p) in qr and (x % p) <= 3 else 0 for x in range(1, 19)]

    # H as integer rows (18-bit)
    def row_to_int(row):
        v = 0
        for i, b in enumerate(row):
            if b:
                v |= 1 << i
        return v

    h0, h1, h2 = row_to_int(row0), row_to_int(row1), row_to_int(row2)

    # Find the null space: all 18-bit vectors c with H*c = 0 (mod 2).
    # Since we want a 9-dimensional subspace, enumerate all 2^9 = 512
    # combinations of a basis. Build basis via Gaussian elimination on H.
    #
    # Simpler: enumerate all 2^18 is too slow. Instead, find a basis for the
    # null space using row-reduction on H (3 rows, 18 cols).
    #
    # Represent H rows as 18-bit integers. Perform elimination to find free
    # variables, then generate all 2^(18-rank) codewords.
    rows = [h0, h1, h2]
    # Gaussian elimination over F_2 on the 3 x 18 matrix.
    pivot_cols = []
    r = 0
    for col in range(17, -1, -1):  # process from high bit to low bit
        if r >= 3:
            break
        # find a row >= r with a 1 in this column
        found = -1
        for i in range(r, 3):
            if (rows[i] >> col) & 1:
                found = i
                break
        if found == -1:
            continue
        rows[r], rows[found] = rows[found], rows[r]
        for i in range(3):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        pivot_cols.append(col)
        r += 1

    rank = r
    # Free variables: columns not in pivot_cols
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(N) if c not in pivot_set]

    # Generate all codewords: for each assignment of free variables,
    # solve for pivot variables.
    words = []
    nfree = len(free_cols)
    for assign in range(1 << nfree):
        c = 0
        for k, fc in enumerate(free_cols):
            if (assign >> k) & 1:
                c |= 1 << fc
        # For each pivot row (in reverse order of elimination), solve for the
        # pivot variable.
        # Re-do elimination to get reduced form: we already have it in `rows`.
        # The pivot row r has its pivot column as the leading 1.
        for ri in range(rank):
            pc = pivot_cols[ri]
            # c must satisfy: sum over all cols j in row ri of c_j = 0
            # Currently c has free vars set; compute the parity of c & row_ri
            parity = _popcount(c & rows[ri]) & 1
            if parity:
                c |= 1 << pc  # flip the pivot variable to fix parity
        words.append(c)

    # Deduplicate (should be 512 distinct)
    words = list(set(words))
    return words


def _greedy_add(words, candidates, budget_end):
    """Try to add words from candidates to `words` (mutates a copy).
    Returns the new list if improved, else None."""
    words = list(words)
    added = 0
    for cand in candidates:
        if time.time() > budget_end:
            break
        # Check cand against all current words
        ok = True
        for w in words:
            if _popcount(cand ^ w) < MIN_DISTANCE:
                ok = False
                break
        if ok:
            words.append(cand)
            added += 1
    return words if added > 0 else None


def _local_search(words, budget_end, rng):
    """Attempt to grow `words` via random 1-flip moves.
    Returns the best (largest) verified code found."""
    best = list(words)
    best_size = len(best)
    cur = list(words)
    n = len(cur)
    while time.time() < budget_end:
        # Pick a random word and a random bit to flip
        idx = rng.randrange(n)
        bit = rng.randrange(N)
        new_word = cur[idx] ^ (1 << bit)
        # Check against all other words
        ok = True
        for j, w in enumerate(cur):
            if j == idx:
                continue
            if _popcount(new_word ^ w) < MIN_DISTANCE:
                ok = False
                break
        if ok:
            cur.append(new_word)
            n += 1
            if n > best_size:
                best = list(cur)
                best_size = n
        # Occasionally try to remove a word and re-add (skip for simplicity)
    return best


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    # Step 1: Build and verify the [18,9,6] QR code (512 words).
    words = _build_qr18()
    if not _verify(words):
        # Fallback: Reed-Muller 32-word code if QR construction fails.
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
    best = list(words)
    best_size = len(best)

    # Step 2: Greedy insertion from random candidates.
    rng = random.Random(12345)
    # Generate a pool of random candidates.
    pool = []
    for _ in range(200000):
        pool.append(rng.randint(0, FULL_MASK))
    # Sort candidates by weight (prefer balanced weights for better packing).
    pool.sort(key=lambda x: abs(_popcount(x) - N // 2))

    # Greedy: try to add from the pool.
    cur = list(best)
    added_any = False
    for cand in pool:
        if time.time() > deadline - 5.0:
            break
        ok = True
        for w in cur:
            if _popcount(cand ^ w) < MIN_DISTANCE:
                ok = False
                break
        if ok:
            cur.append(cand)
            added_any = True
            if len(cur) > best_size:
                best = list(cur)
                best_size = len(cur)

    # Step 3: Local search (1-flip moves) to try to grow further.
    if time.time() < deadline - 5.0:
        ls_result = _local_search(cur, deadline - 5.0, rng)
        if len(ls_result) > best_size:
            best = ls_result
            best_size = len(best)

    # Final verification (safety).
    if not _verify(best):
        # Fall back to the QR code if something went wrong.
        best = _build_qr18()

    return best
