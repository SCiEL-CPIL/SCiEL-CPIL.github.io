"""Ternary covering code K_3(8,1): start from 486-word direct sum, then peel."""
import time
import itertools
import numpy as np


def _tetracode():
    return np.array([[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)])


def _initial_code(q, n, radius):
    # Direct sum of tetracode (covers radius 1 on 4 coords) with all tails on remaining coords
    tet = _tetracode()  # 9 x 4
    tails = np.array(list(itertools.product(range(q), repeat=n - 4)))
    return np.concatenate([np.repeat(tet, len(tails), axis=0),
                           np.tile(tails, (len(tet), 1))], axis=1)


def _covers(code, q, n, radius):
    """Boolean array of length q^n: True if word is within radius of some codeword."""
    total = q ** n
    # Precompute all words as a 2D array
    base = np.arange(total)
    words = np.array([list(reversed([base // (q ** i) % q for i in range(n)])) for base in base]).T
    # words shape (n, total)

    # For each codeword, mark all words within radius as covered
    covered = np.zeros(total, dtype=bool)
    for c in code:
        # Count positions where word differs from c
        diff = (words != c[:, None]).sum(axis=0)
        covered |= (diff <= radius)
    return covered


def _peel(code, q, n, radius, deadline):
    """Try to remove codewords while maintaining coverage."""
    code = list(code)
    total = q ** n
    covered = _covers(np.array(code), q, n, radius)
    # Count how many codewords cover each word
    # For efficiency, track coverage counts
    cov_count = np.zeros(total, dtype=np.int32)
    for c in code:
        diff = (np.array([list(reversed([i // (q ** j) % q for j in range(n)])) for i in range(total)]).T != c[:, None]).sum(axis=0)
        cov_count[diff <= radius] += 1

    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try to remove each codeword
        for idx in range(len(code) - 1, -1, -1):
            if time.time() > deadline:
                break
            c = code[idx]
            # Check if removing c leaves any word uncovered
            diff = (np.array([list(reversed([i // (q ** j) % q for j in range(n)])) for i in range(total)]).T != c[:, None]).sum(axis=0)
            ball = diff <= radius
            # Words only covered by c
            if np.any(ball & (cov_count == 1)):
                continue
            # Safe to remove
            cov_count[ball] -= 1
            code.pop(idx)
            changed = True
    return code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Start from the known 486-word code
    code = _initial_code(q, n, radius)
    best = [list(c) for c in code]

    # Multiple peeling passes
    for attempt in range(20):
        if time.time() > deadline:
            break
        # Try peeling
        peeled = _peel([list(c) for c in code], q, n, radius, deadline)
        if len(peeled) < len(best):
            best = peeled
            code = np.array(peeled)
        # Also try randomized peeling: shuffle order
        if time.time() > deadline:
            break
        shuffled = code.copy()
        np.random.shuffle(shuffled)
        peeled2 = _peel([list(c) for c in shuffled], q, n, radius, deadline)
        if len(peeled2) < len(best):
            best = peeled2
            code = np.array(peeled2)

    # Verify coverage of best
    if len(best) > 0:
        covered = _covers(np.array(best), q, n, radius)
        if not covered.all():
            # Fallback to initial code
            best = [list(c) for c in _initial_code(q, n, radius)]

    return best
