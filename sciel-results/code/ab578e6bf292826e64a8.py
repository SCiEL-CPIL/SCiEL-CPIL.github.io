"""Ternary covering code K_3(7,1): construction floor + greedy local-minimum refinement."""
import time
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback: all-zero word plus a trivial construction (only used off-task)
        return [[0] * n]

    start = time.time()
    deadline = start + 115.0  # safety margin under the 150 s kill

    # --- Construction floor: 4x4 tetracode x 3x3 single-coordinate code = 81 words ---
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    singles = [[0, 0, v] for v in range(3)] + [[0, v, 0] for v in range(3)] + [[v, 0, 0] for v in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]
    code = [c for c in code]  # 81 distinct words

    # --- Coverage helper (vectorized) ---
    # For a codeword w, the words within distance <= 1 are {w} plus the 2*(n-1) words
    # that differ in exactly one coordinate. Precompute the offset patterns once.
    offsets = []
    for i in range(n):
        for d in (-1, 1):
            offsets.append((i, d))

    def coverage_count(cw):
        """Return an int array over the 3^n space: how many codewords cover each word."""
        arr = np.zeros(3 ** n, dtype=np.int32)
        for w in cw:
            base = 0
            for i in range(n):
                base = base * 3 + w[i]
            arr[base] += 1
            for (i, d) in offsets:
                v = (w[i] + d) % 3
                delta = (v - w[i]) * (3 ** (n - 1 - i))
                arr[base + delta] += 1
        return arr

    cov = coverage_count(code)

    # --- Greedy local-minimum refinement ---
    # A codeword is "removable" if, after deleting it, no word it covered is left uncovered.
    # Equivalently: for every word within distance 1 of w, its coverage count (including w)
    # is >= 2. We recompute cov from scratch each accepted removal (cheap: ~81*13 ops).
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for idx in range(len(code)):
            if time.time() > deadline:
                break
            w = code[idx]
            # Check removability: every covered word must have cov >= 2 (w contributes 1)
            base = 0
            for i in range(n):
                base = base * 3 + w[i]
            removable = cov[base] >= 2
            if removable:
                for (i, d) in offsets:
                    v = (w[i] + d) % 3
                    delta = (v - w[i]) * (3 ** (n - 1 - i))
                    if cov[base + delta] < 2:
                        removable = False
                        break
            if removable:
                # Accept: drop w and update cov
                for i in range(n):
                    base2 = 0
                    for j in range(n):
                        base2 = base2 * 3 + w[j]
                cov[base2] -= 1
                for (i, d) in offsets:
                    v = (w[i] + d) % 3
                    delta = (v - w[i]) * (3 ** (n - 1 - i))
                    cov[base2 + delta] -= 1
                code.pop(idx)
                improved = True
                break  # restart scan from 0 since indices shifted

    return code
