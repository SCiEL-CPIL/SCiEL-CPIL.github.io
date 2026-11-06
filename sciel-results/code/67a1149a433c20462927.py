import time
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def _make_dense(exps, coeffs):
        """Convert sparse (exps, coeffs) to dense list."""
        max_deg = max(exps)
        dense = [0] * (max_deg + 1)
        for e, c in zip(exps, coeffs):
            dense[e] = c
        return dense

    def _from_dense(dense):
        """Convert dense list to sparse (exps, coeffs)."""
        exps = []
        coeffs = []
        for i, c in enumerate(dense):
            if c != 0:
                exps.append(i)
                coeffs.append(c)
        return exps, coeffs

    def exhaustive_sign_search(exps, coeffs_magnitudes=None):
        """
        For a fixed set of exponents, try all 2^12 sign patterns.
        If coeffs_magnitudes is None, use all 1s.
        Returns best (score, exps, coeffs) found.
        """
        n = len(exps)
        if n != 12:
            return None

        if coeffs_magnitudes is None:
            coeffs_magnitudes = [1] * 12

        best_local = float('inf')
        best_coeffs = None

        # Vectorized: generate all 2048 sign patterns at once
        # Each row is a sign pattern for the 12 coefficients
        all_patterns = np.array([[1 if (i >> j) & 1 else -1 for j in range(12)] for i in range(2048)], dtype=np.int64)
        # all_patterns shape: (2048, 12)

        max_deg = max(exps)
        # For each sign pattern, build the polynomial and compute square
        # We'll process in batches to keep memory reasonable
        batch_size = 256
        for batch_start in range(0, 2048, batch_size):
            if time.time() > deadline - 3:
                break
            batch_end = min(batch_start + batch_size, 2048)
            batch_patterns = all_patterns[batch_start:batch_end]

            for idx in range(batch_end - batch_start):
                signs = batch_patterns[idx]
                coeffs = [int(signs[j] * coeffs_magnitudes[j]) for j in range(12)]

                p = np.zeros(max_deg + 1, dtype=np.int64)
                for e, c in zip(exps, coeffs):
                    p[e] = c

                q = np.convolve(p, p)
                count = int(np.count_nonzero(q))

                if count < best_local:
                    best_local = count
                    best_coeffs = coeffs[:]

        if best_coeffs is not None:
            return best_local, exps[:], best_coeffs
        return None

    def search_structured_exponents():
        """
        Search over structured exponent sets (arithmetic progressions, geometric-like, etc.)
        and for each, do exhaustive sign search.
        """
        nonlocal best_score, best_candidate

        # Generate candidate exponent sets
        candidate_exps = []

        # 1. Arithmetic progressions: a, a+d, a+2d, ..., a+11d
        for a in range(0, 10):
            for d in range(1, 4):
                exps = [a + i * d for i in range(12)]
                if max(exps) <= 40 and len(set(exps)) == 12:
                    candidate_exps.append(exps)

        # 2. Two blocks: a..a+k-1 and b..b+(11-k)
        for k in range(1, 12):
            for a in range(0, 10):
                block1 = [a + i for i in range(k)]
                for b in range(a + k, 41):
                    block2 = [b + i for i in range(12 - k)]
                    if max(block2) <= 40:
                        exps = sorted(block1 + block2)
                        if len(set(exps)) == 12:
                            candidate_exps.append(exps)

        # 3. Random sparse sets
        random.seed(42)
        for _ in range(50):
            exps = sorted(random.sample(range(0, 41), 12))
            candidate_exps.append(exps)

        # Deduplicate
        seen = set()
        unique_exps = []
        for exps in candidate_exps:
            key = tuple(exps)
            if key not in seen:
                seen.add(key)
                unique_exps.append(exps)

        # Sort by "sparsity" (larger gaps first) as they may have more cancellation potential
        # Actually, let's just evaluate them all with sign search
        for exps in unique_exps:
            if time.time() > deadline - 5:
                break
            result = exhaustive_sign_search(exps)
            if result is not None:
                score, e, c = result
                if score < best_score:
                    _update_best(e, c, score)
                    if best_score <= 13:
                        return

    def search_with_magnitudes():
        """
        Try some exponent sets with coefficient magnitudes other than 1.
        Focus on small sets where we can afford more computation.
        """
        nonlocal best_score, best_candidate

        # For a few promising exponent sets, try different magnitude patterns
        # This is more expensive, so limit to a few sets
        test_exps = [
            list(range(12)),           # 0..11
            [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15],  # two blocks
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],  # even
            [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19],  # three blocks
        ]

        # Try a few magnitude patterns
        magnitude_patterns = [
            [1]*12,
            [1,1,1,1,1,1,2,2,2,2,2,2],
            [1,1,1,1,2,2,2,2,1,1,1,1],
            [1,2,1,2,1,2,1,2,1,2,1,2],
        ]

        for exps in test_exps:
            if time.time() > deadline - 3:
                break
            for mag in magnitude_patterns:
                result = exhaustive_sign_search(exps, mag)
                if result is not None:
                    score, e, c = result
                    if score < best_score:
                        _update_best(e, c, score)
                        if best_score <= 13:
                            return

    # --- Main search ---
    random.seed(42)

    # Phase 1: Exhaustive sign search over structured exponent sets
    search_structured_exponents()

    # Phase 2: Try some with non-unit magnitudes
    if best_score > 13 and time.time() < deadline - 5:
        search_with_magnitudes()

    # Phase 3: If still not great, do a few random restarts with sign search
    if best_score > 13 and time.time() < deadline - 3:
        for _ in range(20):
            if time.time() > deadline - 2:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            result = exhaustive_sign_search(exps)
            if result is not None:
                score, e, c = result
                if score < best_score:
                    _update_best(e, c, score)
                    if best_score <= 13:
                        break

    return best_candidate

# EVOLVE-BLOCK-END
