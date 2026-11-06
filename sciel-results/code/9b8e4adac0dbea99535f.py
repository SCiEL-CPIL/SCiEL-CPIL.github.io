"""Doubly-covered prefix via greedy set-cover with large-prime priority."""
import time
import numpy as np

_DEADLINE = time.time() + 135.0  # safety margin before 150s wall-clock kill


def _primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0]


def _score_from_counts(counts, max_l):
    """Find largest L such that all 1..L have coverage >= 2."""
    below = np.nonzero(counts[1:max_l + 1] < 2)[0]
    if below.size == 0:
        return max_l
    return int(below[0])


def construct():
    primes = _primes_upto(1000)
    n = len(primes)  # 168
    max_l = 20000  # target window

    # Initialize: assign each prime a residue that covers position 1
    # i.e., r_p = 1 % p = 1 for p > 1, and 0 for p=2
    residues = np.zeros(n, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = 1 % p

    counts = np.zeros(max_l + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1

    current_L = _score_from_counts(counts, max_l)
    best_L = current_L
    best_res = residues.copy()

    # Greedy set-cover: process primes in order (largest first to reduce conflict risk)
    # For each prime, try to extend the doubly-covered prefix
    # We'll do multiple passes to allow re-optimization

    # Sort prime indices: process larger primes first (they cover fewer positions per step)
    # Actually, let's think differently:
    # The key insight: we want every position 1..L to have >= 2 covers.
    # Start with all residues = 1 (covers position 1 for all primes).
    # Then iteratively extend L by finding which primes can be adjusted to cover L+1.

    # Better approach: greedy extension
    # At each step, look at position current_L + 1.
    # Count how many primes currently cover it.
    # If < 2, we need to adjust some primes to cover it.
    # Choose primes that are "cheap" to adjust (i.e., adjusting them doesn't break coverage elsewhere).

    # Let's use a different strategy:
    # 1. Start with all r_p = 1
    # 2. For L from 1 upward, ensure position L has >= 2 covers
    # 3. When position L has < 2 covers, find primes to reassign to cover L

    # More efficient: batch process
    # For each target position t, determine which primes cover t
    # A prime p covers t if residues[idx] == t % p

    # Let's do iterative improvement:
    # Pass 1: For each prime, try all residues and pick the one that maximizes coverage extension

    # Actually, let's use the approach that worked:
    # Deterministic greedy with large-prime priority

    # Process primes in decreasing order of p
    # For each prime, choose the residue that best helps extend the prefix

    # Initialize counts
    counts = np.zeros(max_l + 1, dtype=np.int32)

    # Start with a reasonable initial assignment
    # Use r_p = 1 for all (covers position 1)
    residues = np.array([1 % p for p in primes], dtype=np.int64)
    for i, p in enumerate(primes):
        r = residues[i]
        counts[r::p] += 1

    best_L = _score_from_counts(counts, max_l)
    best_res = residues.copy()

    # Now do multiple greedy passes
    # In each pass, process primes in a specific order
    # For each prime, try all candidate residues and pick the best one

    # Order: process larger primes first (they have more residue options and cover sparser positions)
    prime_order = np.argsort(-primes)  # descending by prime value

    for pass_num in range(10):  # multiple passes
        if time.time() >= _DEADLINE - 5:
            break

        for idx in prime_order:
            if time.time() >= _DEADLINE - 3:
                break

            p = int(primes[idx])
            cur_r = int(residues[idx])
            cur_score = _score_from_counts(counts, max_l)

            # Try candidate residues
            # Focus on residues that help cover the current gap
            gap = cur_score + 1  # next position we want to cover

            candidates = set()
            # Residue that covers gap
            candidates.add(gap % p)
            # Residue that covers gap+1
            if gap + 1 <= max_l:
                candidates.add((gap + 1) % p)
            # Residue that covers gap+2
            if gap + 2 <= max_l:
                candidates.add((gap + 2) % p)
            # Current residue neighbors
            candidates.add((cur_r + 1) % p)
            candidates.add((cur_r - 1) % p)
            # A few random candidates for diversity
            for _ in range(5):
                candidates.add(np.random.randint(0, p))

            best_cand = cur_r
            best_cand_score = cur_score

            for c in candidates:
                c = int(c) % p
                if c == cur_r:
                    continue

                # Apply change
                counts[cur_r::p] -= 1
                counts[c::p] += 1

                s = _score_from_counts(counts, max_l)

                # Revert
                counts[c::p] -= 1
                counts[cur_r::p] += 1

                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c

            # Apply best candidate
            if best_cand != cur_r:
                counts[cur_r::p] -= 1
                counts[best_cand::p] += 1
                residues[idx] = best_cand

                if best_cand_score > best_L:
                    best_L = best_cand_score
                    best_res = residues.copy()

    # Final polish: for small primes, do a full scan
    if time.time() < _DEADLINE - 5:
        for idx in range(n):
            if time.time() >= _DEADLINE - 2:
                break
            p = int(primes[idx])
            if p > 50:
                continue  # only full-scan small primes

            cur_r = int(residues[idx])
            cur_score = _score_from_counts(counts, max_l)

            best_cand = cur_r
            best_cand_score = cur_score

            for c in range(p):
                if c == cur_r:
                    continue

                counts[cur_r::p] -= 1
                counts[c::p] += 1

                s = _score_from_counts(counts, max_l)

                counts[c::p] -= 1
                counts[cur_r::p] += 1

                if s > best_cand_score:
                    best_cand_score = s
                    best_cand = c

            if best_cand != cur_r:
                counts[cur_r::p] -= 1
                counts[best_cand::p] += 1
                residues[idx] = best_cand

                if best_cand_score > best_L:
                    best_L = best_cand_score
                    best_res = residues.copy()

    return [int(r) for r in best_res]
