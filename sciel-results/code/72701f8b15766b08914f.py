import time
import numpy as np
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _compute_L(counts, max_L):
    below = np.where(counts[1:max_L + 1] < 2)[0]
    if len(below) == 0:
        return max_L
    return int(below[0])

def _build_counts(residues, primes, L_max):
    counts = np.zeros(L_max + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = int(residues[i])
        if r == 0:
            if p <= L_max:
                counts[p::p] += 1
        else:
            if r <= L_max:
                counts[r::p] += 1
    return counts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    primes = _primes_up_to(1000)
    n_primes = len(primes)
    L_search = 5000

    random.seed(42)

    # Try multiple initial patterns and pick the best
    patterns = [
        lambda i, p: ((i + 2) // 2) % p,
        lambda i, p: i % p,
        lambda i, p: (i * 7 + 3) % p,
        lambda i, p: (i * 13 + 5) % p,
        lambda i, p: (i * 31 + 11) % p,
    ]

    best_residues = None
    best_L = 0
    best_counts = None

    for pat in patterns:
        residues = np.array([pat(i, p) for i, p in enumerate(primes)], dtype=np.int64)
        counts = _build_counts(residues, primes, L_search)
        L = _compute_L(counts, L_search)
        if L > best_L:
            best_L = L
            best_residues = residues.copy()
            best_counts = counts.copy()

    residues = best_residues.copy()
    counts = best_counts.copy()
    current_L = best_L

    # Local search with incremental updates
    while time.time() < deadline - 5.0:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = int(residues[i])
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue

        # Subtract old contribution
        if old_r == 0:
            if p <= L_search:
                counts[p::p] -= 1
        else:
            if old_r <= L_search:
                counts[old_r::p] -= 1

        # Add new contribution
        if new_r == 0:
            if p <= L_search:
                counts[p::p] += 1
        else:
            if new_r <= L_search:
                counts[new_r::p] += 1

        new_L = _compute_L(counts, L_search)

        if new_L >= current_L:
            current_L = new_L
            residues[i] = new_r
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Revert
            if new_r == 0:
                if p <= L_search:
                    counts[p::p] -= 1
            else:
                if new_r <= L_search:
                    counts[new_r::p] -= 1
            if old_r == 0:
                if p <= L_search:
                    counts[p::p] += 1
            else:
                if old_r <= L_search:
                    counts[old_r::p] += 1

    # Final verification with larger L
    L_verify = 20000
    counts_v = _build_counts(best_residues, primes, L_verify)
    final_L = _compute_L(counts_v, L_verify)

    if final_L >= L_verify and time.time() < deadline - 1.0:
        L_verify2 = 100000
        counts_v2 = _build_counts(best_residues, primes, L_verify2)
        final_L = _compute_L(counts_v2, L_verify2)

    return [int(r) for r in best_residues]
# EVOLVE-BLOCK-END
