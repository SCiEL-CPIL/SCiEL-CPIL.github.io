"""Combined solver for Erdos #689 (n=1000) doubly-covered run.

Combines:
  - Current program's NumPy vectorized coverage + prefix scoring
  - Inspiration's simulated annealing with targeted gap repair

Strategy:
  1. Fast NumPy-based coverage computation
  2. Greedy initialization with coverage-deficit heuristic
  3. Simulated annealing that targets the bottleneck (first uncovered integer)
  4. For large primes, use gap-repair: choose residue to cover the bottleneck
"""
import time
import random
import numpy as np

T_START = time.time()
DEADLINE = 115.0
MAX_L = 100000
WORK_L = 2000  # working range for optimization


def _primes_upto(n):
    sieve = bytearray([1]) * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)
N_PRIMES = len(_PRIMES)
_PRIMES_ARR = np.asarray(_PRIMES, dtype=np.int64)


def _build_coverage(residues, limit):
    """Build coverage count array for 1..limit using NumPy."""
    mark = np.zeros(limit + 1, dtype=np.int16)
    for pp, rr in zip(_PRIMES, residues):
        rr = int(rr)
        if rr == 0:
            start = pp
        else:
            start = rr
        if start > limit:
            continue
        mark[start::pp] += 1
    return mark


def _prefix_score(coverage, limit):
    """Find largest L with coverage[1..L] all >= 2."""
    ok = coverage[1:limit + 1] >= 2
    if ok.all():
        return limit
    first = np.argmax(~ok)
    return int(first)


def _verify(residues, limit=MAX_L):
    """Final verification."""
    coverage = _build_coverage(residues, limit)
    return _prefix_score(coverage, limit)


def _initial_greedy(target_L, seed=0):
    """Greedy initialization with coverage-deficit heuristic."""
    rng = random.Random(seed)
    n = N_PRIMES
    residues = [0] * n
    limit = min(target_L + 200, WORK_L)
    coverage = np.zeros(limit + 1, dtype=np.int16)

    # Process small primes first
    order = list(range(n))
    order.sort(key=lambda i: _PRIMES[i])

    for i in order:
        p = _PRIMES[i]
        best_r = 0
        best_gain = -1

        if p <= limit:
            # For small primes, try all residues
            for r in range(p):
                start = r if r > 0 else p
                if start > limit:
                    gain = 0
                else:
                    # Count how many positions with coverage < 2 we'd cover
                    positions = np.arange(start, limit + 1, p)
                    gain = int(np.sum(coverage[positions] < 2))
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        else:
            # Large prime: only one hit in [1, limit]
            # Choose r to maximize coverage deficit at that point
            best_r = 0
            best_gain = -1
            for r in range(1, min(p, limit + 1)):
                if coverage[r] < 2:
                    gain = 2 - coverage[r]
                else:
                    gain = 0
                if gain > best_gain:
                    best_gain = gain
                    best_r = r

        residues[i] = best_r
        start = best_r if best_r > 0 else p
        if start <= limit:
            coverage[start::p] += 1

    return residues, coverage, limit


def _sa_search(residues, coverage, limit, budget, seed=0):
    """Simulated annealing with targeted gap repair."""
    rng = random.Random(seed)
    n = N_PRIMES
    best_res = list(residues)
    best_score = _prefix_score(coverage, limit)
    current_score = best_score
    current_coverage = coverage.copy()

    t_start = time.time()
    T = 5.0
    cooling = 0.9995
    min_T = 0.1

    # Precompute prime indices for fast access
    prime_to_idx = {p: i for i, p in enumerate(_PRIMES)}

    while time.time() - t_start < budget and T > min_T:
        # Find the bottleneck: first uncovered position
        if current_score < limit:
            bottleneck = current_score + 1
        else:
            bottleneck = limit

        # Choose which prime to modify
        # Bias towards primes that cover the bottleneck
        if rng.random() < 0.6 and bottleneck <= limit:
            # Find primes that could cover the bottleneck
            covering_primes = []
            for i in range(n):
                p = _PRIMES[i]
                r = best_res[i]
                # Check if this prime could cover bottleneck with some residue
                if bottleneck <= limit:
                    covering_primes.append(i)
            if covering_primes:
                idx = rng.choice(covering_primes)
            else:
                idx = rng.randint(0, n - 1)
        else:
            idx = rng.randint(0, n - 1)

        p = _PRIMES[idx]
        old_r = best_res[idx]

        # Choose new residue
        if rng.random() < 0.5 and bottleneck <= limit:
            # Gap repair: choose residue that covers the bottleneck
            # We need r such that bottleneck ≡ r (mod p)
            target_r = bottleneck % p
            # Try this and a few neighbors
            candidates = [target_r]
            for dr in [-2, -1, 1, 2, -3, 3]:
                c = (target_r + dr) % p
                if c not in candidates:
                    candidates.append(c)
        else:
            # Random exploration
            if p <= 60:
                candidates = list(range(p))
            else:
                candidates = [rng.randint(0, p - 1) for _ in range(8)]

        best_new_r = old_r
        best_new_score = current_score
        best_new_coverage = None

        for new_r in candidates:
            if new_r == old_r:
                continue

            # Compute delta
            old_start = old_r if old_r > 0 else p
            new_start = new_r if new_r > 0 else p

            # Temporarily apply
            if old_start <= limit:
                current_coverage[old_start::p] -= 1
            if new_start <= limit:
                current_coverage[new_start::p] += 1

            new_score = _prefix_score(current_coverage, limit)

            if new_score > best_new_score:
                best_new_score = new_score
                best_new_r = new_r
                best_new_coverage = current_coverage.copy()

            # Revert
            if new_start <= limit:
                current_coverage[new_start::p] -= 1
            if old_start <= limit:
                current_coverage[old_start::p] += 1

        # Acceptance decision
        delta = best_new_score - current_score
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            # Apply move
            old_start = old_r if old_r > 0 else p
            new_start = best_new_r if best_new_r > 0 else p
            if old_start <= limit:
                current_coverage[old_start::p] -= 1
            if new_start <= limit:
                current_coverage[new_start::p] += 1
            best_res[idx] = best_new_r
            current_score = best_new_score

            if current_score > best_score:
                best_score = current_score
                best_res = list(best_res)

        T *= cooling

    return best_score, best_res


def _targeted_local_search(residues, limit, budget, seed=0):
    """Targeted local search on primes covering the bottleneck."""
    rng = random.Random(seed)
    n = N_PRIMES
    best_res = list(residues)
    coverage = _build_coverage(best_res, limit)
    best_score = _prefix_score(coverage, limit)

    t_start = time.time()
    no_improve = 0

    while time.time() - t_start < budget and no_improve < 200:
        current_score = best_score
        if current_score >= limit:
            break

        bottleneck = current_score + 1

        # Find primes that currently cover the bottleneck
        covering = []
        for i in range(n):
            p = _PRIMES[i]
            r = best_res[i]
            if (bottleneck - r) % p == 0:
                covering.append(i)

        # Also find primes that could cover it (any residue)
        all_indices = list(range(n))
        rng.shuffle(all_indices)

        # Try modifying covering primes first, then others
        search_order = covering + [i for i in all_indices if i not in covering]
        rng.shuffle(search_order)

        improved = False
        for idx in search_order:
            if time.time() - t_start > budget:
                break

            p = _PRIMES[idx]
            old_r = best_res[idx]

            # Generate candidates
            if bottleneck <= limit:
                target_r = bottleneck % p
                candidates = [target_r]
                for dr in range(-5, 6):
                    c = (target_r + dr) % p
                    if c not in candidates:
                        candidates.append(c)
            else:
                candidates = [rng.randint(0, p - 1) for _ in range(10)]

            best_new_r = old_r
            best_new_score = current_score

            old_start = old_r if old_r > 0 else p

            for new_r in candidates:
                if new_r == old_r:
                    continue
                new_start = new_r if new_r > 0 else p

                if old_start <= limit:
                    coverage[old_start::p] -= 1
                if new_start <= limit:
                    coverage[new_start::p] += 1

                s = _prefix_score(coverage, limit)

                if new_start <= limit:
                    coverage[new_start::p] -= 1
                if old_start <= limit:
                    coverage[old_start::p] += 1

                if s > best_new_score:
                    best_new_score = s
                    best_new_r = new_r

            if best_new_r != old_r and best_new_score > current_score:
                new_start = best_new_r if best_new_r > 0 else p
                if old_start <= limit:
                    coverage[old_start::p] -= 1
                if new_start <= limit:
                    coverage[new_start::p] += 1
                best_res[idx] = best_new_r
                current_score = best_new_score
                improved = True

                if current_score > best_score:
                    best_score = current_score

        if not improved:
            no_improve += 1
        else:
            no_improve = 0

    return best_score, best_res


def construct():
    global T_START
    T_START = time.time()

    best_score = 84
    best_res = [((i + 2) // 2) % p for i, p in enumerate(_PRIMES)]

    # Phase 1: Multi-start greedy + SA
    seeds = [42, 123, 456, 789, 1024, 2048, 3000, 5000]

    for s in seeds:
        if time.time() >= DEADLINE - 15:
            break

        # Greedy initialization
        res, coverage, limit = _initial_greedy(1500, seed=s)
        score = _prefix_score(coverage, limit)

        if score > best_score:
            best_score = score
            best_res = list(res)

        # SA refinement
        if time.time() < DEADLINE - 20:
            sa_budget = min(8.0, max(2.0, (DEADLINE - time.time()) / 3.0))
            s2, r2 = _sa_search(res, coverage, limit, budget=sa_budget, seed=s)
            if s2 > best_score:
                best_score = s2
                best_res = list(r2)

        # Targeted local search on the bottleneck
        if time.time() < DEADLINE - 10:
            ls_budget = min(5.0, max(1.0, (DEADLINE - time.time()) / 2.0))
            s3, r3 = _targeted_local_search(best_res, limit, budget=ls_budget, seed=s + 1)
            if s3 > best_score:
                best_score = s3
                best_res = list(r3)

        if best_score >= 1000:
            break

    # Final verification with extended limit
    final_limit = min(best_score + 3000, MAX_L)
    final_score = _verify(best_res, final_limit)
    if final_score > best_score:
        best_score = final_score

    return best_res
