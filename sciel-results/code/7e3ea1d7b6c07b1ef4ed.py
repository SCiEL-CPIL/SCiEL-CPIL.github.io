"""
Greedy set-cover heuristic for double-covering prefix [1, L].

Structural idea (from inspiration):
  - Maintain an incremental coverage-count array.
  - Greedy set-cover: repeatedly find the residue change (prime i -> new residue)
    that maximizes the number of single-covered positions in [1, current_L] that
    become double-covered, or equivalently that extends L the most.
  - After greedy set-cover, do a critical-point repair phase: target the first
    deficient position and try each prime to fix it.
  - Then local search with neutral-move acceptance to escape local minima.

Refinement machinery (from current program):
  - Deadline guard with safety margin.
  - Hill-climbing fallback if greedy gets stuck.
  - Efficient _compute_L via numpy.

We combine: greedy set-cover (structural) + critical repair + local search + hill-climb fallback.
"""
import time
import math
import random
import numpy as np

_DEADLINE = 145.0  # 5s safety margin before 150s kill
_START_TIME = time.time()


def _primes_upto(n):
    """Return list of primes <= n."""
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(n)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


_PRIMES = _primes_upto(1000)
_N = len(_PRIMES)


def _first_start(r, p):
    if r == 0:
        return p
    return r


def _compute_counts(residues, max_check):
    counts = np.zeros(max_check + 1, dtype=np.int32)
    for r, p in zip(residues, _PRIMES):
        start = _first_start(r, p)
        if start <= max_check:
            counts[start::p] += 1
    return counts


def _find_L(counts, max_check):
    below = np.where(counts[1:max_check+1] < 2)[0]
    if len(below) == 0:
        return max_check
    return int(below[0])


def _update_counts(counts, old_r, new_r, p, max_check):
    old_start = _first_start(old_r, p)
    if old_start <= max_check:
        counts[old_start::p] -= 1
    new_start = _first_start(new_r, p)
    if new_start <= max_check:
        counts[new_start::p] += 1


def _compute_L(residues, max_check=100000):
    counts = _compute_counts(residues, max_check)
    return _find_L(counts, max_check)


def _greedy_setcover(residues, max_check, time_limit):
    """
    Greedy set-cover: repeatedly find the residue change that covers the most
    single-covered positions in [1, current_L], thereby extending L.
    """
    residues = list(residues)
    counts = _compute_counts(residues, max_check)
    best_L = _find_L(counts, max_check)

    no_improve = 0
    max_no_improve = 100

    while time.time() < time_limit and no_improve < max_no_improve:
        if best_L >= max_check:
            break

        # Find positions with count == 1 in [1, best_L]
        if best_L > 0:
            single_positions = np.where((counts[1:best_L+1] == 1))[0] + 1
        else:
            single_positions = np.array([], dtype=np.int64)

        if len(single_positions) == 0:
            # No single-covered positions; try to extend beyond best_L
            # Look at positions best_L+1 onward for count < 2
            target = best_L + 1
            if target > max_check:
                break
            # Try to fix position target
            best_delta = 0
            best_i = -1
            best_new_r = -1
            best_old_r = -1
            for i, p in enumerate(_PRIMES):
                old_r = residues[i]
                new_r = target % p
                if new_r == old_r:
                    continue
                _update_counts(counts, old_r, new_r, p, max_check)
                new_L = _find_L(counts, max_check)
                _update_counts(counts, new_r, old_r, p, max_check)
                delta = new_L - best_L
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_new_r = new_r
                    best_old_r = old_r
            if best_i != -1 and best_delta > 0:
                _update_counts(counts, best_old_r, best_new_r, _PRIMES[best_i], max_check)
                residues[best_i] = best_new_r
                best_L += best_delta
                no_improve = 0
            else:
                no_improve += 1
            continue

        # For each prime, find the residue that covers the most single-covered positions
        best_gain = 0
        best_i = -1
        best_new_r = -1
        best_old_r = -1

        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            residues_mod_p = single_positions % p
            unique_residues, counts_per_residue = np.unique(residues_mod_p, return_counts=True)

            best_idx = np.argmax(counts_per_residue)
            candidate_r = int(unique_residues[best_idx])
            candidate_gain = int(counts_per_residue[best_idx])

            if candidate_r == old_r:
                if len(unique_residues) > 1:
                    sorted_idx = np.argsort(-counts_per_residue)
                    for idx in sorted_idx:
                        cr = int(unique_residues[idx])
                        cg = int(counts_per_residue[idx])
                        if cr != old_r:
                            candidate_r = cr
                            candidate_gain = cg
                            break
                    else:
                        continue
                else:
                    continue

            if candidate_gain > best_gain:
                best_gain = candidate_gain
                best_i = i
                best_new_r = candidate_r
                best_old_r = old_r

        if best_i != -1 and best_gain > 0:
            _update_counts(counts, best_old_r, best_new_r, _PRIMES[best_i], max_check)
            residues[best_i] = best_new_r
            new_L = _find_L(counts, max_check)
            if new_L > best_L:
                best_L = new_L
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

    return residues, best_L


def _critical_repair(residues, max_check, time_limit):
    """
    Target the first deficient position and try each prime to fix it.
    """
    residues = list(residues)
    counts = _compute_counts(residues, max_check)
    best_L = _find_L(counts, max_check)

    no_improve = 0
    max_no_improve = 200

    while time.time() < time_limit and no_improve < max_no_improve:
        if best_L >= max_check:
            break

        target_x = best_L + 1
        if target_x > max_check:
            break

        best_delta = 0
        best_i = -1
        best_new_r = -1
        best_old_r = -1

        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            new_r = target_x % p
            if new_r == old_r:
                continue
            _update_counts(counts, old_r, new_r, p, max_check)
            new_L = _find_L(counts, max_check)
            _update_counts(counts, new_r, old_r, p, max_check)
            delta = new_L - best_L
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_new_r = new_r
                best_old_r = old_r

        if best_i != -1 and best_delta > 0:
            _update_counts(counts, best_old_r, best_new_r, _PRIMES[best_i], max_check)
            residues[best_i] = best_new_r
            best_L += best_delta
            no_improve = 0
        else:
            no_improve += 1

    return residues, best_L


def _local_search(residues, max_check, time_limit):
    """
    Local search with neutral-move acceptance to escape local minima.
    """
    residues = list(residues)
    counts = _compute_counts(residues, max_check)
    best_L = _find_L(counts, max_check)

    no_improve = 0
    max_no_improve = 300

    while time.time() < time_limit and no_improve < max_no_improve:
        if best_L >= max_check:
            break

        best_delta = 0
        best_i = -1
        best_new_r = -1
        best_old_r = -1
        best_neutral_i = -1
        best_neutral_r = -1
        best_neutral_old_r = -1

        num_trials = min(80, _N)
        for _ in range(num_trials):
            i = random.randint(0, _N - 1)
            p = _PRIMES[i]
            old_r = residues[i]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue

            _update_counts(counts, old_r, new_r, p, max_check)
            new_L = _find_L(counts, max_check)
            _update_counts(counts, new_r, old_r, p, max_check)

            delta = new_L - best_L
            if delta > best_delta:
                best_delta = delta
                best_i = i
                best_new_r = new_r
                best_old_r = old_r
                best_neutral_i = -1
            elif delta == 0 and best_neutral_i == -1:
                best_neutral_i = i
                best_neutral_r = new_r
                best_neutral_old_r = old_r

        if best_i != -1:
            _update_counts(counts, best_old_r, best_new_r, _PRIMES[best_i], max_check)
            residues[best_i] = best_new_r
            best_L += best_delta
            no_improve = 0
        elif best_neutral_i != -1 and random.random() < 0.3:
            _update_counts(counts, best_neutral_old_r, best_neutral_r, _PRIMES[best_neutral_i], max_check)
            residues[best_neutral_i] = best_neutral_r
            no_improve += 1
        else:
            no_improve += 1

    return residues, best_L


def _hill_climb(residues, max_check, time_limit):
    """
    Hill-climbing: try changing each residue to see if L improves.
    """
    residues = list(residues)
    counts = _compute_counts(residues, max_check)
    current_L = _find_L(counts, max_check)

    improved = True
    while improved and (time.time() - _START_TIME) < time_limit:
        improved = False
        for i, p in enumerate(_PRIMES):
            old_r = residues[i]
            best_r = old_r
            best_L = current_L

            candidates = set()
            candidates.add(old_r)
            for c in range(min(p, 20)):
                candidates.add(c)
            if p > 20:
                for _ in range(5):
                    candidates.add(random.randint(0, p - 1))

            for r in candidates:
                if r == old_r:
                    continue
                _update_counts(counts, old_r, r, p, max_check)
                L = _find_L(counts, max_check)
                _update_counts(counts, r, old_r, p, max_check)
                if L > best_L:
                    best_L = L
                    best_r = r

            if best_r != old_r:
                _update_counts(counts, old_r, best_r, p, max_check)
                residues[i] = best_r
                current_L = best_L
                improved = True

    return residues, current_L


def construct():
    global _START_TIME
    _START_TIME = time.time()

    max_check = 100000
    time_limit = _START_TIME + 140.0  # 10s safety margin

    # Initial residues: simple heuristic
    residues = [0] * _N
    for i, p in enumerate(_PRIMES):
        residues[i] = ((i + 2) // 2) % p

    # Phase 1: Greedy set-cover
    residues, L1 = _greedy_setcover(residues, max_check, time_limit)
    if time.time() > time_limit - 10:
        return residues

    # Phase 2: Critical repair
    residues, L2 = _critical_repair(residues, max_check, time_limit)
    if time.time() > time_limit - 10:
        return residues

    # Phase 3: Local search
    residues, L3 = _local_search(residues, max_check, time_limit)
    if time.time() > time_limit - 10:
        return residues

    # Phase 4: Hill-climb fallback
    residues, L4 = _hill_climb(residues, max_check, time_limit)

    # Final verification
    for i, p in enumerate(_PRIMES):
        assert 0 <= residues[i] < p, f"Invalid residue {residues[i]} for prime {p}"

    return residues
