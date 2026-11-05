"""Combined program: greedy-safe extension + targeted local search + SA with restarts."""
import time
import random
import math
import numpy as np

_DEADLINE = 145.0  # safety margin before 150s wall-clock limit


def _primes_up_to(n):
    """Sieve of Eratosthenes for primes up to n."""
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    primes = _primes_up_to(1000)
    n = len(primes)
    MAX_L = 100000

    # Precompute useful structures
    prime_indices = list(range(n))

    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r != 0 else p
            if start <= MAX_L:
                cov[start::p] += 1
        return cov

    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return int(bad[0])

    def update_coverage(cov, idx, old_r, new_r):
        """
        Update coverage array in-place when changing residue at idx from old_r to new_r.
        """
        p = primes[idx]

        # Remove old
        old_start = old_r if old_r != 0 else p
        if old_start <= MAX_L:
            cov[old_start::p] -= 1

        # Add new
        new_start = new_r if new_r != 0 else p
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    best_L_global = 0
    best_residues_global = [0] * n

    def init_greedy_strong():
        """Strong greedy initialization: process all positions 1..2000."""
        r = [0] * n
        for x in range(1, 2001):
            count = 0
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    count += 1
            while count < 2:
                assigned = False
                for i in range(n):
                    p = primes[i]
                    if r[i] == 0:
                        r[i] = x % p
                        count += 1
                        assigned = True
                        break
                if not assigned:
                    break
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    def init_seed():
        """Seeded initialization with pattern."""
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]

    def init_perturbed():
        """Perturbed seed initialization."""
        r = init_seed()
        for _ in range(10):
            idx = random.randint(0, n-1)
            r[idx] = random.randint(0, primes[idx]-1)
        return r

    def init_random():
        """Random initialization."""
        return [random.randint(0, p-1) for p in primes]

    def init_targeted():
        """Initialize by targeting coverage of early integers."""
        r = [0] * n
        for x in range(1, 501):
            count = 0
            for i, p in enumerate(primes):
                if r[i] != -1:
                    if (x - r[i]) % p == 0:
                        count += 1
            if count < 2:
                for i, p in enumerate(primes):
                    if r[i] == 0 and x % p != 0:
                        r[i] = x % p
                        count += 1
                        if count >= 2:
                            break
        for i in range(n):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    def greedy_safe_extend(residues, coverage, current_L):
        """
        From inspiration 1: extend L by finding primes that can safely change
        to cover L+1 without breaking coverage < 3 at earlier positions.
        """
        improved = True
        while improved and current_L < MAX_L and time.time() < deadline - 10:
            improved = False
            target = current_L + 1
            candidates = []
            for idx in range(n):
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue
                old_r = residues[idx]
                # Check if removing old_r coverage breaks any y < target
                # y must be in [1, target-1] and y % p == old_r
                # cov[y] must be >= 3
                if old_r == 0:
                    # Indices: p, 2p, ... < target
                    if target <= p:
                        candidates.append(idx)
                    else:
                        # Check min cov in p, 2p, ..., floor((target-1)/p)*p
                        # Slice: cov[p:target:p]
                        if target > p:
                            min_cov = np.min(coverage[p:target:p])
                            if min_cov >= 3:
                                candidates.append(idx)
                        else:
                            candidates.append(idx)
                else:
                    # Indices: old_r, old_r+p, ... < target
                    # First index is old_r if old_r >= 1
                    if old_r >= target:
                        candidates.append(idx)
                    else:
                        # Slice: cov[old_r:target:p]
                        if old_r > 0:
                            min_cov = np.min(coverage[old_r:target:p])
                            if min_cov >= 3:
                                candidates.append(idx)
                        else:
                            # old_r == 0 handled above
                            candidates.append(idx)

            if candidates:
                idx = random.choice(candidates)
                p = primes[idx]
                old_r = residues[idx]
                new_r = target % p
                update_coverage(coverage, idx, old_r, new_r)
                residues[idx] = new_r
                new_L = find_first_gap(coverage)
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                else:
                    # Revert
                    update_coverage(coverage, idx, new_r, old_r)
                    residues[idx] = old_r
            else:
                break
        return current_L

    def local_search(residues, coverage, current_L, max_passes=80):
        """Local search with targeted refinement."""
        improved = True
        pass_count = 0
        while improved and pass_count < max_passes and time.time() < deadline - 10:
            improved = False
            pass_count += 1
            shuffled_indices = list(range(n))
            random.shuffle(shuffled_indices)

            for idx in shuffled_indices:
                if time.time() > deadline - 10:
                    break

                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L

                # Generate candidates
                candidates = set()
                candidates.add(old_r)

                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    target_r = target % p
                    candidates.add(target_r)

                # Target positions where coverage is exactly 1 (fixable by this prime)
                check_range = min(current_L + 200, MAX_L)
                if check_range >= 1:
                    one_cov_positions = np.where(coverage[1:check_range+1] == 1)[0] + 1
                    if len(one_cov_positions) > 0:
                        sample_size = min(50, len(one_cov_positions))
                        sampled = random.sample(list(one_cov_positions), sample_size)
                        for pos in sampled:
                            candidates.add(pos % p)

                # Add random candidates
                num_random = min(40, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p-1))

                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)

                # For small primes, try all
                if p <= 50:
                    candidates = set(range(p))

                for r in candidates:
                    if r == old_r:
                        continue
                    if time.time() > deadline - 10:
                        break

                    update_coverage(coverage, idx, old_r, r)
                    new_L = find_first_gap(coverage)
                    update_coverage(coverage, idx, r, old_r)

                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r

                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]

        return current_L

    def simulated_annealing(residues, coverage, current_L, T_init=100.0, T_min=1.0, cooling=0.995, max_passes=200):
        """Simulated annealing phase."""
        T = T_init
        sa_improved = True
        sa_passes = 0
        sa_deadline = deadline - 10

        while sa_improved and sa_passes < max_passes and time.time() < sa_deadline:
            sa_improved = False
            sa_passes += 1
            shuffled = list(range(n))
            random.shuffle(shuffled)

            for idx in shuffled:
                if time.time() > sa_deadline:
                    break

                p = primes[idx]
                old_r = residues[idx]
                new_r = random.randint(0, p - 1)
                if new_r == old_r:
                    continue

                update_coverage(coverage, idx, old_r, new_r)
                new_L = find_first_gap(coverage)
                delta_L = new_L - current_L

                if delta_L >= 0:
                    residues[idx] = new_r
                    current_L = new_L
                    sa_improved = True
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                else:
                    if random.random() < np.exp(delta_L / T):
                        residues[idx] = new_r
                        current_L = new_L
                    else:
                        update_coverage(coverage, idx, new_r, old_r)

                T = max(T_min, T * cooling)

        return current_L

    initializers = [init_greedy_strong, init_seed, init_perturbed, init_random,
                    init_targeted, init_seed, init_perturbed,
                    init_random, init_targeted]

    for init_func in initializers:
        if time.time() > deadline - 10:
            break

        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Greedy safe extend (from inspiration 1)
        current_L = greedy_safe_extend(residues, coverage, current_L)

        # Local search
        current_L = local_search(residues, coverage, current_L, max_passes=80)

        # Simulated annealing
        if time.time() < deadline - 10 and current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]
            current_L = simulated_annealing(residues, coverage, current_L)

        # Restore best if SA ended at a worse point
        if current_L < best_L_global:
            pass  # We already saved best_residues_global

    # Final polish: try to extend the current best solution further
    if best_L_global > 0 and time.time() < deadline - 5:
        residues = best_residues_global[:]
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        # Greedy safe extend again
        current_L = greedy_safe_extend(residues, coverage, current_L)

        # Additional passes focused on the frontier
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            target = current_L + 1
            if target > MAX_L:
                break
            for idx in prime_indices:
                p = primes[idx]
                target_r = target % p
                if residues[idx] == target_r:
                    continue
                old_r = residues[idx]
                update_coverage(coverage, idx, old_r, target_r)
                new_L = find_first_gap(coverage)
                update_coverage(coverage, idx, target_r, old_r)
                if new_L > current_L:
                    update_coverage(coverage, idx, old_r, target_r)
                    residues[idx] = target_r
                    current_L = new_L
                    if current_L > best_L_global:
                        best_L_global = current_L
                        best_residues_global = residues[:]
                    break

    return [int(r) for r in best_residues_global]
