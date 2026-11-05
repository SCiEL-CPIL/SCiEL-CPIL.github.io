"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = [False] * len(sieve[i*i::i])
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n_primes = len(primes)  # 168

    # Max L to check - we want to reach at least 1000, but go higher
    MAX_L = 100000

    def build_coverage(residues):
        """Build the coverage array from scratch."""
        cov = np.zeros(MAX_L + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start <= MAX_L:
                cov[start::p] += 1
        return cov

    def find_first_gap(cov):
        """Find the first x >= 1 where cov[x] < 2."""
        bad = np.where(cov[1:] < 2)[0]
        if len(bad) == 0:
            return MAX_L
        return bad[0]

    def update_coverage(cov, idx, old_r, new_r):
        """
        Update coverage array in-place when changing residue at idx from old_r to new_r.
        """
        p = primes[idx]

        # Remove old
        if old_r == 0:
            old_start = p
        else:
            old_start = old_r
        if old_start <= MAX_L:
            cov[old_start::p] -= 1

        # Add new
        if new_r == 0:
            new_start = p
        else:
            new_start = new_r
        if new_start <= MAX_L:
            cov[new_start::p] += 1

    def time_left():
        return deadline - time.time()

    best_L_global = 0
    best_residues_global = [0] * n_primes

    # Define initializations
    def init_seed():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]

    def init_random():
        return [random.randint(0, p-1) for p in primes]

    def init_perturbed():
        r = init_seed()
        for _ in range(15):
            idx = random.randint(0, n_primes-1)
            r[idx] = random.randint(0, primes[idx]-1)
        return r

    def init_greedy():
        """Greedy initialization targeting early coverage."""
        r = [0] * n_primes
        # For each integer 1..2000, ensure at least 2 primes cover it
        # Greedily assign residues
        for x in range(1, min(2001, MAX_L+1)):
            # Count how many primes already cover x
            count = 0
            for i, p in enumerate(primes):
                if r[i] != 0 or x % p != 0:
                    if (x - r[i]) % p == 0:
                        count += 1
                else:
                    # r[i] == 0 means covers multiples of p
                    if x % p == 0:
                        count += 1

            if count < 2:
                # Find primes that can cover x
                for i, p in enumerate(primes):
                    # Check if this prime already covers x
                    if (x - r[i]) % p == 0:
                        continue
                    # Assign residue to cover x
                    r[i] = x % p
                    count += 1
                    if count >= 2:
                        break

        # Fill any remaining with random
        for i in range(n_primes):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    def init_targeted():
        """Initialize by targeting coverage of early integers more carefully."""
        r = [0] * n_primes
        # For each integer 1..500, ensure at least 2 primes cover it
        for x in range(1, 501):
            count = 0
            for i, p in enumerate(primes):
                if (x - r[i]) % p == 0:
                    count += 1
            if count < 2:
                # Find primes that don't yet cover x
                candidates = []
                for i, p in enumerate(primes):
                    if (x - r[i]) % p != 0:
                        candidates.append(i)
                random.shuffle(candidates)
                for i in candidates:
                    r[i] = x % primes[i]
                    count += 1
                    if count >= 2:
                        break
        # Fill remaining
        for i in range(n_primes):
            if r[i] == 0:
                r[i] = random.randint(0, primes[i]-1)
        return r

    initializers = [init_greedy, init_seed, init_targeted, init_perturbed, init_random, init_greedy, init_perturbed]

    def local_search(residues, coverage, current_L, max_passes=30):
        """
        Exhaustive hill climbing with targeted refinement.
        For each prime, try all residues (or a large subset) and keep the best.
        """
        improved = True
        pass_count = 0

        while improved and pass_count < max_passes and time_left() > 5:
            improved = False
            pass_count += 1

            # Shuffle prime indices to avoid bias
            prime_indices = list(range(n_primes))
            random.shuffle(prime_indices)

            for idx in prime_indices:
                if time_left() < 5:
                    return residues, coverage, current_L

                p = primes[idx]
                old_r = residues[idx]
                best_r_for_idx = old_r
                best_L_for_idx = current_L

                # Generate candidates
                candidates = set()

                # Target the next gap
                target = current_L + 1
                if target <= MAX_L:
                    candidates.add(target % p)

                # Target positions where coverage is exactly 1
                check_range = min(current_L + 300, MAX_L)
                if check_range >= 1:
                    one_cov_positions = np.where(coverage[1:check_range+1] == 1)[0] + 1
                    if len(one_cov_positions) > 0:
                        sample_size = min(80, len(one_cov_positions))
                        if sample_size < len(one_cov_positions):
                            sampled = random.sample(list(one_cov_positions), sample_size)
                        else:
                            sampled = list(one_cov_positions)
                        for pos in sampled:
                            candidates.add(pos % p)

                # Add random candidates
                num_random = min(60, p)
                for _ in range(num_random):
                    candidates.add(random.randint(0, p-1))

                # Add neighbors
                candidates.add((old_r + 1) % p)
                candidates.add((old_r - 1) % p)

                # For small primes, try all
                if p <= 60:
                    candidates = set(range(p))

                for r in candidates:
                    if r == old_r:
                        continue
                    if time_left() < 5:
                        break

                    # Apply change
                    update_coverage(coverage, idx, old_r, r)

                    # Evaluate
                    new_L = find_first_gap(coverage)

                    # Always revert to keep coverage clean for next candidate
                    update_coverage(coverage, idx, r, old_r)

                    if new_L > best_L_for_idx:
                        best_L_for_idx = new_L
                        best_r_for_idx = r

                # Commit best change for this index if found
                if best_r_for_idx != old_r:
                    update_coverage(coverage, idx, old_r, best_r_for_idx)
                    residues[idx] = best_r_for_idx
                    current_L = best_L_for_idx
                    improved = True
                    if current_L > best_L_global:
                        nonlocal best_L_global
                        best_L_global = current_L
                        nonlocal best_residues_global
                        best_residues_global = residues[:]

        return residues, coverage, current_L

    def simulated_annealing(residues, coverage, current_L, steps=5000):
        """
        Simulated annealing for fine-tuning.
        """
        T = 10.0
        T_min = 0.1
        cooling = 0.9995

        for step in range(steps):
            if time_left() < 5:
                break

            # Pick a random prime to change
            idx = random.randint(0, n_primes - 1)
            p = primes[idx]
            old_r = residues[idx]
            new_r = random.randint(0, p - 1)

            if new_r == old_r:
                continue

            # Apply change
            update_coverage(coverage, idx, old_r, new_r)
            new_L = find_first_gap(coverage)

            delta = new_L - current_L

            if delta >= 0 or random.random() < np.exp(delta / max(T, 0.01)):
                # Accept
                current_L = new_L
                residues[idx] = new_r
                if current_L > best_L_global:
                    nonlocal best_L_global
                    best_L_global = current_L
                    nonlocal best_residues_global
                    best_residues_global = residues[:]
            else:
                # Reject
                update_coverage(coverage, idx, new_r, old_r)

            T = max(T * cooling, T_min)

        return residues, coverage, current_L

    # Run multiple initializations with local search and SA
    random.seed(42)

    for init_func in initializers:
        if time_left() < 10:
            break

        residues = init_func()
        coverage = build_coverage(residues)
        current_L = find_first_gap(coverage)

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 1: Exhaustive hill climbing
        residues, coverage, current_L = local_search(residues, coverage, current_L, max_passes=20)

        if current_L > best_L_global:
            best_L_global = current_L
            best_residues_global = residues[:]

        # Phase 2: Simulated annealing for fine-tuning
        if time_left() > 15:
            residues, coverage, current_L = simulated_annealing(residues, coverage, current_L, steps=3000)

            if current_L > best_L_global:
                best_L_global = current_L
                best_residues_global = residues[:]

            # Phase 3: Another round of hill climbing after SA
            if time_left() > 10:
                residues, coverage, current_L = local_search(residues, coverage, current_L, max_passes=10)

                if current_L > best_L_global:
                    best_L_global = current_L
                    best_residues_global = residues[:]

    # Final check: make sure we have a valid result
    if best_L_global == 0:
        # Fallback to seed
        residues = init_seed()
        best_residues_global = residues[:]

    return [int(r) for r in best_residues_global]
# EVOLVE-BLOCK-END
