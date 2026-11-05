"""Combined solver: multi-start greedy + targeted extension + simulated annealing + polish."""
import numpy as np
import time
import random
import math

def _get_primes(n):
    """Sieve of Eratosthenes to get primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    primes = _get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    def build_counts(residues):
        """Build counts array from residues."""
        counts = np.zeros(MAX_L + 1, dtype=np.int8)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1
        counts[0] = 100
        return counts

    def find_L(counts, current_L=0):
        """Find largest L such that all positions 1..L have count >= 2."""
        if current_L >= MAX_L:
            return MAX_L
        if current_L > 0:
            if np.min(counts[1:current_L + 1]) < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return bad[0]
            rest = counts[current_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return current_L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return bad_idx[0]

    def apply_change(counts, residues, i, new_r):
        """Apply a single-prime residue change to counts and residues."""
        p = primes[i]
        old_r = residues[i]
        if old_r == new_r:
            return
        if old_r == 0:
            counts[p::p] -= 1
        else:
            counts[old_r::p] -= 1
        residues[i] = new_r
        if new_r == 0:
            counts[p::p] += 1
        else:
            counts[new_r::p] += 1

    def revert_change(counts, residues, i, old_r):
        """Revert a single-prime residue change."""
        p = primes[i]
        new_r = residues[i]
        if new_r == old_r:
            return
        if new_r == 0:
            counts[p::p] -= 1
        else:
            counts[new_r::p] -= 1
        residues[i] = old_r
        if old_r == 0:
            counts[p::p] += 1
        else:
            counts[old_r::p] += 1

    def greedy_init(seed):
        """Multi-start greedy initialization."""
        random.seed(seed)
        np.random.seed(seed)
        GREEDY_WINDOW = 5000
        residues = [0] * n_primes
        counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)

        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1

            if p <= 200:
                candidates = range(p)
            else:
                candidates = np.random.choice(p, size=min(p, 50), replace=False)

            for r in candidates:
                start = r if r > 0 else p
                if start > GREEDY_WINDOW:
                    continue
                positions = np.arange(start, GREEDY_WINDOW + 1, p)
                deficient = np.sum(counts_greedy[positions] < 2)
                already_covered = np.sum(counts_greedy[positions] >= 2)
                score = deficient - 0.5 * already_covered
                if score > best_score:
                    best_score = score
                    best_r = r

            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= GREEDY_WINDOW:
                counts_greedy[start::p] += 1

        return residues

    def targeted_extension(residues, counts, current_L, time_budget):
        """Targeted greedy extension: directly try to extend L by covering L+1."""
        end_time = start_time + time_budget
        best_L = current_L
        best_residues = residues.copy()

        while current_L < MAX_L and time.time() < end_time:
            x = current_L + 1
            candidates = []
            for i, p in enumerate(primes):
                target_r = x % p
                if residues[i] == target_r:
                    continue
                old_r = residues[i]
                # Check if removing old_r coverage breaks any y in [1, x-1]
                if old_r == 0:
                    if x <= p:
                        candidates.append(i)
                    else:
                        min_cov = np.min(counts[p:x:p])
                        if min_cov >= 3:
                            candidates.append(i)
                else:
                    if old_r >= x:
                        candidates.append(i)
                    else:
                        min_cov = np.min(counts[old_r:x:p])
                        if min_cov >= 3:
                            candidates.append(i)

            if candidates:
                i = random.choice(candidates)
                apply_change(counts, residues, i, x % primes[i])
                new_L = find_L(counts, current_L)
                if new_L > current_L:
                    current_L = new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                else:
                    # Revert if it didn't help
                    revert_change(counts, residues, i, x % primes[i])
            else:
                break

        return best_residues, best_L

    def simulated_annealing(residues, counts, current_L, time_budget):
        """Run simulated annealing for the given time budget."""
        T = 5.0
        cooling = 0.99995
        min_T = 0.05
        best_L = current_L
        best_residues = residues.copy()

        end_time = start_time + time_budget

        while time.time() < end_time:
            i = np.random.randint(n_primes)
            p = primes[i]
            old_r = residues[i]
            new_r = np.random.randint(p)

            if new_r == old_r:
                continue

            # Apply change
            apply_change(counts, residues, i, new_r)

            # Calculate new L
            new_L = find_L(counts, current_L)

            # Acceptance
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
            else:
                delta = new_L - current_L
                if delta > -100:
                    prob = math.exp(delta / T)
                    if random.random() < prob:
                        current_L = new_L
                    else:
                        revert_change(counts, residues, i, old_r)
                else:
                    revert_change(counts, residues, i, old_r)

            T *= cooling
            if T < min_T:
                T = min_T

        return best_residues, best_L

    def targeted_polish(residues, counts, current_L, time_budget):
        """Targeted polish: try all residues for each prime to improve L."""
        best_L = current_L
        best_residues = residues.copy()
        end_time = start_time + time_budget

        improved = True
        while improved and time.time() < end_time:
            improved = False
            for i in range(n_primes):
                if time.time() > end_time:
                    break
                p = primes[i]
                current_r = residues[i]

                best_new_r = current_r
                best_new_L = current_L

                for new_r in range(p):
                    if new_r == current_r:
                        continue
                    if time.time() > end_time:
                        break

                    old_r = residues[i]
                    apply_change(counts, residues, i, new_r)
                    new_L = find_L(counts, current_L)

                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_new_r = new_r

                    # Revert for next trial
                    revert_change(counts, residues, i, old_r)

                if best_new_r != current_r:
                    apply_change(counts, residues, i, best_new_r)
                    current_L = best_new_L
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
                    improved = True

        return best_residues, best_L

    # Phase 1: Multi-start greedy initialization
    best_overall_L = 0
    best_overall_residues = None

    # Try a few different greedy starts
    for seed in [42, 123, 456]:
        residues = greedy_init(seed)
        counts = build_counts(residues)
        current_L = find_L(counts, 0)

        if current_L > best_overall_L:
            best_overall_L = current_L
            best_overall_residues = residues.copy()

    # Phase 2: Targeted greedy extension to push L forward quickly
    residues = best_overall_residues.copy()
    counts = build_counts(residues)
    current_L = best_overall_L

    extension_time = 20.0  # 20 seconds for targeted extension
    residues, current_L = targeted_extension(residues, counts, current_L, extension_time)

    if current_L > best_overall_L:
        best_overall_L = current_L
        best_overall_residues = residues.copy()

    # Phase 3: Simulated annealing for refinement
    residues = best_overall_residues.copy()
    counts = build_counts(residues)
    current_L = best_overall_L

    sa_time = 70.0  # 70 seconds for SA
    residues, current_L = simulated_annealing(residues, counts, current_L, sa_time)

    if current_L > best_overall_L:
        best_overall_L = current_L
        best_overall_residues = residues.copy()

    # Phase 4: Targeted polish
    residues = best_overall_residues.copy()
    counts = build_counts(residues)
    current_L = best_overall_L

    polish_time = 15.0  # Remaining time
    residues, current_L = targeted_polish(residues, counts, current_L, polish_time)

    if current_L > best_overall_L:
        best_overall_L = current_L
        best_overall_residues = residues.copy()

    return best_overall_residues
