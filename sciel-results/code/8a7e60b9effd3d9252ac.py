import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # Primes up to 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                sieve[i * i:limit + 1:i] = [False] * len(sieve[i * i:limit + 1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)
    MAX_L = 100000

    # Helper to build counts array
    def build_counts(residues, window):
        counts = np.zeros(window + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            start = r if r > 0 else p
            if start <= window:
                counts[start::p] += 1
        return counts

    def find_L(counts, current_L, max_L):
        """Find the largest L such that all positions 1..L have count >= 2."""
        if current_L > 0:
            # Check if current prefix is still valid
            if np.min(counts[1:current_L + 1]) < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return bad[0]
            # Try to extend
            L = current_L
            if L < max_L:
                rest = counts[L + 1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    return max_L
                else:
                    return L + bad_idx[0]
        else:
            bad_idx = np.where(counts[1:] < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            return bad_idx[0]

    # --- Phase 1: Multi-start Greedy Initialization ---
    # We try a few greedy strategies and pick the best
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = None

    GREEDY_WINDOW = 5000

    for attempt in range(5):
        residues = [0] * n_primes
        counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)

        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1

            # For small primes, try all residues; for large primes, sample
            if p <= 100:
                candidates = range(p)
            else:
                # Sample residues that cover the current "frontier"
                candidates = np.random.choice(p, size=min(p, 80), replace=False)

            for r in candidates:
                start = r if r > 0 else p
                if start > GREEDY_WINDOW:
                    continue
                # Count positions in [start, GREEDY_WINDOW] with step p that have count < 2
                positions = np.arange(start, GREEDY_WINDOW + 1, p)
                current_counts = counts_greedy[positions]
                deficient = np.sum(current_counts < 2)
                already_covered = np.sum(current_counts >= 2)
                score = deficient - 0.3 * already_covered
                if score > best_score:
                    best_score = score
                    best_r = r

            residues[idx] = best_r
            r = best_r
            start = r if r > 0 else p
            if start <= GREEDY_WINDOW:
                counts_greedy[start::p] += 1

        # Evaluate full length
        full_counts = build_counts(residues, MAX_L)
        L = find_L(full_counts, 0, MAX_L)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = full_counts

    # Start refinement from the best greedy result
    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Simulated Annealing ---
    # Parameters tuned for stability and exploration
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.05

    # Precompute prime indices
    prime_indices = np.arange(n_primes)

    # Main SA loop
    while time.time() < deadline:
        # Pick a random prime to perturb
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        # Apply change
        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1

        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1

        # Calculate new L
        new_L = find_L(counts, current_L, MAX_L)

        # Acceptance
        if new_L >= current_L:
            # Accept
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            # Reject or Accept with probability
            delta = new_L - current_L
            if delta > -100:  # Only compute exp for reasonable deltas
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                else:
                    # Revert
                    residues[i] = old_r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
            else:
                # Revert
                residues[i] = old_r
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1

        T *= cooling_rate
        if T < min_T:
            T = min_T

    return best_residues
