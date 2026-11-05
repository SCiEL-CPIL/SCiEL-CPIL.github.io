import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

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
        if current_L >= max_L:
            return max_L
        # Check if current prefix is still valid
        if current_L > 0 and np.min(counts[1:current_L + 1]) < 2:
            bad = np.where(counts[1:current_L + 1] < 2)[0]
            return bad[0]
        # Try to extend
        if current_L < max_L:
            rest = counts[current_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return max_L
            else:
                return current_L + bad_idx[0]
        return current_L

    # --- Phase 1: Multi-start Greedy Initialization ---
    best_init_L = 0
    best_init_res = [0] * n_primes
    best_init_counts = None

    GREEDY_WINDOW = 8000

    for attempt in range(8):
        residues = [0] * n_primes
        counts_greedy = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)

        for idx in range(n_primes):
            p = primes[idx]
            best_r = 0
            best_score = -1

            if p <= 150:
                candidates = range(p)
            else:
                candidates = np.random.choice(p, size=min(p, 100), replace=False)

            for r in candidates:
                start = r if r > 0 else p
                if start > GREEDY_WINDOW:
                    continue
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

        full_counts = build_counts(residues, MAX_L)
        L = find_L(full_counts, 0, MAX_L)
        if L > best_init_L:
            best_init_L = L
            best_init_res = residues.copy()
            best_init_counts = full_counts

    residues = best_init_res
    counts = best_init_counts
    current_L = best_init_L
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Targeted Greedy Extension ---
    # This phase tries to extend L by finding deficient positions and fixing them
    # by reassigning residues of primes that can cover those positions.
    
    def targeted_extension(residues, counts, current_L, max_L, time_budget):
        """Try to extend L by targeted reassignments."""
        L = current_L
        progress = True
        while progress and time.time() < time_budget:
            progress = False
            if L >= max_L:
                break
            
            # Find the first deficient position after L
            if L + 1 <= max_L:
                rest = counts[L + 1:]
                bad_idx = np.where(rest < 2)[0]
                if len(bad_idx) == 0:
                    L = max_L
                    break
                target_pos = L + bad_idx[0]
            else:
                break
            
            # Find primes that can cover target_pos
            # A prime p covers position x if x % p == r_p (mod p), i.e., r_p = x % p
            # We want to find a prime p where changing r_p to target_pos % p would
            # help cover target_pos without breaking coverage elsewhere.
            
            # Try primes in random order
            perm = np.random.permutation(n_primes)
            found = False
            for i in perm:
                p = primes[i]
                old_r = residues[i]
                new_r = target_pos % p
                
                if new_r == old_r:
                    continue
                
                # Apply change
                start_old = old_r if old_r > 0 else p
                if start_old <= max_L:
                    counts[start_old::p] -= 1
                
                start_new = new_r if new_r > 0 else p
                if start_new <= max_L:
                    counts[start_new::p] += 1
                
                # Check if this helps
                new_L = find_L(counts, L, max_L)
                
                if new_L > L:
                    # Accept - this extends L
                    residues[i] = new_r
                    L = new_L
                    progress = True
                    found = True
                    break
                else:
                    # Revert
                    residues[i] = old_r
                    if start_new <= max_L:
                        counts[start_new::p] -= 1
                    if start_old <= max_L:
                        counts[start_old::p] += 1
            
            if not found:
                # Try a different strategy: look at positions in [L+1, L+50]
                # and find the most frequently deficient position
                window_end = min(L + 50, max_L)
                if L + 1 <= window_end:
                    seg = counts[L + 1:window_end + 1]
                    deficient_counts = np.sum(seg < 2)
                    if deficient_counts == 0:
                        L = window_end
                        progress = True
                        continue
                    # Find position with minimum count
                    min_pos = L + 1 + np.argmin(seg)
                    # Try to fix that position
                    perm2 = np.random.permutation(n_primes)
                    for i in perm2:
                        p = primes[i]
                        old_r = residues[i]
                        new_r = min_pos % p
                        if new_r == old_r:
                            continue
                        
                        start_old = old_r if old_r > 0 else p
                        if start_old <= max_L:
                            counts[start_old::p] -= 1
                        start_new = new_r if new_r > 0 else p
                        if start_new <= max_L:
                            counts[start_new::p] += 1
                        
                        new_L = find_L(counts, L, max_L)
                        if new_L > L:
                            residues[i] = new_r
                            L = new_L
                            progress = True
                            break
                        else:
                            residues[i] = old_r
                            if start_new <= max_L:
                                counts[start_new::p] -= 1
                            if start_old <= max_L:
                                counts[start_old::p] += 1
        
        return L

    # Run targeted extension
    ext_budget = start_time + 60.0
    current_L = targeted_extension(residues, counts, current_L, MAX_L, ext_budget)
    if current_L > best_L:
        best_L = current_L
        best_residues = residues.copy()

    # --- Phase 3: Simulated Annealing with optimized evaluation ---
    T = 10.0
    cooling_rate = 0.99998
    min_T = 0.1

    while time.time() < deadline:
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        start_old = old_r if old_r > 0 else p
        if start_old <= MAX_L:
            counts[start_old::p] -= 1

        start_new = new_r if new_r > 0 else p
        if start_new <= MAX_L:
            counts[start_new::p] += 1

        new_L = find_L(counts, current_L, MAX_L)

        if new_L >= current_L:
            residues[i] = new_r
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            delta = new_L - current_L
            if delta > -100:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    residues[i] = new_r
                    current_L = new_L
                else:
                    residues[i] = old_r
                    if start_new <= MAX_L:
                        counts[start_new::p] -= 1
                    if start_old <= MAX_L:
                        counts[start_old::p] += 1
            else:
                residues[i] = old_r
                if start_new <= MAX_L:
                    counts[start_new::p] -= 1
                if start_old <= MAX_L:
                    counts[start_old::p] += 1

        T *= cooling_rate
        if T < min_T:
            T = min_T

        # Periodic targeted extension during SA
        if int((time.time() - start_time) * 10) % 100 == 0 and time.time() < deadline - 5:
            current_L = targeted_extension(residues, counts, current_L, MAX_L, time.time() + 2)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()

    return best_residues
