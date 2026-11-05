import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

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

    # --- Phase 1: Greedy Construction ---
    GREEDY_WINDOW = 5000

    def build_count_array(residues, primes, window):
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
            if np.min(counts[1:current_L + 1]) < 2:
                bad = np.where(counts[1:current_L + 1] < 2)[0]
                return bad[0]
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
            else:
                return bad_idx[0]

    # Greedy: for each prime, try all residues and pick the one that best
    # covers deficient positions in the early window
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

    # --- Phase 2: Simulated Annealing on full range ---
    counts = np.zeros(MAX_L + 1, dtype=np.int32)
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= MAX_L:
            counts[start::p] += 1

    current_L = find_L(counts, 0, MAX_L)
    best_L = current_L
    best_residues = residues.copy()

    T = 5.0
    cooling = 0.99995
    min_T = 0.05

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

        T *= cooling
        if T < min_T:
            T = min_T

    # --- Phase 3: Local search refinement ---
    # Try each prime's residue in a neighborhood to improve L
    if time.time() < deadline - 10:
        for _ in range(5):  # Multiple passes
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(n_primes):
                p = primes[i]
                old_r = residues[i]
                # Try nearby residues
                for offset in range(-3, 4):
                    new_r = (old_r + offset) % p
                    if new_r == old_r:
                        continue

                    start_old = old_r if old_r > 0 else p
                    if start_old <= MAX_L:
                        counts[start_old::p] -= 1

                    start_new = new_r if new_r > 0 else p
                    if start_new <= MAX_L:
                        counts[start_new::p] += 1

                    new_L = find_L(counts, current_L, MAX_L)

                    if new_L > current_L:
                        residues[i] = new_r
                        current_L = new_L
                        improved = True
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        break
                    else:
                        # Revert
                        if start_new <= MAX_L:
                            counts[start_new::p] -= 1
                        if start_old <= MAX_L:
                            counts[start_old::p] += 1

            if not improved:
                break

    return best_residues
