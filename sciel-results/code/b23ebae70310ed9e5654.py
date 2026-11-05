import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

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

    # State
    residues = np.zeros(n_primes, dtype=np.int32)
    counts = np.zeros(MAX_L + 1, dtype=np.int8)

    def build_counts():
        counts[:] = 0
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1
        counts[0] = 100

    def find_L(current_L=0):
        if current_L >= MAX_L:
            return MAX_L
        if current_L > 0 and np.min(counts[1:current_L + 1]) < 2:
            bad = np.where(counts[1:current_L + 1] < 2)[0]
            return bad[0]
        if current_L < MAX_L:
            rest = counts[current_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) == 0:
                return MAX_L
            return current_L + bad_idx[0]
        return current_L

    def apply_change(i, new_r):
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

    def revert_change(i, old_r):
        p = primes[i]
        new_r = residues[i]
        if old_r == new_r:
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

    # --- Phase 1: Greedy Initialization with window optimization ---
    # For each prime, choose the residue that maximizes coverage of deficient positions
    GREEDY_WINDOW = 5000
    greedy_counts = np.zeros(GREEDY_WINDOW + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_score = -1
        
        if p <= 200:
            candidates = range(p)
        else:
            candidates = np.random.choice(p, size=min(p, 80), replace=False)
        
        for r in candidates:
            start = r if r > 0 else p
            if start > GREEDY_WINDOW:
                continue
            positions = np.arange(start, GREEDY_WINDOW + 1, p)
            deficient = np.sum(greedy_counts[positions] < 2)
            already_covered = np.sum(greedy_counts[positions] >= 2)
            score = deficient - 0.3 * already_covered
            if score > best_score:
                best_score = score
                best_r = r
        
        residues[idx] = best_r
        r = best_r
        start = r if r > 0 else p
        if start <= GREEDY_WINDOW:
            greedy_counts[start::p] += 1

    build_counts()
    current_L = find_L(0)
    best_L = current_L
    best_residues = residues.copy()

    # --- Phase 2: Targeted Greedy Extension ---
    while current_L < MAX_L and time.time() < deadline - 30.0:
        x = current_L + 1
        candidates = []
        for i, p in enumerate(primes):
            target_r = x % p
            if residues[i] == target_r:
                continue
            old_r = residues[i]
            if old_r == 0:
                if x <= p:
                    candidates.append(i)
                    continue
                min_cov = np.min(counts[p:x:p])
                if min_cov >= 3:
                    candidates.append(i)
            else:
                if old_r >= x:
                    candidates.append(i)
                    continue
                if old_r > 0:
                    min_cov = np.min(counts[old_r:x:p])
                    if min_cov >= 3:
                        candidates.append(i)

        if candidates:
            i = random.choice(candidates)
            apply_change(i, x % primes[i])
            new_L = find_L(current_L)
            if new_L > current_L:
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
            else:
                old_r = residues[i]
                revert_change(i, old_r)
        else:
            break

    # --- Phase 3: Simulated Annealing ---
    T = 5.0
    cooling_rate = 0.99995
    min_T = 0.05

    while time.time() < deadline - 10.0:
        i = np.random.randint(n_primes)
        p = primes[i]
        old_r = residues[i]
        new_r = np.random.randint(p)

        if new_r == old_r:
            continue

        apply_change(i, new_r)
        new_L = find_L(current_L)
        delta = new_L - current_L

        if delta > 0:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
        else:
            if delta > -200:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_L = new_L
                else:
                    revert_change(i, old_r)
            else:
                revert_change(i, old_r)

        T *= cooling_rate
        if T < min_T:
            T = min_T

    # --- Phase 4: Final Polish ---
    if time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        for _ in range(50):
            if time.time() > polish_deadline:
                break
            progress = False
            for i in range(n_primes):
                if time.time() > polish_deadline:
                    break
                p = primes[i]
                old_r = residues[i]
                for offset in [-2, -1, 1, 2, -5, 5, -10, 10]:
                    new_r = (old_r + offset) % p
                    if new_r == old_r:
                        continue
                    apply_change(i, new_r)
                    new_L = find_L(current_L)
                    if new_L > current_L:
                        current_L = new_L
                        if new_L > best_L:
                            best_L = new_L
                            best_residues = residues.copy()
                        progress = True
                        break
                    else:
                        revert_change(i, old_r)
            if not progress:
                break

    # Final evaluation
    final_counts = np.zeros(MAX_L + 1, dtype=np.int8)
    for i, p in enumerate(primes):
        r = best_residues[i]
        if r == 0:
            final_counts[p::p] += 1
        else:
            final_counts[r::p] += 1
    final_counts[0] = 100

    if np.min(final_counts[1:best_L + 1]) < 2:
        bad = np.where(final_counts[1:best_L + 1] < 2)[0]
        final_L = bad[0]
    else:
        final_L = best_L
        if final_L < MAX_L:
            rest = final_counts[final_L + 1:]
            bad_idx = np.where(rest < 2)[0]
            if len(bad_idx) > 0:
                final_L = final_L + bad_idx[0]

    return best_residues.tolist()
