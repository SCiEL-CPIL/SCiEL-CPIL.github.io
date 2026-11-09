import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit

    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    n_primes = len(primes)

    # Search for L up to M; extend if we approach the limit.
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)

    # Initialize residues
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        counts[residues[i]::p] += 1

    def calc_L(counts, M):
        # Largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        idx = np.flatnonzero(counts[1:] < 2)
        if idx.size == 0:
            return M
        return int(idx[0])

    def extend_M(new_M):
        nonlocal M, counts
        old_M = M
        old_counts = counts
        counts = np.zeros(new_M + 1, dtype=np.int64)
        counts[:old_M + 1] = old_counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            # first index in (old_M, new_M] congruent to rj mod pj
            start_m = max(0, (old_M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                counts[first_idx::pj] += 1
        M = new_M

    def maybe_extend():
        nonlocal M, counts, current_L
        if best_L >= M - 10:
            extend_M(M * 2)
            current_L = calc_L(counts, M)
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()

    # ---- Phase 1: simulated annealing ----
    T = 1.0
    T_min = 0.05
    cooling_rate = 0.9995

    while time.time() - start_time < deadline - 30:
        move_type = random.random()
        if move_type < 0.65:
            # 1-move targeted at L+1
            x_target = current_L + 1
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_new = x_target % p
            r_old = residues[i]
            if r_new == r_old:
                continue
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            new_L = calc_L(counts, M)
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    maybe_extend()
            else:
                delta = current_L - new_L
                if random.random() < np.exp(-delta / T):
                    current_L = new_L
                else:
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old

        elif move_type < 0.8:
            # 1-move random
            i = random.randint(0, n_primes - 1)
            p = primes[i]
            r_old = residues[i]
            r_new = random.randint(0, p - 1)
            if r_new == r_old:
                continue
            counts[r_old::p] -= 1
            counts[r_new::p] += 1
            residues[i] = r_new
            new_L = calc_L(counts, M)
            if new_L >= current_L:
                current_L = new_L
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    maybe_extend()
            else:
                delta = current_L - new_L
                if random.random() < np.exp(-delta / T):
                    current_L = new_L
                else:
                    counts[r_new::p] -= 1
                    counts[r_old::p] += 1
                    residues[i] = r_old

        else:
            # 2-move: target L+1, then repair a break
            x_target = current_L + 1
            i = random.randint(0, n_primes - 1)
            p1 = primes[i]
            r1_old = residues[i]
            r1_new = x_target % p1
            if r1_new == r1_old:
                continue
            counts[r1_old::p1] -= 1
            counts[r1_new::p1] += 1
            residues[i] = r1_new
            L_after_1 = calc_L(counts, M)
            if L_after_1 >= current_L:
                current_L = L_after_1
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    maybe_extend()
            else:
                # broke something before x_target; try to repair
                broken_x = L_after_1
                j = random.randint(0, n_primes - 1)
                if j == i:
                    j = (j + 1) % n_primes
                p2 = primes[j]
                r2_old = residues[j]
                r2_new = broken_x % p2
                if r2_new != r2_old:
                    counts[r2_old::p2] -= 1
                    counts[r2_new::p2] += 1
                    residues[j] = r2_new
                    L_after_2 = calc_L(counts, M)
                    if L_after_2 >= current_L:
                        current_L = L_after_2
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                            maybe_extend()
                    else:
                        # revert both
                        counts[r2_new::p2] -= 1
                        counts[r2_old::p2] += 1
                        residues[j] = r2_old
                        counts[r1_new::p1] -= 1
                        counts[r1_old::p1] += 1
                        residues[i] = r1_old
                else:
                    counts[r1_new::p1] -= 1
                    counts[r1_old::p1] += 1
                    residues[i] = r1_old

        T = max(T_min, T * cooling_rate)

    # ---- Phase 2: deterministic local polish (try all residues per prime) ----
    while time.time() - start_time < deadline - 8:
        improved = False
        for i in range(n_primes):
            if time.time() - start_time > deadline - 8:
                break
            p = primes[i]
            r_old = residues[i]
            best_r = r_old
            best_local = current_L
            for r_c in range(p):
                if r_c == r_old:
                    continue
                counts[r_old::p] -= 1
                counts[r_c::p] += 1
                local_L = calc_L(counts, M)
                if local_L > best_local:
                    best_local = local_L
                    best_r = r_c
                else:
                    counts[r_c::p] -= 1
                    counts[r_old::p] += 1
            if best_r != r_old:
                counts[r_old::p] -= 1
                counts[best_r::p] += 1
                residues[i] = best_r
                current_L = best_local
                improved = True
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                    maybe_extend()
        if not improved:
            break

    # ---- Phase 3: light SA tail on the polished state ----
    T = 0.15
    while time.time() - start_time < deadline - 1:
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)
        if r_new == r_old:
            continue
        counts[r_old::p] -= 1
        counts[r_new::p] += 1
        residues[i] = r_new
        new_L = calc_L(counts, M)
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                maybe_extend()
        else:
            delta = current_L - new_L
            if random.random() < np.exp(-delta / T):
                current_L = new_L
            else:
                counts[r_new::p] -= 1
                counts[r_old::p] += 1
                residues[i] = r_old
        T = max(0.05, T * 0.999)

    return [int(r) for r in best_residues]
