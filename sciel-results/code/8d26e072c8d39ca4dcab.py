import time
import numpy as np
import random

def _primes_up_to(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)
N_PRIMES = len(PRIMES)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    # Precompute residue positions as boolean masks for [1, L_max]
    L_max = 20000
    masks = []  # masks[i][r] is a boolean array of length L_max+1
    for p in PRIMES:
        pm = []
        for r in range(p):
            m = np.zeros(L_max + 1, dtype=np.bool_)
            if r == 0:
                m[p::p] = True
            else:
                m[r::p] = True
            pm.append(m)
        masks.append(pm)

    def total_deficit(coverage):
        """Sum of max(0, 2 - coverage[x]) for x in [1, L_max]."""
        return int(np.maximum(0, 2 - coverage[1:]).sum())

    def find_L(coverage):
        """Find largest L such that coverage[x] >= 2 for all x in [1, L]."""
        # Find first position where coverage < 2
        bad = np.where(coverage[1:] < 2)[0]
        if len(bad) == 0:
            return L_max
        return int(bad[0])  # bad[0] is 0-indexed, so position is bad[0]+1, L = bad[0]

    def fast_find_L(coverage, current_L):
        """Quick L check starting from current_L."""
        L = current_L
        # Check forward
        x = L + 1
        while x <= L_max and coverage[x] >= 2:
            L = x
            x += 1
        return L

    best_L = 0
    best_residues = None
    best_deficit = float('inf')

    rng = random.Random(12345)

    for restart in range(12):
        if time.time() > deadline - 30:
            break

        # Initialize: random residues
        residues = np.array([rng.randrange(p) for p in PRIMES], dtype=np.int32)
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(N_PRIMES):
            coverage[masks[i][residues[i]]] += 1

        cur_deficit = total_deficit(coverage)
        cur_L = find_L(coverage)

        if cur_L > best_L:
            best_L = cur_L
            best_residues = residues.copy()
        if cur_deficit < best_deficit:
            best_deficit = cur_deficit

        # Simulated annealing
        T = 8.0
        T_min = 0.1
        alpha = 0.99995
        no_improve = 0
        max_no_improve = 5000

        while time.time() < deadline - 30:
            if T < T_min or no_improve > max_no_improve:
                break

            # Pick a random prime
            i = rng.randrange(N_PRIMES)
            p = PRIMES[i]
            r_old = int(residues[i])
            r_new = rng.randrange(p)
            if r_old == r_new:
                T *= alpha
                continue

            # Compute delta deficit using vectorized operations
            old_m = masks[i][r_old]
            new_m = masks[i][r_new]
            affected = old_m | new_m
            # Only positions where at least one mask is true
            c0 = coverage[affected]
            # New coverage at affected positions
            c1 = c0.copy()
            c1[old_m[affected]] -= 1
            c1[new_m[affected]] += 1
            d0 = np.maximum(0, 2 - c0).astype(np.int32)
            d1 = np.maximum(0, 2 - c1).astype(np.int32)
            delta_deficit = int(d1.sum() - d0.sum())

            # Decide accept/reject
            accept = False
            if delta_deficit <= 0:
                accept = True
            else:
                if delta_deficit < T * 8:
                    if rng.random() < np.exp(-delta_deficit / T):
                        accept = True

            if accept:
                coverage[old_m] -= 1
                coverage[new_m] += 1
                residues[i] = r_new
                cur_deficit += delta_deficit
                no_improve = 0
            else:
                no_improve += 1

            T *= alpha

        # Local search: try all residues for each prime (forward + backward)
        for direction in range(2):
            if time.time() > deadline - 15:
                break
            if direction == 0:
                order = list(range(N_PRIMES))
            else:
                order = list(range(N_PRIMES - 1, -1, -1))

            for i in order:
                if time.time() > deadline - 15:
                    break
                p = PRIMES[i]
                r_old = int(residues[i])
                best_local_r = r_old
                best_local_deficit = cur_deficit

                for r in range(p):
                    if r == r_old:
                        continue
                    old_m = masks[i][r_old]
                    new_m = masks[i][r]
                    affected = old_m | new_m
                    c0 = coverage[affected]
                    c1 = c0.copy()
                    c1[old_m[affected]] -= 1
                    c1[new_m[affected]] += 1
                    d0 = np.maximum(0, 2 - c0).astype(np.int32)
                    d1 = np.maximum(0, 2 - c1).astype(np.int32)
                    delta_d = int(d1.sum() - d0.sum())

                    if delta_d < 0:
                        # Apply
                        coverage[old_m] -= 1
                        coverage[new_m] += 1
                        residues[i] = r
                        r_old = r
                        cur_deficit += delta_d
                        best_local_r = r
                        best_local_deficit = cur_deficit

                if time.time() > deadline - 10:
                    break

        # Check final L for this restart
        cur_L = find_L(coverage)
        if cur_L > best_L:
            best_L = cur_L
            best_residues = residues.copy()
        if cur_deficit < best_deficit:
            best_deficit = cur_deficit

    # Final verification
    if best_residues is None:
        best_residues = np.array([0] * N_PRIMES, dtype=np.int32)

    return [int(r) for r in best_residues]
