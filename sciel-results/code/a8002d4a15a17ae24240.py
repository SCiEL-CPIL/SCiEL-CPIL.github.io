import numpy as np
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]

    primes = sieve(1000)
    n_primes = len(primes)
    primes_np = np.array(primes, dtype=np.int64)

    # The range of integers to check.
    MAX_CHECK = 10000
    x_values = np.arange(1, MAX_CHECK + 1)

    # Precompute x mod p for all x and p.
    x_mod_p = x_values[:, None] % primes_np[None, :]

    # Helper: Compute the current score (L)
    def compute_score(residues):
        match = (x_mod_p == residues[None, :])
        counts = match.sum(axis=1)
        invalid = counts < 2
        if not np.any(invalid):
            return MAX_CHECK
        else:
            first_invalid_idx = np.argmax(invalid)
            return int(first_invalid_idx)

    # Helper: Compute coverage array
    def get_coverage(residues):
        cov = np.zeros(MAX_CHECK + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            x = start
            while x <= MAX_CHECK:
                cov[x] += 1
                x += p
        return cov

    # Helper: Try to extend coverage by finding best single move at the gap
    def try_extend(residues, current_score):
        if current_score >= MAX_CHECK:
            return residues, current_score

        gap_idx = current_score
        match_at_gap = (x_mod_p[gap_idx] == residues)
        candidates = np.where(~match_at_gap)[0]

        if len(candidates) == 0:
            return residues, current_score

        M = (x_mod_p == residues[None, :])
        old_counts = M.sum(axis=1)

        C = candidates
        R = x_mod_p[gap_idx, C]

        x_mod_p_cands = x_mod_p[:, C]
        new_matches_cands = (x_mod_p_cands == R[None, :])
        M_cands = M[:, C]

        delta_counts = new_matches_cands.astype(np.int8) - M_cands.astype(np.int8)
        new_counts = old_counts[:, None].astype(np.int8) + delta_counts

        invalid = new_counts < 2
        has_gap = invalid.any(axis=0)
        first_gap_idx = np.argmax(invalid, axis=0)
        scores = np.where(has_gap, first_gap_idx, MAX_CHECK)

        best_idx_in_scores = np.argmax(scores)
        best_candidate_idx = C[best_idx_in_scores]
        best_new_score = scores[best_idx_in_scores]

        if best_new_score > current_score:
            new_residues = residues.copy()
            new_residues[best_candidate_idx] = R[best_idx_in_scores]
            return new_residues, best_new_score

        # If no improvement, try a move that keeps score same but helps at gap
        if best_new_score == current_score:
            # Check if this move actually increases count at gap
            old_count_at_gap = old_counts[gap_idx]
            if old_count_at_gap < 2:
                new_residues = residues.copy()
                new_residues[best_candidate_idx] = R[best_idx_in_scores]
                return new_residues, current_score

        return residues, current_score

    # Helper: Simulated Annealing with vulnerable set optimization
    def simulated_annealing(residues, current_score, time_budget):
        sa_deadline = time.time() + time_budget
        best_res = residues.copy()
        best_sc = current_score
        current_res = residues.copy()
        current_sc = current_score

        # Initial temperature
        T = 10.0
        cooling_rate = 0.999

        # Compute initial coverage
        cov = get_coverage(current_res)

        while time.time() < sa_deadline:
            # Find first hole
            target_x = current_sc + 1
            if target_x > MAX_CHECK:
                break

            # Find safe candidates
            candidates = []
            for i, p in enumerate(primes):
                old_r = current_res[i]
                new_r = target_x % p
                if old_r == new_r:
                    continue

                # Check if changing p to new_r breaks any y < current_sc
                safe = True
                if old_r == 0:
                    start_y = p
                else:
                    start_y = old_r

                y = start_y
                while y < current_sc:
                    if cov[y] == 2:
                        safe = False
                        break
                    y += p

                if safe:
                    candidates.append(i)

            if candidates:
                # Pick a candidate
                if random.random() < 0.8:
                    idx = random.choice(candidates)
                else:
                    idx = random.randint(0, n_primes - 1)

                p = primes[idx]
                old_r = current_res[idx]
                new_r = target_x % p

                if old_r == new_r:
                    continue

                # Compute change in vulnerable set
                old_cov = cov.copy()
                
                # Update coverage
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= MAX_CHECK:
                    cov[y] -= 1
                    y += p

                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= MAX_CHECK:
                    cov[y] += 1
                    y += p

                # Compute new score
                new_sc = current_sc
                for x in range(1, MAX_CHECK + 1):
                    if cov[x] >= 2:
                        new_sc = x
                    else:
                        break

                # Accept or reject based on simulated annealing
                delta = new_sc - current_sc
                if delta > 0 or random.random() < math.exp(delta / T):
                    current_res[idx] = new_r
                    current_sc = new_sc
                    if new_sc > best_sc:
                        best_sc = new_sc
                        best_res = current_res.copy()
                else:
                    # Revert
                    if old_r == 0:
                        start = p
                    else:
                        start = old_r
                    y = start
                    while y <= MAX_CHECK:
                        cov[y] += 1
                        y += p

                    if new_r == 0:
                        start = p
                    else:
                        start = new_r
                    y = start
                    while y <= MAX_CHECK:
                        cov[y] -= 1
                        y += p
                    current_res[idx] = old_r
            else:
                # Stuck. Perform a kick.
                idx = random.randint(0, n_primes - 1)
                p = primes[idx]
                old_r = current_res[idx]
                new_r = random.randint(0, p - 1)

                # Update cov
                if old_r == 0:
                    start = p
                else:
                    start = old_r
                y = start
                while y <= MAX_CHECK:
                    cov[y] -= 1
                    y += p

                if new_r == 0:
                    start = p
                else:
                    start = new_r
                y = start
                while y <= MAX_CHECK:
                    cov[y] += 1
                    y += p

                current_res[idx] = new_r

                # Update L
                new_sc = 0
                for x in range(1, MAX_CHECK + 1):
                    if cov[x] >= 2:
                        new_sc = x
                    else:
                        break
                current_sc = new_sc

            # Cool down
            T *= cooling_rate

        return best_res, best_sc

    # Strategy: Multiple restarts with different initializations
    random.seed(42)
    np.random.seed(42)

    best_residues = None
    best_score = 0

    # Time budget per restart
    restart_time_budget = 12.0

    while time.time() < deadline - 5.0:
        # Different initialization strategies
        init_strategy = random.randint(0, 4)

        if init_strategy == 0:
            # All ones
            current_residues = np.ones(n_primes, dtype=np.int64)
        elif init_strategy == 1:
            # Random residues
            current_residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
        elif init_strategy == 2:
            # Start with r_p = 0 for small primes, 1 for large
            current_residues = np.where(primes_np < 50, 0, 1).astype(np.int64)
        elif init_strategy == 3:
            # All zeros
            current_residues = np.zeros(n_primes, dtype=np.int64)
        else:
            # Greedy initialization: for each x from 1, assign residues to cover it
            current_residues = np.zeros(n_primes, dtype=np.int64)
            for x in range(1, min(50, MAX_CHECK + 1)):
                # Simply set some residues to x % p
                for pi in np.random.permutation(n_primes)[:3]:
                    current_residues[pi] = x % primes[pi]

        current_score = compute_score(current_residues)

        # Phase 1: Greedy single-move extension
        restart_deadline = time.time() + restart_time_budget * 0.3
        while time.time() < restart_deadline and current_score < MAX_CHECK:
            new_res, new_sc = try_extend(current_residues, current_score)
            if new_sc > current_score:
                current_residues = new_res
                current_score = new_sc
            else:
                break

        # Phase 2: Simulated annealing
        restart_deadline2 = time.time() + restart_time_budget * 0.5
        if time.time() < restart_deadline2 and current_score < MAX_CHECK:
            current_residues, current_score = simulated_annealing(
                current_residues, current_score, 
                min(restart_deadline2 - time.time(), restart_time_budget * 0.5)
            )

        # Phase 3: More greedy extension after SA
        restart_deadline3 = time.time() + restart_time_budget * 0.2
        while time.time() < restart_deadline3 and current_score < MAX_CHECK:
            new_res, new_sc = try_extend(current_residues, current_score)
            if new_sc > current_score:
                current_residues = new_res
                current_score = new_sc
            else:
                break

        # Update best
        if current_score > best_score:
            best_score = current_score
            best_residues = current_residues.copy()

        if best_score >= MAX_CHECK:
            break

    # Final: Try to extend the best solution further with remaining time
    if best_residues is not None and time.time() < deadline - 2.0:
        final_res = best_residues.copy()
        final_sc = best_score
        final_deadline = time.time() + min(5.0, deadline - time.time() - 1.0)
        while time.time() < final_deadline and final_sc < MAX_CHECK:
            new_res, new_sc = try_extend(final_res, final_sc)
            if new_sc > final_sc:
                final_res = new_res
                final_sc = new_sc
            else:
                # Try SA one more time with small budget
                if time.time() < final_deadline - 1.0:
                    final_res, final_sc = simulated_annealing(
                        final_res, final_sc, 
                        min(2.0, final_deadline - time.time() - 0.5)
                    )
                if final_sc <= best_score:
                    break
                if final_sc > best_score:
                    best_score = final_sc
                    best_residues = final_res.copy()
                else:
                    break

    if best_residues is None:
        best_residues = np.ones(n_primes, dtype=np.int64)

    return [int(r) for r in best_residues]

if __name__ == "__main__":
    res = construct()
    print(len(res))
