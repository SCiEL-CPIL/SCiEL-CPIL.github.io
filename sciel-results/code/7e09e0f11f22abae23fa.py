"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(1000)
    n = len(primes)  # 168
    primes_arr = np.array(primes, dtype=np.int64)

    # MCMC state
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    residues = np.zeros(n, dtype=np.int64)

    # Initialize residues randomly and build counts
    for i, p in enumerate(primes):
        r = random.randint(0, p - 1)
        residues[i] = r
        if r == 0:
            counts[p::p] += 1
        else:
            counts[r::p] += 1

    def calc_L(counts, M):
        if counts[1] < 2:
            return 0
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    def extend_M(counts, residues, M, new_M):
        new_counts = np.zeros(new_M + 1, dtype=np.int64)
        new_counts[:M + 1] = counts
        for j, pj in enumerate(primes):
            rj = residues[j]
            # Find first index in [M+1, new_M] congruent to rj mod pj
            if rj == 0:
                start_m = (M + 1 + pj - 1) // pj
            else:
                start_m = max(0, (M + 1 - rj + pj - 1) // pj)
            first_idx = rj + start_m * pj
            if first_idx <= new_M:
                new_counts[first_idx::pj] += 1
        return new_counts

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()

    # MCMC parameters
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9998
    stuck_counter = 0

    # Precompute for speed: for each prime, the range of indices it touches in [1, M]
    # We'll use numpy slicing for apply/revert

    def apply_move(counts, i, r_old, r_new, p):
        if r_old == r_new:
            return
        # Subtract old
        if r_old == 0:
            counts[p::p] -= 1
        else:
            counts[r_old::p] -= 1
        # Add new
        if r_new == 0:
            counts[p::p] += 1
        else:
            counts[r_new::p] += 1

    def revert_move(counts, i, r_old, r_new, p):
        if r_old == r_new:
            return
        # Subtract new
        if r_new == 0:
            counts[p::p] -= 1
        else:
            counts[r_new::p] -= 1
        # Add old
        if r_old == 0:
            counts[p::p] += 1
        else:
            counts[r_old::p] += 1

    # Repair function: try to fix the gap at current_L + 1
    def repair(counts, residues, current_L, M):
        gap_x = current_L + 1
        if gap_x > M:
            return False, counts, residues, current_L, M

        # Try to find a prime that doesn't cover gap_x and can be moved to cover it without breaking [1, current_L]
        prime_indices = list(range(n))
        random.shuffle(prime_indices)

        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            r_new = gap_x % p

            if r_old == r_new:
                continue

            # Check if moving from r_old to r_new breaks any count in [1, current_L]
            # We need to check if any y in [1, current_L] with y % p == r_old has count[y] == 2
            # Because that would drop to 1.
            
            # Efficient check using numpy
            if r_old == 0:
                y_vals = np.arange(p, current_L + 1, p)
            else:
                y_vals = np.arange(r_old, current_L + 1, p)
            
            if len(y_vals) > 0:
                # Check if any of these have count == 2
                if np.any(counts[y_vals] == 2):
                    continue

            # Move is valid
            apply_move(counts, i, r_old, r_new, p)
            residues[i] = r_new

            new_L = calc_L(counts, M)
            
            # If we extended M, we might need to update
            if new_L >= M - 10:
                new_M = M * 2
                counts = extend_M(counts, residues, M, new_M)
                M = new_M
                new_L = calc_L(counts, M)

            return True, counts, residues, new_L, M

        return False, counts, residues, current_L, M

    # Main MCMC loop
    while time.time() < deadline:
        # 1. Try repair
        found, counts, residues, current_L, M = repair(counts, residues, current_L, M)
        
        if found:
            if current_L > best_L:
                best_L = current_L
                best_residues = residues.copy()
            stuck_counter = 0
            continue

        # 2. MCMC step (Metropolis-Hastings)
        i = random.randint(0, n - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)

        if r_new == r_old:
            continue

        # Apply move
        apply_move(counts, i, r_old, r_new, p)
        residues[i] = r_new

        new_L = calc_L(counts, M)

        # Extend M if needed
        if new_L >= M - 10:
            new_M = M * 2
            counts = extend_M(counts, residues, M, new_M)
            M = new_M
            new_L = calc_L(counts, M)

        # Acceptance criterion
        if new_L >= current_L:
            current_L = new_L
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                stuck_counter = 0
        else:
            # Delta L is negative
            delta_L = new_L - current_L
            # Energy difference is -delta_L (we want to minimize energy, maximize L)
            # Accept with probability exp(delta_L / T)
            if random.random() < math.exp(delta_L / T):
                current_L = new_L
            else:
                # Revert
                revert_move(counts, i, r_old, r_new, p)
                residues[i] = r_old

        # Cool down
        T = max(T * cooling_rate, T_min)

        # If stuck for too long, reset temperature or do a larger perturbation
        stuck_counter += 1
        if stuck_counter > 1000:
            # Reset temperature to escape local minima
            T = 10.0
            stuck_counter = 0

    # Final evaluation to ensure correctness
    # Rebuild counts for best_residues to get exact L
    # We need to check up to a reasonable limit, say 10000 or M if larger
    # But M might be large. Let's just use the best_L we found, but verify it.
    # Actually, the task asks for the largest L. Our best_L is the best we found.
    # We should return the residues that achieved best_L.
    
    # One final check: ensure best_residues is valid
    # The evaluator will compute the score, so we just return the residues.
    
    return best_residues.tolist()

# EVOLVE-BLOCK-END
