import numpy as np
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit+1, i):
                    sieve[j] = False
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    primes = get_primes(1000)
    num_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)

    # We will maintain a coverage count array.
    # The score is the largest L such that 1..L is doubly covered.
    # We start with a moderate M and extend if necessary.
    M = 4000  # Initial window size. 1..M is tracked.
    # counts[x] = number of primes p such that x % p == residues[i]
    counts = np.zeros(M + 1, dtype=np.int64)
    residues = np.zeros(num_primes, dtype=np.int64)

    def init_counts():
        nonlocal counts
        counts = np.zeros(M + 1, dtype=np.int64)
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                counts[p::p] += 1
            else:
                counts[r::p] += 1

    def extend_M(new_M):
        nonlocal M, counts
        old_M = M
        new_counts = np.zeros(new_M + 1, dtype=np.int64)
        new_counts[:old_M + 1] = counts
        for i, p in enumerate(primes):
            r = residues[i]
            # Find first index in [old_M+1, new_M] congruent to r mod p
            if r == 0:
                # Indices are p, 2p, ...
                # We need the first k*p such that k*p > old_M
                k = (old_M // p) + 1
                start_idx = k * p
            else:
                # Indices are r, r+p, ...
                # We need the first r + k*p such that r + k*p > old_M
                # k*p > old_M - r
                # k > (old_M - r) / p
                if old_M < r:
                    k = 0
                else:
                    k = (old_M - r) // p + 1
                start_idx = r + k * p
            
            if start_idx <= new_M:
                new_counts[start_idx::p] += 1
        
        M = new_M
        counts = new_counts

    def calc_L():
        # Find the largest L such that counts[1..L] >= 2
        # counts[0] is unused for the score (1..L)
        if counts[1] < 2:
            return 0
        # We want the first index >= 1 where count < 2
        # mask[i] is True if counts[i] < 2
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_fail_idx = np.argmax(mask)
        # first_fail_idx is relative to counts[1:], so actual index is first_fail_idx + 1
        # The value at that index fails, so L is (index - 1)
        return int(first_fail_idx)

    def apply_move(i, r_old, r_new):
        p = primes[i]
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

    def revert_move(i, r_old, r_new):
        # Revert means apply the move from r_new back to r_old
        apply_move(i, r_new, r_old)

    def repair_step(current_L):
        """
        Attempts to extend current_L by fixing the first integer that fails.
        Returns (new_L, improved)
        """
        if current_L >= M:
            # Need to extend M to see if we can go further
            extend_M(M * 2)
            return calc_L(), True # Just extending M might reveal more valid prefix if we were capped

        target_x = current_L + 1
        
        # We need to increase coverage of target_x to at least 2.
        # Currently coverage[target_x] < 2.
        # Candidates are primes i where (target_x - residues[i]) % p != 0.
        # Changing such a prime to cover target_x will increase coverage[target_x] by 1.
        # We must ensure that for all x in 1..current_L, coverage[x] remains >= 2.
        # Changing prime i from r_old to r_new:
        # - Decrements coverage[x] for x in 1..current_L where x % p == r_old.
        # - Increments coverage[x] for x in 1..current_L where x % p == r_new.
        # So we need:
        # 1. No x in 1..current_L has coverage[x] == 2 and x % p == r_old. (Otherwise it drops to 1)
        #    (If coverage[x] > 2, dropping to 3 is fine. If coverage[x] == 2, it drops to 1, which is bad.)
        
        candidates = []
        for i, p in enumerate(primes):
            if (target_x - residues[i]) % p != 0:
                candidates.append(i)
        
        if not candidates:
            return current_L, False

        random.shuffle(candidates)
        
        for idx in candidates:
            p = primes[idx]
            r_old = residues[idx]
            r_new = target_x % p
            
            if r_old == r_new:
                continue

            # Check validity:
            # Are there any x in 1..current_L such that x % p == r_old and counts[x] == 2?
            # If yes, this move is invalid.
            
            # Generate x values in 1..current_L congruent to r_old mod p
            if r_old == 0:
                # x = p, 2p, ...
                start_x = p
            else:
                # x = r_old, r_old+p, ...
                start_x = r_old
            
            if start_x > current_L:
                # No x in 1..current_L is affected negatively. Move is safe regarding negative impact.
                valid = True
            else:
                # Create array of x values
                # np.arange(start_x, current_L + 1, p)
                # Check if any of these have counts[x] == 2
                # To avoid creating a huge array if p is small, we can slice counts.
                # indices = np.arange(start_x, current_L + 1, p)
                # if np.any(counts[indices] == 2):
                #     valid = False
                # else:
                #     valid = True
                
                # Optimization:
                # We can just check the slice.
                # But np.arange is fine for current_L ~ few thousands.
                indices = np.arange(start_x, current_L + 1, p)
                if np.any(counts[indices] == 2):
                    continue # Invalid move, skip this candidate

            # Move is valid
            apply_move(idx, r_old, r_new)
            residues[idx] = r_new
            
            new_L = calc_L()
            
            # If we extended M, we might need to update
            # calc_L handles the cap. If new_L == M, it means we hit the boundary.
            # We should extend M if we are close to it.
            if new_L >= M - 10:
                extend_M(M * 2)
                new_L = calc_L()
            
            return new_L, True
        
        return current_L, False

    # Initial random residues
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
    
    init_counts()
    current_L = calc_L()
    best_L = current_L
    best_residues = residues.copy()

    # MCMC parameters
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.99995
    stuck_counter = 0
    MAX_STUCK = 200

    # Main loop
    while time.time() < deadline:
        # 1. Try to repair the gap
        new_L, improved = repair_step(current_L)
        
        if improved:
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
            current_L = new_L
            stuck_counter = 0
            continue

        # 2. If repair failed, try a random MCMC move
        i = random.randint(0, num_primes - 1)
        p = primes[i]
        r_old = residues[i]
        r_new = random.randint(0, p - 1)

        if r_new == r_old:
            continue

        apply_move(i, r_old, r_new)
        residues[i] = r_new

        new_L = calc_L()
        
        # Extend M if needed
        if new_L >= M - 10:
            extend_M(M * 2)
            new_L = calc_L()

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
            # Accept with probability exp(delta_L / T)
            # delta_L is integer, T is float
            if random.random() < math.exp(delta_L / T):
                current_L = new_L
            else:
                # Revert
                revert_move(i, r_old, r_new)
                residues[i] = r_old

        # Cool down
        T = max(T * cooling_rate, T_min)

        # Stuck handling
        stuck_counter += 1
        if stuck_counter > MAX_STUCK:
            # Perturb: change a few random residues
            num_perturb = random.randint(5, 15)
            for _ in range(num_perturb):
                j = random.randint(0, num_primes - 1)
                pj = primes[j]
                r_old_j = residues[j]
                r_new_j = random.randint(0, pj - 1)
                if r_new_j != r_old_j:
                    apply_move(j, r_old_j, r_new_j)
                    residues[j] = r_new_j
            
            current_L = calc_L()
            if current_L < best_L:
                # Revert to best if perturbation was bad
                # We need to revert the perturbations. This is tricky with just counts.
                # Easier: just reset residues to best and re-init counts.
                residues = best_residues.copy()
                init_counts()
                current_L = calc_L()
            
            T = 5.0
            stuck_counter = 0

    # Return best residues found
    return [int(r) for r in best_residues]
