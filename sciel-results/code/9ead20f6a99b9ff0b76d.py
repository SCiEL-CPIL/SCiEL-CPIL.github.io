import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = 118.0  # Safety margin for 120s limit

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
    
    # We will search for L up to M. 
    # Initial M can be smaller, we expand if needed.
    M = 2000
    counts = np.zeros(M + 1, dtype=np.int64)
    
    # Initialize residues
    # Use a greedy initialization: for each prime, pick residue that covers the smallest currently "weak" number
    # But simpler: Random init is often fine if local search is strong.
    # Let's try a slightly smarter init: 
    # For small primes, they cover many numbers. 
    # We'll just use random init for simplicity and rely on the repair loop.
    residues = np.zeros(n_primes, dtype=np.int64)
    for i, p in enumerate(primes):
        residues[i] = random.randint(0, p - 1)
        counts[residues[i]::p] += 1
        
    def calc_L(counts, M):
        # Returns the largest L such that counts[1..L] >= 2
        if counts[1] < 2:
            return 0
        # Find first index >= 1 where count < 2
        mask = counts[1:] < 2
        if not np.any(mask):
            return M
        first_false_idx = np.argmax(mask)
        return first_false_idx

    current_L = calc_L(counts, M)
    best_L = current_L
    best_residues = residues.copy()
    
    # Helper to check if a move is valid (doesn't break prefix [1, L])
    # move: change residue of prime i from r_old to r_new
    # It breaks coverage if any y in [1, L] with y % p == r_old has count[y] == 2
    # Because count[y] will decrease by 1.
    def is_valid_move(counts, i, r_old, r_new, L, p):
        if r_old == r_new:
            return True
        # Check if any y in [1, L] congruent to r_old mod p has count 2
        # y = r_old + k*p
        # Start from first y >= 1
        start_y = r_old
        if start_y == 0:
            start_y = p
        # Iterate y in [start_y, L] step p
        for y in range(start_y, L + 1, p):
            if counts[y] == 2:
                return False
        return True

    # Helper to apply move
    def apply_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_old::p] -= 1
            counts[r_new::p] += 1

    # Helper to revert move
    def revert_move(counts, i, r_old, r_new, p):
        if r_old != r_new:
            counts[r_new::p] -= 1
            counts[r_old::p] += 1

    # Optimization Loop
    # We alternate between "Repair" (targeted moves to fix the gap) and "Perturb" (random moves to escape local minima)
    
    T = 10.0
    T_min = 0.1
    cooling_rate = 0.9995
    
    stuck_counter = 0
    
    while time.time() - start_time < deadline:
        # 1. Try to repair the gap
        # The gap is at current_L + 1
        gap_x = current_L + 1
        
        found_valid_move = False
        
        # Shuffle prime indices to avoid bias
        prime_indices = list(range(n_primes))
        random.shuffle(prime_indices)
        
        for i in prime_indices:
            p = primes[i]
            r_old = residues[i]
            
            # If this prime already covers the gap, it can't help by changing to cover the gap (it's already there)
            # Unless we change it to something else? No, we want to ADD coverage to gap_x.
            # If r_old == gap_x % p, it already covers it. Changing it would REMOVE coverage.
            # So we only consider primes that DO NOT cover gap_x.
            if r_old == gap_x % p:
                continue
                
            r_new = gap_x % p
            
            # Check validity: does this move break coverage for [1, current_L]?
            if is_valid_move(counts, i, r_old, r_new, current_L, p):
                # Apply move
                apply_move(counts, i, r_old, r_new, p)
                residues[i] = r_new
                
                # Recalculate L
                new_L = calc_L(counts, M)
                
                # Since we fixed the gap, new_L should be at least current_L + 1
                # But it might break later? No, is_valid_move ensured [1, current_L] is safe.
                # So new_L >= current_L + 1.
                
                if new_L > best_L:
                    best_L = new_L
                    best_residues = residues.copy()
                    
                    # If we hit the boundary M, extend
                    if new_L >= M - 10:
                        new_M = M * 2
                        old_counts = counts
                        counts = np.zeros(new_M + 1, dtype=np.int64)
                        # Rebuild counts for new range
                        # Copy old part
                        counts[:M+1] = old_counts
                        # Add contributions for new part
                        for j, pj in enumerate(primes):
                            rj = residues[j]
                            # Find first index in [M+1, new_M] congruent to rj mod pj
                            # k = rj + m*pj
                            # We need k >= M+1
                            # m >= (M+1 - rj) / pj
                            start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                            first_idx = rj + start_m * pj
                            if first_idx <= new_M:
                                counts[first_idx::pj] += 1
                        M = new_M
                        # Recalculate current_L
                        current_L = calc_L(counts, M)
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues.copy()
                else:
                    current_L = new_L
                
                found_valid_move = True
                stuck_counter = 0
                break # Restart the loop to find the new gap
        
        if found_valid_move:
            continue
            
        # 2. No valid repair move found. Use Simulated Annealing to escape.
        # Pick a random prime
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        r_old = residues[i]
        
        # Random new residue
        r_new = random.randint(0, p - 1)
        
        if r_new == r_old:
            continue
            
        # Apply move
        apply_move(counts, i, r_old, r_new, p)
        
        new_L = calc_L(counts, M)
        
        if new_L >= current_L:
            current_L = new_L
            residues[i] = r_new
            if new_L > best_L:
                best_L = new_L
                best_residues = residues.copy()
                # Extend M if needed
                if best_L >= M - 10:
                    new_M = M * 2
                    old_counts = counts
                    counts = np.zeros(new_M + 1, dtype=np.int64)
                    counts[:M+1] = old_counts
                    for j, pj in enumerate(primes):
                        rj = residues[j]
                        start_m = max(0, (M + 1 - rj + pj - 1) // pj)
                        first_idx = rj + start_m * pj
                        if first_idx <= new_M:
                            counts[first_idx::pj] += 1
                    M = new_M
                    current_L = calc_L(counts, M)
                    if current_L > best_L:
                        best_L = current_L
                        best_residues = residues.copy()
            stuck_counter = 0
        else:
            # SA acceptance
            delta = current_L - new_L
            prob = np.exp(-delta / T)
            if random.random() < prob:
                current_L = new_L
                residues[i] = r_new
                stuck_counter = 0
            else:
                # Revert
                revert_move(counts, i, r_old, r_new, p)
                stuck_counter += 1
            
        # Cool down
        T = max(T_min, T * cooling_rate)
        
        # If stuck for a long time, restart with a new random seed? 
        # Or just let SA handle it.
        # Given the time limit, we don't restart explicitly, but SA with random moves should explore.

    return [int(r) for r in best_residues]
