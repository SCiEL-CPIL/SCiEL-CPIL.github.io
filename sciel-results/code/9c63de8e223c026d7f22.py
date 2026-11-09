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
    n = len(primes)  # 168 primes

    # Precompute for fast evaluation
    # For each prime index i, we need to check x % primes[i] == residues[i]
    # We'll use a vectorized approach: for a given L, check all x in 1..L
    # For each x, count how many congruences it satisfies
    
    def evaluate(residues, L_max=100000):
        """Return the largest L such that every integer 1..L satisfies at least 2 congruences."""
        residues = np.array(residues)
        primes_arr = np.array(primes)
        
        # We'll check incrementally. Start with a reasonable L and adjust.
        # First, let's do a quick check at small L to see if we even get L>=2
        # For efficiency, we'll compute coverage for a range of x values
        
        # Vectorized check for x in 1..L_max:
        # For each prime p_i with residue r_i, the x values satisfying x ≡ r_i (mod p_i)
        # in 1..L_max are: r_i, r_i + p_i, r_i + 2*p_i, ... <= L_max
        # (if r_i == 0, then x = p_i, 2*p_i, ... <= L_max)
        
        # Build a coverage array
        coverage = np.zeros(L_max + 1, dtype=np.int32)
        for i in range(n):
            p = primes_arr[i]
            r = residues[i]
            if r == 0:
                start = p
            else:
                start = r
            if start > L_max:
                continue
            # x = start, start + p, start + 2p, ... <= L_max
            idx = np.arange(start, L_max + 1, p)
            coverage[idx] += 1
        
        # Find the largest L such that coverage[1..L] >= 2
        # coverage[x] >= 2 for all x in 1..L
        valid = coverage[1:] >= 2
        if not valid[0]:
            return 0
        # Find the first position where valid becomes False
        false_positions = np.where(~valid)[0]
        if len(false_positions) == 0:
            return L_max
        return false_positions[0]

    # Start from the seed solution
    def seed_solution():
        return [((i + 2) // 2) % p for i, p in enumerate(primes)]
    
    best_residues = seed_solution()
    best_score = evaluate(best_residues)
    
    # Hill-climbing with random restarts
    random.seed(42)
    
    def random_solution():
        return [random.randint(0, p - 1) for p in primes]
    
    def hill_climb(residues, max_iters=50000):
        """Local search: try changing one residue at a time, keep if improves."""
        current = list(residues)
        current_score = evaluate(current)
        
        # Precompute which indices to try
        order = list(range(n))
        random.shuffle(order)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            random.shuffle(order)
            for i in order:
                if time.time() > deadline - 5:
                    break
                old_val = current[i]
                p = primes[i]
                # Try a random new value
                if p <= 10:
                    # Try all values for small primes
                    for new_val in range(p):
                        if new_val == old_val:
                            continue
                        current[i] = new_val
                        new_score = evaluate(current)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            break
                    else:
                        current[i] = old_val
                else:
                    # Try a few random values
                    for _ in range(3):
                        new_val = random.randint(0, p - 1)
                        if new_val == old_val:
                            continue
                        current[i] = new_val
                        new_score = evaluate(current)
                        if new_score > current_score:
                            current_score = new_score
                            improved = True
                            break
                    else:
                        current[i] = old_val
                        # Also try a targeted change: set residue to cover a specific uncovered point
                        # Find the first x where coverage < 2
                        # This is more complex, so skip for now
        return current, current_score

    # Main search loop
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        if num_restarts == 1:
            # Start from seed
            current, current_score = hill_climb(best_residues)
        else:
            # Random restart
            current = random_solution()
            current, current_score = hill_climb(current)
        
        if current_score > best_score:
            best_score = current_score
            best_residues = current
        
        # If we've reached a high score, try more targeted improvements
        if best_score >= 200:
            # Try more iterations with the best solution
            if time.time() < deadline - 10:
                improved, new_score = hill_climb(best_residues, max_iters=100000)
                if new_score > best_score:
                    best_score = new_score
                    best_residues = improved

    return best_residues
# EVOLVE-BLOCK-END
