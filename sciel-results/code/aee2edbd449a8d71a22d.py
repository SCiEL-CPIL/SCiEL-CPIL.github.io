import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Precompute for each prime i: list of x in [1, MAX_RANGE] such that x % p == r
    # We'll work with a fixed MAX_RANGE for scoring, but the true score is unbounded (capped at 10^5)
    # However, L is likely small. Let's use a generous range.
    MAX_RANGE = 5000
    
    # Fast scoring function using numpy for vectorized coverage counting
    # For each prime i with residue r_i, the covered positions in [1, MAX_RANGE] are:
    #   r_i, r_i + p_i, r_i + 2*p_i, ... (if r_i > 0)
    #   p_i, 2*p_i, ... (if r_i == 0)
    
    def compute_score_numpy(residues):
        """Compute max L such that 1..L have coverage >= 2."""
        # Build coverage count array
        # For each prime, add 1 to positions x where x % p == r
        # We'll use a list of arrays or just a single count array
        
        # Actually, let's use a different approach for speed:
        # For each x in 1..MAX_RANGE, count how many primes cover it.
        # But that's O(MAX_RANGE * n_primes) = 5000 * 168 = 840000, which is fine.
        
        # Even better: for each prime, generate its covered positions and increment.
        # Total operations: sum(MAX_RANGE / p_i) for all i ≈ MAX_RANGE * sum(1/p_i) ≈ 5000 * 2.5 = 12500
        
        cover_count = [0] * (MAX_RANGE + 1)
        
        for i, p in enumerate(primes):
            r = residues[i]
            if r == 0:
                start_x = p
            else:
                start_x = r
            
            if start_x > MAX_RANGE:
                continue
            
            for x in range(start_x, MAX_RANGE + 1, p):
                cover_count[x] += 1
        
        # Find max L
        L = 0
        for x in range(1, MAX_RANGE + 1):
            if cover_count[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Strategy: Greedy construction with iterative improvement
    # Phase 1: Greedy assignment - for each prime in order, choose residue that maximizes coverage of current frontier
    # Phase 2: Local search with simulated annealing to escape local optima
    
    # Initialize with a randomized start
    random.seed(42)
    
    def random_init():
        return [random.randint(0, p - 1) for p in primes]
    
    # Greedy initialization: process primes in order, for each choose residue that best extends coverage
    def greedy_init():
        residues = [0] * n_primes
        # Start with empty coverage
        # For each prime, try all residues and pick the one that gives best score
        # This is O(n_primes * p_avg * score_cost) which might be slow
        # Let's just use random init and improve
        
        return random_init()
    
    # Main optimization loop
    best_residues = None
    best_score = 0
    
    # Try multiple random restarts with hill climbing
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        residues = random_init()
        current_score = compute_score_numpy(residues)
        
        # Simulated annealing / hill climbing
        temp = 50.0
        cooling = 0.999
        no_improve = 0
        max_no_improve = 2000
        
        while time.time() < deadline and no_improve < max_no_improve:
            # Pick a random prime to modify
            i = random.randint(0, n_primes - 1)
            old_r = residues[i]
            new_r = random.randint(0, primes[i] - 1)
            if new_r == old_r:
                continue
            
            residues[i] = new_r
            new_score = compute_score_numpy(residues)
            
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
                no_improve = 0
            else:
                # Accept with probability exp(delta/temp)
                if temp > 1e-6 and random.random() < math.exp(delta / temp):
                    current_score = new_score
                    no_improve += 1
                else:
                    residues[i] = old_r
                    no_improve += 1
            
            temp *= cooling
            
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
        
        # If we found a good score, do a focused local search
        if best_score > 100:
            # Try to extend further with targeted moves
            residues = best_residues[:]
            current_score = best_score
            temp = 10.0
            cooling = 0.998
            no_improve = 0
            max_no_improve = 5000
            
            while time.time() < deadline and no_improve < max_no_improve:
                # Targeted: find first gap and try to fix it
                # Compute coverage to find first gap
                cover_count = [0] * (MAX_RANGE + 1)
                for i, p in enumerate(primes):
                    r = residues[i]
                    start_x = p if r == 0 else r
                    if start_x <= MAX_RANGE:
                        for x in range(start_x, MAX_RANGE + 1, p):
                            cover_count[x] += 1
                
                gap_x = None
                for x in range(1, MAX_RANGE + 1):
                    if cover_count[x] < 2:
                        gap_x = x
                        break
                
                if gap_x is None or gap_x > MAX_RANGE:
                    break
                
                # Try to fix gap_x: find primes that don't currently cover gap_x
                # and change one to cover it
                candidates = []
                for j, p in enumerate(primes):
                    req_r = gap_x % p
                    if residues[j] != req_r:
                        candidates.append(j)
                
                if candidates:
                    j = random.choice(candidates)
                    old_r_j = residues[j]
                    residues[j] = gap_x % primes[j]
                    test_score = compute_score_numpy(residues)
                    delta = test_score - current_score
                    if delta > 0:
                        current_score = test_score
                        no_improve = 0
                    else:
                        if temp > 1e-6 and random.random() < math.exp(delta / temp):
                            current_score = test_score
                            no_improve += 1
                        else:
                            residues[j] = old_r_j
                            no_improve += 1
                else:
                    no_improve += 1
                
                temp *= cooling
                
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
        
        # Check if we should continue with more restarts
        if best_score >= 1000:
            break
    
    if best_residues is None:
        best_residues = random_init()
    
    return best_residues
