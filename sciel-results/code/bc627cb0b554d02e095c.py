import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START
def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    primes = _sieve(1000)
    P = len(primes)
    
    # Precompute reciprocals as Fractions for exact budget check
    # We need sum(1/p) <= 1
    # To speed up, we can also use floats for quick rejection in SA moves, 
    # but final check must be exact.
    recip_roots = [Fraction(1, p) for p in primes]
    recip_floats = [1.0 / p for p in primes]
    
    # Initial state: Random selection of a subset of primes and residues
    # We want to maximize L.
    # A good starting point is often a greedy packing.
    
    def evaluate(residues, active_mask):
        """
        residues: list of length P, residues[i] is the residue for prime i (if active)
        active_mask: list of bools, whether prime i is used
        Returns: (L, sum_frac, sum_float)
        """
        # Check budget
        # Optimization: Only sum active ones
        # We can maintain running sums in the main loop for speed, 
        # but for simplicity in initial implementation, we compute.
        # Given P~168, this is fast.
        
        # However, checking coverage is the expensive part if done naively for large L.
        # We will check coverage up to a limit.
        
        # First, check budget exactly
        s_frac = Fraction(0)
        s_float = 0.0
        for i in range(P):
            if active_mask[i]:
                s_frac += recip_roots[i]
                s_float += recip_floats[i]
                if s_float > 1.0 + 1e-9: # Quick float check
                    break
        
        if s_float > 1.0 + 1e-9:
            return 0, s_frac, s_float
            
        # If float says > 1, it might be a false positive due to precision? 
        # Unlikely for sum of ~100 terms. But let's be safe.
        # If float <= 1.0, we MUST verify with Fraction.
        if s_frac > 1:
            return 0, s_frac, s_float
        
        # Check coverage
        # We want the largest L such that 1..L is covered.
        # Max possible L is capped at 10^5 by task, but practically limited by primes.
        # If we use all primes, the "density" is ~1.
        # Let's check up to 10000 first, then expand if needed?
        # Actually, just check up to 100000? 
        # 168 primes * 100000 checks = 16.8M ops. Too slow for SA steps.
        
        # Optimized Coverage Check:
        # Use a boolean array or set of covered numbers?
        # Better: For each n in 1..L, check if any (p, r) covers it.
        # n % p == r.
        
        # To make this fast, we can precompute for each prime p, the list of 
        # covered numbers in a range? No, residues change.
        
        # Alternative: 
        # Iterate n from 1 upwards. 
        # For each n, check all active primes.
        # Stop when we find a gap.
        
        # Heuristic: The gap will appear early if coverage is poor.
        # If coverage is good (L > 100), it takes longer.
        
        # Let's limit the check range to 10000 for speed during SA.
        # If we find L > 10000, we can do a deeper check at the end or periodically.
        # Actually, the seed gets ~900. Inspiration 1 gets ~1800.
        # So L is small. Checking up to 5000 is likely sufficient for ranking.
        
        max_check = 5000
        
        # active_primes: list of (p, r)
        active_primes = [(primes[i], residues[i]) for i in range(P) if active_mask[i]]
        
        if not active_primes:
            return 0, s_frac, s_float
            
        # Sort by p? Doesn't matter for correctness, maybe for speed if we break early?
        # No, we must check all.
        
        covered = 0
        for n in range(1, max_check + 1):
            is_cov = False
            for p, r in active_primes:
                if n % p == r:
                    is_cov = True
                    break
            if is_cov:
                covered = n
            else:
                break
                
        return covered, s_frac, s_float

    # Simulated Annealing
    
    # Initialize
    # Start with a random subset of primes that fits the budget
    # Greedy: pick smallest primes first? No, they have largest weight.
    # Pick largest primes first to fit more?
    # Let's start with a random valid set.
    
    active_mask = [False] * P
    residues = [0] * P
    current_sum = Fraction(0)
    
    # Try to add primes in random order, if they fit
    indices = list(range(P))
    random.shuffle(indices)
    
    # To ensure a decent start, let's try to pack it.
    # Sort by weight descending? No, we want to maximize coverage.
    # Small primes are better for coverage (hit more numbers).
    # So we prefer small primes.
    # Let's try to add small primes first if they fit.
    
    for i in sorted(indices, key=lambda x: primes[x]):
        if current_sum + recip_roots[i] <= 1:
            active_mask[i] = True
            current_sum += recip_roots[i]
            residues[i] = random.randint(0, primes[i]-1)
    
    # If this fails to add many, we might have a bad start.
    # Let's force a specific good start: The seed strategy is a good baseline.
    # Seed: i-th prime from 13 upward covers i.
    # Let's try to replicate a strong heuristic start.
    
    # Actually, let's just run SA from this start.
    
    current_L, _, _ = evaluate(residues, active_mask)
    best_L = current_L
    best_residues = residues[:]
    best_mask = active_mask[:]
    best_sum = current_sum
    
    T = 100.0 # Initial temperature
    cooling = 0.999
    
    while time.time() < deadline and T > 0.1:
        
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Change residue of an active prime
            active_indices = [i for i in range(P) if active_mask[i]]
            if not active_indices: continue
            i = random.choice(active_indices)
            old_r = residues[i]
            residues[i] = random.randint(0, primes[i]-1)
            new_L, _, _ = evaluate(residues, active_mask)
            delta = new_L - current_L
            if delta > 0 or random.random() < pow(2.718, delta / T):
                current_L = new_L
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues[:]
                    best_mask = active_mask[:]
                    best_sum = current_sum
            else:
                residues[i] = old_r
                
        elif move_type < 0.6:
            # Toggle a prime (add or remove)
            # Try to add a random inactive prime
            inactive_indices = [i for i in range(P) if not active_mask[i]]
            if inactive_indices:
                i = random.choice(inactive_indices)
                if current_sum + recip_roots[i] <= 1:
                    active_mask[i] = True
                    current_sum += recip_roots[i]
                    residues[i] = random.randint(0, primes[i]-1)
                    new_L, _, _ = evaluate(residues, active_mask)
                    delta = new_L - current_L
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues[:]
                            best_mask = active_mask[:]
                            best_sum = current_sum
                    else:
                        active_mask[i] = False
                        current_sum -= recip_roots[i]
                # else: can't add, skip move (or penalize? just skip)
            else:
                # All active, try to remove one?
                active_indices = [i for i in range(P) if active_mask[i]]
                if active_indices:
                    i = random.choice(active_indices)
                    active_mask[i] = False
                    current_sum -= recip_roots[i]
                    new_L, _, _ = evaluate(residues, active_mask)
                    delta = new_L - current_L
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues[:]
                            best_mask = active_mask[:]
                            best_sum = current_sum
                    else:
                        active_mask[i] = True
                        current_sum += recip_roots[i]
                        
        else:
            # Swap: Remove one active, add one inactive
            active_indices = [i for i in range(P) if active_mask[i]]
            inactive_indices = [i for i in range(P) if not active_mask[i]]
            if active_indices and inactive_indices:
                i_out = random.choice(active_indices)
                i_in = random.choice(inactive_indices)
                # Check budget
                if current_sum - recip_roots[i_out] + recip_roots[i_in] <= 1:
                    active_mask[i_out] = False
                    current_sum -= recip_roots[i_out]
                    active_mask[i_in] = True
                    current_sum += recip_roots[i_in]
                    residues[i_in] = random.randint(0, primes[i_in]-1)
                    new_L, _, _ = evaluate(residues, active_mask)
                    delta = new_L - current_L
                    if delta > 0 or random.random() < pow(2.718, delta / T):
                        current_L = new_L
                        if current_L > best_L:
                            best_L = current_L
                            best_residues = residues[:]
                            best_mask = active_mask[:]
                            best_sum = current_sum
                    else:
                        # Revert
                        active_mask[i_out] = True
                        current_sum += recip_roots[i_out]
                        active_mask[i_in] = False
                        current_sum -= recip_roots[i_in]
                        
        T *= cooling
        
    # Final verification of best solution
    # We need to return the classes for the best solution found.
    # Ensure exact budget check.
    final_sum = Fraction(0)
    for i in range(P):
        if best_mask[i]:
            final_sum += recip_roots[i]
            
    if final_sum > 1:
        # Fallback to seed if SA failed to maintain budget (shouldn't happen)
        primes_seed = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes_seed)]
        
    result = []
    for i in range(P):
        if best_mask[i]:
            result.append([primes[i], best_residues[i]])
            
    return result
# EVOLVE-BLOCK-END
