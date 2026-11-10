import time
import random
import math
import numpy as np

N = 5001 # Range [0, 5000]
DEADLINE = 290.0

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(5003)
P_COUNT = len(PRIMES)

# Split primes into small and large
# Small primes have significant overlap potential
# Large primes are mostly independent
CUT_PRIME = 100 
small_primes = [p for p in PRIMES if p <= CUT_PRIME]
large_primes = [p for p in PRIMES if p > CUT_PRIME]

def construct():
    start_time = time.time()
    
    # Initialize residues for all primes
    residues = {}
    for p in PRIMES:
        residues[p] = 0
        
    # We will optimize small primes using SA, and large primes greedily.
    
    # Helper to compute the union size given residues
    def get_union_size(res_dict):
        # Use a boolean mask
        mask = np.zeros(N, dtype=bool)
        for p in PRIMES:
            r = res_dict[p]
            mask[r::p] = True
        return np.sum(mask)
        
    # Helper to compute marginal cost of changing residue for prime p
    # This is expensive if done naively.
    
    # Strategy:
    # 1. Optimize small primes.
    # 2. For large primes, pick best residue given current union.
    
    # Initial guess for small primes: Random? Or 0?
    # Let's start with 0 for all.
    for p in small_primes:
        residues[p] = 0
        
    # Calculate current union for small primes only?
    # No, large primes affect the total score, but their optimal choice depends on small primes.
    # So the objective for small primes should be:
    # Minimize (Union of Small) + (Sum of Marginal Costs of Large)
    
    # To speed up, we can approximate the cost of large primes.
    # Or just evaluate the full union for the SA steps.
    # N=5000, P~669.
    # Updating the mask for one prime change is O(N/p).
    # For small p, this is fast.
    
    # Let's use SA on small primes.
    
    # State: residues of small primes
    # Objective: Total covered elements
    
    # Precompute masks for small primes?
    # No, just update.
    
    current_mask = np.zeros(N, dtype=bool)
    for p in small_primes:
        current_mask[residues[p]::p] = True
        
    # For large primes, we will determine their best residue later.
    # But for the SA objective, we need to account for them.
    # Approximation: The cost of a large prime p is 1 unless it hits an already covered element.
    # Probability of hitting an already covered element is roughly |Current Union| / N.
    # This is a rough estimate.
    
    # Better: Just evaluate the full union for the SA objective.
    # It might be slow, but we have 290s.
    # Number of small primes ~ 25 (up to 100).
    # SA steps: 100,000?
    # Each step: change one residue, update mask, count sum.
    # Update mask: O(N/p). For p=2, 2500 ops.
    # Count sum: O(N) = 5000 ops.
    # Total ~ 7500 ops per step.
    # 100,000 steps * 7500 = 750,000,000 ops.
    # In Python, this is too slow.
    
    # Optimization:
    # Maintain the count of covered elements.
    # When changing residue for p from r_old to r_new:
    # 1. Remove r_old::p from mask.
    # 2. Add r_new::p to mask.
    # 3. The change in sum is (newly added) - (newly removed).
    # newly added: elements in r_new::p that were False in mask.
    # newly removed: elements in r_old::p that are True in mask AND not in r_new::p.
    # Since r_old != r_new, the sets are disjoint.
    # So:
    # delta = sum(mask[r_new::p] == False) - sum(mask[r_old::p] == True)
    # Wait, we must update mask.
    
    # Let's implement this efficient update.
    
    def calculate_delta(p, r_old, r_new, mask):
        # Elements in r_new::p that are not covered
        added = 0
        # Elements in r_old::p that are covered (and will be uncovered if not in r_new::p)
        # Since disjoint, all covered in r_old::p will be uncovered.
        removed = 0
        
        # Check added
        for x in range(r_new, N, p):
            if not mask[x]:
                added += 1
                
        # Check removed
        for x in range(r_old, N, p):
            if mask[x]:
                removed += 1
                
        return added - removed
        
    # This is still O(N/p).
    # For small p, this is fast.
    
    # Let's run SA on small primes.
    
    T = 10.0
    cooling = 0.999
    rng = np.random.default_rng()
    
    best_mask = current_mask.copy()
    best_score = N - np.sum(best_mask)
    best_small_res = residues.copy()
    
    steps = 0
    while time.time() - start_time < DEADLINE - 20:
        # Pick a random small prime
        idx = rng.integers(0, len(small_primes))
        p = small_primes[idx]
        old_r = residues[p]
        
        # Pick new residue
        new_r = rng.integers(0, p)
        while new_r == old_r:
            new_r = rng.integers(0, p)
            
        delta = calculate_delta(p, old_r, new_r, current_mask)
        
        # Accept
        if delta < 0 or (T > 0.01 and rng.random() < math.exp(-delta / T)):
            # Apply
            # Update mask
            current_mask[old_r::p] = False
            current_mask[new_r::p] = True
            residues[p] = new_r
            
            # Note: delta is the change in covered count.
            # We want to minimize covered count.
            # So if delta < 0, we improved.
            
            # Check if this is a new best for small primes?
            # We need to evaluate the full score including large primes to compare.
            # But full evaluation is expensive.
            # Let's just keep the best small primes config based on small union size?
            # No, that's not correct.
            
            # Let's do a full evaluation every K steps.
            steps += 1
            if steps % 100 == 0:
                # Evaluate full score
                full_mask = current_mask.copy()
                # Greedily assign large primes
                for lp in large_primes:
                    best_cost = N
                    best_r = 0
                    for r in range(lp):
                        # Cost is number of new elements
                        # new_elements = sum(full_mask[r::lp] == False)
                        # This is O(N/lp). lp is large, so this is small.
                        cost = np.sum(~full_mask[r::lp])
                        if cost < best_cost:
                            best_cost = cost
                            best_r = r
                    full_mask[best_r::lp] = True
                    residues[lp] = best_r
                    
                score = N - np.sum(full_mask)
                if score > best_score:
                    best_score = score
                    best_small_res = residues.copy()
                    best_mask = full_mask.copy() # Store full mask? No, just residues.
                    # Store best residues
                    best_residues = residues.copy()
                    
        T *= cooling
        
    # Final step: Determine large primes for the best small primes config
    # We need to reconstruct the best residues.
    # If we didn't find any improvement, use initial?
    # Let's ensure we have a best_residues.
    
    if 'best_residues' not in locals():
        best_residues = residues.copy()
        
    # Re-evaluate large primes for best_residues
    final_mask = np.zeros(N, dtype=bool)
    for p in small_primes:
        final_mask[best_residues[p]::p] = True
        
    for lp in large_primes:
        best_cost = N
        best_r = 0
        for r in range(lp):
            cost = np.sum(~final_mask[r::lp])
            if cost < best_cost:
                best_cost = cost
                best_r = r
        final_mask[best_r::lp] = True
        best_residues[lp] = best_r
        
    # Construct result
    result = [x for x in range(N) if not final_mask[x]]
    return result
