import time
import random
import numpy as np
from fractions import Fraction

# Precompute primes up to 1000
def _get_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, is_prime in enumerate(sieve) if is_prime]

PRIMES = _get_primes(1000)
# We can only use primes such that sum(1/p) <= 1. 
# The sum of reciprocals of all primes up to 1000 is approx 2.2, so we must select a subset.
# However, the problem statement says "sum of 1/p at most 1".
# Wait, the seed program uses primes from 13 upwards. Sum of 1/p for p>=13 up to 1000 is approx 0.92.
# If we include smaller primes (2, 3, 5, 7), the sum exceeds 1 quickly?
# 1/2 + 1/3 + 1/5 + 1/7 + 1/11 + 1/13 ...
# 0.5 + 0.333 + 0.2 = 1.033 > 1. So we cannot use {2,3,5}.
# We need a subset of primes P <= 1000 such that sum_{p in P} 1/p <= 1.
# To maximize coverage, we likely want small primes because they have smaller residues (denser coverage per unit cost? No, 1/p is the cost).
# Actually, cost is 1/p. Smaller p means higher cost.
# Larger p means lower cost, but the residue class covers 1/p of the integers.
# The "density" of coverage provided by a class (p, r) is exactly 1/p.
# So the total density of coverage is sum 1/p.
# If sum 1/p <= 1, we can potentially cover up to 100% of integers.
# The challenge is to arrange residues to avoid gaps.

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    # Target coverage length
    # The task says L is capped at 10^5, but "L >= 999 would realize... at x=1000".
    # The seed covers 1..L where L is the largest such that 1..L are covered.
    # The seed program returns classes for primes 13..997.
    # Let's aim for a large L. If we cover 1..10000, score is 10000.
    # The seed score is 1.0, which corresponds to L=999? Or is the score normalized?
    # "The score is the largest L... Maximize".
    # "search score is value/seed". Seed value is likely the L from the seed program.
    # Seed program: primes from 13 to 997.
    # Let's estimate seed L.
    # It maps i-th prime (starting 13) to cover i.
    # p_0=13 covers 1. p_1=17 covers 2. ...
    # This is a very specific construction.
    # Let's just maximize L.
    
    # We need to select a subset of primes and residues.
    # Strategy: 
    # 1. Start with a greedy set of primes that fits in budget 1.
    # 2. Use Simulated Annealing to optimize residues and potentially swap primes.
    
    # Budget: sum(1/p) <= 1.
    # Let's precompute reciprocals as Fractions for exactness, but use floats for SA speed, 
    # and verify with Fraction at the end.
    
    # Initial Selection: 
    # Greedy: pick primes with smallest 1/p (largest primes) to fit many? 
    # No, we want to cover small integers. Small integers are hard to cover if we only use large primes?
    # Actually, any residue class (p, r) covers {r, r+p, r+2p, ...}.
    # To cover 1, we need a prime p and r=1. Cost 1/p.
    # To cover 2, we need a prime p and r=2 (or r=1 if p=2, but p=2 is expensive).
    
    # Let's define the state:
    # selected_primes: list of (p, r)
    # We want to maximize L such that 1..L are covered.
    
    # To make SA efficient, we need to evaluate "new coverage" quickly.
    # We will track a boolean array `covered` of size MAX_L (e.g., 20000).
    
    MAX_L = 20000
    covered = np.zeros(MAX_L, dtype=bool)
    
    # Initial state: Greedy selection
    # Sort primes by 1/p (ascending cost, i.e., descending p) ? 
    # Or sort by potential to cover small numbers?
    # Let's start with a set of primes that sums to ~1.
    # 1/2 + 1/3 + 1/5 = 1.033 (Too high)
    # 1/2 + 1/3 + 1/7 + 1/11 ...
    # Let's just pick a random valid set initially?
    # Better: Start with the seed's set of primes (13..997) if it's valid?
    # Seed sum: sum(1/p for p in primes[13:1000]) ~ 0.92. Valid.
    # But maybe we can do better by swapping in smaller primes if we drop some large ones?
    # Small primes are "expensive" but cover "dense" patterns.
    
    # Let's try to include 2, 3, 5, 7?
    # 1/2+1/3+1/5 = 1.033. Too much.
    # Drop 5? 1/2+1/3 = 0.833. Room for 1/7 (0.142) -> 0.975. Room for 1/11 (0.09) -> 1.065 (No).
    # So {2, 3, 7, 11, 13, ...} might work.
    
    # Let's initialize with a greedy approach:
    # Sort primes by 1/p (cost). 
    # Actually, let's just use a random valid set of ~20-30 primes.
    
    # Helper to check budget
    def calc_budget(primes_list):
        return sum(1.0/p for p in primes_list)
    
    # Let's start with the primes from the seed, but randomize residues.
    # Seed primes: 13, 17, 19, ...
    # This set has sum ~0.92.
    # We can try to swap out large primes for smaller ones if budget allows.
    
    # Let's build an initial set:
    # Try to include 2, 3. Cost 0.833. Remaining 0.166.
    # 1/7 = 0.142. Remaining 0.024.
    # 1/43 = 0.023. 
    # So {2, 3, 7, 43} is a valid set. Sum ~ 0.999.
    # But 4 primes is too few to cover 1..10000 densely?
    # With 4 primes, the number of classes is 4. The "holes" will be large.
    # We need many classes to fill gaps.
    # So we need many small-cost primes (large primes).
    # The seed uses ~150 primes.
    
    # Strategy:
    # 1. Start with a set of large primes that fits in budget 1.
    # 2. Optimize residues using SA to maximize contiguous coverage from 1.
    
    # Let's generate a set of primes that sums to <= 1.
    # Sort primes by value (descending) to pick smallest costs first.
    # This maximizes the number of classes.
    # Primes <= 1000.
    # Sum of 1/p for p > 100 is small.
    # Let's pick primes from 1000 down to P_min such that sum <= 1.
    
    sorted_primes_desc = sorted(PRIMES, reverse=True)
    initial_primes = []
    current_sum = 0.0
    for p in sorted_primes_desc:
        if current_sum + 1.0/p <= 1.0:
            initial_primes.append(p)
            current_sum += 1.0/p
        else:
            # Try smaller primes? No, we want max count, so we stick to largest primes.
            # But maybe we should skip some large ones to fit a medium one?
            # For now, greedy max count is a good start.
            pass
            
    # This will give us a set of primes, mostly large.
    # Let's see how many.
    # Sum 1/p for p in [1000, 997, ...]
    # Approx integral of 1/(x log x) dx from X to 1000.
    # ln(ln(1000)) - ln(ln(X)) ~ 1.
    # ln(1000) ~ 6.9. ln(6.9) ~ 1.9.
    # We need diff 1. ln(ln(X)) ~ 0.9. ln(X) ~ 2.5. X ~ 12.
    # So primes from 12 to 1000 sum to ~1.
    # This matches the seed (13 to 997).
    
    # So the seed's prime set is already near-optimal in terms of count/density.
    # The improvement must come from RESIDUE OPTIMIZATION.
    
    # State: list of (p, r)
    # Initialize residues randomly
    state = [(p, random.randint(0, p-1)) for p in initial_primes]
    
    # Precompute masks for each prime?
    # For each prime p, and residue r, the covered indices are r, r+p, ...
    # We can precompute for each prime p a list of offsets? No, p varies.
    # We can precompute for each prime p, a boolean mask of length MAX_L?
    # That would be 168 primes * 20000 booleans = 3.3M booleans. Manageable.
    # But residue changes the mask.
    # Mask for (p, r) is just shifted mask for (p, 0).
    # Let's precompute base masks for each prime in initial_primes.
    
    # However, if we swap primes, we need new masks.
    # Let's keep the prime set fixed during SA, only optimize residues.
    # Then maybe do a second phase to swap primes.
    
    # Phase 1: Optimize residues for fixed prime set.
    
    # Precompute base masks
    # mask[p_idx][r] is not possible to store all r.
    # But we can compute coverage incrementally.
    
    # Let's use a different representation for speed.
    # `covered` array.
    # To evaluate a move (change r for prime i from r_old to r_new):
    # 1. Remove coverage of (p_i, r_old).
    # 2. Add coverage of (p_i, r_new).
    # 3. Calculate new L (largest prefix of 1..L covered).
    
    # Removing/Adding is O(MAX_L / p_i) operations on the array?
    # No, we can use numpy slicing.
    # indices = np.arange(r_old, MAX_L, p_i)
    # covered[indices] = False (if we assume exclusive ownership? No, multiple primes can cover same int)
    # We need a count array! `cover_count`
    
    cover_count = np.zeros(MAX_L, dtype=np.int8)
    
    # Initialize cover_count
    for p, r in state:
        indices = np.arange(r, MAX_L, p)
        cover_count[indices] += 1
        
    # Function to get L from cover_count
    def get_L(counts):
        # Find first 0 in counts[1:] (index 1 corresponds to integer 1)
        # If all are > 0, L = MAX_L
        # Use np.nonzero
        zeros = np.nonzero(counts[1:] == 0)[0]
        if len(zeros) == 0:
            return MAX_L
        return zeros[0] + 1 # Because we started at index 1
        
    current_L = get_L(cover_count)
    best_L = current_L
    best_state = [tuple(s) for s in state]
    
    # SA Parameters
    T = 500.0
    T_min = 1.0
    cooling = 0.999
    
    # Precompute indices for each prime to speed up updates?
    # For a given p, the sets of indices for each r are disjoint and partition 0..MAX_L-1 (almost).
    # Actually, for a fixed p, the indices for r are r, r+p, ...
    # We can precompute a list of arrays `indices_by_prime_residue`?
    # That's too much memory.
    # But we can precompute `base_indices` for p=0? No.
    
    # Let's just use np.arange in the loop. It's fast enough for 150 primes * 10000 steps?
    # 150 * 10000 = 1.5M np.arange calls. Might be slow.
    
    # Optimization: 
    # Instead of updating the whole array, we can just track the change in L?
    # Hard.
    
    # Let's try to optimize the update.
    # We will store for each prime index i, the current residue r_i.
    # When changing r_i:
    # old_indices = np.arange(r_old, MAX_L, p)
    # new_indices = np.arange(r_new, MAX_L, p)
    # cover_count[old_indices] -= 1
    # cover_count[new_indices] += 1
    
    # This is O(MAX_L/p) operations. For small p, this is large.
    # But we only have large primes (p >= 13).
    # MAX_L = 20000. p=13 -> 1500 ops.
    # 1500 ops * 2 (remove/add) * 10000 steps = 30M ops. Should be okay in 140s?
    # NumPy is vectorized, so it's fast.
    
    # Let's run SA.
    
    steps = 0
    while time.time() < deadline:
        # Pick a random prime index
        i = random.randint(0, len(state) - 1)
        p, r_old = state[i]
        
        # Pick new residue
        if random.random() < 0.5:
            r_new = random.randint(0, p - 1)
        else:
            # Local move
            r_new = (r_old + random.choice([-1, 1])) % p
            
        if r_new == r_old:
            continue
            
        # Apply move
        old_indices = np.arange(r_old, MAX_L, p)
        new_indices = np.arange(r_new, MAX_L, p)
        
        cover_count[old_indices] -= 1
        cover_count[new_indices] += 1
        
        new_L = get_L(cover_count)
        
        # Accept/Reject
        if new_L > current_L:
            # Always accept
            state[i] = (p, r_new)
            current_L = new_L
            if current_L > best_L:
                best_L = current_L
                best_state = [tuple(s) for s in state]
        else:
            # Calculate delta in "score". 
            # We want to maximize L. 
            # Using L directly as energy is tricky for SA because L is discrete and sparse.
            # Let's use a proxy: number of covered integers in 1..1000? 
            # Or just use L.
            # If new_L < current_L, delta = new_L - current_L < 0.
            # Accept with probability exp(delta / T).
            # But T is in units of L? L can be 10000. T=500 is small.
            # Let's scale T.
            delta = new_L - current_L
            if delta < 0:
                # prob = exp(delta / T)
                # If delta is -100 and T=500, exp(-0.2) ~ 0.8. Good.
                if random.random() < np.exp(delta / T):
                    state[i] = (p, r_new)
                    current_L = new_L
                else:
                    # Revert
                    cover_count[old_indices] += 1
                    cover_count[new_indices] -= 1
            
        T *= cooling
        if T < T_min:
            T = T_min
            # Restart? Or just continue.
            
        steps += 1
        
    # Final verification with Fractions
    # Check budget
    budget = sum(Fraction(1, p) for p, r in best_state)
    if budget > 1:
        # Fallback to seed if invalid (should not happen)
        primes = [p for p in range(13, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]
        
    return [list(s) for s in best_state]
