import time
import numpy as np
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Generate primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Strategy: Use primes >= 13 as base, then add small primes to fill gaps
    # Sum of 1/p for p in [13, 997] is about 0.92
    # We have about 0.08 budget left for small primes
    
    # First, let's compute the sum for primes >= 13
    large_primes = [p for p in primes if p >= 13]
    small_primes = [p for p in primes if p < 13]  # [2, 3, 5, 7, 11]
    
    # Sum of reciprocals for large primes
    large_sum = Fraction(0, 1)
    for p in large_primes:
        large_sum += Fraction(1, p)
    
    remaining_budget = Fraction(1, 1) - large_sum
    
    # We can use some small primes if they fit in remaining budget
    # 1/11 ≈ 0.0909, 1/7 ≈ 0.1429, etc.
    # remaining_budget is about 0.08, so we can only use 1/11 if we drop some large primes
    
    # Better approach: Use ALL primes >= 13, and try to add 11 if possible
    # by dropping a few large primes
    
    # For now, let's use all large primes and see what L we get
    # Then try to improve by swapping
    
    # Phase 1: Assign residues to large primes to maximize coverage
    # Key insight: We want to cover as many of [1, L] as possible
    # For each prime p, residue r covers {r, r+p, r+2p, ...}
    
    # The seed approach: i-th prime gets residue (i+1) % p
    # This is suboptimal. We need a better assignment.
    
    # Let's try a different strategy:
    # For each prime p, choose r to cover the most uncovered numbers in [1, L_target]
    # But this is what the greedy does, and it's slow.
    
    # Alternative: Use a known covering system structure
    # The classic covering system uses:
    # 2|0, 3|1, 5|2, 7|5, 11|... 
    # But we need sum 1/p <= 1
    
    # Let me try a different structural approach:
    # Use primes in a specific order and assign residues to create overlapping coverage
    
    # Actually, let's try the most direct approach that can reach L=999:
    # Use a precomputed optimal or near-optimal covering system
    
    # Based on the Erdős-Ruzsa result, for x=1000, C=1, we need a covering system
    # with sum 1/p <= 1 that covers [1, 999]
    
    # Let's construct one:
    # Use all primes from 13 to 997 (163 primes), sum ≈ 0.92
    # Add prime 11: sum becomes 0.92 + 0.0909 ≈ 1.011 > 1, too much
    # So we need to drop some primes to add 11
    
    # Drop the largest primes to make room for 11
    # 1/11 = 0.0909, we need to free up about 0.011 of budget
    # Drop primes with 1/p > 0.011, i.e., p < 91
    # Primes < 91: 13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89
    # 1/13 ≈ 0.0769, that's way more than 0.011
    
    # This approach is getting complicated. Let me try a simpler structural change:
    
    # Use a covering system based on the Chinese Remainder Theorem structure
    # Or use a known result: the primes 2,3,5,7,11,13,... can be arranged
    
    # Actually, the simplest high-scoring approach from inspiration 1 was to use
    # primes >= 13 and greedily assign residues. Let me optimize that.
    
    # The key improvement: instead of covering sequentially, use a global optimization
    # to assign residues that maximize the minimum gap
    
    # For speed, let's use a heuristic:
    # 1. Start with all primes >= 13
    # 2. For each prime, assign the residue that covers the most "critical" numbers
    # 3. Critical numbers are those with few covering options
    
    # Let's try a different tactic: use the fact that for large L, we need
    # the covering system to be "dense" - each number should be covered by multiple primes
    
    # New structural approach: Use a lattice-based covering
    # Assign residues such that the union of arithmetic progressions covers [1, L]
    
    # For the time budget, let me implement a fast greedy with better heuristics:
    
    L_target = 1000
    covered = np.zeros(L_target + 1, dtype=bool)
    
    # Use all primes >= 13
    candidate_primes = large_primes[:]
    
    # Sort by size (larger primes are cheaper and cover more spread out)
    # Actually, for coverage density, smaller primes are better (denser)
    # But we have budget constraints
    
    # Let's try: assign residues to maximize coverage of [1, L_target]
    # For each prime, the best residue is the one that covers the most uncovered numbers
    
    used_primes = set()
    current_sum = Fraction(0, 1)
    classes = []
    
    # Precompute for each prime, which residue covers what
    # For speed, iterate primes in order of increasing p (smaller primes cover more densely)
    
    # Sort primes: we want to use small primes first for density
    # But we need to respect the budget
    
    # Let's try a different order: use primes in descending order of 1/p (i.e., ascending p)
    # No wait, 1/p decreases with p, so ascending p means descending 1/p
    
    # Actually, let's just try all primes >= 13 and see
    # For each, pick the residue that covers the most currently uncovered numbers
    
    # To make this fast, limit L_target and use numpy
    
    # Phase 1: Use primes >= 13, assign best residue for each
    # Process in order of increasing p (smaller primes first for better coverage)
    
    primes_to_use = sorted(large_primes)  # ascending order
    
    for p in primes_to_use:
        if time.time() > deadline - 5:
            break
        
        cost = Fraction(1, p)
        if current_sum + cost > Fraction(1, 1):
            continue
        
        # Find residue that covers most uncovered numbers in [1, L_target]
        best_r = 0
        best_gain = -1
        
        # For each possible residue r in [0, p-1], count uncovered numbers
        # x in [1, L_target] with x % p == r
        # That's floor((L_target - r) / p) if r <= L_target, else 0
        # But we need to count only UNCOVERED ones
        
        # For speed, approximate: assume uniform coverage
        # Gain ≈ (number of x in [1,L] with x%p==r) * (1 - current_coverage_density)
        
        # More precise: iterate
        # For small p, this is fast. For large p, p can be up to 997, so p iterations
        
        # Let's do it properly but efficiently
        # Count uncovered x in [1, L_target] with x % p == r
        
        # Create array of x values: 1, 2, ..., L_target
        # For each r, x = r, r+p, r+2p, ... <= L_target
        # Count how many are not covered
        
        # For speed, use numpy
        # But p can be large, so let's be smart
        
        # Alternative: track uncovered count per residue class
        # This is complex. Let's just iterate for now, but limit L_target
        
        # Actually, for p > L_target, each residue class has at most 1 element
        # For p <= L_target, we can compute quickly
        
        if p > L_target:
            # Each residue class has at most one element in [1, L_target]
            # x = r if 1 <= r <= L_target, else x = r + p (but r+p > L_target if p > L_target)
            # So only r in [1, L_target] gives a valid x
            # Gain is 1 if x=r is uncovered, else 0
            for r in range(1, min(p, L_target + 1)):
                if not covered[r]:
                    gain = 1
                else:
                    gain = 0
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
                if gain == 1:
                    break  # Can't do better than 1
        else:
            # p <= L_target, iterate over residues
            # For each r in [0, p-1], count uncovered x with x%p==r
            # x values: r, r+p, r+2p, ... <= L_target
            # If r == 0, x values are p, 2p, 3p, ...
            # If r > 0, x values are r, r+p, r+2p, ...
            
            # For speed, precompute uncovered mask
            # Then for each r, count uncovered in arithmetic progression
            
            # This is O(p * L/p) = O(L) per prime, which is too slow for 163 primes
            
            # Let's use a different approach: maintain a count array for residues
            # But that's complex. For now, let's limit the search
            
            # Simpler: just try a few promising residues
            # The best residue is likely the one that covers the most uncovered numbers
            # Approximate by checking a few candidates
            
            # Candidate residues: 0, 1, 2, ..., and residues of recently uncovered numbers
            # For speed, just check r = 0 and r = 1 and a few others
            
            # Actually, let's just do the full iteration but with early termination
            # For large p, most residues have 0 or 1 elements, so we can break early
            
            checked = 0
            for r in range(p):
                checked += 1
                if checked > 50 and p > 100:
                    break  # Limit work for large primes
                
                # Count uncovered x with x % p == r
                if r == 0:
                    # x = p, 2p, 3p, ...
                    start = p
                else:
                    start = r
                
                count = 0
                x = start
                while x <= L_target:
                    if not covered[x]:
                        count += 1
                    x += p
                
                if count > best_gain:
                    best_gain = count
                    best_r = r
                    if count == 0:
                        continue
                # Early break if we found a good one
                if best_gain >= 5 and p < 50:
                    break
        
        if best_gain <= 0:
            # This prime doesn't help, skip
            continue
        
        # Add this class
        classes.append([p, best_r])
        used_primes.add(p)
        current_sum += cost
        
        # Update covered
        if best_r == 0:
            start = p
        else:
            start = best_r
        x = start
        while x <= L_target:
            covered[x] = True
            x += p
    
    # Now try to add small primes to fill gaps
    # Check which numbers in [1, L_target] are uncovered
    uncovered = [i for i in range(1, L_target + 1) if not covered[i]]
    
    if uncovered:
        # Try to add small primes to cover the earliest uncovered numbers
        # Sort small primes by 1/p (descending, i.e., ascending p)
        small_primes_sorted = sorted(small_primes, key=lambda p: -1/p)
        # [2, 3, 5, 7, 11] in order of decreasing 1/p
        
        for p in small_primes_sorted:
            if time.time() > deadline - 2:
                break
            
            cost = Fraction(1, p)
            if current_sum + cost > Fraction(1, 1):
                continue
            
            if p in used_primes:
                continue
            
            # Find best residue to cover uncovered numbers
            best_r = 0
            best_gain = -1
            
            for r in range(p):
                # Count uncovered x with x % p == r
                if r == 0:
                    start = p
                else:
                    start = r
                
                count = 0
                x = start
                while x <= L_target:
                    if not covered[x]:
                        count += 1
                    x += p
                
                if count > best_gain:
                    best_gain = count
                    best_r = r
            
            if best_gain <= 0:
                continue
            
            classes.append([p, best_r])
            used_primes.add(p)
            current_sum += cost
            
            # Update covered
            if best_r == 0:
                start = p
            else:
                start = best_r
            x = start
            while x <= L_target:
                covered[x] = True
                x += p
    
    # Calculate L
    L = 0
    for i in range(1, L_target + 1):
        if covered[i]:
            L = i
        else:
            break
    
    return classes
