import time
import numpy as np
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # 1. Precompute primes up to 1000
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    primes = sieve(1000)
    
    # Candidate primes to consider (small primes are more effective)
    # We will select subsets of these
    candidate_primes = [p for p in primes if p <= 200]
    
    # Helper to check coverage length
    def get_coverage_length(residues, primes_set, max_L=10000):
        # residues: dict {p: r}
        # primes_set: list of primes
        # Returns largest L such that 1..L is covered
        if not primes_set:
            return 0
        
        # Create a boolean array for coverage
        # We check up to max_L
        covered = np.zeros(max_L + 1, dtype=bool)
        
        for p in primes_set:
            r = residues[p]
            # Mark numbers i such that i % p == r
            # i = r, r+p, r+2p, ...
            # Note: r can be 0. If r=0, numbers are p, 2p, ...
            # If r>0, numbers are r, r+p, ...
            # We only care about i >= 1.
            
            start = r
            if start == 0:
                start = p
            if start > max_L:
                continue
                
            # Vectorized marking
            indices = np.arange(start, max_L + 1, p)
            covered[indices] = True
        
        # Find the largest L such that covered[1:L+1] is all True
        # We can find the first False in covered[1:]
        # If all are True, L = max_L
        # Use np.argmax on ~covered[1:]
        # If ~covered[1:] is all False, argmax returns 0 (first element is False? No, if all False, argmax is 0)
        # Wait, if covered[1] is False, ~covered[1] is True, argmax is 0 -> L=0.
        # If covered[1..100] is True, covered[101] is False.
        # ~covered[101] is True. Index in ~covered[1:] is 100.
        # So L = index.
        
        not_covered = ~covered[1:]
        if not np.any(not_covered):
            return max_L
        
        first_gap = np.argmax(not_covered)
        return first_gap
    
    # Local Search on residues for a fixed set of primes
    def local_search(primes_set, max_iterations=10000):
        # Initialize residues
        # Greedy init: for each p, pick r that covers most new numbers in 1..100
        residues = {}
        covered = np.zeros(101, dtype=bool)
        
        for p in primes_set:
            best_r = 0
            best_gain = -1
            for r in range(p):
                start = r if r > 0 else p
                if start > 100:
                    gain = 0
                else:
                    indices = np.arange(start, 101, p)
                    gain = np.sum(~covered[indices])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            residues[p] = best_r
            start = best_r if best_r > 0 else p
            if start <= 100:
                indices = np.arange(start, 101, p)
                covered[indices] = True
        
        current_L = get_coverage_length(residues, primes_set)
        
        for _ in range(max_iterations):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try to change one residue
            p = random.choice(primes_set)
            old_r = residues[p]
            new_r = random.randint(0, p - 1)
            if new_r == old_r:
                continue
                
            residues[p] = new_r
            new_L = get_coverage_length(residues, primes_set)
            
            if new_L > current_L:
                current_L = new_L
                improved = True
            else:
                residues[p] = old_r
            
            if not improved and random.random() < 0.01:
                # Random restart or perturbation?
                # Just continue
                pass
        
        return current_L, residues

    # Generate candidate prime sets
    # We want sets where sum(1/p) <= 1
    # Greedy selection of primes to maximize potential coverage
    # We'll generate several sets
    
    best_L = 0
    best_classes = []
    
    # Set 1: {2, 3} + greedy fill
    # Set 2: {2, 5} + greedy fill
    # Set 3: {3, 5} + greedy fill
    # Set 4: {2, 7} + greedy fill
    # ...
    
    base_sets = [
        [2, 3],
        [2, 5],
        [2, 7],
        [3, 5],
        [3, 7],
        [5, 7],
        [2, 3, 5] # Invalid, but let's see if we can drop one? No, sum > 1.
    ]
    
    # Filter valid bases
    valid_bases = []
    for base in base_sets:
        s = sum(1.0/p for p in base)
        if s <= 1.0:
            valid_bases.append(base)
    
    # Also try single primes
    for p in primes[:10]:
        valid_bases.append([p])
    
    # For each valid base, greedily add more primes
    all_candidate_sets = set()
    
    for base in valid_bases:
        current_set = list(base)
        current_sum = sum(1.0/p for p in current_set)
        
        # Try to add primes from candidate_primes
        # Sort by 1/p descending (smallest primes first)
        for p in candidate_primes:
            if p in current_set:
                continue
            if current_sum + 1.0/p <= 1.0:
                current_set.append(p)
                current_sum += 1.0/p
        
        # Store the set as a tuple for uniqueness
        all_candidate_sets.add(tuple(sorted(current_set)))
        
        # Also try a version where we don't just add smallest, but maybe skip some?
        # For now, just the greedy fill.
        
    # Limit the number of sets to avoid timeout
    # Sort sets by size? Or just take first 50?
    # Let's take a diverse set.
    # We'll just iterate over the first 100 unique sets found.
    
    sets_to_try = list(all_candidate_sets)[:100]
    
    for p_set in sets_to_try:
        if time.time() > deadline - 10:
            break
            
        L, residues = local_search(list(p_set), max_iterations=2000)
        
        if L > best_L:
            best_L = L
            # Construct classes
            classes = [[p, residues[p]] for p in p_set]
            best_classes = classes
            
    # Final verification
    if best_classes:
        # Verify budget
        s = Fraction(0)
        for p, r in best_classes:
            s += Fraction(1, p)
        if s > 1:
            # Should not happen, but fallback
            return [[2, 1]]
        
        # Verify coverage
        # The local search already checked, but double check for safety
        # If best_L is high, re-verify
        if best_L > 100:
            # Re-calculate L exactly
            covered = np.zeros(best_L + 10, dtype=bool)
            for p, r in best_classes:
                start = r if r > 0 else p
                if start <= best_L + 10:
                    indices = np.arange(start, best_L + 11, p)
                    covered[indices] = True
            
            # Check 1..best_L
            if np.all(covered[1:best_L+1]):
                return best_classes
            else:
                # Find actual L
                not_covered = ~covered[1:best_L+1]
                if np.any(not_covered):
                    actual_L = np.argmax(not_covered)
                else:
                    actual_L = best_L
                
                # If actual_L is significantly less, we might have a bug in local search
                # But we return the classes anyway, the evaluator will calculate L
                # So we just return the classes.
                
    # Fallback
    return [[2, 1]]
