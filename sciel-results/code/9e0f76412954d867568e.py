"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
from math import isqrt
import numpy as np

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

PRIMES = _primes_up_to(1000)
PRIME_SET = set(PRIMES)

def construct():
    """
    New approach: Systematic residue optimization using vectorized NumPy operations.
    
    Strategy:
    1. Select primes in a specific order (small to large, but skip 2 to save budget)
    2. For each prime, try all residues and pick the one that maximizes contiguous coverage from 1
    3. Use NumPy for fast coverage checks
    """
    # We'll build classes incrementally
    classes = []
    used_primes = set()
    budget = Fraction(1, 1)
    
    # Target: cover as much as possible up to 100000
    MAX_L = 100000
    
    # Precompute a coverage array that we'll update
    # covered[n] = True if n is covered
    covered = np.zeros(MAX_L + 1, dtype=bool)
    
    # We'll try primes in order of increasing size (but skip 2, 3 if needed to save budget)
    # Actually, let's try a greedy approach: for each candidate prime, pick the residue
    # that extends the contiguous coverage from 1 the most
    
    # Start with no primes
    current_max_covered = 0
    
    # Try adding primes one by one
    # We'll iterate through primes and for each, if budget allows, add it with optimal residue
    for p in PRIMES:
        cost = Fraction(1, p)
        if budget < cost:
            continue
        if p in used_primes:
            continue
        
        # Try all residues and find the one that maximizes contiguous coverage from 1
        best_residue = 0
        best_max_covered = -1
        
        for r in range(p):
            # Create mask for this residue: positions where n % p == r
            # n ranges from 1 to MAX_L
            # n % p == r means n = r + k*p
            # For r=0: n = p, 2p, 3p, ...
            # For r>0: n = r, r+p, r+2p, ...
            
            # Vectorized: create array of n from 1 to MAX_L
            # mask = (np.arange(1, MAX_L+1) % p) == r
            
            # To be efficient, let's compute the new coverage
            # new_covered = covered OR mask
            
            # Find the longest prefix of True in new_covered
            # i.e., find the largest L such that new_covered[1:L+1] are all True
            
            # Efficient way: use cumulative sum
            # But we need to find first False
            
            # Let's compute mask efficiently
            if r == 0:
                # n = p, 2p, 3p, ...
                indices = np.arange(p, MAX_L + 1, p)
            else:
                indices = np.arange(r, MAX_L + 1, p)
            
            # Create new covered array
            new_covered = covered.copy()
            new_covered[indices] = True
            
            # Find longest prefix of True from index 1
            # We can use: find first index where new_covered is False
            # But new_covered[0] is always False (we don't cover 0)
            
            # Find first False in new_covered[1:]
            # If all True, then MAX_L is covered
            false_indices = np.where(~new_covered[1:])[0]
            if len(false_indices) == 0:
                max_cov = MAX_L
            else:
                max_cov = false_indices[0]  # This is the count of True from index 1
            
            if max_cov > best_max_covered:
                best_max_covered = max_cov
                best_residue = r
        
        # Add this prime with best residue
        classes.append([p, best_residue])
        used_primes.add(p)
        budget -= cost
        
        # Update covered array
        if best_residue == 0:
            indices = np.arange(p, MAX_L + 1, p)
        else:
            indices = np.arange(best_residue, MAX_L + 1, p)
        covered[indices] = True
        
        current_max_covered = best_max_covered
        
        # If we've covered everything, break
        if current_max_covered >= MAX_L:
            break
    
    # Now, we might have remaining budget. Let's try to use it to extend coverage
    # by adding more primes with optimal residues
    
    # But first, let's see what our current coverage is
    false_indices = np.where(~covered[1:])[0]
    if len(false_indices) == 0:
        current_max_covered = MAX_L
    else:
        current_max_covered = false_indices[0]
    
    # Try to add more primes to extend coverage
    for p in PRIMES:
        if p in used_primes:
            continue
        cost = Fraction(1, p)
        if budget < cost:
            continue
        
        # Try all residues
        best_residue = None
        best_max_covered = current_max_covered
        
        for r in range(p):
            if r == 0:
                indices = np.arange(p, MAX_L + 1, p)
            else:
                indices = np.arange(r, MAX_L + 1, p)
            
            new_covered = covered.copy()
            new_covered[indices] = True
            
            false_indices = np.where(~new_covered[1:])[0]
            if len(false_indices) == 0:
                max_cov = MAX_L
            else:
                max_cov = false_indices[0]
            
            if max_cov > best_max_covered:
                best_max_covered = max_cov
                best_residue = r
        
        if best_residue is not None and best_max_covered > current_max_covered:
            classes.append([p, best_residue])
            used_primes.add(p)
            budget -= cost
            
            if best_residue == 0:
                indices = np.arange(p, MAX_L + 1, p)
            else:
                indices = np.arange(best_residue, MAX_L + 1, p)
            covered[indices] = True
            
            current_max_covered = best_max_covered
            
            if current_max_covered >= MAX_L:
                break
    
    return classes

# EVOLVE-BLOCK-END
