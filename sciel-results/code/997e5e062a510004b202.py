"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from functools import lru_cache
import random

def _compute_scores(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A = list(A)
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def _geometric_set(base, exponents):
    """Generate set {base^e_i} for given base and exponents."""
    A = set()
    for e in exponents:
        val = base ** e
        if val > 10000:
            continue
        A.add(val)
    return sorted(A)

def _search_geometric():
    """Search for best geometric progression-like sets."""
    best_score = float('inf')
    best_set = None
    
    # Try different bases
    for base in range(2, 11):
        # Find max exponent such that base^e <= 10000
        max_e = 0
        val = 1
        while val * base <= 10000:
            val *= base
            max_e += 1
        
        # We need 20 distinct values from base^e_i
        # If max_e < 19, we can't get 20 distinct powers
        if max_e < 19:
            continue
        
        # Try all subsets of exponents of size 20 from [0, max_e]
        # This is C(max_e+1, 20) which for max_e=19 is C(20,20)=1, for max_e=20 is C(21,20)=21
        # For larger max_e, this grows but let's cap it
        if max_e + 1 > 30:
            # Too many combinations, sample some
            # Try consecutive exponents
            for start in range(0, max_e - 19 + 1):
                exponents = list(range(start, start + 20))
                A = _geometric_set(base, exponents)
                if len(A) == 20:
                    score = _compute_scores(A)
                    if score < best_score:
                        best_score = score
                        best_set = A
            # Also try some random subsets
            for _ in range(100):
                exponents = random.sample(range(0, max_e + 1), 20)
                A = _geometric_set(base, exponents)
                if len(A) == 20:
                    score = _compute_scores(A)
                    if score < best_score:
                        best_score = score
                        best_set = A
        else:
            # Try all combinations
            for exponents in combinations(range(0, max_e + 1), 20):
                A = _geometric_set(base, exponents)
                if len(A) == 20:
                    score = _compute_scores(A)
                    if score < best_score:
                        best_score = score
                        best_set = A
    
    return best_set, best_score

def _search_perturbed_geometric():
    """Search for perturbed geometric sets: A = {b^e_i * k_i} where k_i are small coprime integers."""
    best_score = float('inf')
    best_set = None
    
    # Try base 2 primarily, with small perturbations
    for base in [2, 3]:
        # Generate base powers
        powers = []
        val = 1
        while val <= 10000:
            powers.append(val)
            val *= base
        
        if len(powers) < 20:
            continue
        
        # For base 2, we have powers up to 2^13 = 8192, so 14 powers (2^0 to 2^13)
        # We need 20 distinct values, so we need perturbations
        
        # Try: take some powers and multiply by small coprime factors
        # Strategy: use a mix of pure powers and perturbed versions
        
        # Let's try a structured approach:
        # For each of 20 slots, pick a base power and a small multiplier
        # Ensure all values are distinct and <= 10000
        
        # Small multipliers to consider: 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        small_multipliers = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        
        # For base 2, powers are 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192
        # We need 20 values. Let's try to use all 14 powers with multipliers 1,
        # and add 6 more by using smaller powers with larger multipliers
        
        # Actually, let's try a different approach:
        # Use a grid of (power_index, multiplier) pairs
        
        candidates = []
        for p_idx, p_val in enumerate(powers):
            for m in small_multipliers:
                val = p_val * m
                if val <= 10000:
                    candidates.append((p_idx, m, val))
        
        # Now we need to select 20 distinct values from candidates
        # such that the set has good sum/product properties
        
        # This is a combinatorial optimization problem. Let's use a greedy/heuristic approach.
        
        # First, let's try to find a good set by random sampling
        random.seed(42)
        for trial in range(20000):
            # Select 20 random candidates ensuring distinct values
            selected = set()
            A = []
            for _ in range(20):
                p_idx, m, val = random.choice(candidates)
                if val not in selected:
                    selected.add(val)
                    A.append(val)
            
            if len(A) != 20:
                continue
            
            A.sort()
            score = _compute_scores(A)
            if score < best_score:
                best_score = score
                best_set = A
        
    return best_set, best_score

def _local_search(start_set, max_iter=50000):
    """Perform local search to improve a given set."""
    A = list(start_set)
    best_score = _compute_scores(A)
    best_A = A[:]
    
    for _ in range(max_iter):
        # Try to replace one element with another value
        i = random.randint(0, len(A) - 1)
        old_val = A[i]
        
        # Try a random new value
        new_val = random.randint(1, 10000)
        if new_val in A:
            continue
        
        A[i] = new_val
        A.sort()
        
        score = _compute_scores(A)
        if score < best_score:
            best_score = score
            best_A = A[:]
        else:
            A[i] = old_val
            A.sort()
    
    return best_A, best_score

def construct():
    # First, try geometric progressions
    geo_set, geo_score = _search_geometric()
    
    # Try perturbed geometric
    pert_set, pert_score = _search_perturbed_geometric()
    
    # Choose the best starting point
    if geo_set is not None and (pert_set is None or geo_score < pert_score):
        start_set = geo_set
        best_score = geo_score
    elif pert_set is not None:
        start_set = pert_set
        best_score = pert_score
    else:
        start_set = list(range(1, 21))
        best_score = _compute_scores(start_set)
    
    # Local search from the best starting point
    improved_set, improved_score = _local_search(start_set, max_iter=30000)
    
    if improved_score < best_score:
        return improved_set
    else:
        return start_set

# EVOLVE-BLOCK-END
