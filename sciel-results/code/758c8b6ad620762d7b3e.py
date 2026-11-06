import time
import random
import numpy as np

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A."""
    n = len(A)
    sums = set()
    prods = set()
    # Optimization: Use lists and set conversion, or direct addition.
    # Direct addition is usually fine for n=20.
    for i in range(n):
        ai = A[i]
        for j in range(n):
            sums.add(ai + A[j])
            prods.add(ai * A[j])
    return max(len(sums), len(prods))

def _is_valid(A, n=20, maximum=10000):
    """Check if A has n distinct positive integers <= maximum."""
    if len(A) != n:
        return False
    if len(set(A)) != n:
        return False
    if any(x <= 0 for x in A):
        return False
    if any(x > maximum for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s limit
    
    rng = random.Random(42)
    
    best_A = None
    best_score = float('inf')
    
    # Strategy: Model A as union of two disjoint GPs with small integer bases.
    # A = {a1 * b1^k for k=0..k1} U {a2 * b2^k for k=0..k2}
    # where a1, a2 are starting values, b1, b2 are bases (integers >= 2).
    # We search over small bases (2, 3, 4, 5) and starting values.
    
    # To ensure we have exactly 20 distinct positive integers <= 10000,
    # we will generate candidates and filter/adjust.
    
    # Possible bases for GP
    bases = [2, 3, 4, 5]
    
    # We will try pairs of bases (b1, b2).
    # For each pair, we vary the starting values a1, a2 and the split k1, k2.
    # k1 + k2 + 2 = 20 => k1 + k2 = 18.
    # So k1 can range from 0 to 18, and k2 = 18 - k1.
    # Actually, the number of terms in GP1 is k1+1, and in GP2 is k2+1.
    # Total terms = (k1+1) + (k2+1) = k1 + k2 + 2.
    # We need total terms = 20, so k1 + k2 = 18.
    
    # However, the terms must be distinct and <= 10000.
    # If b is large, the GP grows fast, so k1, k2 must be small.
    # If b is small (e.g., 2), we can have more terms.
    
    # Let's iterate over possible splits and bases.
    
    # To speed up, we can precompute GP terms for given a, b, k.
    # But a can vary.
    
    # Instead of exhaustive search over a1, a2 (which can be up to 10000),
    # we can sample a1, a2 from a reasonable range, or use local search.
    
    # Let's use a multi-start local search.
    # Initial candidates: random a1, a2, and fixed bases/splits.
    
    # We'll define a function to generate a candidate from (a1, b1, k1, a2, b2, k2).
    # It returns a sorted list of 20 distinct integers if possible, else None.
    
    def _generate_candidate(a1, b1, k1, a2, b2, k2):
        # k1, k2 are the exponents (0 to k1, 0 to k2).
        # Number of terms: (k1+1) + (k2+1) = 20.
        if (k1 + 1) + (k2 + 1) != 20:
            return None
        
        terms1 = []
        current = a1
        for _ in range(k1 + 1):
            if current > 10000:
                return None
            if current <= 0:
                return None
            terms1.append(current)
            current *= b1
            
        terms2 = []
        current = a2
        for _ in range(k2 + 1):
            if current > 10000:
                return None
            if current <= 0:
                return None
            terms2.append(current)
            current *= b2
            
        A = terms1 + terms2
        if len(A) != 20:
            return None
        if len(set(A)) != 20:
            return None
        if any(x > 10000 for x in A):
            return None
        return sorted(A)
    
    # We will sample random (a1, b1, k1, a2, b2, k2) tuples.
    # k1, k2 are constrained by k1+k2=18.
    # b1, b2 from [2,3,4,5].
    # a1, a2: we need to choose such that the GP fits in 10000.
    # For a given b and k, the max value is a * b^k <= 10000 => a <= 10000 / b^k.
    # So a must be in [1, 10000 // b^k].
    
    # Let's sample k1 from 0 to 18.
    # For each k1, k2 = 18 - k1.
    # Sample b1, b2.
    # Sample a1, a2 within valid ranges.
    
    # To improve efficiency, we can precompute valid (b, k, a_max) combinations.
    # But sampling is fine if we do enough iterations.
    
    # Let's do 1000 random starts.
    for _ in range(1000):
        if time.time() > deadline:
            break
            
        # Choose k1
        k1 = rng.randint(0, 18)
        k2 = 18 - k1
        
        # Choose b1, b2
        b1 = rng.choice(bases)
        b2 = rng.choice(bases)
        
        # Calculate max a for b1, k1
        # a1 * b1^k1 <= 10000 => a1 <= 10000 // (b1^k1)
        # If b1^k1 > 10000, then a1 must be 0, which is invalid.
        # So we need b1^k1 <= 10000.
        
        max_a1 = 10000 // (b1 ** k1)
        if max_a1 < 1:
            continue
            
        max_a2 = 10000 // (b2 ** k2)
        if max_a2 < 1:
            continue
            
        # Sample a1, a2
        a1 = rng.randint(1, max_a1)
        a2 = rng.randint(1, max_a2)
        
        A = _generate_candidate(a1, b1, k1, a2, b2, k2)
        if A is None:
            continue
            
        # Local refinement: try to swap elements between the two GPs or perturb them.
        # Since the structure is rigid (GP), perturbing might break the GP structure.
        # But the goal is to minimize max(|A+A|, |A*A|), not to maintain GP structure.
        # So we can use the GP candidate as a starting point for general local search.
        
        current_A = A[:]
        current_score = _score(current_A)
        
        # Simple local search: try changing each element by small deltas
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            indices = list(range(20))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline - 2:
                    break
                    
                old_val = current_A[i]
                # Try small perturbations
                delta_range = max(10, int(old_val * 0.05))
                
                for _ in range(5):
                    delta = rng.randint(-delta_range, delta_range)
                    new_val = old_val + delta
                    
                    if new_val <= 0 or new_val > 10000:
                        continue
                    if new_val in current_A and new_val != old_val:
                        continue
                        
                    new_A = current_A[:]
                    new_A[i] = new_val
                    new_A.sort()
                    
                    s = _score(new_A)
                    
                    if s < current_score:
                        current_A = new_A
                        current_score = s
                        improved = True
                        break
                
        if current_score < best_score:
            best_score = current_score
            best_A = current_A[:]
            
    # If no candidate was found, fallback to a simple GP
    if best_A is None:
        # Fallback: GP with base 2
        A = []
        current = 1
        while len(A) < 20 and current <= 10000:
            A.append(current)
            current *= 2
        # If we have fewer than 20, fill with large random numbers
        if len(A) < 20:
            existing = set(A)
            while len(A) < 20:
                val = rng.randint(1, 10000)
                if val not in existing:
                    A.append(val)
                    existing.add(val)
        best_A = sorted(A[:20])
        
    return sorted(best_A)
