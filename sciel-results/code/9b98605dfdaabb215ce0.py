"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = 30
    best_val = None
    best_coeffs = None
    best_frac = None

    # The seed solution: [0]*28 + [1, -1] gives 1/29 - 1/30 = 1/870
    # |1/870| ≈ 0.001149
    seed_val = Fraction(1, 870)
    best_val = abs(seed_val)
    best_coeffs = [0] * 28 + [1, -1]
    best_frac = seed_val

    # Try all pairs (i, j) with i < j, coefficient[i-1]=1, coefficient[j-1]=-1
    # This gives 1/i - 1/j = (j-i)/(i*j). Minimize |j-i|/(i*j).
    # The minimum over all pairs is achieved by consecutive large indices:
    # i=29, j=30: 1/870 (already have this)
    # i=28, j=30: 2/(28*30) = 1/420 ≈ 0.00238 (worse)
    # i=29, j=30 is the best pair.

    # Try triples: coefficient[i-1]=1, coefficient[j-1]=-1, coefficient[k-1]=1 or -1
    # We want to find small nonzero sums of the form ±1/a ± 1/b ± 1/c ...
    
    # Strategy: enumerate subsets of size up to ~8 with signs, but use meet-in-the-middle
    # for larger subsets. For now, do a targeted search.
    
    # First: check all 2-element combinations (done above, best is 1/870)
    # Then: check all 3-element combinations
    # sum = s1/i + s2/j + s3/k where s1,s2,s3 in {-1,1}
    # We want |sum| to be small and nonzero.
    
    # For 3 elements: enumerate i < j < k, and all 8 sign combinations.
    # That's C(30,3) * 8 = 4060 * 8 = 32480 combinations. Fast.
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for k in range(j + 1, n + 1):
                # 8 sign combos for three terms
                # We can skip symmetric ones: flipping all signs gives same |value|
                # So only 4 distinct |values| per triple
                # s1=1, s2=1, s3=1: 1/i+1/j+1/k (always positive, large)
                # s1=1, s2=1, s3=-1: 1/i+1/j-1/k
                # s1=1, s2=-1, s3=1: 1/i-1/j+1/k
                # s1=1, s2=-1, s3=-1: 1/i-1/j-1/k
                # s1=-1, s2=1, s3=1: -1/i+1/j+1/k (same |val| as 1/i-1/j-1/k)
                # etc.
                
                # Just compute all 8:
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        for s3 in (1, -1):
                            val = Fraction(s1, i) + Fraction(s2, j) + Fraction(s3, k)
                            if val == 0:
                                continue
                            av = abs(val)
                            if best_val is None or av < best_val:
                                best_val = av
                                best_frac = val
                                coeffs = [0] * n
                                coeffs[i-1] = s1
                                coeffs[j-1] = s2
                                coeffs[k-1] = s3
                                best_coeffs = coeffs
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if time.time() > deadline - 3:
        return best_coeffs
    
    # 4-element combinations: C(30,4) * 16 = 27405 * 16 = 438480. Still feasible.
    # But we can be smarter: for each 4-tuple, we want s1/i+s2/j+s3/k+s4/l close to 0.
    # Group as (s1/i + s2/j) and (s3/k + s4/l). 
    # Actually, let's just do it directly for 4 elements.
    
    # For efficiency, precompute all pairs' signed sums
    # pair_sum[i][j] for i<j: the 4 distinct signed sums (up to global sign)
    # Actually, let's enumerate 4-element subsets and for each, try all 16 sign combos.
    
    # C(30,4) = 27405, times 16 = 438,480. Each involves 4 Fraction additions.
    # This should run in a few seconds.
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            for k in range(j + 1, n + 1):
                for l in range(k + 1, n + 1):
                    # 16 sign combos, but global sign flip is symmetric, so 8 distinct |vals|
                    for s1 in (1, -1):
                        for s2 in (1, -1):
                            for s3 in (1, -1):
                                for s4 in (1, -1):
                                    val = Fraction(s1, i) + Fraction(s2, j) + Fraction(s3, k) + Fraction(s4, l)
                                    if val == 0:
                                        continue
                                    av = abs(val)
                                    if best_val is None or av < best_val:
                                        best_val = av
                                        best_frac = val
                                        coeffs = [0] * n
                                        coeffs[i-1] = s1
                                        coeffs[j-1] = s2
                                        coeffs[k-1] = s3
                                        coeffs[l-1] = s4
                                        best_coeffs = coeffs
                    if time.time() > deadline - 5:
                        break
                if time.time() > deadline - 5:
                    break
            if time.time() > deadline - 5:
                break
        if time.time() > deadline - 5:
            break
    
    if time.time() > deadline - 3:
        return best_coeffs
    
    # 5-element: C(30,5) * 32 = 142506 * 32 = 4,560,192. This might be too slow with Fractions.
    # Let's use a meet-in-the-middle approach for 5+ elements.
    
    # Meet-in-the-middle for 5 elements: split into 2 and 3.
    # Precompute all signed sums for pairs (2 elements) and triples (3 elements).
    # Then for each pair-sum and triple-sum over disjoint index sets, check if their sum is small.
    
    # Actually, let's do 5-element with a smarter approach:
    # Fix 3 elements, enumerate their 8 sign combos to get partial sums.
    # For the remaining 2 elements, enumerate 4 sign combos.
    # Total: C(30,5) * 8 * 4 = 142506 * 32 = 4.56M. With Fraction arithmetic this is slow.
    
    # Alternative: use floating point for screening, then exact for candidates.
    
    # Let's try a different approach for 5 elements:
    # Precompute all 2-element signed sums: for each pair (i,j), i<j, compute 4 values:
    #   1/i+1/j, 1/i-1/j, -1/i+1/j, -1/i-1/j
    # But we only need to track the value and which indices/signs.
    
    # Actually, let's just do a targeted search for 5 elements using float screening.
    
    if time.time() < deadline - 10:
        # Precompute 2-element signed sums (4 per pair)
        # And 3-element signed sums (8 per triple)
        # Then combine disjoint pairs and triples.
        
        # For 5 elements: choose 2 indices for the pair, 3 for the triple, disjoint.
        # Number of ways: C(30,5) * C(5,2) = 142506 * 10 = 1,425,060 splits.
        # For each split: 4 pair-signs * 8 triple-signs = 32 combos.
        # Total: ~45.6M. Too many with Fractions.
        
        # Use float for screening:
        # Precompute float values for all 2-element signed sums and 3-element signed sums.
        
        pair_sums = {}  # (i, j) -> list of (float_val, (s1, s2))
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                vals = []
                for s1 in (1, -1):
                    for s2 in (1, -1):
                        vals.append((s1/i + s2/j, (s1, s2)))
                pair_sums[(i, j)] = vals
        
        triple_sums = {}  # (i, j, k) -> list of (float_val, (s1, s2, s3))
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                for k in range(j + 1, n + 1):
                    vals = []
                    for s1 in (1, -1):
                        for s2 in (1, -1):
                            for s3 in (1, -1):
                                vals.append((s1/i + s2/j + s3/k, (s1, s2, s3)))
                    triple_sums[(i, j, k)] = vals
        
        # Now for each 5-element subset, try all splits into 2+3
        # This is still a lot. Let's just do a random/local search instead.
        
        # Actually, let's try a local search / hill climbing approach.
        # Start from current best, flip individual coefficients, or swap.
        
        import random
        random.seed(42)
        
        current = list(best_coeffs)
        current_val = abs(best_frac)
        
        def eval_coeffs(coeffs):
            s = Fraction(0)
            for idx, c in enumerate(coeffs):
                if c != 0:
                    s += Fraction(c, idx + 1)
            return s
        
        # Local search: try flipping each position's coefficient
        for iteration in range(500000):
            if time.time() > deadline - 5:
                break
            
            # Try flipping a random position
            pos = random.randint(0, n - 1)
            old_val = current[pos]
            new_val = -old_val if old_val != 0 else (1 if random.random() < 0.5 else -1)
            if new_val == old_val:
                new_val = -old_val if old_val != 0 else 1
            
            current[pos] = new_val
            new_sum = eval_coeffs(current)
            
            if new_sum != 0:
                new_abs = abs(new_sum)
                if new_abs < current_val:
                    current_val = new_abs
                    best_val = new_abs
                    best_frac = new_sum
                    best_coeffs = list(current)
                else:
                    # Revert with some probability (simulated annealing-like)
                    current[pos] = old_val
            else:
                # Zero sum is invalid; revert
                current[pos] = old_val
        
        # Also try adding/removing a term
        for iteration in range(200000):
            if time.time() > deadline - 5:
                break
            
            pos = random.randint(0, n - 1)
            old_val = current[pos]
            if old_val == 0:
                new_val = 1 if random.random() < 0.5 else -1
            else:
                # Remove it
                new_val = 0
            
            current[pos] = new_val
            new_sum = eval_coeffs(current)
            
            if new_sum != 0:
                new_abs = abs(new_sum)
                if new_abs < current_val:
                    current_val = new_abs
                    best_val = new_abs
                    best_frac = new_sum
                    best_coeffs = list(current)
                else:
                    current[pos] = old_val
            else:
                current[pos] = old_val
    
    return best_coeffs
# EVOLVE-BLOCK-END
