"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    max_val = P['maximum']
    
    # Precompute LCM of all divisors
    from math import gcd
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    all_lcm = 1
    for d in divisors:
        all_lcm = lcm(all_lcm, d)
    
    # The period is all_lcm. We need to find a window of length L
    # where for each divisor d_i, there's exactly one multiple in the window
    # and all these multiples are distinct.
    
    # Strategy: Try different window lengths L starting from a reasonable minimum.
    # The minimum possible span is at least the max gap between consecutive multiples
    # of the "tightest" constraints. But a better approach is to search for the
    # optimal alignment.
    
    # Since all_lcm might be large, let's compute it first.
    # 6,10,14,15,21,22,26,33,35,39
    # 6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # Primes: 2,3,5,7,11,13
    # LCM = 2*3*5*7*11*13 = 30030
    
    # So the pattern repeats every 30030.
    # We need to find a start offset s such that the set { (k_i * d_i) mod 30030 } 
    # for some integers k_i, when lifted to the range [s, s+L], are distinct and
    # minimize L.
    
    # Actually, it's simpler: For any interval [A, B], we need to pick one multiple
    # of d_i in [A, B] for each i, and they must all be distinct.
    # This is possible iff for every pair (i,j), there exist multiples m_i of d_i 
    # and m_j of d_j in [A,B] such that m_i != m_j.
    # This is equivalent to saying that the "collision" points don't block the interval.
    
    # A more direct search:
    # The span L must be at least the maximum over all i of (d_i - 1) ? No.
    # Actually, if the span is L, we can fit a multiple of d_i in the interval 
    # unless the interval is too short and misaligned.
    
    # Let's try a heuristic search over the period 30030.
    # For a given span L, check if there's a start point A such that
    # for each d_i, there is a multiple in [A, A+L] and all are distinct.
    
    # Minimum possible L:
    # Consider d=39. We need a multiple of 39.
    # Consider d=35. We need a multiple of 35.
    # If L < 39, we can have at most one multiple of 39.
    # The "density" argument suggests L should be around 100-200.
    
    # Let's do a binary search on L? Or just iterate L upwards.
    # For a fixed L, we need to check if a valid A exists.
    # A valid A must satisfy:
    # 1. For each i, there exists k such that A <= k*d_i <= A+L.
    #    This means A mod d_i must be in a specific set of "good" residues.
    #    Specifically, let r_i = A mod d_i. Then there is a multiple in [A, A+L]
    #    iff there exists k such that k*d_i - r_i >= 0 and k*d_i - r_i <= L.
    #    i.e., ceil(r_i/d_i)*d_i <= A+L.
    #    Since r_i < d_i, the smallest multiple >= A is ceil(A/d_i)*d_i.
    #    Let m_i(A) = ceil(A/d_i)*d_i.
    #    We need m_i(A) <= A+L.
    
    # 2. All m_i(A) must be distinct.
    
    # This is a constraint satisfaction problem on A mod LCM.
    
    # Let's try a simpler approach: Randomized local search or brute force over
    # a subset of the period.
    
    # Given the small LCM (30030), we can afford to check all A in [0, 30030)
    # for a given L.
    
    # What is the minimum L?
    # Let's try L = 100, 110, 120, etc.
    
    # Optimization: Precompute for each d_i, the "bad" intervals of A where
    # no multiple fits in [A, A+L].
    # For a fixed d, the condition "no multiple of d in [A, A+L]" is
    # equivalent to A mod d being in (L+1, d-1] ?
    # Let r = A mod d. The next multiple is at A + (d-r) if r>0, or A if r=0.
    # Distance to next multiple is (d - r) % d.
    # We need (d - r) % d <= L.
    # So if (d - r) % d > L, then A is "bad" for d.
    # Note: (d-r)%d is 0 if r=0, else d-r.
    # So bad if d-r > L => r < d-L.
    # Also if L >= d, then always good.
    
    # So for each d_i, the bad residues are {0, 1, ..., d_i-L-1} if L < d_i.
    # If L >= d_i, all residues are good.
    
    # We need A mod d_i NOT in bad_set_i for all i.
    # AND m_i(A) must be distinct.
    
    # Let's start with L = 100 and increase.
    
    deadline = time.time() + 60  # 60s limit, keep margin
    best_result = None
    best_L = float('inf')
    
    # Lower bound for L:
    # We need to fit 10 distinct numbers.
    # The "tightest" divisor is 39.
    # If L < 39, we can only have 1 multiple of 39.
    # Let's just iterate L from a reasonable lower bound.
    # Lower bound: max(d_i) - min(d_i) ? No.
    # Let's try L from 100 to 500.
    
    # Actually, let's be smarter.
    # The span must be at least the maximum gap between any two "required" points.
    # But we can choose the points.
    
    # Let's try L = 100 first.
    
    for L in range(100, 500):
        if time.time() > deadline - 10:
            break
            
        # Check if there exists A in [0, all_lcm) such that:
        # 1. For all i, A mod d_i is not in bad_set_i
        # 2. All m_i(A) are distinct
        
        found = False
        # We can iterate A in steps, or use CRT-like filtering.
        # For now, simple iteration over A in [0, all_lcm) might be too slow
        # if we do it for many L. 30030 * 400 = 12M checks. Might be okay in Py?
        # 12M * 10 divisors = 120M ops. Too slow in pure Python (10s+).
        
        # Optimization: Precompute bad masks for each L?
        # Or iterate A and check constraints.
        
        # Let's try a different approach:
        # For a fixed L, the valid A's are those where A mod d_i is "good" for all i.
        # This is a system of congruences.
        
        # Let's just try a few random A's for each L?
        # Or iterate A but break early if constraints fail.
        
        # Let's try iterating A from 0 to all_lcm-1.
        # To speed up, we can precompute for each d_i the set of good residues.
        
        # Actually, let's just try to find ANY solution with a small L using
        # a more targeted search.
        
        # Since the period is small, let's just do it.
        # But 30030 * 10 = 300k ops per L.
        # 400 L's * 300k = 120M ops.
        # In Python, 120M simple ops might take 10-20s.
        # We have 60s. So it's feasible.
        
        # Let's optimize the inner loop.
        
        # Precompute: for each d_i, list of good residues for this L.
        # good_residues[i] = [r for r in range(d_i) if (d_i - r) % d_i <= L]
        # Note: (d_i - r) % d_i is 0 if r=0, else d_i-r.
        # So condition is: (0 if r==0 else d_i-r) <= L
        # i.e., r == 0 or r >= d_i - L.
        
        # So good residues are {0} U [d_i-L, d_i-1].
        # If L >= d_i, all residues are good.
        
        good_masks = []
        for d in divisors:
            if L >= d:
                good_masks.append(None)  # All good
            else:
                # Good if r == 0 or r >= d - L
                # Bad if 1 <= r <= d - L - 1
                bad_start = 1
                bad_end = d - L  # exclusive
                # We can represent this as a set or just check in loop
                good_masks.append((d, bad_start, bad_end))
        
        # Now iterate A
        # To speed up, we can check the "tightest" constraint first.
        # Sort divisors by d_i - L (ascending), so the one with smallest "good" set
        # is checked first? No, the one with largest bad set.
        
        # Let's just iterate A.
        for A in range(all_lcm):
            valid = True
            multiples = []
            for i, d in enumerate(divisors):
                r = A % d
                if good_masks[i] is not None:
                    d_val, bs, be = good_masks[i]
                    if bs <= r < be:
                        valid = False
                        break
                # Compute multiple
                if r == 0:
                    m = A
                else:
                    m = A + (d - r)
                multiples.append(m)
            
            if not valid:
                continue
                
            # Check distinctness
            if len(set(multiples)) == n:
                # Found a solution!
                # Construct the result
                result = [int(m) for m in multiples]
                # Ensure they are within [1, 1000000]
                # Since A < 30030 and L < 500, max m < 30030 + 500 = 30530.
                # Min m >= 1 (since A>=0, m>=A>=0, but m must be positive.
                # If m=0, it's not positive.
                # Check if any m is 0.
                if any(m == 0 for m in result):
                    # Shift A? Or just reject this A.
                    # If m=0, it means A=0 and r=0 for some d.
                    # We can just skip this A.
                    continue
                
                if L < best_L:
                    best_L = L
                    best_result = result
                found = True
                break
        
        if found:
            break
    
    if best_result is None:
        # Fallback: return the divisors themselves (valid but large span)
        # Or try a larger L
        # Let's just return a valid solution, even if not optimal.
        # The divisors themselves are distinct and valid.
        # Span = 39 - 6 = 33. Wait, they are distinct?
        # 6,10,14,15,21,22,26,33,35,39. Yes, all distinct.
        # Are they positive? Yes.
        # Are they <= 1000000? Yes.
        # So this is a valid fallback.
        best_result = P['divisors'][:]
    
    return best_result
# EVOLVE-BLOCK-END
