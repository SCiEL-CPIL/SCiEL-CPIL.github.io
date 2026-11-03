"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    def score_set(U):
        if not U:
            return 0.0
        U_set = set(U)
        max_U = max(U_set)
        if max_U == 0:
            return 0.0
        
        # Compute sumset and difference set
        # For small sets, O(n^2) is fine.
        # To be safe for slightly larger sets, we can use numpy if needed, but n will be small.
        sumset = set()
        diffset = set()
        for a in U_set:
            for b in U_set:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n = len(U_set)
        if n == 0:
            return 0.0
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        
        try:
            exponent = 1 + math.log(ratio) / math.log(2 * max_U + 1)
        except:
            return 0.0
        return exponent

    # Strategy: Search for geometric progressions U = {0, 1, B, B^2, ..., B^(n-1)}
    # The size of the sumset for such a set with base B is roughly n(n+1)/2 if B is large enough
    # (no carries/overlaps), and diffset is roughly n^2 - 1 (or n^2 depending on symmetry).
    # Specifically, if B > n, all sums a_i + a_j are distinct for i<=j? No.
    # For geometric progression with base B, the sums are unique if B > 1.
    # Let U = {0, 1, B, B^2, ..., B^{n-1}}.
    # Sumset size: The number of distinct sums of two elements.
    # Since B is large, the representation is unique in base B.
    # Sum of B^i and B^j (i<=j). If i<j, it's B^i + B^j. If i=j, it's 2B^i.
    # If 2B^i < B^{i+1} (i.e., B > 2), these are all distinct.
    # So |U+U| = n(n+1)/2 (pairs with replacement) + (n-1) (pairs i<j where order doesn't matter for sum? No, sum is commutative).
    # Actually, distinct values in U+U:
    # Indices (i, j) with 0 <= i <= j < n.
    # Value is B^i + B^j.
    # If i < j, value is B^i + B^j.
    # If i == j, value is 2 B^i.
    # Are these all distinct?
    # B^i + B^j = B^k + B^l.
    # If B is large enough, yes.
    # So |U+U| = n(n+1)/2.
    #
    # Diffset size:
    # Differences B^i - B^j.
    # i != j.
    # B^i - B^j = B^j (B^{i-j} - 1).
    # These are distinct for different (i,j) pairs?
    # If i > j, B^i - B^j is positive.
    # If i < j, B^i - B^j is negative.
    # 0 is included (i=j).
    # Number of positive differences: n(n-1)/2.
    # Number of negative differences: n(n-1)/2.
    # Plus 0.
    # Are any positive differences equal to negative ones? No, signs differ.
    # Are any positive differences equal to each other?
    # B^i - B^j = B^k - B^l.
    # If B is large, likely distinct.
    # So |U-U| = n(n-1) + 1.
    
    # Ratio R = |U-U| / |U+U| = (n^2 - n + 1) / (n(n+1)/2) = 2(n^2 - n + 1) / (n^2 + n)
    # As n -> infinity, R -> 2.
    # Score S = 1 + log(R) / log(2 * B^(n-1) + 1)
    # S = 1 + log(R) / ( (n-1) * log(2B) )  (approx)
    
    # To maximize S, we want to maximize log(R) / (n-1).
    # R(n) = 2(n^2 - n + 1) / (n^2 + n)
    # Let's check R for small n:
    # n=1: R = 2(1)/1 = 2. (U={0} or {1}? U must contain 0. If n=1, U={0}. Max=0. Invalid. So n>=2.)
    # If U={0, 1}, n=2. Max=1.
    #   Sumset: {0,1,2}. Size 3.
    #   Diffset: {-1, 0, 1}. Size 3.
    #   R = 1.
    #   Formula: 2(4-2+1)/(4+2) = 6/6 = 1. Correct.
    #   S = 1 + log(1)/log(3) = 1.
    
    # n=3: U={0, 1, B}.
    #   R = 2(9-3+1)/(9+3) = 14/12 = 1.166.
    #   log(R) ~ 0.154.
    #   Denom: log(2B^2).
    #   If B=2, U={0,1,2}. Not geometric with B>2.
    #   Let's just evaluate specific sets.
    
    best_score = 0.0
    best_set = [0, 1]
    
    # Evaluate geometric progressions
    # U = {0, 1, B, B^2, ..., B^(n-1)}
    # We need B >= 2.
    
    # Search space for n: 2 to 10
    # Search space for B: 2 to 100
    
    for n in range(2, 12):
        for B in range(2, 101):
            if time.time() > deadline:
                return best_set
            
            # Construct U
            U = [0]
            current = 1
            for i in range(1, n):
                U.append(current)
                current *= B
            
            # Check if max U is too large?
            # Score penalizes max U.
            # If B is large, max U is huge, log(2 max U) is large, score drops.
            # So small B is likely better.
            
            s = score_set(U)
            if s > best_score:
                best_score = s
                best_set = U
                
    # Also try a few other structured sets
    # Arithmetic progression?
    # U = {0, 1, 2, ..., n-1}
    # Sumset size: 2n-1.
    # Diffset size: 2n-1.
    # Ratio: 1.
    # Score: 1.
    # This is the baseline.
    
    # Try "Sparse" sets with random gaps?
    # But geometric is analytically strong for ratio -> 2.
    
    # Let's do a quick local search around the best geometric set found
    current_best = list(best_set)
    current_score = best_score
    
    # Try adding/removing elements or modifying last element
    # We'll stick to the geometric family primarily as it's deterministic and fast.
    
    # One more check: The paper mentions size 2003.
    # Our n is small (2-11).
    # Does larger n help?
    # S(n, B) = 1 + log(2(n^2-n+1)/(n^2+n)) / log(2 B^{n-1} + 1)
    # For B=2:
    # n=2: 1 + log(1)/log(3) = 1.
    # n=3: U={0,1,2}. Sum={0,1,2,3,4} (5). Diff={-2..2} (5). R=1. S=1.
    #   Wait, geometric B=2, n=3 is {0,1,2}.
    # n=4: U={0,1,2,4}.
    #   Sumset: 0..8?
    #   0,1,2,4, 1+1=2, 1+2=3, 1+4=5, 2+2=4, 2+4=6, 4+4=8.
    #   {0,1,2,3,4,5,6,8}. Size 8.
    #   Diffset:
    #   Pos: 1,2,4, 2-1=1, 4-1=3, 4-2=2. {1,2,3,4}.
    #   Neg: {-1,-2,-3,-4}.
    #   0.
    #   Size 9.
    #   R = 9/8 = 1.125.
    #   log(1.125) ~ 0.117.
    #   log(2*4+1) = log(9) ~ 2.197.
    #   S = 1 + 0.117/2.197 = 1.053.
    #   This is > 1.0!
    
    # n=5: U={0,1,2,4,8}.
    #   Max=8. Log(17) ~ 2.83.
    #   Sumset size?
    #   Powers of 2. Sums are unique.
    #   Pairs (i,j) i<=j.
    #   Count = n(n+1)/2 = 15.
    #   Diffset size: n(n-1)+1 = 21.
    #   R = 21/15 = 1.4.
    #   log(1.4) ~ 0.336.
    #   S = 1 + 0.336/2.83 = 1.118.
    
    # n=6: U={0,1,2,4,8,16}.
    #   Max=16. Log(33) ~ 3.49.
    #   Sumset size: 6*7/2 = 21.
    #   Diffset size: 6*5+1 = 31.
    #   R = 31/21 ~ 1.476.
    #   log(1.476) ~ 0.389.
    #   S = 1 + 0.389/3.49 = 1.111.
    #   Slight drop.
    
    # n=7: U={0,1,2,4,8,16,32}.
    #   Max=32. Log(65) ~ 4.17.
    #   Sumset: 7*8/2 = 28.
    #   Diffset: 7*6+1 = 43.
    #   R = 43/28 ~ 1.535.
    #   log(1.535) ~ 0.428.
    #   S = 1 + 0.428/4.17 = 1.102.
    
    # It seems n=5 or 6 with B=2 is good.
    # Let's verify B=3.
    # n=3, B=3: U={0,1,3}.
    #   Sum: 0,1,2,3,4,6. Size 6.
    #   Diff: -3..3. Size 7.
    #   R = 7/6 = 1.166.
    #   Log(7) ~ 1.94.
    #   S = 1 + log(1.166)/1.94 = 1 + 0.153/1.94 = 1.079.
    #   Better than B=2, n=3 (which was 1.0).
    
    # n=4, B=3: U={0,1,3,9}.
    #   Max=9. Log(19) ~ 2.94.
    #   Sumset: 4*5/2 = 10.
    #   Diffset: 4*3+1 = 13.
    #   R = 13/10 = 1.3.
    #   log(1.3) ~ 0.262.
    #   S = 1 + 0.262/2.94 = 1.089.
    
    # n=5, B=3: U={0,1,3,9,27}.
    #   Max=27. Log(55) ~ 4.0.
    #   Sumset: 5*6/2 = 15.
    #   Diffset: 5*4+1 = 21.
    #   R = 21/15 = 1.4.
    #   log(1.4) ~ 0.336.
    #   S = 1 + 0.336/4.0 = 1.084.
    
    # So B=2, n=5 seems strong (~1.118).
    # Let's check B=2, n=4 again. 1.053.
    # Let's check B=2, n=6. 1.111.
    # Let's check B=2, n=7. 1.102.
    
    # What about B=2, n=8?
    # Max=128. Log(257) ~ 5.54.
    # Sumset: 36. Diffset: 65.
    # R = 65/36 = 1.805.
    # log(1.805) ~ 0.59.
    # S = 1 + 0.59/5.54 = 1.106.
    
    # It looks like the peak is around n=5, B=2.
    # Let's run the loop to be sure and pick the best.
    
    return best_set
# EVOLVE-BLOCK-END
