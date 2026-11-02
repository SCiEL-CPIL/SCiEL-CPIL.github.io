"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score for a set U of nonnegative integers containing 0."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n == 0:
        return 0.0
    maxU = max(U)
    if maxU == 0:
        return 0.0
    
    # Compute sumset U+U using numpy for speed
    arr = np.array(U, dtype=np.int64)
    # All pairwise sums
    sums = arr[:, None] + arr[None, :]
    sumset = set(sums.flatten().tolist())
    
    # Compute difference set U-U
    diffs = arr[:, None] - arr[None, :]
    diffset = set(diffs.flatten().tolist())
    
    ratio = len(diffset) / len(sumset)
    denom = math.log(2 * maxU + 1)
    if denom == 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom

def _score_set(U):
    """Alias for compute_score."""
    return compute_score(U)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    best_U = [0, 1, 3]
    best_score = compute_score(best_U)
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Use structured sets that are known to have good sumset/diffset properties.
    # The key insight: we want |U-U| large and |U+U| small relative to log(maxU).
    # For a set U, |U-U| <= |U|^2 and |U+U| >= |U|.
    # The ratio |U-U|/|U+U| is maximized when the set is "Sidon-like" for sums (few collisions)
    # but has many distinct differences.
    
    # Actually, for a Sidon set (B_2 sequence), |U+U| ~ |U|^2, and |U-U| ~ |U|^2.
    # But we want the RATIO to be large. 
    # If U is an arithmetic progression {0, d, 2d, ..., (n-1)d}, then:
    # |U+U| = 2n-1, |U-U| = 2n-1. Ratio = 1. Score = 1 + log(1)/log(...) = 1.
    # We need ratio > 1.
    # If U is a geometric progression {0, 1, b, b^2, ...}, then:
    # |U+U| is roughly n(n+1)/2 (if b is large enough to avoid collisions).
    # |U-U| is roughly n(n-1) + 1 (all differences are distinct if b is large).
    # Ratio ~ (n(n-1)) / (n(n+1)/2) = 2(n-1)/(n+1) ~ 2.
    # So geometric progressions give a ratio approaching 2.
    # Score = 1 + log(2) / log(2 * b^(n-1) + 1) ~ 1 + log(2) / ((n-1)log(b)*log(2)) = 1 + 1/((n-1)log(b)).
    # This approaches 1 as n or b grows. So we want small n and small b?
    # But maxU = b^(n-1). If b=2, n=3: U={0,1,2}, maxU=2. log(5)=1.609. log(2)=0.693. Score=1+0.693/1.609=1.43.
    # If b=2, n=4: U={0,1,2,4}, maxU=4. log(9)=2.197. log(2)=0.693. Score=1+0.693/2.197=1.316.
    # If b=2, n=5: U={0,1,2,4,8}, maxU=8. log(17)=2.833. Score=1+0.693/2.833=1.244.
    # So smaller sets with base 2 are better?
    # Let's check U={0,1,2}: sumset={0,1,2,3,4}, diffset={-2,-1,0,1,2}. Ratio=1. Score=1.
    # Wait, my geometric progression analysis assumed no collisions. For {0,1,2}, sums are 0,1,2,1,2,3,2,3,4 -> {0,1,2,3,4}. Diff: -2,-1,0,1,2. Ratio=1.
    # For {0,1,2,4}: sums: 0,1,2,4, 1,2,3,5, 2,3,4,6, 4,5,6,8 -> {0,1,2,3,4,5,6,8}. Size 8.
    # diffs: -4,-3,-2,0, -3,-2,-1,1, -2,-1,0,2, 0,1,2,4 -> {-4,-3,-2,-1,0,1,2,3,4}. Size 9.
    # Ratio = 9/8 = 1.125. Score = 1 + log(1.125)/log(9) = 1 + 0.1178/2.197 = 1.0536.
    # This is worse than 1.43? No, 1.43 was for n=3, b=2 which is {0,1,2} but I calculated ratio=1 there.
    # Let's re-evaluate {0,1,2}. Sumset={0,1,2,3,4} (size 5). Diffset={-2,-1,0,1,2} (size 5). Ratio=1. Score=1.
    # So geometric progressions don't help if collisions occur.
    
    # Let's try a set that is a "difference basis" or something.
    # The problem asks to maximize 1 + log(|U-U|/|U+U|)/log(2 max U + 1).
    # To maximize this, we want |U-U|/|U+U| to be large, and maxU to be small.
    # The maximum possible ratio is when |U+U| is minimal and |U-U| is maximal.
    # Minimal |U+U| for a set of size n is n (if U is a Sidon set with no repeated sums? No, minimal is n if U is {0,1,...,n-1}? No, {0,1,...,n-1} has sumset size 2n-1).
    # Actually, the minimum size of the sumset of a set of n integers is n, achieved when the set is an arithmetic progression? No, AP has size 2n-1.
    # The minimum size of the sumset is n, but only if the set is such that all sums are distinct? That's impossible for n>2 because a+a=b+b implies 2a=2b.
    # Wait, U+U is a set. The minimum size of U+U for a set U of size n is n. This is achieved if U is a "sum-free" set? No.
    # Actually, by Cauchy-Davenport, |U+U| >= 2|U|-1 for prime moduli, but in integers it's different.
    # For U = {0, 1, ..., n-1}, |U+U| = 2n-1.
    # For U = {0, 2, 4, ..., 2(n-1)}, |U+U| = 2n-1.
    # It is known that |U+U| >= 2n-1 for any set of n integers. Equality holds for arithmetic progressions.
    # So the minimum size of the sumset is 2n-1.
    # The maximum size of the difference set is n(n-1)+1 (all differences distinct).
    # So the maximum ratio is (n(n-1)+1)/(2n-1).
    # For large n, this is ~ n/2.
    # So we want n to be as large as possible, but maxU must be small.
    # If we take an arithmetic progression, ratio is 1.
    # If we take a set with all distinct differences (a Golomb ruler), then |U-U| = n(n-1)+1.
    # But what is |U+U| for a Golomb ruler?
    # For a Golomb ruler, the sums are not necessarily distinct.
    # However, if we choose the elements to grow exponentially, we can ensure both sums and differences are distinct.
    # Let U = {0, 1, 3, 7, 15, ...} (2^k - 1).
    # Then all sums and differences are distinct.
    # |U+U| = n(n+1)/2 (since a+b is unique for a<=b? No, a+b is unique for all pairs (a,b) with a!=b? No, a+b=b+a. So number of distinct sums is n(n+1)/2).
    # |U-U| = n(n-1)+1.
    # Ratio = (n(n-1)+1) / (n(n+1)/2) = 2(n^2-n+1)/(n^2+n) ~ 2.
    # So the ratio approaches 2.
    # Score = 1 + log(2) / log(2 * maxU + 1).
    # maxU = 2^(n-1) - 1.
    # log(2 * maxU + 1) = log(2^(n) - 1) ~ n log 2.
    # Score ~ 1 + log(2) / (n log 2) = 1 + 1/n.
    # So for n=3, Score ~ 1.33. For n=4, Score ~ 1.25.
    # This is consistent with my earlier calculation.
    
    # So the best strategy is to find a set with ratio close to 2 and small maxU.
    # The set {0, 1, 3} has maxU=3. Sumset={0,1,2,3,4,6} (size 6). Diffset={-3,-2,-1,0,1,2,3} (size 7). Ratio=7/6=1.166. Score=1+log(1.166)/log(7)=1+0.153/1.945=1.078.
    # The set {0, 1, 3, 7} has maxU=7. Sumset size 10. Diffset size 13. Ratio=1.3. Score=1+log(1.3)/log(15)=1+0.262/2.708=1.097.
    # The set {0, 1, 3, 7, 15} has maxU=15. Sumset size 15. Diffset size 21. Ratio=1.4. Score=1+log(1.4)/log(31)=1+0.336/3.434=1.098.
    # The set {0, 1, 3, 7, 15, 31} has maxU=31. Sumset size 21. Diffset size 31. Ratio=1.476. Score=1+log(1.476)/log(63)=1+0.389/4.143=1.094.
    # So the score peaks around n=5 or 6.
    
    # Let's try to optimize the set using local search, starting from {0, 1, 3, 7, 15}.
    
    U = [0, 1, 3, 7, 15]
    best_score = compute_score(U)
    best_U = list(U)
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        n = len(U)
        maxU = max(U)
        
        # Try adding a new element
        if n < 10:
            # Try adding an element that is roughly 2*maxU + 1 or similar
            new_val = 2 * maxU + 1
            if new_val not in U:
                new_U = U + [new_val]
                new_U.sort()
                s = compute_score(new_U)
                if s > best_score:
                    best_score = s
                    best_U = new_U
                    U = list(new_U)
                    continue
        
        # Try local moves
        moves = []
        for i in range(1, n):
            # Shift element i
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_val = U[i] + delta
                if new_val > 0 and new_val not in U:
                    new_U = list(U)
                    new_U[i] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        # Try replacing an element
        for i in range(1, n):
            for _ in range(5):
                new_val = random.randint(1, maxU + 5)
                if new_val not in U:
                    new_U = list(U)
                    new_U[i] = new_val
                    new_U.sort()
                    moves.append(new_U)
        
        if not moves:
            continue
        
        best_move_score = best_score
        best_move_U = U
        
        for new_U in moves:
            if time.time() > deadline:
                break
            s = compute_score(new_U)
            if s > best_move_score:
                best_move_score = s
                best_move_U = new_U
        
        if best_move_U != U:
            U = best_move_U
            best_score = best_move_score
            best_U = list(U)
    
    return best_U

# EVOLVE-BLOCK-END
