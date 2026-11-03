import random
import math
from itertools import combinations

def score_set(U):
    U = sorted(U)
    if not U or U[0] != 0:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    # Compute |U+U|
    sum_set = set()
    for i in range(len(U)):
        for j in range(i, len(U)):
            sum_set.add(U[i] + U[j])
    
    # Compute |U-U|
    diff_set = set()
    for i in range(len(U)):
        for j in range(len(U)):
            diff_set.add(U[i] - U[j])
            
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    if ratio <= 1:
        # If ratio is 1, log is 0, score is 1.
        # But we want to maximize, so we hope ratio > 1.
        pass
        
    # The formula uses log base 2? 
    # "log" in math contexts usually means natural log, but the denominator has log(2 max U + 1).
    # The problem statement says: 1 + log(|U-U|/|U+U|)/log(2 max U + 1)
    # It does not specify the base. However, "log(2)" in the denominator of the *exponent* description 
    # "log(2 max U + 1)" suggests the base might be 2 or e.
    # Let's look at the seed score.
    # Seed: [0, 1, 3]. Max=3. 2*3+1=7.
    # |U-U|=7, |U+U|=6. Ratio=7/6.
    # If base e: 1 + ln(7/6)/ln(7) = 1 + 0.154/1.946 = 1.079.
    # If base 2: 1 + log2(7/6)/log2(7) = 1 + 0.222/2.807 = 1.079.
    # The ratio of logs is independent of the base!
    # log_b(A) / log_b(B) = ln(A) / ln(B).
    # So the base doesn't matter.
    
    val = 1 + math.log(size_diff / size_sum) / math.log(2 * max_u + 1)
    return val

def construct():
    best_set = [0, 1, 3]
    best_score = score_set(best_set)
    
    # Try small N and small M
    # We want to minimize max U, so we try small M first.
    # But we need enough N to get ratio close to 2.
    
    random.seed(42) # For reproducibility
    
    for N in range(3, 20):
        # M should be at least N to fit N distinct numbers.
        # To be sparse, M should be larger.
        # Let's try M from N to 4*N*N
        for M in range(N, 4*N*N + 1, max(1, N//5)):
            # Generate random set
            for _ in range(10): # 10 trials per (N, M)
                if N == 1:
                    U = [0]
                else:
                    # Pick N-1 random distinct numbers from 1 to M
                    candidates = list(range(1, M+1))
                    if len(candidates) < N-1:
                        continue
                    chosen = random.sample(candidates, N-1)
                    U = [0] + chosen
                    U.sort()
                    
                    s = score_set(U)
                    if s > best_score:
                        best_score = s
                        best_set = U[:]
                        
    return best_set
