import itertools
import math

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    # log base 2 or natural log? 
    # The formula uses log, usually natural log in math, but ratio of logs is base independent.
    # denominator: log(2 * max_u + 1)
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    best_score = 0
    best_set = [0, 1, 3]
    
    # Search for sets of size 3 to 6
    # Range up to 20 seems sufficient for small N
    # We can extend range if needed, but 20 is small enough for brute force
    max_val = 20
    
    for n in range(3, 7):
        # Choose n-1 elements from 1..max_val, add 0
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            U = [0] + list(combo)
            # Optional: check if 0 is smallest, it is.
            score = calculate_score(U)
            if score > best_score:
                best_score = score
                best_set = U
                
    return best_set
