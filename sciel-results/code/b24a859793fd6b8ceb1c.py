import time
import random
from itertools import combinations

def is_sidon_4(subset):
    # subset is a tuple of 4 distinct integers
    # Check if all unordered pair sums (including diagonals) are unique
    sums = set()
    for i in range(4):
        for j in range(i, 4):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def has_sidon_4(a_set):
    for comb in combinations(a_set, 4):
        if is_sidon_4(comb):
            return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = list(range(12))
    best_score = 4 # Baseline max Sidon size
    
    # We want to find a set with max Sidon size 3.
    # This means NO 4-subset is Sidon.
    
    # Strategy 1: Random search in small ranges
    # Bias towards smaller ranges to increase collision probability
    for _ in range(500000):
        if time.time() > deadline:
            break
        
        # Try small ranges first
        if random.random() < 0.5:
            r = random.randint(12, 20)
        else:
            r = random.randint(20, 40)
            
        if r < 11:
            continue
            
        s = random.sample(range(r + 1), 12)
        
        # Check if any 4-subset is Sidon
        if not has_sidon_4(s):
            # Found a set with max Sidon size 3
            return s
            
    # If we didn't find a 3, return the baseline (which is 4)
    # We could also try to find a 4 that is "more dense" but score is same.
    return best_set
