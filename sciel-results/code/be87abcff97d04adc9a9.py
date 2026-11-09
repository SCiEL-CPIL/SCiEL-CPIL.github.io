import random
import time
from itertools import combinations

def is_sidon_4(subset):
    """Check if a tuple of 4 integers is a Sidon set."""
    # subset is (a, b, c, d)
    a, b, c, d = subset
    sums = [
        a + a, a + b, a + c, a + d,
        b + b, b + c, b + d,
        c + c, c + d,
        d + d
    ]
    # Check for duplicates
    # Optimization: sort and check adjacent, or use set
    # Set is fast enough for 10 elements
    return len(set(sums)) == 10

def has_sidon_subset_4(S):
    """Check if any subset of S of size 4 is Sidon."""
    # S is a list of 12 distinct integers
    # We can iterate through all combinations of 4
    for comb in combinations(S, 4):
        if is_sidon_4(comb):
            return True
    return False

def get_max_sidon_size_exact(S):
    """Find the exact size of the largest Sidon subset of S."""
    # We primarily care about distinguishing 3 vs 4+
    # If no size 4, it's 3 (since size 3 is almost always possible for 12 distinct ints)
    # Actually, can max Sidon size be 2?
    # A set of 3 distinct ints is Sidon iff it's not an AP.
    # If S has 12 elements, it's very likely to contain a non-AP triple.
    # So if no size 4, size is 3.
    
    if has_sidon_subset_4(S):
        # It could be 5, 6...
        # For the purpose of minimization, 4 is better than 5.
        # We might want to know if it's 4 or >4 to prioritize.
        # But finding a set with max size 3 is the primary goal.
        # If we can't find 3, we return the best we have (min 4).
        # To differentiate, we can check for size 5.
        # But C(12,5) = 792. Still fast.
        # Let's just return 4 for now, or refine if needed.
        # For now, assume if we fail to find 3, we are stuck with 4+.
        # We will try to minimize it further if we have time.
        return 4 
        
    return 3

def construct():
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for safety
    
    best_set = None
    best_score = float('inf')
    
    # Strategy: 
    # 1. Try to find a set with max Sidon size 3.
    # 2. If not found, keep the set with min max Sidon size (likely 4).
    
    # Heuristic: Sets from small ranges are more likely to have collisions.
    # Try ranges [0, 15], [0, 20], [0, 30]
    
    # Initial best: a random set from a small range
    best_set = random.sample(range(0, 20), 12)
    best_score = get_max_sidon_size_exact(best_set)
    
    # Local Search / Random Search
    while time.time() - start_time < time_limit:
        # Pick a strategy
        # 50% random new set from small range
        # 30% random new set from medium range
        # 20% mutate best_set
        
        r = random.random()
        if r < 0.5:
            # Small range
            pool_size = random.randint(15, 25)
            S = random.sample(range(0, pool_size), 12)
        elif r < 0.8:
            # Medium range
            pool_size = random.randint(25, 40)
            S = random.sample(range(0, pool_size), 12)
        else:
            # Mutate best_set
            S = best_set[:]
            idx = random.randint(0, 11)
            # Pick a new value
            # Bias towards values that cause collisions
            # Just pick random from 0..30 first
            new_val = random.randint(0, 30)
            # Ensure distinct
            attempts = 0
            while new_val in S and attempts < 10:
                new_val = random.randint(0, 30)
                attempts += 1
            if new_val in S:
                continue
            S[idx] = new_val
        
        # Evaluate
        # Fast check: does it have a Sidon subset of size 4?
        if not has_sidon_subset_4(S):
            # Max Sidon size is 3. This is the optimal target.
            # We can return immediately or check if it's even better (2)?
            # Size 2 is impossible for 12 distinct ints (will always have a non-AP triple).
            # So 3 is the minimum possible.
            return S
        
        # If it has a Sidon subset of size 4, score is >= 4.
        # Calculate exact score to track best "bad" set.
        # We know it's at least 4.
        # Is it 5?
        # To save time, we might just treat it as 4 unless we have lots of time.
        # But if we find a set with score 4, and later one with score 5, we prefer 4.
        # So we need to distinguish.
        
        # Check for size 5
        # C(12,5) = 792.
        # is_sidon for size 5: 15 sums.
        # This is fast.
        
        has_5 = False
        for comb in combinations(S, 5):
            # Check if Sidon
            # a,b,c,d,e
            a,b,c,d,e = comb
            sums = [
                a+a, a+b, a+c, a+d, a+e,
                b+b, b+c, b+d, b+e,
                c+c, c+d, c+e,
                d+d, d+e,
                e+e
            ]
            if len(set(sums)) == 15:
                has_5 = True
                break
        
        if has_5:
            score = 5 # Could be higher, but 5 is worse than 4
            # Check 6? C(12,6)=924. 21 sums.
            # If we have time, check 6.
            # But 4 is the best "bad" score.
        else:
            score = 4
            
        if score < best_score:
            best_score = score
            best_set = S[:]
            
        # If best_score is 4, we are doing okay.
        # We can keep searching for 3.
        
    return best_set
