import random
import time

def _score_set(S):
    """Count distinct d > 0 such that ∃a: a, a+d, a+2d all in S."""
    S_set = set(S)
    distinct_d = set()
    for a in S:
        for d in range(1, 513 - a):
            if a + d in S_set and a + 2*d in S_set:
                distinct_d.add(d)
    return len(distinct_d)

def _dense_seeds(n=32):
    """Return n integers from [1,512] that participate in many 3-APs."""
    # Count for each x in [1,512] how many 3-APs it participates in
    # A 3-AP is (a, a+d, a+2d). x participates if x=a, x=a+d, or x=a+2d for some a,d.
    # For x as middle: a = x-d, a+2d = x+d, need 1<=x-d and x+d<=512, d>=1
    # For x as first: a=x, need x+d, x+2d in [1,512], d>=1
    # For x as last: a+2d=x, a=x-2d, need a>=1, d>=1
    # Total participation count for x:
    # as middle: d can be 1..min(x-1, 512-x)
    # as first: d can be 1..(512-x)//2
    # as last: d can be 1..(x-1)//2
    counts = []
    for x in range(1, 513):
        c_mid = min(x - 1, 512 - x)
        c_first = (512 - x) // 2
        c_last = (x - 1) // 2
        total = c_mid + c_first + c_last
        counts.append((total, x))
    counts.sort(reverse=True)
    return [x for _, x in counts[:n]]

def construct():
    # Pre-compute dense seeds
    seeds = _dense_seeds(32)
    
    # Initial set: top 32 dense seeds
    current = set(seeds)
    current_score = _score_set(current)
    
    # Simulated annealing
    best = set(current)
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110  # seconds, leave buffer for 120s limit
    
    # All candidates: all numbers 1..512
    all_nums = list(range(1, 513))
    
    temp = 5.0
    cooling = 0.9995
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        # Pick a random element to remove
        to_remove = random.choice(list(current))
        # Pick a random element to add (not already in current)
        candidates_to_add = [x for x in all_nums if x not in current]
        to_add = random.choice(candidates_to_add)
        
        new_set = set(current)
        new_set.discard(to_remove)
        new_set.add(to_add)
        
        new_score = _score_set(new_set)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best = set(current)
                best_score = current_score
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
    
    return sorted(best)
