"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    # To speed up, we can iterate a, then b, check c
    # But for n=32, O(n^2) is fine (1024 checks per score)
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def construct():
    """
    Construct a 32-element set in 1..512 maximizing the number of distinct 
    3-term arithmetic progression differences.
    Uses Simulated Annealing with a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Initial random set
    current_set = set(random.sample(range(1, 513), 32))
    current_score = _count_progression_diffs(current_set)
    
    best_set = set(current_set)
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    # Pre-compute valid range for potential additions/removals
    # We will use a random move strategy: 
    # 1. Remove a random element from current_set
    # 2. Add a random element from 1..512 not in current_set
    
    while time.time() < deadline and T > min_T:
        # Generate neighbor
        # Remove one element
        to_remove = random.choice(list(current_set))
        new_set = set(current_set)
        new_set.discard(to_remove)
        
        # Add one element
        # To avoid infinite loop if new_set is full (it's 31 elements, so always space)
        # We pick from 1..512
        to_add = random.randint(1, 512)
        # It's possible to pick to_remove, which is a no-op move, but rare enough 
        # or we can just accept it as a 0-delta move.
        # Better: pick from available? 
        # Generating a random int and checking membership is O(1).
        # If to_add is in new_set, it's a no-op (score unchanged). 
        # To ensure movement, we can retry a few times or just accept.
        # Let's just add it. If it's already there, set size stays 31? 
        # No, new_set has 31 elements. If to_add is in new_set, size stays 31. 
        # We need size 32.
        # So we must ensure to_add is not in new_set.
        
        # Efficient way to pick a random element NOT in a set of size 31 from 512:
        # Just pick random until not in set. Expected trials < 1.07.
        while to_add in new_set:
            to_add = random.randint(1, 512)
            
        new_set.add(to_add)
        
        new_score = _count_progression_diffs(new_set)
        
        delta = new_score - current_score
        
        # Acceptance probability
        if delta > 0 or random.random() < pow(2.718281828, delta / T):
            current_set = new_set
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = set(current_set)
        
        T *= cooling_rate
        
        # Check time periodically
        if time.time() > deadline - 0.5:
            break

    return list(best_set)
# EVOLVE-BLOCK-END
