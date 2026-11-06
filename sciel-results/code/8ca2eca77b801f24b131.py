"""Strategy: Algebraic Construction + Vectorized Greedy Completion.
1. Start with a structured base set (e.g., intervals or modular grids) that guarantees many small differences.
2. Use a fast vectorized score calculation.
3. Perform a greedy completion phase: iteratively pick the element x in 1..512 \ S that maximizes the number of NEW distinct differences introduced.
4. Use simulated annealing only on the remaining few slots if time permits, or just return the greedy result.
"""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    if not S:
        return 0
    S_arr = np.array(sorted(S), dtype=np.int32)
    n = len(S_arr)
    if n < 3:
        return 0
    
    # Vectorized check:
    # For each pair (i, j) with i < j, d = S[j] - S[i].
    # We need to check if S[i] + 2*d is in S.
    # Instead of O(n^2) python loops, we can use sets for lookup which is O(n^2) in C speed.
    # Or better:
    # diffs = set()
    # S_set = set(S)
    # for a in S:
    #   for b in S:
    #     d = b - a
    #     if d > 0 and a + 2*d in S_set:
    #       diffs.add(d)
    # return len(diffs)
    
    # Let's stick to the optimized Python set version as n=32 is small, 
    # but we will call it frequently.
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _get_diff_set(S):
    """Return the set of differences."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return diffs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    n_target = 32
    max_val = 512
    
    best_set = None
    best_score = -1
    
    # Strategy 1: Start with a dense interval.
    # {1, ..., 32} has diffs 1..15.
    # Let's try a few different structured starts.
    
    start_candidates = [
        list(range(1, 33)),           # 1..32
        list(range(1, 17)) + list(range(33, 49)), # Two intervals
        list(range(1, 33, 2)) + list(range(2, 34, 2)), # Same as 1..32
        # A "checkerboard" like structure?
        # Let's try a random structured start:
        # Pick a random base b, then take b + i*step for i in 0..31?
        # If step=1, it's an interval.
        # If step=2, it's 1,3,5..65. Diffs are even numbers?
        # Actually, mixing intervals is good.
    ]
    
    # Add some random starts
    for _ in range(10):
        start_candidates.append(random.sample(range(1, max_val + 1), n_target))
        
    # We will run a "Greedy + SA" pipeline on each start.
    # But 10 starts * (Greedy + SA) might be too slow if not optimized.
    
    # Let's focus on the best start first, then try others if time permits.
    
    # Pre-calculate all possible elements
    all_elements = set(range(1, max_val + 1))
    
    for start_set in start_candidates:
        if time.time() > deadline - 10.0:
            break
            
        S = set(start_set)
        if len(S) != n_target:
            continue
            
        current_diffs = _get_diff_set(S)
        current_score = len(current_diffs)
        
        if current_score > best_score:
            best_score = current_score
            best_set = list(S)
            
        # Phase 1: Greedy Completion/Improvement
        # We have 32 elements. We can swap elements.
        # Greedy: Find the best swap (remove x, add y) that maximizes score.
        # To speed up, we don't recompute from scratch.
        # But n=32 is small, so O(n * (n^2)) = O(n^3) = 32768 ops per swap evaluation.
        # 32 removals * 512 additions = 16384 evaluations.
        # 16384 * 1024 (inner loop) = 16M ops. Too slow for Python in 1s.
        
        # Optimization: 
        # When removing x, we lose diffs that depend on x.
        # When adding y, we gain diffs that depend on y.
        # This is complex to track incrementally without bugs.
        
        # Alternative Fast Heuristic:
        # "Local Search with Random Restarts" but faster.
        # Since n=32, we can afford ~1000 SA steps per second if optimized.
        
        # Let's do a simple, fast SA on this start.
        T = 1.0
        S_list = list(S)
        S_set = set(S_list)
        
        # Fast Score Function using Numpy?
        # Actually, for n=32, Python sets are fine if we minimize overhead.
        
        for step in range(2000):
            if time.time() > deadline - 5.0:
                break
                
            # Pick a random element to remove
            idx = random.randint(0, n_target - 1)
            x_remove = S_list[idx]
            
            # Pick a random element to add
            # Bias towards elements that are "close" to existing ones?
            # Or just random.
            x_add = random.randint(1, max_val)
            
            if x_add in S_set:
                continue
                
            # Tentative move
            S_set.discard(x_remove)
            S_set.add(x_add)
            
            # Calculate new score
            new_score = _count_progression_diffs(S_set)
            
            # Accept or reject
            if new_score > current_score or (random.random() < np.exp((new_score - current_score) / T)):
                # Accept
                S_list[idx] = x_add
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(S_list)
            else:
                # Reject
                S_set.discard(x_add)
                S_set.add(x_remove)
            
            # Cool down
            T *= 0.999
            if T < 0.01:
                T = 0.01

    # Final check
    if best_set is None:
        return list(range(1, 33))
        
    return best_set
# EVOLVE-BLOCK-END
