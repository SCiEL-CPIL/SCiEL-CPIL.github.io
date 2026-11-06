import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    best_set = []
    
    def is_valid_addition(S, x, sums_set):
        """Check if adding x to S preserves unique triple sums.
        S is sorted list, sums_set is the set of all existing triple sums."""
        n = len(S)
        # New triple sums involving x:
        # (x,x,x), (x,x,s_i), (x,s_i,s_j) where s_i <= s_j in S
        new_sums = []
        new_sums.append(3 * x)
        for s in S:
            new_sums.append(2 * x + s)
        for i in range(n):
            for j in range(i, n):
                new_sums.append(x + S[i] + S[j])
        
        # Check for duplicates among new_sums
        if len(new_sums) != len(set(new_sums)):
            return False
        
        # Check against existing sums
        for s in new_sums:
            if s in sums_set:
                return False
        
        return True
    
    def build_greedy(candidate_order):
        """Build a set greedily following candidate_order."""
        S = []
        sums_set = set()
        for x in candidate_order:
            if x in S:
                continue
            if is_valid_addition(S, x, sums_set):
                S.append(x)
                # Update sums_set with new triple sums
                n = len(S) - 1  # previous size
                # Add (x,x,x), (x,x,s_i), (x,s_i,s_j)
                sums_set.add(3 * x)
                for i in range(n):
                    sums_set.add(2 * x + S[i])
                for i in range(n):
                    for j in range(i, n):
                        sums_set.add(x + S[i] + S[j])
        return S
    
    # Try multiple strategies
    # Strategy 1: Random permutations of [1, 4096]
    # Strategy 2: Random subsets
    
    rng = random.Random(42)
    
    # First, try a deterministic greedy from 1 upward as baseline
    baseline = build_greedy(range(1, 4097))
    if len(baseline) > len(best_set):
        best_set = baseline
    
    # Try random restarts
    for attempt in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Random permutation of a subset
        # Use random sampling from [1, 4096]
        # Try different subset sizes and ranges
        subset_size = rng.randint(500, 2000)
        candidates = rng.sample(range(1, 4097), subset_size)
        rng.shuffle(candidates)
        
        S = build_greedy(candidates)
        if len(S) > len(best_set):
            best_set = S
    
    # Try some more targeted strategies
    # Larger numbers give more room
    for attempt in range(100):
        if time.time() > deadline - 3.0:
            break
        
        # Try with emphasis on larger numbers
        candidates = list(range(2000, 4097))
        rng.shuffle(candidates)
        
        S = build_greedy(candidates)
        if len(S) > len(best_set):
            best_set = S
    
    return best_set
