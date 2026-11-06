import time
import random
from itertools import combinations

def is_sidon_fast(S):
    """Check if a list of integers is a Sidon set including diagonal sums."""
    mask = 0
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            bit = 1 << s
            if mask & bit:
                return False
            mask |= bit
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon_fast(subset):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    best_set = None
    best_score = 13  # Max possible is 12
    
    # Phase 1: Generate diverse initial candidates
    candidates = []
    
    # Dense sets in small ranges - most promising for minimizing Sidon size
    # Range [0, 15] gives 16 values, we pick 12
    for _ in range(50):
        S = sorted(random.sample(range(0, 16), 12))
        candidates.append(S)
    
    # Even tighter: [0, 13]
    for _ in range(30):
        S = sorted(random.sample(range(0, 14), 12))
        candidates.append(S)
    
    # [0, 12] - only 13 values, pick 12
    for _ in range(20):
        S = sorted(random.sample(range(0, 13), 12))
        candidates.append(S)
    
    # Consecutive-like with small gaps
    for start in range(0, 5):
        for pattern in [[0,1,2,3,4,5,6,7,8,9,10,11],
                        [0,1,2,3,4,5,6,7,8,9,10,12],
                        [0,1,2,3,4,5,6,7,8,9,11,12],
                        [0,1,2,3,4,5,6,7,8,10,11,12],
                        [0,1,2,3,4,5,6,7,9,10,11,12],
                        [0,1,2,3,4,5,6,8,9,10,11,12],
                        [0,1,2,3,4,5,7,8,9,10,11,12],
                        [0,1,2,3,4,6,7,8,9,10,11,12],
                        [0,1,2,3,5,6,7,8,9,10,11,12],
                        [0,1,2,4,5,6,7,8,9,10,11,12],
                        [0,1,3,4,5,6,7,8,9,10,11,12],
                        [0,2,3,4,5,6,7,8,9,10,11,12]]:
            S = [start + x for x in pattern]
            if S[-1] <= 128:
                candidates.append(S)
    
    # Evaluate initial candidates
    for S in candidates:
        if time.time() > deadline:
            break
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S[:]
    
    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)
    
    S = best_set[:]
    
    # Phase 2: Local search - try to improve by swapping elements
    rng = random.Random(42)
    improved = True
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline and improved and no_improve_count < max_no_improve:
        improved = False
        no_improve_count += 1
        
        # Try all single-element swaps
        for i in range(12):
            if time.time() > deadline:
                break
            
            # Try moving element i to a new value in a small range
            # Focus on values that create more collisions
            tried = set()
            for _ in range(30):
                if time.time() > deadline:
                    break
                
                # Sample from a tight range around existing values
                min_val = min(S)
                max_val = max(S)
                new_val = rng.randint(max(0, min_val - 2), min(128, max_val + 2))
                
                if new_val in tried or new_val in S:
                    continue
                tried.add(new_val)
                
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                
                score = max_sidon_size(new_S)
                if score < best_score:
                    best_score = score
                    S = new_S
                    improved = True
                    no_improve_count = 0
                    break
                elif score == best_score and rng.random() < 0.1:
                    # Accept equal to explore
                    S = new_S
                    improved = True
                    no_improve_count = 0
                    break
        
        if not improved:
            # Try more aggressive moves: swap two elements
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                i, j = rng.sample(range(12), 2)
                new_i = rng.randint(0, min(128, max(S) + 3))
                new_j = rng.randint(0, min(128, max(S) + 3))
                
                if new_i in S or new_j in S or new_i == new_j:
                    continue
                
                new_S = S[:]
                new_S[i] = new_i
                new_S[j] = new_j
                new_S.sort()
                
                score = max_sidon_size(new_S)
                if score < best_score:
                    best_score = score
                    S = new_S
                    improved = True
                    no_improve_count = 0
                    break
                elif score == best_score and rng.random() < 0.05:
                    S = new_S
                    improved = True
                    no_improve_count = 0
                    break
    
    # Phase 3: Final systematic polish
    if time.time() < deadline:
        for i in range(12):
            if time.time() > deadline:
                break
            for new_val in range(0, min(20, max(S) + 5)):
                if new_val in S:
                    continue
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                score = max_sidon_size(new_S)
                if score < best_score:
                    best_score = score
                    S = new_S
                    best_set = S[:]
    
    # Return the best set found
    result = best_set if best_set is not None else S
    result = sorted(result)
    
    # Ensure validity
    assert len(result) == 12
    assert len(set(result)) == 12
    assert all(0 <= x <= 128 for x in result)
    
    return result
