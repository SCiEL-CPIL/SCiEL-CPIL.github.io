import time
import random
from itertools import combinations

def is_sidon_fast(S):
    """Check if a list of integers is a Sidon set using bitmask."""
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
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon_fast(subset):
                return k
    return 2 # Any 2 distinct elements form a Sidon set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Initialize with dense sets. APs are good candidates for small max Sidon size.
    # Try a few initial candidates: Random dense, AP, etc.
    candidates = []
    
    # 1. Random dense sets
    for _ in range(5):
        S = random.sample(range(0, 25), 12)
        S.sort()
        candidates.append(S)
        
    # 2. Arithmetic Progressions
    # a, a+d, ..., a+11d. Range [0, 128].
    # Max sum 2*(a+11d) <= 256 -> a+11d <= 128.
    for d in range(1, 11):
        for a in range(0, 128 - 11*d + 1):
            S = [a + i*d for i in range(12)]
            candidates.append(S)
            
    # 3. Other structured sets? e.g. squares?
    # But dense is key.
    
    best_set = None
    best_score = 13 # Max possible is 12, so 13 is safe init
    
    # Evaluate initial candidates
    for S in candidates:
        if time.time() > deadline: break
        score = max_sidon_size(S)
        if score < best_score:
            best_score = score
            best_set = S[:]

    if best_set is None:
        best_set = list(range(12))
        best_score = max_sidon_size(best_set)

    S = best_set[:]
    
    # Local Search
    # We want to minimize score.
    # If score is already 4 (likely minimum for 12 elements in this range?), we might stop or try to verify.
    # Actually, min max Sidon size for 12 elements is likely 4 or 5.
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(12))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Try moving S[i] to a random value in a dense range
            # We keep the set dense to maintain collisions
            # Range [0, 30] is a good balance
            for _ in range(10): # Fewer swaps per position, more positions
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 30)
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
                    improved = True
                    # Reset swap count for this i if improved? Or just break?
                    break
                elif score == best_score and random.random() < 0.05:
                    # Accept equal to escape local minima
                    S = new_S
                    best_set = S[:]
                    # Don't set improved=True to allow other moves in this pass
                    # But we don't want to loop forever on equal scores
                    # So we just update S and continue
            
            if improved:
                # If we found a better score, restart the local search from this new state
                # This helps escape local minima more effectively
                indices = list(range(12))
                random.shuffle(indices)

    return best_set
