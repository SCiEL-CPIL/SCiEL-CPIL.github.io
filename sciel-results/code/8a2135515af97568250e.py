import time
import random
from itertools import combinations

def is_sidon(S):
    """Check if a list of integers is a Sidon set (unique pair sums including diagonals)."""
    sums = set()
    n = len(S)
    for i in range(n):
        si = S[i]
        for j in range(i, n):
            s = si + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # Check from largest possible down to smallest
    for k in range(n, 1, -1):
        for subset in combinations(S, k):
            if is_sidon(subset):
                return k
    return 2 # Any 2 distinct elements form a Sidon set

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Initial guess: random
    S = random.sample(range(129), 12)
    S.sort()
    best_set = S[:]
    best_score = max_sidon_size(S)
    
    # Local search
    while time.time() < deadline:
        improved = False
        for i in range(12):
            if time.time() > deadline:
                break
            # Try swapping S[i] with a random value
            for _ in range(50): # Limit swaps per position
                if time.time() > deadline:
                    break
                new_val = random.randint(0, 128)
                if new_val in S:
                    continue
                
                # Create new set
                new_S = S[:]
                new_S[i] = new_val
                new_S.sort()
                
                # Compute score
                score = max_sidon_size(new_S)
                
                if score < best_score:
                    best_score = score
                    S = new_S
                    best_set = S
                    improved = True
                elif score == best_score and random.random() < 0.05:
                    # Occasionally accept equal score to escape local minima
                    S = new_S
                    best_set = S
                    improved = True
        
        if not improved:
            # Stuck, restart with random
            S = random.sample(range(129), 12)
            S.sort()
            score = max_sidon_size(S)
            if score < best_score:
                best_score = score
                best_set = S

    if best_set is None:
        return list(range(12))
        
    return best_set
