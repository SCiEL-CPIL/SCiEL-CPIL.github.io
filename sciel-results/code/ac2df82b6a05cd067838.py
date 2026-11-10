import itertools
import random
import time

def is_sidon(S):
    """
    Check if set S is a Sidon set.
    Condition: All sums s_i + s_j for i <= j are distinct.
    """
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Find the size of the largest Sidon subset of S.
    S is a list of distinct integers.
    """
    N = len(S)
    # We know N=12.
    # Check k from N down to 3. 
    # Note: A set of size 2 is always Sidon (unless a+a == b+b, which implies a=b, but elements are distinct).
    # Actually, for size 2, {a,b}, sums are a+a, a+b, b+b. 
    # If a+a == b+b -> a=b (impossible for distinct).
    # If a+a == a+b -> a=b (impossible).
    # So size 2 is always Sidon.
    # We want to minimize the max size. The minimum possible is 3? 
    # Can we have max size 2? 
    # That would mean no 3-element subset is Sidon.
    # A 3-element subset {a,b,c} is Sidon if a+a, a+b, a+c, b+b, b+c, c+c are all distinct.
    # Collisions like a+c == b+b are common.
    # It is theoretically possible to have max Sidon size 2, but it's very hard for 12 elements.
    # Most likely the minimum is 3.
    
    for k in range(N, 2, -1):
        for comb in itertools.combinations(S, k):
            if is_sidon(comb):
                return k
    return 2

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s budget, leaving 15s safety margin
    
    # Initial candidate: The densest possible set is [0, 1, ..., 11]
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    current_best = list(best_set)
    current_score = best_score
    
    # Try some structured candidates first
    structured_candidates = [
        list(range(12)),
        list(range(0, 24, 2)), # Even numbers 0..22
        [0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32], # 4 clusters of 3
        [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23], # 3 clusters of 4
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20, 21]  # 2 clusters of 5, 1 of 2? No, 5+5+2=12.
    ]
    
    for cand in structured_candidates:
        if len(cand) != 12 or len(set(cand)) != 12:
            continue
        score = max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand
            current_best = cand
            current_score = score
            if best_score == 3:
                return best_set
                
    # Local search
    # We will mutate one element at a time.
    # To increase density, we should prefer small values.
    # Range [0, 30] is a good balance.
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # Occasionally do a random restart to escape local minima
        if attempts % 100 == 0:
            # Generate a random dense set
            # Pick 12 distinct numbers from [0, 25]
            pool = list(range(25))
            random.shuffle(pool)
            current_best = pool[:12]
            current_score = max_sidon_size(current_best)
            if current_score < best_score:
                best_score = current_score
                best_set = current_best
                if best_score == 3:
                    return best_set
            continue
            
        # Mutation
        S = list(current_best)
        idx = random.randint(0, 11)
        old_val = S[idx]
        
        # Try to replace with a value that creates collisions.
        # Small values are more likely to collide with other small values.
        # Let's pick a new value from [0, 30]
        new_val = random.randint(0, 30)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in S:
            # Try a few times to find a distinct value
            for _ in range(5):
                new_val = random.randint(0, 30)
                if new_val != old_val and new_val not in S:
                    break
            else:
                continue
                
        S[idx] = new_val
        
        score = max_sidon_size(S)
        
        if score < current_score:
            current_best = S
            current_score = score
            if score < best_score:
                best_score = score
                best_set = S
                if best_score == 3:
                    return best_set
        elif score == current_score:
            # Accept sideways moves to explore
            if random.random() < 0.5:
                current_best = S
                current_score = score
                if score < best_score:
                    best_score = score
                    best_set = S
                    if best_score == 3:
                        return best_set

    return best_set
