import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # 45s out of 60s
    
    n = 20
    max_val = 10000
    
    def score(A):
        A = list(set(A))
        if len(A) != n:
            return float('inf')
        if max(A) > max_val:
            return float('inf')
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(n):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    best_score = float('inf')
    best_set = list(range(1, n+1))
    
    # Helper to generate GP
    def gen_gp(ratio, start, length):
        A = []
        val = start
        for _ in range(length):
            A.append(val)
            val *= ratio
        return A
    
    # Candidate 1: Single GP
    for ratio in [2, 3]:
        for start in [1, 2]:
            A = gen_gp(ratio, start, n)
            if max(A) <= max_val:
                s = score(A)
                if s < best_score:
                    best_score = s
                    best_set = A
    
    # Candidate 2: Two clusters GP
    # A1 = {start1 * ratio^i}, A2 = {start2 * ratio^i}
    # Ensure distinct
    for ratio in [2, 3]:
        for start1 in [1, 2]:
            for start2 in [3, 4, 5, 10]:
                A1 = gen_gp(ratio, start1, 10)
                A2 = gen_gp(ratio, start2, 10)
                A = A1 + A2
                if len(set(A)) == n and max(A) <= max_val:
                    s = score(A)
                    if s < best_score:
                        best_score = s
                        best_set = A
                        
    # Local search
    def local_search(A, time_limit):
        A = list(A)
        current_score = score(A)
        while time.time() < time_limit:
            improved = False
            # Random swap
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            if i == j: continue
            old_i = A[i]
            old_j = A[j]
            
            # Try replacing A[i] with a new value
            # Pick a new value from a candidate pool
            # Candidate pool: numbers close to current, or random
            # To keep it fast, just try a few random numbers
            for _ in range(5):
                new_val = random.randint(1, max_val)
                if new_val in A and A.index(new_val) != i:
                    continue
                A[i] = new_val
                s = score(A)
                if s < current_score:
                    current_score = s
                    improved = True
                    break
                else:
                    A[i] = old_i
            
            if not improved:
                # Try swapping two elements
                new_i = random.randint(1, max_val)
                new_j = random.randint(1, max_val)
                if new_i in A or new_j in A:
                    # Handle duplicates carefully
                    pass 
                A[i] = new_i
                A[j] = new_j
                if len(set(A)) == n and max(A) <= max_val:
                    s = score(A)
                    if s < current_score:
                        current_score = s
                        improved = True
                    else:
                        A[i] = old_i
                        A[j] = old_j
                else:
                    A[i] = old_i
                    A[j] = old_j
                    
            if time.time() > time_limit - 5:
                break
                
        return A, current_score

    # Run local search on best found so far
    if time.time() < deadline - 10:
        best_set, best_score = local_search(best_set, deadline - 10)
        
    return best_set
