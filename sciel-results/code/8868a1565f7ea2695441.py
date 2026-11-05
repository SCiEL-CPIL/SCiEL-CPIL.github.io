import random
import time
import math

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # Leave 5s margin for return overhead
    
    n = 20
    max_val = 10000
    
    def get_score(A):
        # A is a list of 20 distinct ints
        # Compute A+A and A*A sizes
        sums = set()
        prods = set()
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
        return max(len(sums), len(prods))
    
    def random_set():
        return random.sample(range(1, max_val + 1), n)
    
    # Initial candidates
    candidates = []
    
    # 1. Arithmetic progression 1..20 (Seed)
    candidates.append(list(range(1, 21)))
    
    # 2. Shifted AP
    candidates.append(list(range(2, 22)))
    
    # 3. Geometric-ish (powers of 2, truncated/scaled)
    # 1, 2, 4, ..., 2^13 (14 elems). Add 15..20? 
    # Let's just use random for variety.
    
    # 4. Random sets
    for _ in range(20):
        candidates.append(random_set())
        
    # 5. Hybrid: Small AP + Large AP
    c5 = list(range(1, 11)) + list(range(100, 110))
    candidates.append(c5)
    
    c6 = list(range(1, 11)) + list(range(1000, 1010))
    candidates.append(c6)
    
    # 6. "Fibonacci" like
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    # Ensure distinct and <= 10000
    # Fibs grow fast. 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765
    # All <= 10000. Distinct? 1,1 is not distinct.
    # Fix: remove duplicate 1.
    fibs = list(dict.fromkeys(fibs)) # Remove duplicates preserving order
    if len(fibs) < n:
         # Pad with randoms if needed
         while len(fibs) < n:
             fibs.append(random.randint(1, max_val))
    # Ensure distinct
    fibs = list(dict.fromkeys(fibs))
    if len(fibs) >= n:
        candidates.append(fibs[:n])

    # Find best initial candidate
    best_set = candidates[0]
    best_score = get_score(best_set)
    
    for c in candidates[1:]:
        s = get_score(c)
        if s < best_score:
            best_score = s
            best_set = c[:]
            
    # Local Search / Simulated Annealing
    def improve(A, time_limit):
        current = A[:]
        current_score = get_score(current)
        best = current[:]
        best_score_local = current_score
        
        T = 100.0
        cooling = 0.9999
        # Min T to allow some exploration
        min_T = 0.5
        
        while time.time() - start_time < time_limit:
            # Move: pick random index, replace with new random value
            idx = random.randint(0, n - 1)
            old_val = current[idx]
            new_val = random.randint(1, max_val)
            
            # Ensure distinctness
            if new_val in current:
                continue
            
            current[idx] = new_val
            new_score = get_score(current)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score_local:
                    best_score_local = new_score
                    best = current[:]
            else:
                # Accept with prob exp(-(new_score - current_score) / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = 1.0
                else:
                    prob = math.exp(-delta / T)
                
                if random.random() < prob:
                    current_score = new_score
                else:
                    current[idx] = old_val
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best, best_score_local

    # Run improvement on the best initial candidate
    improved_set, improved_score = improve(best_set, timeout)
    
    if improved_score < best_score:
        best_set = improved_set
        best_score = improved_score
        
    # If we have time, try another random start and improve it
    # This helps if the first basin of attraction is poor.
    if time.time() - start_time < timeout - 15:
        c_rand = random_set()
        s_rand = get_score(c_rand)
        # Only improve if it looks promising or just to explore
        # Actually, just run improvement on a random start
        improved_set2, improved_score2 = improve(c_rand, timeout)
        
        if improved_score2 < best_score:
            best_set = improved_set2
            best_score = improved_score2

    return best_set
