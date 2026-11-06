import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    
    def calc_initial_counts(A):
        counts = [0] * (target_hi + 1)
        A_list = list(A)
        for i in range(len(A_list)):
            a = A_list[i]
            for j in range(i, len(A_list)):
                b = A_list[j]
                s = a + b
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def calc_score(counts):
        uncovered = 0
        peak = 0
        for s in range(target_lo, target_hi + 1):
            c = counts[s]
            if c == 0:
                uncovered += 1
            if c > peak:
                peak = c
        return uncovered * 10000 + peak

    def local_search_sa(initial_A):
        A = list(initial_A)
        A_set = set(A)
        counts = calc_initial_counts(A)
        curr_score = calc_score(counts)
        
        best_A = A[:]
        best_score = curr_score
        
        # Simulated Annealing Parameters
        T = 50.0
        cooling = 0.9995
        min_T = 0.01
        
        while time.time() < deadline and T > min_T:
            r = random.random()
            
            if r < 0.7:
                # Swap
                if not A:
                    continue
                x = random.choice(A)
                y = random.randint(0, n)
                if y in A_set:
                    continue
                
                old_A = A[:]
                old_counts = counts[:]
                
                # Remove x
                A.remove(x)
                A_set.remove(x)
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s = 2 * x
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
                
                # Add y
                A.append(y)
                A_set.add(y)
                for z in A:
                    if z == y:
                        continue
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s = 2 * y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
                    
                new_score = calc_score(counts)
                delta_score = new_score - curr_score
                
                if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_A = A[:]
                else:
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
            elif r < 0.85:
                # Insert
                y = random.randint(0, n)
                if y in A_set:
                    continue
                
                old_A = A[:]
                old_counts = counts[:]
                
                A.append(y)
                A_set.add(y)
                for z in A:
                    if z == y:
                        continue
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s = 2 * y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
                    
                new_score = calc_score(counts)
                delta_score = new_score - curr_score
                
                if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_A = A[:]
                else:
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
            else:
                # Delete
                if not A:
                    continue
                x = random.choice(A)
                
                old_A = A[:]
                old_counts = counts[:]
                
                A.remove(x)
                A_set.remove(x)
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s = 2 * x
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
                    
                new_score = calc_score(counts)
                delta_score = new_score - curr_score
                
                if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                    curr_score = new_score
                    if curr_score < best_score:
                        best_score = curr_score
                        best_A = A[:]
                else:
                    A = old_A
                    A_set = set(A)
                    counts = old_counts
                    
            T *= cooling
            
        return best_A, best_score

    best_global_A = None
    best_global_score = float('inf')
    
    sizes = [15, 20, 25, 30, 35, 40]
    
    for k in sizes:
        if time.time() > deadline:
            break
            
        # Run multiple restarts for this size
        # We will just loop until time is up for this size, or until we find a very good solution
        restarts = 0
        while time.time() < deadline:
            init_A = random.sample(range(n + 1), k)
            A_res, score_res = local_search_sa(init_A)
            
            if score_res < best_global_score:
                best_global_score = score_res
                best_global_A = A_res
            
            # Check if we found a very good solution
            unc = score_res // 10000
            peak = score_res % 10000
            if unc == 0 and peak <= 2:
                break
                
            restarts += 1
            if restarts >= 5:
                break # Limit restarts per size to allow trying other sizes
        
        if best_global_score // 10000 == 0 and best_global_score % 10000 <= 2:
            break

    if best_global_A is None:
        return list(range(64, 129))
        
    return best_global_A
