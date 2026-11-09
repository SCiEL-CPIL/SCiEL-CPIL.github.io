import time
import random
import numpy as np
from itertools import combinations

def _score_set(A):
    """Compute max(|A+A|, |A*A|) for a set A of positive integers."""
    A_list = list(A)
    n = len(A_list)
    
    # Sum set A+A with equal summands allowed
    sum_set = set()
    for i in range(n):
        for j in range(i, n):
            sum_set.add(A_list[i] + A_list[j])
    
    # Product set A*A with equal factors allowed
    prod_set = set()
    for i in range(n):
        for j in range(i, n):
            prod_set.add(A_list[i] * A_list[j])
    
    return max(len(sum_set), len(prod_set))

def _score_set_fast(A):
    """Faster scoring using numpy for sum set, but product set needs care."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Sum set: use broadcasting
    sums = A_arr[:, None] + A_arr[None, :]
    sum_set = set(np.unique(sums).tolist())
    
    # Product set
    prods = A_arr[:, None] * A_arr[None, :]
    prod_set = set(np.unique(prods).tolist())
    
    return max(len(sum_set), len(prod_set))

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s margin
    
    best_score = float('inf')
    best_set = None
    
    # Strategy 1: Start with structured sets (AP, GP, mixed)
    # Try various structured families
    
    # Family 1: Arithmetic Progression
    for start in range(1, 50):
        for step in range(1, 100):
            A = [start + i * step for i in range(20)]
            if max(A) > 10000:
                continue
            if len(set(A)) != 20:
                continue
            s = _score_set_fast(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
            if time.time() > deadline:
                return best_set if best_set else list(range(1, 21))
    
    # Family 2: Geometric Progression (integer)
    for start in range(1, 20):
        for ratio in range(2, 50):
            A = [start * (ratio ** i) for i in range(20)]
            if max(A) > 10000:
                continue
            if len(set(A)) != 20:
                continue
            s = _score_set_fast(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
            if time.time() > deadline:
                return best_set if best_set else list(range(1, 21))
    
    # Family 3: Mixed - small AP + large GP
    for ap_start in range(1, 10):
        for ap_step in range(1, 5):
            for ap_len in range(5, 15):
                ap = [ap_start + i * ap_step for i in range(ap_len)]
                if max(ap) > 5000:
                    continue
                gp_len = 20 - ap_len
                for gp_start in range(1, 10):
                    for gp_ratio in range(2, 30):
                        gp = [gp_start * (gp_ratio ** i) for i in range(gp_len)]
                        if max(gp) > 10000:
                            continue
                        A = ap + gp
                        if len(set(A)) != 20:
                            continue
                        if max(A) > 10000:
                            continue
                        s = _score_set_fast(A)
                        if s < best_score:
                            best_score = s
                            best_set = A[:]
                        if time.time() > deadline:
                            return best_set if best_set else list(range(1, 21))
    
    # Family 4: Powers of 2
    A = [1 << i for i in range(20)]
    if max(A) <= 10000:
        s = _score_set_fast(A)
        if s < best_score:
            best_score = s
            best_set = A[:]
    
    # Family 5: Small numbers with structure
    # Try sets with many small numbers to create collisions
    for base in range(1, 10):
        # Create set with duplicates in sum/product structure
        A = []
        # Add base, 2*base, 3*base, ... but limit to 20 elements
        for i in range(1, 21):
            val = base * i
            if val <= 10000:
                A.append(val)
        if len(A) == 20 and len(set(A)) == 20:
            s = _score_set_fast(A)
            if s < best_score:
                best_score = s
                best_set = A[:]
    
    # Local search from best found
    if best_set is not None and time.time() < deadline - 10:
        current = best_set[:]
        current_score = best_score
        
        # Random local search
        random.seed(42)
        for iteration in range(10000):
            if time.time() > deadline - 5:
                break
            
            # Try to replace one element
            idx = random.randint(0, 19)
            old_val = current[idx]
            
            # Try nearby values or random values
            if random.random() < 0.5:
                new_val = old_val + random.randint(-10, 10)
            else:
                new_val = random.randint(1, 10000)
            
            if new_val < 1 or new_val > 10000:
                continue
            if new_val in current:
                continue
            
            current[idx] = new_val
            s = _score_set_fast(current)
            
            if s < current_score:
                current_score = s
                if s < best_score:
                    best_score = s
                    best_set = current[:]
            else:
                current[idx] = old_val
        
        # Simulated annealing-like refinement
        if time.time() < deadline - 5:
            current = best_set[:]
            current_score = best_score
            temp = 10.0
            for iteration in range(20000):
                if time.time() > deadline - 3:
                    break
                
                idx = random.randint(0, 19)
                old_val = current[idx]
                new_val = old_val + random.randint(-5, 5)
                
                if new_val < 1 or new_val > 10000:
                    continue
                if new_val in current:
                    continue
                
                current[idx] = new_val
                s = _score_set_fast(current)
                
                delta = s - current_score
                if delta < 0 or random.random() < np.exp(-delta / max(temp, 0.1)):
                    current_score = s
                    if s < best_score:
                        best_score = s
                        best_set = current[:]
                else:
                    current[idx] = old_val
                
                temp *= 0.999
    
    return best_set if best_set else list(range(1, 21))
