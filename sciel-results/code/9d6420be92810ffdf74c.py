"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import gcd
import time
import math

def construct():
    def check_pairwise_coprime(values):
        vals = sorted(set(values))
        n = len(vals)
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(vals[i], vals[j]) != 1:
                    return False
        return True
    
    def evaluate(A, B):
        sums = [a + b for a in A for b in B]
        distinct_sums = set(sums)
        if check_pairwise_coprime(distinct_sums):
            return len(distinct_sums)
        return 0

    # Known baseline
    A_test = [1, 7, 13]
    B_test = [4, 10, 16]
    best_A = A_test[:]
    best_B = B_test[:]
    best_count = evaluate(A_test, B_test)
    
    # Simulated Annealing
    random.seed(12345)
    start_time = time.time()
    timeout = 55
    
    # Initial state
    A = sorted(random.sample(range(1, 129), 3))
    B = sorted(random.sample(range(1, 129), 3))
    current_count = evaluate(A, B)
    
    if current_count > best_count:
        best_count = current_count
        best_A = A[:]
        best_B = B[:]
        
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    while time.time() - start_time < timeout and T > min_T:
        if best_count == 9:
            break
            
        # Mutate
        if random.random() < 0.5:
            idx = random.randint(0, 2)
            old_val = A[idx]
            new_val = random.randint(1, 128)
            if new_val in A and new_val != old_val:
                continue
            A[idx] = new_val
            A.sort()
            new_count = evaluate(A, B)
            
            if new_count >= current_count:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_A = A[:]
                    best_B = B[:]
            else:
                # Accept worse with probability
                delta = new_count - current_count
                if random.random() < math.exp(delta / T):
                    current_count = new_count
                else:
                    A[idx] = old_val
                    A.sort()
        else:
            idx = random.randint(0, 2)
            old_val = B[idx]
            new_val = random.randint(1, 128)
            if new_val in B and new_val != old_val:
                continue
            B[idx] = new_val
            B.sort()
            new_count = evaluate(A, B)
            
            if new_count >= current_count:
                current_count = new_count
                if current_count > best_count:
                    best_count = current_count
                    best_A = A[:]
                    best_B = B[:]
            else:
                delta = new_count - current_count
                if random.random() < math.exp(delta / T):
                    current_count = new_count
                else:
                    B[idx] = old_val
                    B.sort()
                    
        T *= cooling
        
    return [best_A, best_B]
# EVOLVE-BLOCK-END
