import random
import time
from math import gcd

def is_pairwise_coprime(lst):
    n = len(lst)
    for i in range(n):
        for j in range(i + 1, n):
            if gcd(lst[i], lst[j]) != 1:
                return False
    return True

def score_candidate(A, B):
    sums = set()
    for a in A:
        for b in B:
            sums.add(a + b)
    distinct_sums = list(sums)
    if not is_pairwise_coprime(distinct_sums):
        return 0
    return len(distinct_sums)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds to be safe
    
    best_A = [1, 7, 13]
    best_B = [4, 10, 16]
    best_score = score_candidate(best_A, best_B)
    
    # Random initialization
    A = [random.randint(1, 128) for _ in range(3)]
    B = [random.randint(1, 128) for _ in range(3)]
    # Ensure distinct elements in A and B
    while len(set(A)) < 3:
        A[random.randint(0, 2)] = random.randint(1, 128)
    while len(set(B)) < 3:
        B[random.randint(0, 2)] = random.randint(1, 128)
        
    current_score = score_candidate(A, B)
    if current_score > best_score:
        best_score = current_score
        best_A = A.copy()
        best_B = B.copy()
        
    no_improvement_count = 0
    max_no_improvement = 100000
    
    while time.time() < deadline and no_improvement_count < max_no_improvement:
        # Pick a random set to modify
        if random.random() < 0.5:
            set_to_modify = 'A'
            idx = random.randint(0, 2)
            old_val = A[idx]
            new_val = random.randint(1, 128)
            if new_val == old_val:
                continue
            if new_val in A and A.count(new_val) < 2: # Allow duplicates temporarily? No, must be distinct.
                # Check if adding new_val keeps distinctness
                # Actually, just swap
                pass
            # Ensure distinctness
            if new_val in A and new_val != old_val:
                continue
            A[idx] = new_val
            modified_set = A
            other_set = B
        else:
            set_to_modify = 'B'
            idx = random.randint(0, 2)
            old_val = B[idx]
            new_val = random.randint(1, 128)
            if new_val == old_val:
                continue
            if new_val in B and new_val != old_val:
                continue
            B[idx] = new_val
            modified_set = B
            other_set = A
            
        new_score = score_candidate(A, B)
        
        if new_score > current_score:
            current_score = new_score
            no_improvement_count = 0
            if current_score > best_score:
                best_score = current_score
                best_A = A.copy()
                best_B = B.copy()
        else:
            # Revert
            if set_to_modify == 'A':
                A[idx] = old_val
            else:
                B[idx] = old_val
            no_improvement_count += 1
            
            # If stuck, try a bigger jump
            if no_improvement_count % 1000 == 0:
                # Random restart or large perturbation
                for _ in range(10):
                    if random.random() < 0.5:
                        idx = random.randint(0, 2)
                        A[idx] = random.randint(1, 128)
                    else:
                        idx = random.randint(0, 2)
                        B[idx] = random.randint(1, 128)
                # Ensure distinctness
                while len(set(A)) < 3:
                    A[random.randint(0, 2)] = random.randint(1, 128)
                while len(set(B)) < 3:
                    B[random.randint(0, 2)] = random.randint(1, 128)
                current_score = score_candidate(A, B)
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
                    best_B = B.copy()
                no_improvement_count = 0

    return [best_A, best_B]
