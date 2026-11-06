import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    best_score = 0
    best_A = None
    best_B = None
    
    def evaluate(A, B):
        sums = set()
        for a in A:
            for b in B:
                sums.add(a + b)
        if len(sums) < 2:
            return 0, []
        
        # Check pairwise coprime
        s_list = list(sums)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if gcd(s_list[i], s_list[j]) != 1:
                    return 0, []
        
        return len(s_list), s_list

    def random_init():
        # Pick 3 distinct numbers from 1..128
        A = random.sample(range(1, 129), 3)
        B = random.sample(range(1, 129), 3)
        return A, B

    def local_search(A, B, max_steps=1000):
        nonlocal best_score, best_A, best_B
        
        current_score, _ = evaluate(A, B)
        if current_score > best_score:
            best_score = current_score
            best_A = A.copy()
            best_B = B.copy()
            
        for _ in range(max_steps):
            if time.time() > deadline:
                return
            
            # Pick a random set (A or B) and a random index to change
            if random.random() < 0.5:
                idx = random.randint(0, 2)
                old_val = A[idx]
                A[idx] = random.randint(1, 128)
                # Ensure distinctness within A
                if A.count(A[idx]) > 1:
                    A[idx] = old_val
                    continue
            else:
                idx = random.randint(0, 2)
                old_val = B[idx]
                B[idx] = random.randint(1, 128)
                if B.count(B[idx]) > 1:
                    B[idx] = old_val
                    continue
            
            new_score, _ = evaluate(A, B)
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_A = A.copy()
                    best_B = B.copy()
            else:
                # Revert
                if random.random() < 0.5:
                    A[idx] = old_val
                else:
                    B[idx] = old_val
                    
    # Run multiple restarts
    while time.time() < deadline:
        A, B = random_init()
        local_search(A, B, max_steps=500)
        
        # If we found 9, we can stop early as it's the max possible
        if best_score == 9:
            break
            
    if best_A is None:
        # Fallback to seed if something went wrong
        return [[1, 7, 13], [4, 10, 16]]
        
    return [best_A, best_B]

# EVOLVE-BLOCK-START
# The actual entry point is construct()
# EVOLVE-BLOCK-END
