import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    def get_scores(A):
        A_set = set(A)
        sums = set()
        prods = set()
        for i in range(len(A)):
            for j in range(i, len(A)):
                sums.add(A[i] + A[j])
                prods.add(A[i] * A[j])
        return max(len(sums), len(prods))
    
    # Initial candidate: 1..20
    best_A = list(range(1, 21))
    best_score = get_scores(best_A)
    
    # Try a few other structured starts
    candidates = [
        list(range(1, 21)),
        [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 20, 24, 27, 30, 32, 36, 40, 45, 48], # Divisors of 48? No, just some small composites
        [2**i for i in range(20)], # GP
    ]
    
    for cand in candidates:
        if len(set(cand)) != 20 or any(x > 10000 for x in cand):
            continue
        score = get_scores(cand)
        if score < best_score:
            best_score = score
            best_A = list(cand)
            
    # Hill Climbing
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Reset or perturb
            # Simple restart from best_A with random perturbation
            A = list(best_A)
            # Swap 2 random elements with random new ones
            idxs = random.sample(range(20), 2)
            for idx in idxs:
                new_val = random.randint(1, 10000)
                while new_val in A:
                    new_val = random.randint(1, 10000)
                A[idx] = new_val
            score = get_scores(A)
            if score < best_score:
                best_score = score
                best_A = list(A)
                no_improve_count = 0
            else:
                no_improve_count += 1
            continue
            
        A = list(best_A)
        # Pick a random element to replace
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try a random replacement
        # Bias towards small numbers for product collisions, or neighbors for sum stability
        if random.random() < 0.5:
            new_val = random.randint(1, 100)
        else:
            new_val = random.randint(1, 10000)
            
        if new_val == old_val:
            continue
        if new_val in A:
            continue
            
        A[idx] = new_val
        score = get_scores(A)
        
        if score < best_score:
            best_score = score
            best_A = list(A)
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Revert
            A[idx] = old_val
            
    return best_A
