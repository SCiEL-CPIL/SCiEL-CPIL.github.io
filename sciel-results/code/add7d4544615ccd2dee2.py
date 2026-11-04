import time
import random
from math import log

def score_set(U):
    U = sorted(set(U))
    if 0 not in U:
        return -1
    n = len(U)
    if n < 2:
        return -1
    
    sums = set()
    diffs = set()
    
    for i in range(n):
        ui = U[i]
        for j in range(i, n):
            sums.add(ui + U[j])
        for j in range(i + 1, n):
            d = U[j] - ui
            diffs.add(d)
            diffs.add(-d)
    diffs.add(0)
    
    sum_count = len(sums)
    diff_count = len(diffs)
    
    if sum_count == 0:
        return -1
        
    ratio = diff_count / sum_count
    if ratio <= 1:
        return -1
        
    max_u = max(U)
    D = 2 * max_u + 1
    if D <= 1:
        return -1
        
    return 1 + log(ratio) / log(D)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    best_set = [0, 1, 4, 9, 12]
    best_score = score_set(best_set)
    
    # Candidate pool for new elements
    max_val = 30
    candidates = list(range(1, max_val + 1))
    
    # Local search
    current_set = list(best_set)
    current_score = best_score
    
    no_improve = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 5 and no_improve < max_no_improve:
        improved = False
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap
            if len(current_set) > 1:
                i = random.randint(1, len(current_set) - 1)
                old_val = current_set[i]
                new_val = random.choice(candidates)
                if new_val != old_val and new_val not in current_set:
                    new_set = current_set.copy()
                    new_set[i] = new_val
                    new_set = sorted(new_set)
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
        elif move_type < 0.7:
            # Add
            if len(current_set) < 8:
                new_val = random.choice(candidates)
                if new_val not in current_set:
                    new_set = sorted(current_set + [new_val])
                    new_score = score_set(new_set)
                    if new_score > current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
        else:
            # Remove
            if len(current_set) > 2:
                i = random.randint(1, len(current_set) - 1)
                new_set = [x for j, x in enumerate(current_set) if j != i]
                new_score = score_set(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
        
        if improved:
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
        else:
            no_improve += 1
            
        # Occasionally restart with a random set
        if no_improve > 500 and random.random() < 0.01:
            size = random.randint(4, 6)
            current_set = [0] + random.sample(range(1, 25), size - 1)
            current_set = sorted(current_set)
            current_score = score_set(current_set)
            no_improve = 0

    return best_set

if __name__ == "__main__":
    U = construct()
    print("Result:", U)
    print("Score:", score_set(U))
