import time
import random
from itertools import combinations

def _max_sidon_size(A):
    """
    Exhaustively computes the largest subset of A with all unordered pair sums unique.
    A is a list of distinct integers.
    """
    n = len(A)
    if n == 0:
        return 0
    
    A = sorted(A)
    
    best_size = 0
    # Check subsets in decreasing size order
    for size in range(n, 0, -1):
        found = False
        for subset in combinations(A, size):
            # Check if subset is Sidon
            sums = set()
            is_sidon = True
            # Optimization: check diagonals first? No, just all pairs i<=j
            for i in range(len(subset)):
                ai = subset[i]
                for j in range(i, len(subset)):
                    s = ai + subset[j]
                    if s in sums:
                        is_sidon = False
                        break
                    sums.add(s)
                if not is_sidon:
                    break
            if is_sidon:
                best_size = size
                found = True
                break
        if found:
            break
            
    return best_size

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Initial candidate: Arithmetic Progression
    # AP is dense, likely to have small max Sidon subset
    current_best = list(range(12))
    current_best_val = _max_sidon_size(current_best)
    
    # If we already found 3, we can't do better (min possible is 1, but 12 distinct ints in [0,128] must have Sidon subset >= 4? 
    # Actually, can max Sidon size be 3?
    # If max Sidon size is 3, then no 4 elements form a Sidon set.
    # Let's see if we can find one.
    
    if current_best_val == 3:
        return current_best

    # Local Search
    # We want to minimize the max Sidon subset size.
    
    def get_neighbor(A):
        A = list(A)
        idx = random.randint(0, len(A) - 1)
        old_val = A[idx]
        
        # Strategy: 70% local move, 30% global move
        if random.random() < 0.7:
            # Keep it dense
            new_val = random.randint(max(0, old_val - 20), min(128, old_val + 20))
        else:
            new_val = random.randint(0, 128)
            
        if new_val in A and new_val != old_val:
            # Ensure distinctness
            for _ in range(20):
                new_val = random.randint(0, 128)
                if new_val not in A:
                    break
            else:
                return A # Failed to generate valid neighbor
        
        A[idx] = new_val
        return A

    # Hill Climb
    # We accept moves that reduce the max Sidon size.
    # We also accept moves that keep the size same but change the set (to escape local minima) 
    # with a small probability, or just explore.
    
    # However, if we find 3, we stop.
    # If we are at 4, we try to find 3.
    # If we are at 5, we try to find 4.
    
    no_improve_counter = 0
    
    while time.time() < deadline:
        neighbor = get_neighbor(current_best)
        if neighbor == current_best:
            continue
            
        val = _max_sidon_size(neighbor)
        
        if val < current_best_val:
            current_best = neighbor
            current_best_val = val
            no_improve_counter = 0
            if current_best_val == 3:
                break
        elif val == current_best_val:
            # Accept equal value moves with some probability to explore
            # This helps escape local minima where all neighbors are worse
            if random.random() < 0.1:
                current_best = neighbor
                no_improve_counter += 1
            else:
                no_improve_counter += 1
        else:
            # Worse move
            # Occasionally accept worse moves (Simulated Annealing style)
            # But we are minimizing, so be careful.
            # If we are stuck at 4 and want 3, we might need to go through 5?
            # Unlikely. 3 is "more constrained" than 4.
            # So we should stick to strict improvement or equal.
            no_improve_counter += 1
            
        # If stuck for too long, try a random restart?
        # Or just keep going.
        
    return current_best
