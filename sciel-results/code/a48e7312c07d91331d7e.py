"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recips = [Fraction(1, d) for d in denominators]
    
    # Helper to calculate sum and check validity
    def get_sum(indices):
        s = Fraction(0)
        for i in indices:
            s += recips[i]
        return s

    def get_deficit(indices):
        s = get_sum(indices)
        if s >= 1:
            return None
        return 1 - s

    # Initial candidates
    candidates = set()
    
    # 1. Greedy
    greedy = []
    total = Fraction(0)
    for i in range(n):
        if total + recips[i] < 1:
            greedy.append(i)
            total += recips[i]
    candidates.add(tuple(sorted(greedy)))
    
    # 2. Known good solution: 1/2 + 1/3 + 1/7 + 1/43 = 1 - 1/1806
    # Indices: 2->0, 3->1, 7->5, 43->41
    candidates.add((0, 1, 5, 41))
    
    # 3. Random initializations
    for _ in range(20):
        if time.time() > deadline:
            break
        k = random.randint(4, 15)
        subset = random.sample(range(n), k)
        candidates.add(tuple(sorted(subset)))
        
    best_deficit = None
    best_indices = None
    
    # Evaluate initial candidates
    for cand in candidates:
        d = get_deficit(list(cand))
        if d is not None:
            if best_deficit is None or d < best_deficit:
                best_deficit = d
                best_indices = list(cand)
                
    if best_indices is None:
        # Fallback to greedy if all failed (unlikely)
        return greedy

    # Local Search
    current = list(best_indices)
    current_deficit = best_deficit
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding one element
        current_sum = get_sum(current)
        for i in range(n):
            if i not in current:
                new_sum = current_sum + recips[i]
                if new_sum < 1:
                    new_def = 1 - new_sum
                    if new_def < current_deficit:
                        current.append(i)
                        current.sort()
                        current_deficit = new_def
                        current_sum = new_sum
                        improved = True
                        break # Restart loop with new state
        
        if improved:
            continue
            
        # Try swapping one out, one in
        for i_out in current:
            for i_in in range(n):
                if i_in == i_out or i_in in current:
                    continue
                new_sum = current_sum - recips[i_out] + recips[i_in]
                if new_sum < 1:
                    new_def = 1 - new_sum
                    if new_def < current_deficit:
                        # Apply swap
                        current.remove(i_out)
                        current.append(i_in)
                        current.sort()
                        current_sum = new_sum
                        current_deficit = new_def
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue

        # Try swapping one out, two in
        for i_out in current:
            base_sum = current_sum - recips[i_out]
            for i_in1 in range(n):
                if i_in1 == i_out or i_in1 in current:
                    continue
                sum1 = base_sum + recips[i_in1]
                if sum1 >= 1:
                    continue
                for i_in2 in range(n):
                    if i_in2 == i_out or i_in2 == i_in1 or i_in2 in current:
                        continue
                    new_sum = sum1 + recips[i_in2]
                    if new_sum < 1:
                        new_def = 1 - new_sum
                        if new_def < current_deficit:
                            current.remove(i_out)
                            current.append(i_in1)
                            current.append(i_in2)
                            current.sort()
                            current_sum = new_sum
                            current_deficit = new_def
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # Try swapping two out, one in
        for i_out1 in range(len(current)):
            for i_out2 in range(i_out1 + 1, len(current)):
                i1 = current[i_out1]
                i2 = current[i_out2]
                base_sum = current_sum - recips[i1] - recips[i2]
                for i_in in range(n):
                    if i_in == i1 or i_in == i2 or i_in in current:
                        continue
                    new_sum = base_sum + recips[i_in]
                    if new_sum < 1:
                        new_def = 1 - new_sum
                        if new_def < current_deficit:
                            current.remove(i1)
                            current.remove(i2)
                            current.append(i_in)
                            current.sort()
                            current_sum = new_sum
                            current_deficit = new_def
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return current
# EVOLVE-BLOCK-END
