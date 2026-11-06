import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 55
    
    # Precompute reciprocals
    reciprocals = [Fraction(1, i) for i in range(25)]
    
    best_S = set()
    best_size = 0
    
    def is_valid_fast(S):
        if not S:
            return True
        
        total_sum = sum((reciprocals[a] for a in S), Fraction(0))
        
        for a in S:
            target = reciprocals[a]
            sum_others = total_sum - target
            
            if target > sum_others:
                continue
            
            others = [b for b in S if b != a]
            sub_sums = {Fraction(0)}
            for b in others:
                new_sums = set()
                for s in sub_sums:
                    new_sums.add(s + reciprocals[b])
                sub_sums = sub_sums | new_sums
                
            if target in sub_sums:
                return False
                
        return True

    # Initial Greedy
    S = set()
    for a in range(24, 0, -1):
        if time.time() - start_time > timeout:
            break
        new_S = S | {a}
        if is_valid_fast(new_S):
            S = new_S
            
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
        
    current_S = S
    
    # Randomized Local Search
    while time.time() - start_time < timeout:
        candidates_to_add = [a for a in range(1, 25) if a not in current_S]
        random.shuffle(candidates_to_add)
        
        moved = False
        for a in candidates_to_add:
            if time.time() - start_time > timeout:
                break
            new_S = current_S | {a}
            if is_valid_fast(new_S):
                current_S = new_S
                if len(current_S) > best_size:
                    best_S = current_S
                    best_size = len(current_S)
                moved = True
                break
        
        if not moved:
            # Try to swap
            if current_S:
                to_remove = list(current_S)
                random.shuffle(to_remove)
                for a in to_remove:
                    if time.time() - start_time > timeout:
                        break
                    new_S = current_S - {a}
                    current_S = new_S
                    
                    # Try to add a different element
                    candidates_to_add = [b for b in range(1, 25) if b not in current_S]
                    random.shuffle(candidates_to_add)
                    for b in candidates_to_add:
                        if time.time() - start_time > timeout:
                            break
                        next_S = current_S | {b}
                        if is_valid_fast(next_S):
                            current_S = next_S
                            if len(current_S) > best_size:
                                best_S = current_S
                                best_size = len(current_S)
                            break
                    break

    return sorted(best_S)
