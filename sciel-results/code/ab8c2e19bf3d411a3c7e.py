import random
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    candidates = list(range(1, 25))
    
    best_set = []
    
    # Helper to check if a set is valid
    def is_valid(S):
        if not S:
            return True
        S_set = set(S)
        for x in S_set:
            target = recip[x]
            # Get reciprocals of other elements
            others = [recip[y] for y in S_set if y != x]
            # Sort in descending order for pruning
            others.sort(reverse=True)
            
            # Precompute suffix sums for pruning
            suffix_sums = [0] * (len(others) + 1)
            for i in range(len(others) - 1, -1, -1):
                suffix_sums[i] = suffix_sums[i+1] + others[i]
            
            def can_make(target, idx, current_sum):
                if current_sum == target:
                    return True
                if idx == len(others) or current_sum > target:
                    return False
                # Prune if even adding all remaining cannot reach target
                if current_sum + suffix_sums[idx] < target:
                    return False
                
                # Option 1: Include others[idx]
                if can_make(target, idx + 1, current_sum + others[idx]):
                    return True
                
                # Option 2: Exclude others[idx]
                if can_make(target, idx + 1, current_sum):
                    return True
                
                return False
            
            if can_make(target, 0, Fraction(0)):
                return False
        
        return True
    
    # Randomized greedy search
    num_restarts = 1000
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_set = []
        shuffled_candidates = candidates[:]
        random.shuffle(shuffled_candidates)
        
        for x in shuffled_candidates:
            # Check if adding x keeps the set valid
            # We can optimize by only checking violations involving x or new sums
            # But for N=24, full check is okay if fast enough
            # Let's try adding x
            new_set = current_set + [x]
            if is_valid(new_set):
                current_set = new_set
        
        if len(current_set) > len(best_set):
            best_set = current_set
    
    return best_set
