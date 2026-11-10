import random
import time

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    # S is a list of integers
    # Check subsets from largest to smallest
    n = len(S)
    # Max possible size is n
    # We can iterate k from n down to 1
    # But for n=12, we can just check all subsets of size k
    # C(12, 6) = 924, C(12, 5) = 792, C(12, 4) = 495
    # Total subsets is 4095. Very fast.
    
    # We want the max k such that there exists a subset of size k that is Sidon.
    # We can check k = 12, 11, ...
    # But usually the max size is small (3-6).
    # Let's check k from 12 down to 1.
    
    # Optimization: if we find a Sidon subset of size k, we can return k.
    # We need to check if ANY subset of size k is Sidon.
    
    # To be efficient, we can generate combinations.
    from itertools import combinations
    
    for k in range(n, 0, -1):
        for comb in combinations(S, k):
            if is_sidon(list(comb)):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = 60.0 # Safety margin
    
    # Initial candidate: consecutive integers
    best_set = list(range(12))
    best_size = max_sidon_size(best_set)
    
    # Try to find a set with smaller max Sidon size
    # We will perform a local search
    # Start with a random set in a small range to increase collisions
    # Or start with the seed and perturb
    
    current_set = list(range(12))
    current_size = best_size
    
    # Heuristic: sets with smaller diameter tend to have smaller max Sidon subsets?
    # Actually, for fixed size, smaller diameter means more collisions, so smaller Sidon subsets.
    # Let's try to compress the set.
    
    # Try a few random sets in [0, 40]
    random.seed(42)
    
    for _ in range(10000):
        if time.time() > deadline:
            break
            
        # Generate a random set of 12 distinct integers in [0, 60]
        s = random.sample(range(61), 12)
        s.sort()
        
        size = max_sidon_size(s)
        
        if size < current_size:
            current_size = size
            current_set = s
            best_size = current_size
            best_set = s
            
            if current_size <= 3:
                break
                
    # If we didn't find size 3, try local search from best found
    if best_size > 3:
        # Local search
        for _ in range(5000):
            if time.time() > deadline:
                break
            
            # Pick a random element to change
            idx = random.randint(0, 11)
            old_val = current_set[idx]
            
            # Try replacing with a random value in [0, 128]
            # Or maybe keep it close to maintain density
            new_val = random.randint(0, 128)
            
            if new_val in current_set:
                continue
                
            # Create new set
            new_set = current_set.copy()
            new_set[idx] = new_val
            new_set.sort()
            
            new_size = max_sidon_size(new_set)
            
            if new_size < current_size:
                current_size = new_size
                current_set = new_set
                best_size = current_size
                best_set = new_set
                
                if current_size <= 3:
                    break
            elif new_size == current_size and random.random() < 0.1:
                # Sometimes accept same size to escape local minima?
                # But we want to minimize, so this is risky.
                # Let's stick to strict improvement or neutral moves with probability.
                current_set = new_set
                # Don't update best if not improved, but keep current for exploration
                # Actually, if we move to a neutral state, we might get stuck.
                # Let's just accept if it's better.
                # If we want to explore, we should accept neutral moves.
                # But max_sidon_size is expensive-ish.
                # Let's only accept better moves for now.
                current_set = old_set # revert? No, we want to explore.
                # Let's just revert for now to keep it simple and fast.
                # Actually, let's just try better moves.
                
    return best_set
