"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def is_sidon(subset):
    """
    Check if a subset has all unordered pair sums (i<=j) unique.
    """
    sums = set()
    k = len(subset)
    for i in range(k):
        for j in range(i, k):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(arr):
    """
    Find the largest subset of arr with all pair sums unique (including diagonals).
    arr is a list of 12 distinct integers.
    """
    n = len(arr)
    # We want to find the max k such that there exists a subset of size k that is Sidon.
    # Check from n down to 1.
    for size in range(n, 0, -1):
        # Optimization: if size is large, few combinations.
        # If size is small, many combinations, but check is fast.
        for subset in combinations(arr, size):
            if is_sidon(subset):
                return size
    return 1

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Start with an Arithmetic Progression, which is known to have many collisions
    # and thus likely small Sidon subsets.
    best_arr = list(range(0, 128, 128 // 12))[:12]
    # Ensure distinct and within range
    seen = set()
    unique_best = []
    for x in best_arr:
        if x not in seen:
            seen.add(x)
            unique_best.append(x)
    while len(unique_best) < 12:
        # Fill with random if AP wasn't distinct enough (it is, but just in case)
        val = random.randint(0, 128)
        if val not in seen:
            seen.add(val)
            unique_best.append(val)
    best_arr = unique_best[:12]
    best_arr.sort()
    
    best_score = max_sidon_size(best_arr)
    
    # If we already found a very good score (e.g., 3), we can stop early or keep searching for 2?
    # Min possible is 2 (any 3 distinct ints in AP have collision? No, {0,1,3} is Sidon. 
    # {0,1,2} is not. 
    # Can we have max Sidon size 2? 
    # That means every 3-subset has a collision.
    # a+b=c+d? For 3 elements {a,b,c}, sums are 2a, a+b, 2b, a+c, b+c, 2c.
    # Collision if a+c=2b (AP).
    # So if every triple is an AP, max size is 2.
    # But you can't have 12 distinct ints where every triple is an AP.
    # So 3 is the likely minimum.
    
    if best_score <= 3:
        return best_arr

    # Local Search
    # We want to minimize best_score.
    current_arr = best_arr[:]
    current_score = best_score
    
    no_improve_count = 0
    max_no_improve = 2000
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        # Generate a neighbor
        candidate = current_arr[:]
        
        # Strategy: Replace one element with a random value from the range
        # or swap two elements with new random values to increase collisions.
        
        # 1. Single element replacement
        idx = random.randint(0, 11)
        old_val = candidate[idx]
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val in candidate and new_val != old_val:
            # Try a few other values if collision
            for _ in range(5):
                new_val = random.randint(0, 128)
                if new_val not in candidate or new_val == old_val:
                    break
            if new_val in candidate and new_val != old_val:
                continue
        
        candidate[idx] = new_val
        candidate.sort()
        
        # Evaluate
        score = max_sidon_size(candidate)
        
        if score < best_score:
            best_score = score
            best_arr = candidate[:]
            current_arr = candidate[:]
            current_score = score
            no_improve_count = 0
            
            if best_score <= 3:
                break
        else:
            no_improve_count += 1
            # Simulated Annealing style: accept worse with low prob to escape local minima
            # But since we want to minimize, and the landscape is discrete,
            # simple hill climbing with restarts is often better.
            # Let's try a small perturbation if we are stuck.
            if no_improve_count % 100 == 0:
                # Random restart to a new AP-like structure or random set
                # Actually, just randomizing one or two elements more aggressively
                for _ in range(random.randint(1, 3)):
                    idx = random.randint(0, 11)
                    candidate[idx] = random.randint(0, 128)
                # Re-evaluate the perturbed candidate? 
                # This might be slow. Let's just continue hill climbing.
                pass
        
        # Occasionally try swapping two elements
        if random.random() < 0.3:
            candidate = current_arr[:]
            i, j = random.sample(range(12), 2)
            # Swap values with random new values
            val1 = random.randint(0, 128)
            val2 = random.randint(0, 128)
            
            # Check distinctness
            # Remove old values, add new ones
            temp_set = set(candidate)
            temp_set.remove(candidate[i])
            temp_set.remove(candidate[j])
            
            if val1 == val2:
                continue
            if val1 in temp_set or val2 in temp_set:
                continue
                
            candidate[i] = val1
            candidate[j] = val2
            candidate.sort()
            
            score = max_sidon_size(candidate)
            if score < best_score:
                best_score = score
                best_arr = candidate[:]
                current_arr = candidate[:]
                current_score = score
                no_improve_count = 0
                if best_score <= 3:
                    break
            else:
                no_improve_count += 1

    return best_arr
# EVOLVE-BLOCK-END
