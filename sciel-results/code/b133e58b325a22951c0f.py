import itertools
import random
import time

def is_sidon(s):
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def max_sidon_size(s):
    # Exhaustive search for max Sidon subset
    # For n=12, subsets of size 6 are 924, size 5 are 792.
    # Total checks are small.
    n = len(s)
    for k in range(n, 3, -1):
        for subset in itertools.combinations(s, k):
            if is_sidon(subset):
                return k
    return 3

def construct():
    # Local search to minimize max Sidon subset size
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    random.seed(42)
    start_time = time.time()
    max_time = 50.0
    
    # Try multiple random restarts
    while time.time() - start_time < max_time:
        # 1. Random initialization
        current_set = random.sample(range(129), 12)
        current_score = max_sidon_size(current_set)
        
        if current_score < best_score:
            best_set = current_set[:]
            best_score = current_score
            
        # 2. Local Search: Try to improve current_set
        improved = True
        while improved and time.time() - start_time < max_time:
            improved = False
            for i in range(12):
                if time.time() - start_time >= max_time:
                    break
                original_val = current_set[i]
                # Try a few random replacements
                for _ in range(30):
                    new_val = random.randint(0, 128)
                    if new_val in current_set and new_val != original_val:
                        continue
                    new_set = current_set[:]
                    new_set[i] = new_val
                    if len(set(new_set)) != 12:
                        continue
                    
                    new_score = max_sidon_size(new_set)
                    if new_score < current_score:
                        current_set = new_set
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_set = current_set[:]
                            best_score = current_score
                        break # Accept first improvement for this i
                # If improved, current_set changed. 
                # We continue to next i with the new current_set.
        
        # If we found a set with score 3, we can stop as it's likely optimal
        if best_score <= 3:
            break
            
    return best_set
