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
    n = len(s)
    for k in range(n, 2, -1):
        for subset in itertools.combinations(s, k):
            if is_sidon(subset):
                return k
    return 2

def construct():
    # Start with AP
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    if best_score <= 3:
        return best_set

    # Local Search
    current_set = best_set[:]
    current_score = best_score
    
    start_time = time.time()
    time_limit = 55
    
    # Simple random restart local search
    # Since we can only do ~100-200 evaluations, 
    # we do a few random restarts and a short local search from each.
    
    for restart in range(20):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            s = current_set[:]
        else:
            s = random.sample(range(129), 12)
            
        score = max_sidon_size(s)
        if score < current_score:
            current_score = score
            current_set = s
            if current_score <= 3:
                return current_set
        
        # Local search from s
        # Try to improve by swapping one element
        # Limit local search steps to 5 to keep time low
        for step in range(5):
            if time.time() - start_time > time_limit:
                break
                
            i = random.randint(0, 11)
            old_val = s[i]
            # Try a few random replacements
            improved = False
            for _ in range(10):
                new_val = random.randint(0, 128)
                if new_val == old_val:
                    continue
                if new_val in s:
                    continue
                    
                s[i] = new_val
                new_score = max_sidon_size(s)
                
                if new_score < score:
                    score = new_score
                    improved = True
                    if score < current_score:
                        current_score = score
                        current_set = s[:]
                        if current_score <= 3:
                            return current_set
                    break
                else:
                    s[i] = old_val
            
            if not improved:
                break
                
    return current_set
