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
    for k in range(n, 3, -1):
        for subset in itertools.combinations(s, k):
            if is_sidon(subset):
                return k
    return 3

def construct():
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    if best_score <= 4:
        return best_set

    # Local search to try to reduce max_sidon_size to 4
    current_set = best_set[:]
    current_score = best_score
    
    # We want to find a set with score 4
    # We can try random swaps/moves
    for _ in range(5000):
        # Random perturbation
        new_set = current_set[:]
        i, j = random.sample(range(12), 2)
        # Swap or move
        if random.random() < 0.5:
            new_set[i], new_set[j] = new_set[j], new_set[i]
        else:
            # Move i to a random position in [0, 128]
            new_val = random.randint(0, 128)
            if new_val not in new_set:
                new_set[i] = new_val
        
        if len(set(new_set)) != 12:
            continue
            
        new_score = max_sidon_size(new_set)
        
        if new_score < current_score:
            current_set = new_set
            current_score = new_score
            if current_score <= 4:
                break
        elif new_score == current_score:
            # Optional: accept with small prob to escape local minima
            # But for now, let's just stick to improving or trying many randoms
            pass

    return current_set
