import random
import math
import itertools
from collections import Counter

def is_sidon(subset):
    """
    Check if a subset is a Sidon set.
    """
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """
    Find the size of the largest Sidon subset of S.
    """
    n = len(S)
    for k in range(n, 0, -1):
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def count_collisions(S):
    """
    Count the total number of sum collisions.
    """
    sums = []
    for i in range(len(S)):
        for j in range(i, len(S)):
            sums.append(S[i] + S[j])
    counter = Counter(sums)
    collisions = 0
    for count in counter.values():
        if count > 1:
            collisions += count * (count - 1) // 2
    return collisions

def construct():
    rng = random.Random(42)
    
    # Start with a dense arithmetic progression
    current_S = list(range(12))
    current_score = count_collisions(current_S)
    
    best_S = current_S.copy()
    best_score = current_score
    
    # Simulated Annealing to maximize collisions
    T = 5.0
    cooling = 0.9995
    iterations = 10000
    
    for i in range(iterations):
        idx = rng.randint(0, 11)
        old_val = current_S[idx]
        
        # Perturbation: change one element
        # Bias towards small numbers to increase collisions
        new_val = rng.randint(0, 40)
        
        if new_val in current_S and new_val != old_val:
            continue
            
        current_S[idx] = new_val
        new_score = count_collisions(current_S)
        
        delta = new_score - current_score
        if delta > 0 or (T > 1e-6 and rng.random() < math.exp(delta / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_S = current_S.copy()
        else:
            current_S[idx] = old_val
            
        T *= cooling
        
    # Final Polish: Local search on best_S using EXACT evaluation
    # We want to minimize max_sidon_size.
    best_exact_size = max_sidon_size(best_S)
    
    # Try to improve with random local search
    for _ in range(100):
        idx = rng.randint(0, 11)
        old_val = best_S[idx]
        new_val = rng.randint(0, 60)
        if new_val in best_S and new_val != old_val:
            continue
        candidate = best_S.copy()
        candidate[idx] = new_val
        size = max_sidon_size(candidate)
        if size < best_exact_size:
            best_S = candidate
            best_exact_size = size

    return best_S
