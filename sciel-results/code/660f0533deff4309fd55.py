import time
import random
from itertools import combinations

def _is_sidon_with_diagonals(subset):
    """Check if all unordered pair sums (including diagonals) are unique."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def _has_sidon_subset_of_size_k(ambient, k):
    """Check if ambient contains a Sidon subset of size k."""
    n = len(ambient)
    # If k is 1, always true. If k is 2, always true (distinct integers).
    if k <= 2:
        return True
    
    # Use combinations, but prune if possible? 
    # For n=12, k=5, there are C(12,5) = 792 combinations.
    # This is very fast.
    for subset in combinations(ambient, k):
        if _is_sidon_with_diagonals(subset):
            return True
    return False

def _max_sidon_size_ambient(ambient):
    """Find the largest Sidon subset of ambient set."""
    n = len(ambient)
    # Check from largest down
    # We primarily care if it's >= 5 or < 5.
    for size in range(n, 1, -1):
        if _has_sidon_subset_of_size_k(ambient, size):
            return size
    return 1

def _evaluate_ambient(ambient):
    """Return the max Sidon subset size for this ambient set."""
    return _max_sidon_size_ambient(ambient)

def construct():
    deadline = time.time() + 75  # safety margin
    
    best_score = 12
    best_ambient = list(range(12))
    
    # Strategy: Find a set with no 5-element Sidon subset.
    # This implies max size <= 4.
    
    # 1. Structured Candidates: APs and their variants
    candidates = []
    
    # APs
    for start in range(0, 120):
        for step in range(1, 11):
            cand = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
                
    # "Block" APs: Union of small APs
    # e.g., {0,1,2}, {10,11,12}, etc.
    for block_gap in range(3, 20):
        for block_len in [3, 4]:
            if block_len * (12 // block_len) < 12: continue
            # Try to pack 12 elements into blocks of length block_len
            # This is complex to generate generally, so we stick to simple structures or random.
            
    # 2. Random Search with Targeted Objective
    # We want to minimize max_sidon_size.
    # If we find size 4, we stop or keep searching for 3/2.
    
    random.seed(42)
    
    # Initial population from structured
    for cand in candidates:
        if time.time() > deadline: break
        score = _evaluate_ambient(cand)
        if score < best_score:
            best_score = score
            best_ambient = cand
        if best_score <= 4:
            # We found a very good candidate. Try to polish it to 3 or 2?
            # But 4 is likely the floor for 12 elements in [0,128] unless we are very clever.
            # Let's continue searching to see if we can find 3.
            pass

    # Local Search / Simulated Annealing
    # Start from the best found so far
    current_ambient = best_ambient[:]
    current_score = best_score
    
    # If we haven't found anything better than 5, start from a random AP or random set
    if current_score > 4:
        # Try a dense AP again, maybe with random offsets
        current_ambient = sorted(random.sample(range(0, 129), 12))
        current_score = _evaluate_ambient(current_ambient)
        if current_score < best_score:
            best_score = current_score
            best_ambient = current_ambient[:]

    for iteration in range(20000):
        if time.time() > deadline:
            break
            
        # Perturbation: Swap 1 or 2 elements
        if random.random() < 0.7:
            # Swap 1
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current_ambient:
                continue
            trial = current_ambient[:]
            trial[i] = new_val
            trial = sorted(trial)
        else:
            # Swap 2
            i, j = random.sample(range(12), 2)
            new_val1 = random.randint(0, 128)
            new_val2 = random.randint(0, 128)
            if new_val1 in current_ambient or new_val2 in current_ambient:
                continue
            if new_val1 == new_val2:
                continue
            trial = current_ambient[:]
            trial[i] = new_val1
            trial[j] = new_val2
            trial = sorted(trial)
            if len(set(trial)) != 12:
                continue

        trial_score = _evaluate_ambient(trial)
        
        # Accept if better
        if trial_score < current_score:
            current_score = trial_score
            current_ambient = trial
            if trial_score < best_score:
                best_score = trial_score
                best_ambient = trial
        else:
            # Accept worse with some probability to escape local minima (Simulated Annealing)
            # Temperature decreases over time
            T = 1.0 - (iteration / 20000.0)
            if random.random() < T * 0.5:
                current_score = trial_score
                current_ambient = trial

        # Early exit if we found size 2? (Unlikely)
        if best_score <= 3:
             # Try to find 2?
             pass

    return best_ambient
