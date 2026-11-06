import time
import random
from itertools import combinations

def _is_sidon(subset):
    """Check if a tuple/list of integers is a Sidon set."""
    s = sorted(subset)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            total = s[i] + s[j]
            if total in sums:
                return False
            sums.add(total)
    return True

def _count_sidon_subsets(ambient, k):
    """Count the number of k-element Sidon subsets in ambient."""
    count = 0
    for subset in combinations(ambient, k):
        if _is_sidon(subset):
            count += 1
    return count

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    best_ambient = list(range(12))
    best_score_4 = _count_sidon_subsets(best_ambient, 4)
    best_score_3 = _count_sidon_subsets(best_ambient, 3)
    
    # We want to minimize best_score_4.
    # If best_score_4 == 0, max Sidon size is 3 (assuming best_score_3 > 0).
    # If we can get best_score_4 == 0, we are done (score 3).
    # Can we get best_score_3 == 0? (Max size 2).
    # This is much harder. 3-Sidon sets are very common.
    # Let's aim for 0 four-Sidon subsets first.
    
    random.seed(42)
    
    # Initial candidates
    candidates = []
    # 1. Random sets
    for _ in range(50):
        candidates.append(sorted(random.sample(range(129), 12)))
    # 2. APs
    for start in range(0, 100):
        for step in range(1, 10):
            cand = [start + i * step for i in range(12)]
            if all(0 <= x <= 128 for x in cand):
                candidates.append(cand)
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline: break
        c4 = _count_sidon_subsets(cand, 4)
        if c4 < best_score_4:
            best_score_4 = c4
            best_ambient = cand
            best_score_3 = _count_sidon_subsets(cand, 3)
            if best_score_4 == 0:
                # Found a set with no 4-Sidon subsets
                # Check if we can do better (3-Sidon subsets to 0?)
                # But 3 is likely the floor.
                # We can try to polish to see if 3-Sidon is also 0 (unlikely)
                pass

    # Local Search
    current_ambient = best_ambient[:]
    current_c4 = best_score_4
    current_c3 = best_score_3
    
    # If we haven't found 0 four-Sidon subsets, continue searching
    # If we have, we might try to minimize 3-Sidon subsets just in case,
    # but the primary goal is to keep 4-Sidon at 0.
    
    for iteration in range(100000):
        if time.time() > deadline:
            break
            
        # Perturbation: Change one element
        i = random.randint(0, 11)
        old_val = current_ambient[i]
        new_val = random.randint(0, 128)
        
        if new_val == old_val or new_val in current_ambient:
            continue
            
        trial = current_ambient[:]
        trial[i] = new_val
        trial = sorted(trial)
        
        trial_c4 = _count_sidon_subsets(trial, 4)
        
        # Objective: Minimize c4. If c4 is 0, minimize c3?
        # Actually, if c4 > 0, we just want to reduce it.
        # If c4 == 0, we want to keep it 0.
        # If we reduce c4, accept.
        # If c4 is same, and c4 == 0, we can accept to explore.
        # If c4 is same and c4 > 0, accept with some probability.
        
        if trial_c4 < current_c4:
            current_ambient = trial
            current_c4 = trial_c4
            current_c3 = _count_sidon_subsets(current_ambient, 3)
            if current_c4 < best_score_4:
                best_score_4 = current_c4
                best_ambient = current_ambient
                best_score_3 = current_c3
        else:
            # Accept worse with probability
            # If we are at 0, we don't want to go to 1.
            # So if current_c4 == 0 and trial_c4 > 0, reject.
            if current_c4 == 0 and trial_c4 > 0:
                continue
                
            T = 1.0 - (iteration / 100000.0)
            if random.random() < T * 0.5:
                current_ambient = trial
                current_c4 = trial_c4
                current_c3 = _count_sidon_subsets(current_ambient, 3)
                # Update best if c4 is better (unlikely if we accept worse)
                if current_c4 < best_score_4:
                    best_score_4 = current_c4
                    best_ambient = current_ambient
                    best_score_3 = current_c3

    return best_ambient
