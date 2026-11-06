"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_strong_sidon(subset):
    """Check if subset has all unordered pair sums unique including diagonals.
    i.e., for all pairs (a,b) with a<=b from subset, the sum a+b is unique."""
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(ambient):
    """Find the largest subset of ambient that is a strong Sidon set."""
    n = len(ambient)
    best = 0
    # Try subsets from largest to smallest
    for size in range(n, 0, -1):
        for subset in combinations(ambient, size):
            if _is_strong_sidon(subset):
                return size
        # If we got here, no subset of this size works
    return best

def _evaluate(ambient):
    """Return the max Sidon subset size for the given ambient set."""
    return _max_sidon_size(ambient)

def construct():
    deadline = time.time() + 85  # 85s deadline, 5s safety margin
    best_set = list(range(12))
    best_score = _evaluate(best_set)
    
    # Try a few random initializations
    rng = random.Random(42)
    
    # Start with a structured set that has many collisions
    # Use multiples of a small number to create collisions
    # e.g., {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} has many collisions
    # Better: cluster values to create many a+b=c+d collisions
    
    # Initial candidate: cluster around a few values
    initial_candidates = [
        list(range(12)),
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # Spread out but with some structure
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ]
    
    # Generate random initial sets
    for _ in range(50):
        s = sorted(rng.sample(range(129), 12))
        initial_candidates.append(s)
    
    # Also try structured sets
    for base in range(0, 120, 10):
        s = sorted(set([base + i for i in range(12)]))
        if len(s) == 12:
            initial_candidates.append(s)
    
    # Evaluate all initial candidates
    for cand in initial_candidates:
        if len(set(cand)) != 12:
            continue
        if any(x < 0 or x > 128 for x in cand):
            continue
        score = _evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
    
    # Local search: try swapping elements
    for iteration in range(200000):
        if time.time() > deadline:
            break
        if best_score <= 1:
            break
        
        cur = best_set[:]
        cur.sort()
        
        # Pick a random element to replace
        idx = rng.randint(0, 11)
        old_val = cur[idx]
        
        # Try a few replacement values
        tried = 0
        found_improvement = False
        while tried < 20:
            new_val = rng.randint(0, 128)
            if new_val == old_val or new_val in cur:
                tried += 1
                continue
            tried += 1
            
            trial = cur[:]
            trial[idx] = new_val
            trial.sort()
            
            score = _evaluate(trial)
            if score < best_score:
                best_score = score
                best_set = trial
                found_improvement = True
                break
            elif score == best_score:
                # Accept with some probability for diversity
                if rng.random() < 0.3:
                    best_set = trial
                    found_improvement = True
                    break
        
        if not found_improvement:
            # Try a different strategy: swap two elements
            if rng.random() < 0.3:
                i, j = rng.sample(range(12), 2)
                trial = best_set[:]
                trial[i], trial[j] = trial[j], trial[i]
                # This doesn't change the set, so skip
                # Instead, try replacing one element with a value near another
                idx2 = rng.randint(0, 11)
                old_val2 = best_set[idx2]
                # Try values near other elements
                for other in best_set:
                    if other == old_val2:
                        continue
                    for delta in [-2, -1, 1, 2, 3]:
                        new_val2 = other + delta
                        if 0 <= new_val2 <= 128 and new_val2 not in best_set and new_val2 != old_val2:
                            trial2 = best_set[:]
                            trial2[idx2] = new_val2
                            score2 = _evaluate(trial2)
                            if score2 < best_score:
                                best_score = score2
                                best_set = trial2
                                break
                    if best_score <= 1:
                        break
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
