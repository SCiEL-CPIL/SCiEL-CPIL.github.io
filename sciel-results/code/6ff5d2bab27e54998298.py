"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_dissociated(subset):
    """Check if all reciprocal subset sums are distinct using integer arithmetic.
    
    We use the common denominator approach: for subset S, the reciprocal sum is
    sum(1/s for s in S). To check distinctness, we compare fractions exactly.
    We use the fact that two sums are equal iff their cross-multiplied values are equal.
    
    For efficiency with small sets, we enumerate all 2^k subset sums as fractions
    represented by (numerator, denominator) in reduced form, or better:
    use the common denominator LCM of all elements in the full set [1..22].
    
    Actually, let's use a simpler approach: for a given subset, compute all 2^k subset
    sums as fractions using Python's Fraction or by using a common denominator.
    
    Since k is at most ~12-15, 2^k is at most ~32768, which is manageable.
    """
    k = len(subset)
    if k == 0:
        return True
    
    # Use a large common denominator to avoid floating point issues
    # The LCM of numbers 1..22 is 232792560
    # For any subset, the LCM of its elements divides this
    # So we can use this as a universal denominator
    
    LCM_ALL = 232792560  # LCM(1, 2, ..., 22)
    
    # For each element s in subset, 1/s = (LCM_ALL // s) / LCM_ALL
    # So subset sum = sum of (LCM_ALL // s) for s in subset, all over LCM_ALL
    # Two subset sums are distinct iff the numerators are distinct
    
    weights = [LCM_ALL // s for s in subset]
    
    # Now check if all 2^k subset sums of weights are distinct
    # Use a set to track seen sums
    n = len(weights)
    seen = set()
    # Generate all subset sums
    # For small n, we can do this iteratively
    current_sums = [0]
    for w in weights:
        current_sums = current_sums + [x + w for x in current_sums]
    
    if len(current_sums) != len(set(current_sums)):
        return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    # Start with the known good set of size 9
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Verify initial set
    if not _is_dissociated(best_set):
        # Fallback to a smaller known valid set
        best_set = [1, 2, 3, 5, 7, 11]
        best_size = len(best_set)
    
    # Try to improve by adding elements
    # The full universe is [1, 2, ..., 22]
    universe = list(range(1, 23))
    
    # Try greedy addition: repeatedly try to add the best element that maintains dissociation
    # Use randomized restarts for robustness
    
    random.seed(42)
    
    for restart in range(200):
        if time.time() > DEADLINE:
            break
        
        # Start from current best or a random perturbation
        if restart == 0:
            current = list(best_set)
        else:
            # Random subset of the best set, possibly with some random elements
            current = [x for x in best_set if random.random() < 0.7]
            # Add 1-3 random elements from universe not in current
            available = [x for x in universe if x not in current]
            if available:
                add_count = min(len(available), random.randint(1, 3))
                for x in random.sample(available, add_count):
                    current.append(x)
        
        # Greedy: try to add elements one by one
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            # Try adding each candidate element not in current
            candidates = [x for x in universe if x not in current]
            # Shuffle for randomization
            random.shuffle(candidates)
            
            for cand in candidates:
                if time.time() > DEADLINE:
                    break
                trial = current + [cand]
                if _is_dissociated(trial):
                    current = trial
                    changed = True
                    break
        
        # Update best if improved
        if len(current) > best_size:
            best_size = len(current)
            best_set = list(current)
        
        # Also try: start from best_set and try to add more
        if time.time() > DEADLINE:
            break
        trial = list(best_set)
        changed = True
        while changed and time.time() < DEADLINE:
            changed = False
            candidates = [x for x in universe if x not in trial]
            random.shuffle(candidates)
            for cand in candidates:
                if time.time() > DEADLINE:
                    break
                trial2 = trial + [cand]
                if _is_dissociated(trial2):
                    trial = trial2
                    changed = True
                    break
        if len(trial) > best_size:
            best_size = len(trial)
            best_set = list(trial)
    
    return best_set
# EVOLVE-BLOCK-END
