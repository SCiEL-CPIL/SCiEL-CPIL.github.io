import time
import random
import math
from itertools import combinations

def _score(A):
    """Compute max(|A+A|, |A*A|) for set A (list of distinct ints)."""
    n = len(A)
    sumset = set()
    prodset = set()
    for i in range(n):
        for j in range(i, n):
            sumset.add(A[i] + A[j])
            prodset.add(A[i] * A[j])
    return max(len(sumset), len(prodset))

def _valid(A, max_val=10000):
    """Check if A is valid: 20 distinct positive integers <= max_val."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    if any(x < 1 for x in A):
        return False
    if any(x > max_val for x in A):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Initial set: 1 to 20
    best_A = list(range(1, 21))
    best_score = _score(best_A)
    
    # Try some structured initial sets
    candidate_sets = [
        list(range(1, 21)),          # 1..20
        list(range(2, 22)),          # 2..21
        [2**i for i in range(20)],   # powers of 2 (but max is 2^19=524288 > 10000, invalid)
    ]
    
    # Filter valid candidates
    for cs in candidate_sets:
        if _valid(cs):
            s = _score(cs)
            if s < best_score:
                best_score = s
                best_A = list(cs)
    
    # Local search: try to reduce score by modifying elements
    random.seed(42)
    
    def _try_improve(A, score):
        """Try single-element modifications to improve score."""
        n = len(A)
        best_A_local = list(A)
        best_score_local = score
        
        # Try replacing each element with a nearby value
        for i in range(n):
            old_val = A[i]
            # Try values in a range around old_val
            for delta in range(-50, 51):
                if delta == 0:
                    continue
                new_val = old_val + delta
                if new_val < 1 or new_val > 10000:
                    continue
                # Check distinctness
                if new_val in A and new_val != old_val:
                    continue
                # Create new set
                new_A = list(A)
                new_A[i] = new_val
                s = _score(new_A)
                if s < best_score_local:
                    best_score_local = s
                    best_A_local = new_A
        
        return best_A_local, best_score_local
    
    # Iterative local search
    current_A = list(best_A)
    current_score = best_score
    
    while time.time() < deadline:
        new_A, new_score = _try_improve(current_A, current_score)
        if new_score < current_score:
            current_A = new_A
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_A = list(current_A)
        else:
            # No improvement; try random perturbation
            # Pick a random element and change it significantly
            i = random.randint(0, 19)
            new_val = random.randint(1, 10000)
            if new_val in current_A:
                continue
            new_A = list(current_A)
            new_A[i] = new_val
            s = _score(new_A)
            if s < current_score:
                current_A = new_A
                current_score = s
                if current_score < best_score:
                    best_score = current_score
                    best_A = list(current_A)
            elif time.time() > deadline - 5:
                break
    
    return best_A
