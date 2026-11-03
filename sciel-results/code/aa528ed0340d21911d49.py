import math
import time
import random
import numpy as np
import itertools
from functools import lru_cache

def _score(U):
    """Score a set U. Returns 0 for invalid sets."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Use NumPy for faster computation
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute all pairwise sums
    sums_arr = U_arr[:, None] + U_arr[None, :]
    unique_sums = np.unique(sums_arr)
    
    # Compute all pairwise differences
    diffs_arr = U_arr[:, None] - U_arr[None, :]
    unique_diffs = np.unique(diffs_arr)
    
    ratio = len(unique_diffs) / len(unique_sums)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _is_golomb_ruler(U):
    """Check if U is a Golomb ruler (all pairwise differences are distinct)."""
    U = sorted(set(U))
    n = len(U)
    if n <= 2:
        return True
    diffs = set()
    for i in range(n):
        for j in range(i+1, n):
            d = U[j] - U[i]
            if d in diffs:
                return False
            diffs.add(d)
    return True

def _score_fast(U):
    """Faster scoring for small sets using Python sets (avoids NumPy overhead)."""
    if not U or 0 not in U:
        return 0
    U = sorted(set(U))
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    sumset = set()
    diffset = set()
    n = len(U)
    for i in range(n):
        for j in range(n):
            sumset.add(U[i] + U[j])
            diffset.add(U[i] - U[j])
    
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return 0
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 1.0
    return 1 + math.log(ratio) / denom

def _local_search(U, max_iterations=500, deadline=None):
    """Local search to refine a set with structured perturbations."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = U[:]
    current_U = U[:]
    current_score = best_score
    
    for i in range(max_iterations):
        if deadline and time.time() > deadline:
            break
        
        # Try adding a random element
        max_u = max(current_U) if current_U else 0
        if random.random() < 0.5:
            new_val = random.randint(1, max(max_u * 2, 10))
        else:
            new_val = random.randint(1, max_u + 10)
        
        candidate = sorted(set(current_U + [new_val]))
        s = _score(candidate)
        if s > current_score:
            current_U = candidate
            current_score = s
            if s > best_score:
                best_score = s
                best_U = candidate[:]
        
        # Try removing an element (not 0)
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            candidate = sorted(set(current_U[:idx] + current_U[idx+1:]))
            s = _score(candidate)
            if s > current_score:
                current_U = candidate
                current_score = s
                if s > best_score:
                    best_score = s
                    best_U = candidate[:]
        
        # Try shifting an element
        if len(current_U) > 1:
            idx = random.randint(1, len(current_U) - 1)
            shift = random.randint(-5, 5)
            if shift == 0:
                continue
            new_val = current_U[idx] + shift
            if new_val >= 0:
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) >= len(current_U) - 1:  # Allow at most one collision
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
        
        # Try a larger perturbation
        if len(current_U) > 1 and random.random() < 0.3:
            idx = random.randint(1, len(current_U) - 1)
            new_val = random.randint(1, max(max_u * 3, 20))
            candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
            if len(candidate) >= len(current_U) - 1:
                s = _score(candidate)
                if s > current_score:
                    current_U = candidate
                    current_score = s
                    if s > best_score:
                        best_score = s
                        best_U = candidate[:]
        
        # Structured perturbation: try to maintain Golomb-like properties
        if len(current_U) > 2 and random.random() < 0.2:
            # Try moving an element to a position that preserves difference uniqueness
            idx = random.randint(1, len(current_U) - 1)
            # Try positions near existing elements
            nearby_positions = []
            for j in range(len(current_U)):
                if j != idx:
                    for delta in range(-3, 4):
                        pos = current_U[j] + delta
                        if pos >= 0 and pos not in current_U:
                            nearby_positions.append(pos)
            
            if nearby_positions:
                new_val = random.choice(nearby_positions)
                candidate = sorted(set([current_U[j] if j != idx else new_val for j in range(len(current_U))]))
                if len(candidate) >= len(current_U) - 1:
                    s = _score(candidate)
                    if s > current_score:
                        current_U = candidate
                        current_score = s
                        if s > best_score:
                            best_score = s
                            best_U = candidate[:]
                        
    return best_U

def _exhaustive_search_small(max_val, n_min, n_max, deadline):
    """Exhaustive search for small sets."""
    best_score = 0
    best_set = [0, 1]
    
    for n in range(n_min, n_max + 1):
        if time.time() > deadline:
            break
        # We need to choose n-1 elements from 1..max_val
        for combo in itertools.combinations(range(1, max_val + 1), n - 1):
            if time.time() > deadline:
                break
            U = (0,) + combo
            s = _score_fast(list(U))
            if s > best_score:
                best_score = s
                best_set = list(U)
    
    return best_set, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 20s safety margin
    
    # Known good starting points including Golomb rulers
    candidates = [
        # Golomb rulers (maximize |U-U| for given diameter)
        [0, 1, 4],
        [0, 1, 4, 9],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 39],
        [0, 1, 4, 10, 18, 23, 31, 39, 47],
        # Other good candidates
        [0, 1, 2, 4, 8],
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
        [0, 1, 2, 4, 8],
    ]
    
    best_U = [0, 1]
    best_score = 0
    
    # Evaluate candidates
    for c in candidates:
        s = _score(c)
        if s > best_score:
            best_score = s
            best_U = c[:]
    
    # Phase 1: Exhaustive search for small sets
    # Try n=3,4,5 with max_val up to 30
    for max_val in [20, 30, 40]:
        if time.time() > deadline:
            break
        n_max = 5 if max_val <= 30 else 4
        found_set, found_score = _exhaustive_search_small(max_val, 3, n_max, deadline)
        if found_score > best_score:
            best_score = found_score
            best_U = found_set[:]
    
    # Phase 2: Local search refinement with more iterations
    if time.time() < deadline - 15:
        refined = _local_search(best_U, max_iterations=1500, deadline=deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_U = refined
    
    # Phase 3: Try random restarts with local search
    random.seed(42)
    for _ in range(30):
        if time.time() > deadline - 10:
            break
        
        n = random.randint(3, 9)
        max_val = random.randint(15, 80)
        
        # Generate a random set with 0 and n-1 other elements
        other_elements = random.sample(range(1, max_val), min(n - 1, max_val - 1))
        U = sorted([0] + other_elements)
        
        s = _score(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        
        # Refine with local search
        if time.time() < deadline - 5:
            refined = _local_search(U, max_iterations=400, deadline=deadline)
            s_r = _score(refined)
            if s_r > best_score:
                best_score = s_r
                best_U = refined
    
    # Phase 4: Final high-precision polish - try adding/removing elements to best set
    if time.time() < deadline - 3:
        current_set = set(best_U)
        max_current = max(best_U)
        
        # Try adding elements in a wider range
        for new_elem in range(1, min(max_current + 40, 150)):
            if time.time() > deadline:
                break
            if new_elem in current_set:
                continue
            new_set = sorted(current_set | {new_elem})
            s = _score(new_set)
            if s > best_score:
                best_score = s
                best_U = new_set
                current_set = set(new_set)
                max_current = max(best_U)
        
        # Try removing elements
        if len(best_U) > 2:
            for elem in list(best_U):
                if time.time() > deadline:
                    break
                if elem == 0:
                    continue
                new_set = sorted(set(best_U) - {elem})
                if len(new_set) < 2:
                    continue
                s = _score(new_set)
                if s > best_score:
                    best_score = s
                    best_U = new_set
    
    # Phase 5: Final local search polish with tighter tolerance
    if time.time() < deadline - 1:
        refined = _local_search(best_U, max_iterations=500, deadline=deadline)
        s = _score(refined)
        if s > best_score:
            best_score = s
            best_U = refined
    
    return best_U
