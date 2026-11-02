"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from itertools import combinations

# EVOLVE-BLOCK-START
def _score(U):
    """Compute the score for a set U."""
    U = sorted(set(U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset U+U
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    sumset_size = len(sumset)
    
    # Compute difference set U-U
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size == 0:
        return 0.0
    
    ratio = diffset_size / sumset_size
    log_arg = 2 * max_u + 1
    
    if log_arg <= 1:
        return 0.0
    
    score = 1 + math.log(ratio) / math.log(log_arg)
    return score


def _local_search(U, max_time, rng):
    """Perform local search to improve the set U."""
    U = sorted(set(U))
    best_score = _score(U)
    best_U = list(U)
    start_time = time.time()
    
    # Try adding elements
    max_u = max(U)
    
    while time.time() - start_time < max_time * 0.7:
        improved = False
        
        # Try adding a new element
        # Candidate elements: near existing elements, or gaps
        candidates = set()
        for u in U:
            for delta in range(-3, 4):
                v = u + delta
                if v >= 0 and v not in U:
                    candidates.add(v)
        
        # Also try some larger jumps
        if max_u > 0:
            for _ in range(20):
                v = rng.randint(0, max_u + max_u // 3 + 5)
                if v not in U:
                    candidates.add(v)
        
        # Try each candidate addition
        for v in sorted(candidates):
            new_U = sorted(U + [v])
            new_score = _score(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                max_u = max(U)
                improved = True
                break
        
        if not improved:
            # Try removing an element
            removed = False
            for idx in range(len(U)):
                if U[idx] == 0:
                    continue  # Must keep 0
                new_U = U[:idx] + U[idx+1:]
                if len(new_U) < 2:
                    continue
                new_score = _score(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    max_u = max(U)
                    removed = True
                    break
            
            if not removed:
                # Try swapping: remove one, add another
                swapped = False
                for idx in range(len(U)):
                    if U[idx] == 0:
                        continue
                    U_without = U[:idx] + U[idx+1:]
                    # Try adding candidates to U_without
                    for v in sorted(candidates):
                        if v in U_without:
                            continue
                        new_U = sorted(U_without + [v])
                        new_score = _score(new_U)
                        if new_score > best_score + 1e-12:
                            best_score = new_score
                            best_U = new_U
                            U = new_U
                            max_u = max(U)
                            swapped = True
                            break
                    if swapped:
                        break
                
                if not swapped:
                    # No improvement, break
                    break
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin from 120s limit
    
    rng = random.Random(42)
    
    best_U = [0, 1, 3]
    best_score = _score(best_U)
    
    # Strategy 1: Start with small sets and grow
    # Try various initial structures
    
    # Geometric-like sets
    for base in [2, 3, 4, 5]:
        for max_power in range(3, 8):
            U = [0]
            val = 1
            for _ in range(max_power):
                U.append(val)
                val *= base
            if time.time() > deadline:
                break
            score = _score(U)
            if score > best_score:
                best_score = score
                best_U = list(U)
    
    # Strategy 2: Local search from good starting points
    starting_sets = [
        [0, 1, 3],
        [0, 1, 3, 9],
        [0, 1, 3, 9, 27],
        [0, 2, 4, 8, 16, 32],
        [0, 1, 2, 4, 8, 16, 32],
        [0, 1, 4, 16, 64],
        [0, 3, 9, 27, 81],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 2, 5, 10, 20, 40],
    ]
    
    for start_set in starting_sets:
        if time.time() > deadline - 10:
            break
        U, score = _local_search(list(start_set), deadline - time.time() - 5, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Strategy 3: Randomized construction with local search
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        # Try random starts
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            # Generate a random set
            size = rng.randint(4, 15)
            max_val = rng.randint(20, 200)
            U = [0]
            while len(U) < size:
                v = rng.randint(1, max_val)
                if v not in U:
                    U.append(v)
            U = sorted(U)
            
            U, score = _local_search(U, min(remaining_time - 5, 10), rng)
            if score > best_score:
                best_score = score
                best_U = U
    
    # Final polish: try more aggressive local search on best
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        U, score = _local_search(list(best_U), remaining_time - 2, rng)
        if score > best_score:
            best_score = score
            best_U = U
    
    return sorted(best_U)
# EVOLVE-BLOCK-END
