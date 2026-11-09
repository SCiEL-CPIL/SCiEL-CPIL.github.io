"""
Snake-in-the-box in Q_9: construct longest induced path.

Approach: Iterated Local Search (ILS) with 2-opt perturbations.
Distinct from Simulated Annealing on tail moves, this approach uses
structural changes (reversing segments) to escape local optima where
the tail is stuck, followed by greedy extension.

Key components:
1. Greedy initialization.
2. Greedy extension phase.
3. 2-opt perturbation when stuck: reverse a random segment to change
   the adjacency structure, potentially opening new extension paths.
4. Validity check for 2-opt moves is performed in O(L^2) which is fast
   for L < 200.
"""

import random
import time

N = 9
MAX_VERTEX = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_path(path):
    """Check if the given path is a valid induced path in Q_N."""
    if not path:
        return True
    used = set(path)
    for i in range(len(path) - 1):
        if (path[i] ^ path[i+1]).bit_count() != 1:
            return False
    # Check induced property: non-consecutive vertices must differ by >= 2 bits
    for i in range(len(path)):
        for j in range(i + 2, len(path)):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def _greedy_init(start=0):
    """Quick greedy initialization."""
    path = [start]
    used = {start}
    while True:
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check induced property against all previous vertices
            valid = True
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    valid = False
                    break
            if valid:
                candidates.append(v)
        if not candidates:
            break
        next_v = random.choice(candidates)
        path.append(next_v)
        used.add(next_v)
    return path

def _try_extend(path, used):
    """Attempt to extend the path by one valid vertex. Returns True if successful."""
    cur = path[-1]
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used:
            continue
        valid = True
        for p in path[:-1]:
            if (v ^ p).bit_count() == 1:
                valid = False
                break
        if valid:
            candidates.append(v)
    
    if not candidates:
        return False
    
    next_v = random.choice(candidates)
    path.append(next_v)
    used.add(next_v)
    return True

def _apply_2opt(path, i, j):
    """
    Apply 2-opt move: reverse path[i:j+1].
    Returns True if the new path is valid, False otherwise.
    Modifies path in place if valid, otherwise restores original.
    """
    original = path[i:j+1]
    path[i:j+1] = original[::-1]
    
    if _is_valid_path(path):
        return True
    else:
        path[i:j+1] = original
        return False

def construct():
    """
    Construct the longest snake in Q_9 using ILS with 2-opt perturbations.
    Returns a list of distinct integers representing the vertices in order.
    """
    start_time = time.time()
    time_limit = 115.0
    
    random.seed(42)
    
    best_path = _greedy_init(0)
    best_len = len(best_path)
    
    current_path = list(best_path)
    current_used = set(current_path)
    
    while time.time() - start_time < time_limit:
        # Phase 1: Greedy Extension
        while True:
            if not _try_extend(current_path, current_used):
                break
        
        # Update best if improved
        if len(current_path) > best_len:
            best_len = len(current_path)
            best_path = list(current_path)
            
        # Phase 2: Perturbation (2-opt)
        # Only perturb if we are stuck (couldn't extend) and path is long enough
        if len(current_path) > 10:
            # Try a few random 2-opt moves
            improved_structure = False
            for _ in range(20):
                L = len(current_path)
                if L < 3:
                    break
                i = random.randint(1, L - 2)
                j = random.randint(i + 1, L - 1)
                # Ensure we don't pick trivial small reversals that do nothing
                if j - i < 2:
                    continue
                    
                if _apply_2opt(current_path, i, j):
                    # Rebuild used set (expensive but rare enough)
                    current_used = set(current_path)
                    improved_structure = True
                    break
            
            # If no 2-opt worked, try a "pivot" or "retract" as fallback?
            # Or just restart?
            # Let's try a simple tail pivot to change direction
            if not improved_structure and len(current_path) >= 2:
                prev = current_path[-2]
                last = current_path[-1]
                candidates = []
                for b in range(N):
                    v = prev ^ (1 << b)
                    if v in current_used or v == last:
                        continue
                    valid = True
                    for p in current_path[:-2]:
                        if (v ^ p).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        candidates.append(v)
                if candidates:
                    new_last = random.choice(candidates)
                    current_path[-1] = new_last
                    current_used.remove(last)
                    current_used.add(new_last)
        else:
            # If path is too short, just restart with a new greedy init
            current_path = _greedy_init(random.randint(0, MAX_VERTEX-1))
            current_used = set(current_path)
            if len(current_path) > best_len:
                best_len = len(current_path)
                best_path = list(current_path)

    return best_path

# EVOLVE-BLOCK-END
