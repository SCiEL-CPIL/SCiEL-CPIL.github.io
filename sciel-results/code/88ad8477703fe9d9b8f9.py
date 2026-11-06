"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 9
SEED_LENGTH = 104
DEADLINE = 115.0  # safety margin before 120s kill

# EVOLVE-BLOCK-START
def _is_safe(v, path, used):
    """Check if v can be appended to path without violating induced path property."""
    if v in used:
        return False
    # v must differ from all non-consecutive vertices by at least 2 bits
    # Consecutive is path[-1], which we already know differs by exactly 1 bit
    for i in range(len(path) - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, used, start_bit_order=None):
    """Extend path greedily. Returns extended path and used set."""
    if start_bit_order is None:
        start_bit_order = list(range(N))
    
    while True:
        cur = path[-1]
        found = False
        for b in start_bit_order:
            v = cur ^ (1 << b)
            if _is_safe(v, path, used):
                path.append(v)
                used.add(v)
                found = True
                break
        if not found:
            break
    return path, used

def _backtrack_search(path, used, depth_limit=None, deadline=None):
    """
    Backtracking search to extend the current path.
    Uses constraint-based pruning: if no valid next vertex exists, backtrack.
    Returns the longest path found.
    """
    best_path = list(path)
    best_len = len(path)
    
    if deadline is None:
        deadline = time.time() + DEADLINE - time.time()
    
    def backtrack(path, used):
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return
        
        cur = path[-1]
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v not in used and _is_safe(v, path, used):
                candidates.append(v)
        
        if not candidates:
            if len(path) > best_len:
                best_len = len(path)
                best_path = list(path)
            return
        
        # Heuristic: try candidates that leave more options for the future
        # Score candidates by how many valid extensions they allow
        scored = []
        for v in candidates:
            # Count how many of v's neighbors are still available
            ext_count = 0
            for b2 in range(N):
                v2 = v ^ (1 << b2)
                if v2 not in used and v2 != cur:
                    # Quick check: v2 must not be at distance 1 from any path[:-1]
                    valid = True
                    for i in range(len(path) - 1):
                        if (v2 ^ path[i]).bit_count() == 1:
                            valid = False
                            break
                    if valid:
                        ext_count += 1
            scored.append((-ext_count, v))
        
        scored.sort()
        
        for _, v in scored:
            path.append(v)
            used.add(v)
            backtrack(path, used)
            path.pop()
            used.discard(v)
            
            if time.time() > deadline:
                return
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
    
    backtrack(path, used)
    return best_path

def _random_restart_greedy(seed, deadline):
    """Do a greedy search with a random bit ordering, then try to extend with backtracking."""
    bit_order = list(range(N))
    random.Random(seed).shuffle(bit_order)
    
    path = [0]
    used = {0}
    
    # Greedy phase
    path, used = _greedy_extend(path, used, bit_order)
    
    # If we haven't beaten best yet, try backtracking extension
    if len(path) > SEED_LENGTH:
        # Try to extend further with limited backtracking
        extended = _backtrack_search(path, used, deadline=deadline)
        if len(extended) > len(path):
            path = extended
    
    return path

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # First, get the seed greedy result
    seed_path = [0]
    seed_used = {0}
    seed_path, seed_used = _greedy_extend(seed_path, seed_used)
    best_path = list(seed_path)
    best_len = len(best_path)
    
    # Try multiple random restarts with different bit orderings
    seed_counter = 0
    while time.time() < deadline - 5.0:  # leave 5s margin
        seed_counter += 1
        path = _random_restart_greedy(seed_counter, deadline)
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        # If we've found a very good result, spend remaining time on backtracking
        if best_len > SEED_LENGTH + 20 and time.time() < deadline - 10.0:
            # Try deeper backtracking from the best path
            extended = _backtrack_search(best_path, set(best_path), deadline=deadline)
            if len(extended) > best_len:
                best_len = len(extended)
                best_path = list(extended)
    
    return best_path
# EVOLVE-BLOCK-END
