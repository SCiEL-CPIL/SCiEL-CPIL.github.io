"""Snake-in-the-box in Q_9: improved construction via heuristic search."""
import random
import sys

N = 9
TOTAL_VERTICES = 1 << N

# EVOLVE-BLOCK-START

def _is_valid_extension(v, path):
    """Check if v can be appended to path to maintain induced path property."""
    for i, p in enumerate(path):
        if i == len(path) - 1:
            continue
        # v must not be adjacent to any non-consecutive vertex
        if (v ^ p).bit_count() == 1:
            return False
    return True

def _try_extend_greedy(path, used, order):
    """Try to extend path using given bit order. Returns True if extended."""
    cur = path[-1]
    for b in order:
        v = cur ^ (1 << b)
        if v in used:
            continue
        if _is_valid_extension(v, path):
            path.append(v)
            used.add(v)
            return True
    return False

def _construct_greedy(order):
    """Construct a snake using greedy extension with given bit order."""
    path = [0]
    used = {0}
    while True:
        if not _try_extend_greedy(path, used, order):
            break
    return path

def _construct_greedy_with_restarts(num_restarts=50, seed=42):
    """Try multiple random bit orders and return the best snake found."""
    rng = random.Random(seed)
    best_path = []
    for _ in range(num_restarts):
        order = list(range(N))
        rng.shuffle(order)
        path = _construct_greedy(order)
        if len(path) > len(best_path):
            best_path = path
    return best_path

def _try_backtrack_improve(path, max_depth=50000):
    """Try to improve the path using limited backtracking."""
    if not path:
        return path
    
    best_path = list(path)
    best_len = len(best_path)
    nodes_visited = 0
    
    def backtrack(path, used, depth):
        nonlocal best_path, best_len, nodes_visited
        if nodes_visited > max_depth:
            return
        nodes_visited += 1
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        cur = path[-1]
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if not _is_valid_extension(v, path):
                continue
            path.append(v)
            used.add(v)
            backtrack(path, used, depth + 1)
            path.pop()
            used.remove(v)
            if nodes_visited > max_depth:
                return
    
    used = set(path)
    backtrack(path, used, 0)
    return best_path

def _local_search_improve(path, iterations=10000):
    """Try to improve the path by local search: remove and re-insert vertices."""
    if not path:
        return path
    
    best_path = list(path)
    best_len = len(best_path)
    used = set(best_path)
    
    for _ in range(iterations):
        # Try to extend from the end
        cur = best_path[-1]
        extended = False
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            if _is_valid_extension(v, best_path):
                best_path.append(v)
                used.add(v)
                extended = True
                break
        if extended:
            continue
        
        # Try removing a middle vertex and re-extending
        if len(best_path) > 2:
            # Remove a random middle vertex
            idx = random.randint(1, len(best_path) - 2)
            removed = best_path[idx]
            best_path.pop(idx)
            used.remove(removed)
            
            # Try to extend from the new end
            cur = best_path[-1]
            for b in range(N):
                v = cur ^ (1 << b)
                if v in used:
                    continue
                if _is_valid_extension(v, best_path):
                    best_path.append(v)
                    used.add(v)
                    break
            
            # If we couldn't extend, try to put the removed vertex back
            if len(best_path) == len(best_path) - 1:  # didn't extend
                # Try inserting removed vertex back
                if not _is_valid_extension(removed, best_path):
                    # Can't put it back, revert
                    pass
                else:
                    # Find position to insert
                    pass
            
            # If path got shorter, revert
            if len(best_path) < best_len:
                # Revert: this is complex, just continue
                pass
            else:
                best_len = len(best_path)
    
    return best_path

def construct():
    """Construct a snake in Q_9 by combining greedy with multiple strategies."""
    # Strategy 1: Greedy with natural order
    path1 = _construct_greedy(list(range(N)))
    
    # Strategy 2: Greedy with reversed order
    path2 = _construct_greedy(list(range(N-1, -1, -1)))
    
    # Strategy 3: Multiple random restarts
    path3 = _construct_greedy_with_restarts(num_restarts=100, seed=42)
    
    # Strategy 4: Try different initial vertices
    best_path = max([path1, path2, path3], key=len)
    
    for start in range(1, 10):
        # Start from a different vertex
        path = [start]
        used = {start}
        order = list(range(N))
        while True:
            if not _try_extend_greedy(path, used, order):
                break
        if len(path) > len(best_path):
            best_path = path
    
    # Strategy 5: Try to improve the best path with limited backtracking
    if len(best_path) < 120:
        improved = _try_backtrack_improve(best_path, max_depth=200000)
        if len(improved) > len(best_path):
            best_path = improved
    
    return best_path

# EVOLVE-BLOCK-END
