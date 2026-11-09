"""Snake-in-the-box in Q_9: Bidirectional growth with limited backtracking repair."""
import time
import random
import numpy as np

N = 9
SEED_TIME = 150.0  # evaluator wall-clock kill; keep margin
DEADLINE = SEED_TIME - 20.0  # 130s safety margin
T0 = time.time()

def _elapsed():
    return time.time() - T0

def _deadline_hit():
    return _elapsed() >= DEADLINE

# Precompute popcount for all 9-bit values
_POPCOUNT_TABLE = np.array([i.bit_count() for i in range(512)], dtype=np.int32)

def _popcount(arr):
    """Vectorized popcount for array of small integers (< 512)."""
    return _POPCOUNT_TABLE[arr.astype(np.int64)]

def _is_valid_extension(v, path_arr, used_set, end_index):
    """Check if v can be appended at position end_index (0 or 1)."""
    if v in used_set:
        return False
    if len(path_arr) == 0:
        return True
    
    diffs = np.bitwise_xor(path_arr, v)
    bc = _popcount(diffs)
    
    # The vertex it connects to (end vertex) has bc==1; all others must have bc>=2
    if end_index == 1:
        # Append: connecting vertex is path_arr[-1]
        if len(path_arr) > 1:
            others_bc = bc[:-1]
        else:
            others_bc = np.array([], dtype=np.int32)
    else:
        # Prepend: connecting vertex is path_arr[0]
        if len(path_arr) > 1:
            others_bc = bc[1:]
        else:
            others_bc = np.array([], dtype=np.int32)
    
    if len(others_bc) > 0 and np.any(others_bc == 1):
        return False
    return True

def _count_safe_options(v, path_arr, used_set):
    """Count how many neighbors of v are still safe to extend."""
    count = 0
    for b in range(N):
        u = v ^ (1 << b)
        if u in used_set:
            continue
        if len(path_arr) == 0:
            count += 1
            continue
        diffs = np.bitwise_xor(path_arr, u)
        bc = _popcount(diffs)
        if np.any(bc == 1):
            continue
        count += 1
    return count

def _get_candidates(cur, path_arr, used_set, end_index):
    """Get all valid neighbors for extending the end."""
    candidates = []
    for b in range(N):
        v = cur ^ (1 << b)
        if v in used_set:
            continue
        if not _is_valid_extension(v, path_arr, used_set, end_index):
            continue
        candidates.append(v)
    return candidates

def _extend_end_with_backtrack(path, used_set, end_index, max_depth=4, rng=None):
    """
    Attempt to extend the snake at the specified end using limited backtracking.
    Returns the new path if extended, otherwise the original path.
    """
    if rng is None:
        rng = random.Random()
        
    best_ext = 0
    best_path = path[:]
    
    # We treat this as a DFS on the extension
    def dfs(current_path, current_used, depth):
        nonlocal best_ext, best_path
        
        if _deadline_hit():
            return
        
        if len(current_path) > len(best_path):
            best_path = current_path[:]
            best_ext = len(current_path)
            
        if depth >= max_depth:
            return
            
        cur = current_path[0] if end_index == 0 else current_path[-1]
        candidates = _get_candidates(cur, np.array(current_path, dtype=np.int64), current_used, end_index)
        
        if not candidates:
            return
            
        # Score candidates with Warnsdorff
        scored = []
        for v in candidates:
            # For speed, we might skip complex scoring in deep backtracking
            # but let's use a simple heuristic: prefer vertices with fewer future options
            # Actually, in backtracking, we just want to find *any* valid extension.
            # Randomization helps escape local minima.
            scored.append((rng.random(), v))
        scored.sort(key=lambda x: x[0])
        
        for _, v in scored:
            if _deadline_hit():
                return
                
            new_path = current_path[:]
            new_used = current_used.copy()
            
            if end_index == 1:
                new_path.append(v)
            else:
                new_path.insert(0, v)
            new_used.add(v)
            
            dfs(new_path, new_used, depth + 1)
            
    dfs(path, used_set, 0)
    
    return best_path

def _bidirectional_grow_with_repair(start_vertex, rng, repair_depth=3):
    """
    Grow a snake bidirectionally. If an end gets stuck, attempt to repair it
    using limited backtracking.
    """
    path = [start_vertex]
    used = {start_vertex}
    path_arr = np.array(path, dtype=np.int64)
    
    grow_left = False
    
    while not _deadline_hit():
        # Decide which end to grow
        if grow_left:
            cur = path[0]
            end_index = 0
        else:
            cur = path[-1]
            end_index = 1
        
        # Find all valid neighbors
        candidates = _get_candidates(cur, path_arr, used, end_index)
        
        if not candidates:
            # Stuck. Try to repair this end with backtracking
            # We pass the current path and try to extend from this end
            # Note: _extend_end_with_backtrack modifies the path list
            repaired_path = _extend_end_with_backtrack(path, used, end_index, repair_depth, rng)
            
            if len(repaired_path) > len(path):
                # Update path and used set
                # The repair function returns a new list, so we need to rebuild used set
                # This is inefficient. Let's optimize the repair function to return the added vertices or update in place.
                # For now, accept the overhead for simplicity, but it's O(L) to rebuild set.
                path = repaired_path
                used = set(path)
                path_arr = np.array(path, dtype=np.int64)
            else:
                # Repair failed, try the other end
                grow_left = not grow_left
                continue
        
        # Warnsdorff: pick candidate with fewest safe options
        scored = []
        for v in candidates:
            opts = _count_safe_options(v, path_arr, used)
            scored.append((opts, rng.random(), v))
        scored.sort(key=lambda x: (x[0], x[1]))
        
        v = scored[0][2]
        
        if end_index == 1:
            path.append(v)
            path_arr = np.append(path_arr, v)
        else:
            path.insert(0, v)
            path_arr = np.insert(path_arr, 0, v)
        used.add(v)
        
        grow_left = not grow_left
    
    return path

def _validate(path):
    """Validate that path is a valid induced path in Q_N."""
    n = len(path)
    if n == 0:
        return False
    seen = set(path)
    if len(seen) != n:
        return False
    for i in range(n):
        if path[i] < 0 or path[i] >= (1 << N):
            return False
    for i in range(n - 1):
        d = (path[i] ^ path[i+1]).bit_count()
        if d != 1:
            return False
    for i in range(n):
        for j in range(i + 2, n):
            if (path[i] ^ path[j]).bit_count() == 1:
                return False
    return True

def construct():
    global T0
    T0 = time.time()
    
    best_path = None
    best_len = 0
    
    # 1. Baseline Greedy
    path_greedy, used_greedy = [0], {0}
    while not _deadline_hit():
        cur = path_greedy[-1]
        nxt = None
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used_greedy:
                continue
            ok = True
            for p in path_greedy[:-1]:
                if (v ^ p).bit_count() == 1:
                    ok = False
                    break
            if ok:
                nxt = v
                break
        if nxt is None:
            break
        path_greedy.append(nxt)
        used_greedy.add(nxt)
    
    if len(path_greedy) > best_len:
        best_len = len(path_greedy)
        best_path = path_greedy[:]
    
    rng = random.Random(42)
    
    # 2. Bidirectional Growth with Repair
    start_vertices = [0, (1 << N) - 1]
    for _ in range(10):
        start_vertices.append(rng.randint(0, (1 << N) - 1))
    
    attempts = 0
    for sv in start_vertices:
        if _deadline_hit():
            break
        for trial in range(10):
            if _deadline_hit():
                break
            path = _bidirectional_grow_with_repair(sv, rng, repair_depth=3)
            if len(path) > best_len:
                if _validate(path):
                    best_len = len(path)
                    best_path = path[:]
            attempts += 1
    
    # 3. Randomized Restarts
    while not _deadline_hit():
        sv = rng.randint(0, (1 << N) - 1)
        path = _bidirectional_grow_with_repair(sv, rng, repair_depth=2) # Slightly less depth for speed in many restarts
        if len(path) > best_len:
            if _validate(path):
                best_len = len(path)
                best_path = path[:]
        attempts += 1
        if attempts > 500:
            break

    if best_path is None:
        return [0]
    
    if not _validate(best_path):
        return path_greedy
    
    return best_path
