"""
Snake-in-the-Box in Q9 using Iterated Local Search with Truncation,
Reordered Greedy, and Insert Moves.
"""
import time
import random
import numpy as np

N = 9
TOTAL_VERTS = 1 << N  # 512

# Precompute Hamming-1 neighbor info: for each vertex, the list of its 9 neighbors.
_NEIGHBORS = [[v ^ (1 << b) for b in range(N)] for v in range(TOTAL_VERTS)]

def _is_valid_extend(v, path):
    """
    v is adjacent to path[-1]. Check that v is NOT adjacent to any path[i]
    for i in range(len(path) - 1). (i.e., i < len(path) - 1)
    """
    if v in path:
        return False  # will be caught by caller, but safe
    L = len(path)
    for i in range(L - 1):
        if (v ^ path[i]).bit_count() == 1:
            return False
    return True

def _greedy_extend(path, rng, used_set):
    """
    Extend path from the end using randomized greedy.
    """
    while True:
        cur = path[-1]
        bits = list(range(N))
        rng.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            if _is_valid_extend(v, path):
                path.append(v)
                used_set.add(v)
                found = True
                break
        if not found:
            break
    return path, used_set

def _greedy_prepend(path, rng, used_set):
    """
    Extend path from the beginning using randomized greedy.
    v must be adjacent to path[0] and NOT adjacent to path[i] for i >= 1.
    """
    while True:
        cur = path[0]
        bits = list(range(N))
        rng.shuffle(bits)
        found = False
        for b in bits:
            v = cur ^ (1 << b)
            if v in used_set:
                continue
            # v adjacent to cur. Check v against path[1:]
            ok = True
            for j in range(1, len(path)):
                if (v ^ path[j]).bit_count() == 1:
                    ok = False
                    break
            if ok:
                path.insert(0, v)
                used_set.add(v)
                found = True
                break
        if not found:
            break
    return path, used_set

def _try_extend_ends(path, used_set, rng):
    """
    Try to extend from both ends greedily (one step each).
    Returns (path, used_set, extended_any)
    """
    extended = False
    
    # Try extend from the end
    cur = path[-1]
    bits = list(range(N))
    rng.shuffle(bits)
    for b in bits:
        v = cur ^ (1 << b)
        if v in used_set:
            continue
        if _is_valid_extend(v, path):
            path.append(v)
            used_set.add(v)
            extended = True
            break
    
    # Try extend from the beginning
    cur = path[0]
    bits = list(range(N))
    rng.shuffle(bits)
    for b in bits:
        v = cur ^ (1 << b)
        if v in used_set:
            continue
        ok = True
        for j in range(1, len(path)):
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if ok:
            path.insert(0, v)
            used_set.add(v)
            extended = True
            break
    
    return path, used_set, extended

def _try_insert(path, used_set, v):
    """
    Try to insert vertex v at some position i (0 <= i <= len(path)-1).
    If i == len(path)-1, it's an extension at the end.
    If i == 0, prepend.
    Otherwise, between path[i] and path[i+1].
    
    For insertion between path[i] and path[i+1]:
    - v must be adjacent to path[i] AND path[i+1]
    - v must NOT be adjacent to any path[j] for j != i, i+1
    
    For extension at end (i = len(path)-1):
    - v adjacent to path[-1]
    - v not adjacent to path[0:-1]
    
    For prepend (i = 0):
    - v adjacent to path[0]
    - v not adjacent to path[1:]
    """
    L = len(path)
    
    # Case: extend at end
    if (v ^ path[-1]).bit_count() == 1:
        # Check v not adjacent to path[0:L-1]
        ok = True
        for j in range(L - 1):
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if ok:
            return path + [v]
    
    # Case: prepend
    if (v ^ path[0]).bit_count() == 1:
        ok = True
        for j in range(1, L):
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if ok:
            return [v] + path
    
    # Case: insert between i and i+1
    for i in range(L - 1):
        a, b = path[i], path[i+1]
        if (v ^ a).bit_count() != 1 or (v ^ b).bit_count() != 1:
            continue
        ok = True
        for j in range(L):
            if j == i or j == i + 1:
                continue
            if (v ^ path[j]).bit_count() == 1:
                ok = False
                break
        if ok:
            return path[:i+1] + [v] + path[i+1:]
    
    return None

def _collect_insert_candidates(path, used_set):
    """
    Collect candidate vertices for insertion.
    A vertex v can be inserted between a and b (consecutive in path)
    if v is adjacent to both a and b, and v is not in used_set.
    
    Also include vertices that can extend ends (for completeness).
    """
    candidates = set()
    
    # End extensions
    for cur in [path[0], path[-1]]:
        for nb in _NEIGHBORS[cur]:
            if nb not in used_set:
                candidates.add(nb)
    
    # Internal insertions: between each consecutive pair (a, b)
    # The two common neighbors of a and b (other than a, b themselves) are candidates.
    # a and b differ in exactly 1 bit (say bit k).
    # A vertex v adjacent to both a and b must satisfy:
    #   v = a ^ (1<<k) ^ (1<<l) = b ^ (1<<l) for l != k
    # There are N-1 such vertices.
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        diff = a ^ b
        k = diff.bit_length() - 1  # the differing bit
        for l in range(N):
            if l == k:
                continue
            v = a ^ (1 << k) ^ (1 << l)  # = b ^ (1 << l)
            if v not in used_set:
                candidates.add(v)
    
    return candidates

def _make_initial_greedy(start, rng):
    """
    Make an initial snake starting at 'start' using randomized greedy.
    """
    path = [start]
    used = {start}
    path, used = _greedy_extend(path, rng, used)
    return path, used

def _multi_start_greedy(rng, n_starts=30):
    """
    Do multiple random greedy starts to find a good initial solution.
    """
    best = None
    best_len = 0
    for _ in range(n_starts):
        start = rng.randint(0, TOTAL_VERTS - 1)
        path, used = _make_initial_greedy(start, rng)
        # Also try to extend from the beginning
        path, used = _greedy_prepend(path, rng, used)
        if len(path) > best_len:
            best_len = len(path)
            best = path
    return best

def construct():
    """
    Construct the longest induced path in Q_9.
    """
    deadline = time.time() + 110.0  # Safety margin
    rng = random.Random(12345)
    
    # --- Phase 1: Multi-start greedy to find a good initial solution ---
    best_path = _multi_start_greedy(rng, n_starts=40)
    if best_path is None:
        best_path = [0]
    best_used = set(best_path)
    best_len = len(best_path)
    
    # Try to extend the best path further
    current_path = list(best_path)
    current_used = set(best_used)
    current_path, current_used = _greedy_extend(current_path, rng, current_used)
    current_path, current_used = _greedy_prepend(current_path, rng, current_used)
    if len(current_path) > best_len:
        best_path = list(current_path)
        best_used = set(current_used)
        best_len = len(current_path)
    
    # --- Phase 2: Iterated Local Search ---
    current_path = list(best_path)
    current_used = set(best_used)
    
    stagnation = 0
    max_stagnation = 20
    
    while time.time() < deadline - 3.0:
        made_progress = False
        
        # 2a: Try to extend ends
        current_path, current_used, ext = _try_extend_ends(current_path, current_used, rng)
        if ext:
            # Keep extending
            while True:
                current_path, current_used, ext2 = _try_extend_ends(current_path, current_used, rng)
                if not ext2:
                    break
            if len(current_path) > best_len:
                best_path = list(current_path)
                best_used = set(current_used)
                best_len = len(current_path)
            stagnation = 0
            continue
        
        # 2b: Try insert moves
        candidates = _collect_insert_candidates(current_path, current_used)
        # Randomize order
        cand_list = list(candidates)
        rng.shuffle(cand_list)
        
        inserted = False
        for v in cand_list:
            if time.time() > deadline - 3.0:
                break
            new_path = _try_insert(current_path, current_used, v)
            if new_path is not None:
                current_path = new_path
                current_used = set(current_path)
                inserted = True
                if len(current_path) > best_len:
                    best_path = list(current_path)
                    best_used = set(current_used)
                    best_len = len(current_path)
                break  # Restart loop after one insertion
        
        if inserted:
            stagnation = 0
            continue
        
        # 2c: Perturbation - truncate and re-extend
        stagnation += 1
        
        # Random truncation: remove k vertices from a random position
        trunc_len = rng.randint(2, min(8, max(1, len(current_path) // 6)))
        
        # Decide where to truncate: from end, from start, or middle
        mode = rng.randint(0, 2)
        
        if mode == 0:
            # Truncate from the end
            if len(current_path) > trunc_len + 3:
                new_len = len(current_path) - trunc_len
                current_path = current_path[:new_len]
                current_used = set(current_path)
        elif mode == 1:
            # Truncate from the start
            if len(current_path) > trunc_len + 3:
                new_len = len(current_path) - trunc_len
                current_path = current_path[new_len:]
                current_used = set(current_path)
        else:
            # Truncate a middle segment and reconnect ends if possible
            # Simpler: just truncate from the end for now
            if len(current_path) > trunc_len + 3:
                new_len = len(current_path) - trunc_len
                current_path = current_path[:new_len]
                current_used = set(current_path)
        
        # Re-extend greedily from both ends
        current_path, current_used = _greedy_extend(current_path, rng, current_used)
        current_path, current_used = _greedy_prepend(current_path, rng, current_used)
        
        if len(current_path) > best_len:
            best_path = list(current_path)
            best_used = set(current_used)
            best_len = len(current_path)
            stagnation = 0
        
        # If stuck for too long, restart from best with a new perturbation
        if stagnation > max_stagnation:
            current_path = list(best_path)
            current_used = set(best_used)
            stagnation = 0
    
    return best_path

# EVOLVE-BLOCK-START
# The construct function is the entry point.
# EVOLVE-BLOCK-END
