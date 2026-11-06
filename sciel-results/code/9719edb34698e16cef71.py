"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
import math
from collections import deque

def _bfs_reset_length(trans):
    """Compute exact shortest reset word length via BFS on subset automaton."""
    start = tuple(range(N))
    if len(set(start)) == 1:
        return 0
    visited = {start: 0}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        d = visited[s]
        for letter in range(2):
            new_set = tuple(sorted(set(trans[letter][st] for st in s)))
            if len(new_set) == 1:
                return d + 1
            if new_set not in visited:
                visited[new_set] = d + 1
                queue.append(new_set)
    return -1

def _bfs_reset_length_bitmask(a, b):
    """
    Compute exact shortest reset word length using bitmask representation.
    a, b: lists of length N, a[i] and b[i] are destinations.
    Returns the reset length, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    # Precompute subset transitions for all 2^N subsets
    # trans[letter][mask] = new_mask
    # But 2^12 = 4096, so we can precompute on the fly or use a dict cache
    
    # For speed, we'll compute transitions lazily with caching
    cache_a = {}
    cache_b = {}
    
    def apply_a(mask):
        if mask in cache_a:
            return cache_a[mask]
        result = 0
        for i in range(N):
            if mask & (1 << i):
                result |= (1 << a[i])
        cache_a[mask] = result
        return result
    
    def apply_b(mask):
        if mask in cache_b:
            return cache_b[mask]
        result = 0
        for i in range(N):
            if mask & (1 << i):
                result |= (1 << b[i])
        cache_b[mask] = result
        return result
    
    # BFS on bitmask states
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        m = queue.popleft()
        d = visited[m]
        for apply in (apply_a, apply_b):
            new_m = apply(m)
            if new_m == 0:
                continue
            # Check if singleton
            if new_m & (new_m - 1) == 0:
                return d + 1
            if new_m not in visited:
                visited[new_m] = d + 1
                queue.append(new_m)
    
    return -1

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Phase 1: Quick structural families to get a good starting point
    # Family: a = cycle, b = i+1 except last maps to k
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[N-1] = k
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
    
    # Family: a = cycle, b = i+1 except one position maps to 0
    for pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[pos] = 0
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
    
    # Family: a = cycle, b = identity except one maps to 0
    for pos in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = list(range(N))
        b[pos] = 0
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
    
    # Family: a = cycle, b = i -> max(i-1, 0)
    a = [(i + 1) % N for i in range(N)]
    b = [max(i-1, 0) for i in range(N)]
    L = _bfs_reset_length_bitmask(a, b)
    if L > best_len:
        best_len = L
        best_a = a[:]
        best_b = b[:]
    
    # Family: a = cycle, b = i -> (i*2) % N
    a = [(i + 1) % N for i in range(N)]
    b = [(i * 2) % N for i in range(N)]
    L = _bfs_reset_length_bitmask(a, b)
    if L > best_len:
        best_len = L
        best_a = a[:]
        best_b = b[:]
    
    # Phase 2: Simulated Annealing
    if best_a is None:
        # Fallback initial
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = _bfs_reset_length_bitmask(best_a, best_b)
    
    # Ensure we have a synchronizing automaton to start
    if best_len <= 0:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = _bfs_reset_length_bitmask(best_a, best_b)
    
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_len = best_len
    
    # Simulated annealing parameters
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.9995
    T = T_start
    step = 0
    max_steps = 50000
    
    while time.time() < deadline and step < max_steps:
        step += 1
        T *= cooling_rate
        
        # Generate a mutation
        # Mutate one transition in a or b
        if random.random() < 0.5:
            # Mutate a
            idx = random.randint(0, N-1)
            old_val = cur_a[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            cur_a[idx] = new_val
        else:
            # Mutate b
            idx = random.randint(0, N-1)
            old_val = cur_b[idx]
            new_val = random.randint(0, N-1)
            if new_val == old_val:
                continue
            cur_b[idx] = new_val
        
        # Evaluate
        new_len = _bfs_reset_length_bitmask(cur_a, cur_b)
        
        if new_len > cur_len:
            # Better, always accept
            cur_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            # Worse or equal, accept with probability exp(-delta/T)
            delta = cur_len - new_len
            if delta > 0 and random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_len = new_len
            else:
                # Reject
                if random.random() < 0.5:
                    cur_a[idx] = old_val
                else:
                    cur_b[idx] = old_val
                cur_len = best_len if cur_len <= 0 else cur_len
    
    # Return the best found
    if best_len > 0:
        return [best_a, best_b]
    else:
        # Fallback
        return [[(i + 1) % N for i in range(N)], [0] * N]
