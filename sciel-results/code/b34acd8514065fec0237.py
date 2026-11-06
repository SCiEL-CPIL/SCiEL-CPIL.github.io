N = 12
import time
from collections import deque

def _bfs_reset_length_bitmask(a, b):
    """
    Compute exact shortest reset word length using bitmask representation.
    a, b: lists of length N, a[i] and b[i] are destinations.
    Returns the reset length, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    
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
        
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        m = queue.popleft()
        d = visited[m]
        for apply in (apply_a, apply_b):
            new_m = apply(m)
            if new_m == 0:
                continue
            if new_m & (new_m - 1) == 0:
                return d + 1
            if new_m not in visited:
                visited[new_m] = d + 1
                queue.append(new_m)
    return -1

def construct():
    deadline = time.time() + 50
    
    best_a = None
    best_b = None
    best_len = 0
    
    # Try the standard Pinwheel variants
    # Variant 1: a=cycle, b=cycle with one break to 0
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = [(i + 1) % N for i in range(N)]
        b[N-1] = k
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
            
    # Variant 2: a=cycle, b=identity with one break
    for k in range(N):
        a = [(i + 1) % N for i in range(N)]
        b = list(range(N))
        b[k] = 0
        L = _bfs_reset_length_bitmask(a, b)
        if L > best_len:
            best_len = L
            best_a = a[:]
            best_b = b[:]
            
    # Variant 3: a=cycle, b=shift with clamp
    a = [(i + 1) % N for i in range(N)]
    b = [max(i-1, 0) for i in range(N)]
    L = _bfs_reset_length_bitmask(a, b)
    if L > best_len:
        best_len = L
        best_a = a[:]
        best_b = b[:]
        
    # If we found 121, return immediately
    if best_len == 121:
        return [best_a, best_b]
        
    # If not, do a quick local search from the best found
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_len = _bfs_reset_length_bitmask(best_a, best_b)
        
    cur_a = best_a[:]
    cur_b = best_b[:]
    cur_len = best_len
    
    # Simple Hill Climbing
    steps = 0
    while time.time() < deadline and steps < 10000:
        steps += 1
        # Mutate one transition
        if steps % 2 == 0:
            idx = steps % N
            old = cur_a[idx]
            cur_a[idx] = (old + 1) % N
        else:
            idx = steps % N
            old = cur_b[idx]
            cur_b[idx] = (old + 1) % N
            
        new_len = _bfs_reset_length_bitmask(cur_a, cur_b)
        
        if new_len > cur_len:
            cur_len = new_len
            if new_len > best_len:
                best_len = new_len
                best_a = cur_a[:]
                best_b = cur_b[:]
                if best_len == 121:
                    break
        else:
            # Revert
            if steps % 2 == 0:
                cur_a[idx] = old
            else:
                cur_b[idx] = old
                
    if best_len > 0:
        return [best_a, best_b]
    else:
        # Fallback
        return [[(i + 1) % N for i in range(N)], [0] * N]
