import itertools
from collections import deque

def construct(n):
    if n == 1:
        return "1"
    
    # LRU Greedy with Fallback to Max-Overlap
    
    # 1. Try LRU
    try:
        return _construct_lru(n)
    except Exception:
        pass
    
    # 2. Fallback to Max-Overlap (Current Program Logic)
    return _construct_max_overlap(n)

def _construct_lru(n):
    start_perm = tuple(range(1, n + 1))
    current_window = deque(start_perm[:n-1])
    visited = {start_perm}
    last_seen = {i: -1 for i in range(1, n + 1)}
    time = 0
    word = list(start_perm)
    
    fact = 1
    for i in range(2, n + 1):
        fact *= i
    total_perms = fact
    
    for _ in range(total_perms - 1):
        best_symbol = -1
        best_lru_time = -1
        
        for s in range(1, n + 1):
            new_window = tuple(list(current_window) + [s])
            if new_window not in visited:
                if last_seen[s] > best_lru_time:
                    best_lru_time = last_seen[s]
                    best_symbol = s
        
        if best_symbol == -1:
            raise Exception("LRU Stuck")
            
        word.append(best_symbol)
        visited.add(new_window)
        last_seen[best_symbol] = time
        time += 1
        current_window.append(best_symbol)
        current_window.popleft()
        
    return "".join(map(str, word))

def _construct_max_overlap(n):
    # Copy of current program logic
    perms = sorted(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    prefix_to_indices = {}
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix = p[:k]
            if prefix not in prefix_to_indices:
                prefix_to_indices[prefix] = []
            prefix_to_indices[prefix].append(i)
    
    visited = [False] * N
    visited[0] = True
    current_word = list(perms[0])
    
    for _ in range(N - 1):
        found = False
        for k in range(n - 1, 0, -1):
            suffix = tuple(current_word[-k:])
            candidates = prefix_to_indices.get(suffix, [])
            
            next_idx = -1
            for idx in candidates:
                if not visited[idx]:
                    next_idx = idx
                    break
            
            if next_idx != -1:
                visited[next_idx] = True
                next_perm = perms[next_idx]
                current_word.extend(next_perm[k:])
                found = True
                break
        
        if not found:
            for i in range(N):
                if not visited[i]:
                    visited[i] = True
                    next_perm = perms[i]
                    current_word.extend(next_perm)
                    found = True
                    break
                    
    return "".join(map(str, current_word))
