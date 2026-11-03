import time
import itertools
import sys
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 140.0 # Leave 10s margin
    
    if n <= 4:
        return _standard_construct(n)
    
    perms = list(itertools.permutations(range(1, n + 1)))
    N = len(perms)
    
    baseline_word = _standard_construct(n)
    baseline_len = len(baseline_word)
    
    # Precompute prefix map for fast overlap lookup
    # prefix_map[k][prefix_tuple] = list of perm indices
    prefix_map = [defaultdict(list) for _ in range(n)]
    for i, p in enumerate(perms):
        for k in range(1, n):
            prefix_map[k][p[:k]].append(i)
            
    # Nearest Neighbor Heuristic
    visited = [False] * N
    path = []
    current = 0
    visited[0] = True
    path.append(0)
    
    for step in range(N - 1):
        best_next = -1
        p_curr = perms[current]
        
        # Check overlaps from high to low
        for k in range(n - 1, 0, -1):
            suffix = p_curr[n-k:]
            candidates = prefix_map[k].get(suffix, [])
            for v in candidates:
                if not visited[v]:
                    best_next = v
                    break
            if best_next != -1:
                break
                
        if best_next == -1:
            # Fallback: find any unvisited
            for v in range(N):
                if not visited[v]:
                    best_next = v
                    break
                    
        if best_next == -1:
            break
            
        visited[best_next] = True
        path.append(best_next)
        current = best_next
        
        if time.time() - start_time > deadline - 10.0:
             break
             
    if len(path) < N:
        return baseline_word
        
    # Reconstruct word
    word = list(perms[path[0]])
    for i in range(1, N):
        prev_p = perms[path[i-1]]
        curr_p = perms[path[i]]
        # Find max overlap
        k = 0
        for k in range(n - 1, 0, -1):
            if prev_p[n-k:] == curr_p[:k]:
                break
        word.extend(curr_p[k:])
        
    if _is_valid_superpermutation(word, n):
        if len(word) < baseline_len:
            return "".join(map(str, word))
            
    return baseline_word

def _standard_construct(n):
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _is_valid_superpermutation(word, n):
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i+n])
        if len(set(window)) != n:
            continue
        seen.add(window)
    all_perms = set(itertools.permutations(range(1, n+1)))
    return all_perms.issubset(seen)
