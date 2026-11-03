import numpy as np
from itertools import permutations
import time
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Classic construction
    classic_word = _classic_construct(n)
    best_word = classic_word
    best_len = len(classic_word)
    
    if n >= 4 and n <= 7:
        if n == 7:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)
        else:
            t = _tsp_construct(n, deadline)
            if t is not None:
                if len(t) < best_len:
                    best_word = t
                    best_len = len(t)

    return best_word

def _classic_construct(n):
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return "".join(map(str, word))

def _tsp_construct(n, deadline):
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    perm_strs = ["".join(map(str, p)) for p in perms]
    
    # 1. Build Overlap Matrix
    overlap = np.zeros((N, N), dtype=np.int8)
    
    for k in range(n - 1, 0, -1):
        suffix_groups = {}
        prefix_groups = {}
        
        for idx in range(N):
            s = perm_strs[idx][n-k:]
            p = perm_strs[idx][:k]
            
            if s not in suffix_groups:
                suffix_groups[s] = []
            suffix_groups[s].append(idx)
            
            if p not in prefix_groups:
                prefix_groups[p] = []
            prefix_groups[p].append(idx)
        
        for key in suffix_groups:
            if key in prefix_groups:
                is_list = suffix_groups[key]
                js_list = prefix_groups[key]
                for i in is_list:
                    for j in js_list:
                        if overlap[i, j] < k:
                            overlap[i, j] = k
                        
        if time.time() > deadline - 60:
            return None

    # 2. Greedy Path Construction
    cost = n - overlap.astype(np.float32)
    
    visited = [False] * N
    visited[0] = True
    path = [0]
    curr = 0
    
    for _ in range(N - 1):
        # Find next
        # Vectorized min
        unvisited_mask = ~np.array(visited)
        costs = cost[curr].copy()
        costs[~unvisited_mask] = np.inf
        next_idx = np.argmin(costs)
        
        visited[next_idx] = True
        path.append(next_idx)
        curr = next_idx
        
        if time.time() > deadline - 30:
            return None

    # 3. Simulated Annealing
    # Current Cost
    current_cost = 0
    for i in range(N - 1):
        current_cost += cost[path[i], path[i+1]]
        
    best_path = path[:]
    best_cost = current_cost
    
    T = 10.0  # Initial Temperature
    cooling_rate = 0.99
    
    # Run until deadline
    while time.time() < deadline - 5:
        if T < 0.01:
            break
            
        # Pick random i, j
        i = random.randint(1, N - 2)
        j = random.randint(i + 1, N - 1)
        
        # Calculate Delta Cost
        # Edges involved:
        # Old: (i-1, i), (j, j+1)
        # New: (i-1, j), (i, j+1)
        # Note: if j == N-1, edge (j, j+1) doesn't exist.
        
        old_cost = 0
        new_cost = 0
        
        # Edge (i-1, i)
        old_cost += cost[path[i-1], path[i]]
        # Edge (i-1, j)
        new_cost += cost[path[i-1], path[j]]
        
        # Edge (j, j+1) if exists
        if j < N - 1:
            old_cost += cost[path[j], path[j+1]]
            new_cost += cost[path[i], path[j+1]]
        else:
            # If j is last, we only change the first edge?
            # No, reversing path[i:j+1] changes:
            # ... a, p_i, ..., p_j ...
            # to ... a, p_j, ..., p_i ...
            # Edge a->p_i becomes a->p_j.
            # Edge p_j->next (if any) becomes p_i->next.
            # If j is last, there is no p_j->next.
            # So we only change a->p_i to a->p_j.
            # Wait, if j is last, the segment is p_i...p_j.
            # Reversing it makes p_j...p_i.
            # The node before i is a.
            # Old edge: a->p_i. New edge: a->p_j.
            # The node after j is none.
            # So yes, only one edge changes.
            pass
            
        delta = new_cost - old_cost
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            # Accept
            path[i:j+1] = path[i:j+1][::-1]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path[:]
                
        T *= cooling_rate
        
    # Reconstruct String
    s = perm_strs[best_path[0]]
    for k in range(1, N):
        p_prev = perm_strs[best_path[k-1]]
        p_curr = perm_strs[best_path[k]]
        # Overlap is precomputed, but we need to append the rest
        ov = overlap[best_path[k-1], best_path[k]]
        s += p_curr[ov:]
        
    # Verify
    windows = set()
    for i in range(len(s) - n + 1):
        w = tuple(map(int, s[i:i+n]))
        windows.add(w)
        
    if len(windows) == N:
        return s
    else:
        return None
