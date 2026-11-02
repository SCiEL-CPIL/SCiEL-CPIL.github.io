import time
import sys
import math
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    if n == 1:
        return "1"
    
    # Generate all permutations
    perms = list(permutations(range(1, n + 1)))
    N = len(perms)
    
    # Precompute overlap matrix M using efficient prefix/suffix maps
    M = [[0] * N for _ in range(N)]
    
    # Build prefix maps: for each k, map prefix tuple -> list of indices
    prefix_maps = [{} for _ in range(n)]
    
    for k in range(1, n):
        p_map = prefix_maps[k]
        for idx, p in enumerate(perms):
            pre = p[:k]
            if pre not in p_map:
                p_map[pre] = []
            p_map[pre].append(idx)
            
    # Fill M: for each i, find max k such that perms[i][n-k:] == perms[j][:k]
    for i in range(N):
        for k in range(n - 1, 0, -1):
            suf = perms[i][n-k:]
            if suf in prefix_maps[k]:
                for j in prefix_maps[k][suf]:
                    if M[i][j] == 0:
                        M[i][j] = k
    
    # Greedy Nearest Neighbor initialization
    visited = [False] * N
    path = [0]
    visited[0] = True
    
    for _ in range(N - 1):
        last = path[-1]
        best_j = -1
        best_m = -1
        for j in range(N):
            if not visited[j]:
                m = M[last][j]
                if m > best_m:
                    best_m = m
                    best_j = j
        if best_j == -1:
            break
        path.append(best_j)
        visited[best_j] = True
    
    # Calculate current overlap sum
    def get_overlap_sum(p):
        s = 0
        for t in range(len(p) - 1):
            s += M[p[t]][p[t+1]]
        return s
    
    current_sum = get_overlap_sum(path)
    
    # Or-opt local search: move one node to another position
    improved = True
    iteration = 0
    max_iterations = 50
    
    while improved and time.time() < deadline - 10:
        improved = False
        iteration += 1
        if iteration > max_iterations:
            break
            
        for k in range(1, N - 1):  # Only move internal nodes
            u = path[k-1]
            v = path[k]
            w = path[k+1]
            
            # Gain from removing v
            gain_remove = M[u][w] - M[u][v] - M[v][w]
            
            # Try inserting v at all other positions
            for p in range(0, N - 1):
                if p == k - 1 or p == k:
                    continue
                
                # Determine neighbors at insertion position p
                if p < k:
                    a = path[p-1] if p > 0 else -1
                    b = path[p]
                else:
                    # p > k: adjust indices
                    if p == N - 1:
                        a = path[N-1]
                        b = -1
                    else:
                        b = path[p+1]
                        if p - 1 < k:
                            a = path[p-1]
                        else:
                            a = path[p]
                
                # Calculate insertion gain
                if a == -1 and b == -1:
                    gain_insert = 0
                elif a == -1:
                    gain_insert = M[v][b]
                elif b == -1:
                    gain_insert = M[a][v]
                else:
                    gain_insert = M[a][v] + M[v][b] - M[a][b]
                    
                total_gain = gain_remove + gain_insert
                
                if total_gain > 0:
                    # Perform the move
                    path.pop(k)
                    if p > k:
                        p -= 1
                    path.insert(p, v)
                    current_sum += total_gain
                    improved = True
                    break
            
            if improved:
                break
    
    # 2-opt local search: swap two adjacent pairs
    improved = True
    iteration = 0
    max_iterations = 30
    
    while improved and time.time() < deadline - 5:
        improved = False
        iteration += 1
        if iteration > max_iterations:
            break
        
        for i in range(0, N - 2):
            if time.time() > deadline - 5:
                break
            a = path[i]
            b = path[i+1]
            c = path[i+2]
            
            # Try reversing the segment from i to i+2
            # Old: ... a, b, c ...
            # New: ... c, b, a ...
            
            prev_a = path[i-1] if i > 0 else -1
            next_c = path[i+3] if i+3 < N else -1
            
            old_cost = 0
            if prev_a != -1:
                old_cost += M[prev_a][a]
            old_cost += M[a][b] + M[b][c]
            if next_c != -1:
                old_cost += M[c][next_c]
            
            new_cost = 0
            if prev_a != -1:
                new_cost += M[prev_a][c]
            new_cost += M[c][b] + M[b][a]
            if next_c != -1:
                new_cost += M[a][next_c]
            
            if new_cost < old_cost:
                path[i:i+3] = [c, b, a]
                current_sum += (old_cost - new_cost)
                improved = True
    
    # Construct the word
    word = list(perms[path[0]])
    for t in range(1, len(path)):
        i = path[t-1]
        j = path[t]
        ov = M[i][j]
        word += perms[j][ov:]
    
    # Verify
    if len(word) < n:
        return "1"
        
    seen = set()
    for i in range(len(word) - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            seen.add(w)
            
    if len(seen) == N:
        return "".join(map(str, word))
    else:
        # Fallback: concatenate all permutations
        result = []
        for p in perms:
            result.extend(p)
        return "".join(map(str, result))
