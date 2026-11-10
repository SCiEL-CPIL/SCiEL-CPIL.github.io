"""Improved superpermutation construction for n=7 (and smaller n)."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
from collections import deque
import numpy as np
import random

def construct(n):
    deadline = time.time() + 115  # safety margin before 120s limit
    t_start = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    if n < 7:
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
                best_overlap = 0
                max_possible = min(len(new), len(block))
                for o in range(max_possible, -1, -1):
                    if new[len(new) - o:] == block[:o]:
                        best_overlap = o
                        break
                new += block[best_overlap:]
            word = new
        return "".join(map(str, word))

    # For n=7
    # Step 1: Construct W_6
    word6 = [1]
    perms6 = []
    for m in range(2, 7):
        perms, seen = [], set()
        for i in range(len(word6) - m + 2):
            window = tuple(word6[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            best_overlap = 0
            max_possible = min(len(new), len(block))
            for o in range(max_possible, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word6 = new
        if m == 6:
            perms6 = perms # List of 720 perms in order

    P = [tuple(p) for p in perms6]
    N = len(P) # 720
    
    # Identify q_last
    q_last = tuple(word6[-6:])
    
    # Precompute L(q, p)
    L_matrix = np.zeros((N, N), dtype=np.int8)
    for i, q in enumerate(P):
        for j, p in enumerate(P):
            max_k = 0
            if q[-6:] == p[:6]: max_k = 6
            elif q[-5:] == p[:5]: max_k = 5
            elif q[-4:] == p[:4]: max_k = 4
            elif q[-3:] == p[:3]: max_k = 3
            elif q[-2:] == p[:2]: max_k = 2
            elif q[-1:] == p[:1]: max_k = 1
            else: max_k = 0
            L_matrix[i][j] = max_k
            
    C = 13 - L_matrix
    
    C_start = np.zeros(N, dtype=np.int8)
    for i, p in enumerate(P):
        max_k = 0
        if q_last[-6:] == p[:6]: max_k = 6
        elif q_last[-5:] == p[:5]: max_k = 5
        elif q_last[-4:] == p[:4]: max_k = 4
        elif q_last[-3:] == p[:3]: max_k = 3
        elif q_last[-2:] == p[:2]: max_k = 2
        elif q_last[-1:] == p[:1]: max_k = 1
        else: max_k = 0
        C_start[i] = 13 - max_k
        
    path = list(range(N))
    
    def calc_cost(path):
        cost = C_start[path[0]]
        for i in range(N - 1):
            cost += C[path[i]][path[i+1]]
        return cost
        
    current_cost = calc_cost(path)
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N - 1):
            if time.time() > deadline - 5:
                break
            a = path[i]
            b = path[i+1]
            
            delta = 0
            if i > 0:
                left = path[i-1]
                delta += (C[left][b] - C[left][a])
            else:
                delta += (C_start[b] - C_start[a])
                
            delta += (C[b][a] - C[a][b])
            
            if i < N - 2:
                right = path[i+2]
                delta += (C[a][right] - C[b][right])
                
            if delta < 0:
                path[i], path[i+1] = b, a
                current_cost += delta
                improved = True
        
        if not improved and time.time() < deadline - 10:
            for _ in range(100):
                i = random.randint(0, N - 2)
                a = path[i]
                b = path[i+1]
                
                delta = 0
                if i > 0:
                    left = path[i-1]
                    delta += (C[left][b] - C[left][a])
                else:
                    delta += (C_start[b] - C_start[a])
                    
                delta += (C[b][a] - C[a][b])
                
                if i < N - 2:
                    right = path[i+2]
                    delta += (C[a][right] - C[b][right])
                    
                if delta < 0:
                    path[i], path[i+1] = b, a
                    current_cost += delta
                    improved = True
                elif delta == 0:
                    if random.random() < 0.1:
                        path[i], path[i+1] = b, a
                        improved = True

    word = list(word6)
    for idx in path:
        p = P[idx]
        block = list(p) + [7] + list(p)
        best_overlap = 0
        max_possible = min(len(word), len(block))
        for o in range(max_possible, -1, -1):
            if word[len(word) - o:] == block[:o]:
                best_overlap = o
                break
        word += block[best_overlap:]
        
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
