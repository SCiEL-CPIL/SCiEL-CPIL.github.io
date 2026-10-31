import numpy as np
import random

N = 50

def is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_max_sidon(N, max_k=10):
    best_set = []
    for k in range(1, max_k + 1):
        # DFS to find a set of size k
        # Use recursion
        def dfs(current, start):
            if len(current) == k:
                return current[:]
            # Pruning: if we can't reach k, return None
            if len(current) + (N - start) < k:
                return None
            
            for i in range(start, N):
                new_set = current + [i]
                if is_sidon(new_set):
                    res = dfs(new_set, i + 1)
                    if res is not None:
                        return res
            return None
        
        found = dfs([], 0)
        if found:
            best_set = found
        else:
            break
            
    return best_set

def construct():
    # Find largest Sidon set
    S = find_max_sidon(N)
    if not S:
        S = [0]
        
    h = np.zeros(N)
    for i in S:
        h[i] = 1.0 / len(S)
        
    return list(h)

def compute_ratio(h):
    h = np.asarray(h)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_peak = np.max(conv)
    if max_peak == 0:
        return 0.0
    return l2_sq / (l1 * max_peak)
