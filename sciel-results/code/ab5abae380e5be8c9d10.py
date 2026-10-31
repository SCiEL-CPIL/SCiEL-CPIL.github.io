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

def greedy_sidon(N, max_restarts=500):
    best_set = []
    # Theoretical max size for Sidon set in [0, N) is approx sqrt(N)
    # For N=50, sqrt(50) ~ 7.07. So 7 or 8 is the target.
    for _ in range(max_restarts):
        indices = list(range(N))
        random.shuffle(indices)
        current_set = []
        for i in indices:
            new_set = current_set + [i]
            if is_sidon(new_set):
                current_set = new_set
        if len(current_set) > len(best_set):
            best_set = current_set
            # Early termination if we find a very large set
            if len(best_set) >= 8:
                break
    return best_set

def construct():
    # Find largest Sidon set using greedy heuristic with random restarts
    S = greedy_sidon(N)
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
