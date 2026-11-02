import numpy as np

def find_greedy_sidon_set(n, target_size=10):
    """
    Finds a large Sidon set in [0, n) using a greedy heuristic.
    """
    S = []
    sums = set()
    
    # Try to build a set of size target_size
    # Greedy: pick the smallest available number that doesn't create a duplicate sum
    # To maximize size, we might want to spread them out?
    # Actually, for Sidon sets, the greedy "smallest first" often works well to find *a* set, 
    # but to find a *large* one, we might need to be smarter.
    # However, for N=50, a set of size ~7 is expected.
    
    # A better greedy: iterate through candidates, add if valid.
    # If we get stuck, backtrack? No, keep it simple and fast.
    
    # Let's try to construct a set of size K.
    # We can try to maximize K.
    
    best_set = []
    
    # Standard greedy:
    for i in range(n):
        valid = True
        new_sums = []
        for j in S:
            s = i + j
            if s in sums:
                valid = False
                break
            new_sums.append(s)
        if valid:
            s_ii = 2*i
            if s_ii in sums:
                valid = False
            else:
                new_sums.append(s_ii)
        
        if valid:
            S.append(i)
            sums.update(new_sums)
            
    return S

def construct():
    # Find a large Sidon set in 0..49
    sidon_set = find_greedy_sidon_set(50)
    h = [0.0] * 50
    k = len(sidon_set)
    if k == 0:
        return [1.0] + [0.0]*49
    val = 1.0 / k
    for idx in sidon_set:
        h[idx] = val
    return h
