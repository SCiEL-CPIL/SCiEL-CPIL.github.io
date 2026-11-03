import numpy as np
import time

N = 50
DELTA = 0.5 / N

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _construct_sidon_greedy(n, order=None):
    """
    Construct a maximal Sidon set in [0, n-1] using a greedy approach.
    To maximize the size, we try to add elements that leave the most room for others.
    A simple heuristic is to iterate through indices and add if possible.
    To improve, we can try multiple orderings or use a 'least restrictive' heuristic.
    For N=50, a simple greedy might suffice to get K~7.
    """
    if order is None:
        order = list(range(n))
    
    S = []
    used_sums = set()
    
    for i in order:
        # Check if i can be added
        valid = True
        # Check self-sum 2*i
        if (2 * i) in used_sums:
            valid = False
        else:
            # Check sums with existing elements
            for s in S:
                if (s + i) in used_sums:
                    valid = False
                    break
        
        if valid:
            S.append(i)
            # Update used_sums
            used_sums.add(2 * i)
            for s in S[:-1]:
                used_sums.add(s + i)
                
    return S

def construct():
    # Try a few different greedy orders to find a larger Sidon set
    # 1. Natural order
    # 2. Reverse order
    # 3. Prime-like order?
    
    best_set = []
    
    # Heuristic: Start with 0, then pick largest possible gaps?
    # Or just try natural order first.
    sets_to_try = [
        list(range(N)),
        list(range(N-1, -1, -1)),
        # Try to spread out: 0, N//2, N//4, ...
        [0, N//2, N//4, N//8, N//16] + [i for i in range(N) if i not in [0, N//2, N//4, N//8, N//16]]
    ]
    
    for order in sets_to_try:
        S = _construct_sidon_greedy(N, order)
        if len(S) > len(best_set):
            best_set = S
            
    # If greedy fails to find a large set, fall back to a known small Sidon set
    # For N=50, we expect K ~ 7.
    # If best_set is small, we might need a more robust search.
    # But for now, let's assume greedy works.
    
    K = len(best_set)
    if K == 0:
        # Fallback to uniform
        h = np.ones(N) / N
        return list(h)
        
    # Create uniform weights on the Sidon set
    h = np.zeros(N)
    for idx in best_set:
        h[idx] = 1.0 / K
        
    return list(h)
