import numpy as np
import time
import random
from itertools import combinations

N = 50

def _compute_ratio(h):
    """Compute R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    norm1 = np.sum(h)
    if norm1 == 0:
        return 0.0
    norm2_sq = np.dot(h, h)
    # Autoconvolution (circular or linear? Assume linear for step heights on [-1/4,1/4])
    # For discrete steps, autoconvolution is the circular convolution in frequency domain
    # But more likely: h*h means the autoconvolution as a sequence
    # Let's compute linear autoconvolution
    autoconv = np.convolve(h, h, mode='full')
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm == 0:
        return 0.0
    return norm2_sq / (norm1 * sup_norm)

def _max_multiplicity(S):
    """Compute max multiplicity of differences in set S"""
    if len(S) < 2:
        return 0
    counts = {}
    S_sorted = sorted(S)
    for i in range(len(S_sorted)):
        for j in range(i + 1, len(S_sorted)):
            d = S_sorted[j] - S_sorted[i]
            counts[d] = counts.get(d, 0) + 1
    return max(counts.values()) if counts else 0

def _greedy_sidon(max_iter=10000, seed=None):
    """Randomized greedy construction of a large Sidon set"""
    if seed is not None:
        random.seed(seed)
    
    S = set()
    # Start with a random element
    S.add(random.randint(0, N - 1))
    
    for _ in range(max_iter):
        # Try to add a random candidate
        candidates = list(set(range(N)) - S)
        if not candidates:
            break
        
        # Shuffle and try
        random.shuffle(candidates)
        added = False
        for c in candidates:
            new_S = S | {c}
            if _max_multiplicity(list(new_S)) == 1:
                S = new_S
                added = True
                break
        if not added:
            break
    
    return S

def _local_search(S, max_swaps=500, seed=None):
    """Try to improve Sidon set size by swaps"""
    if seed is not None:
        random.seed(seed)
    
    S = set(S)
    best_size = len(S)
    
    for _ in range(max_swaps):
        # Try removing one element and adding another
        if len(S) < 2:
            break
        
        to_remove = random.choice(list(S))
        S.remove(to_remove)
        
        # Try to add multiple elements
        added_any = False
        candidates = list(set(range(N)) - S)
        random.shuffle(candidates)
        
        for c in candidates:
            new_S = S | {c}
            if _max_multiplicity(list(new_S)) == 1:
                S = new_S
                added_any = True
                break
        
        # Also try adding the removed element back plus others
        if not added_any:
            # Maybe we should try a different strategy: remove two, add three?
            # For simplicity, just try single swaps
            S.add(to_remove)
        
        if len(S) > best_size:
            best_size = len(S)
    
    return S

def _simulated_annealing(S, iterations=5000, seed=None):
    """Simulated annealing to find larger Sidon sets"""
    if seed is not None:
        random.seed(seed)
    
    S = set(S)
    best_S = set(S)
    best_size = len(S)
    
    T = 1.0
    cooling = 0.999
    
    for i in range(iterations):
        if random.random() < 0.5:
            # Try adding
            candidates = list(set(range(N)) - S)
            if candidates:
                c = random.choice(candidates)
                new_S = S | {c}
                m = _max_multiplicity(list(new_S))
                if m == 1:
                    S = new_S
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                elif m <= 2:
                    # Accept with probability
                    if random.random() < np.exp(-1.0 / T):
                        S = new_S
        else:
            # Try removing
            if S:
                c = random.choice(list(S))
                S.discard(c)
        
        T *= cooling
    
    return best_S

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    best_S = set()
    best_size = 0
    best_ratio = 0.0
    
    # Phase 1: Multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline - 30:
        seed = attempts
        S = _greedy_sidon(max_iter=2000, seed=seed)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        attempts += 1
    
    # Phase 2: Local search from best
    if time.time() < deadline - 15:
        S = _local_search(best_S, max_swaps=200, seed=42)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        S = _simulated_annealing(best_S, iterations=3000, seed=123)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Create the vector
    h = [0.0] * N
    for idx in best_S:
        h[idx] = 1.0
    
    return h

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()
# EVOLVE-BLOCK-END

def _construct_impl():
    start_time = time.time()
    deadline = 145.0
    
    best_S = set()
    best_size = 0
    
    # Phase 1: Multiple randomized greedy attempts
    attempts = 0
    while time.time() < deadline - 30:
        seed = attempts
        S = _greedy_sidon(max_iter=2000, seed=seed)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        attempts += 1
    
    # Phase 2: Local search from best
    if time.time() < deadline - 15:
        S = _local_search(best_S, max_swaps=200, seed=42)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5:
        S = _simulated_annealing(best_S, iterations=3000, seed=123)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
    
    h = [0.0] * N
    for idx in best_S:
        h[idx] = 1.0
    
    return h
