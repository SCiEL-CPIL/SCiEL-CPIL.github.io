import numpy as np
import itertools
import time
import math
import random

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def generate_weight_1_vectors():
    """Generate all weight-1 vectors in {-2, 0, 2}^D (norm^2 = 4)"""
    candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    return np.array(candidates, dtype=np.int64)

def greedy_mis_init(conflict_bits, N):
    """
    Greedy MIS using bitsets for fast neighbor lookup.
    Returns a bitmask (Python int) of the selected vertices.
    """
    degrees = [c.bit_count() for c in conflict_bits]
    order = list(range(N))
    order.sort(key=lambda x: degrees[x])
    
    selected_mask = 0
    for v in order:
        if (conflict_bits[v] & selected_mask) == 0:
            selected_mask |= (1 << v)
            
    return selected_mask

def simulated_annealing(conflict_bits, N, deadline, rng):
    """
    Simulated Annealing to maximize the size of the independent set.
    State: selected_mask (Python int)
    """
    curr_mask = greedy_mis_init(conflict_bits, N)
    curr_size = curr_mask.bit_count()
    
    best_mask = curr_mask
    best_size = curr_size
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    while time.time() < deadline:
        if rng.random() < 0.5:
            v = rng.integers(N)
            if (curr_mask >> v) & 1:
                continue
            if (conflict_bits[v] & curr_mask) == 0:
                curr_mask |= (1 << v)
                curr_size += 1
                if curr_size > best_size:
                    best_size = curr_size
                    best_mask = curr_mask
        else:
            v = rng.integers(N)
            if not ((curr_mask >> v) & 1):
                continue
            curr_mask &= ~(1 << v)
            curr_size -= 1
            if rng.random() < math.exp(-1.0 / T):
                pass
            else:
                curr_mask |= (1 << v)
                curr_size += 1
        
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_mask

def construct():
    t0 = time.time()
    deadline = t0 + 115
    
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    weight1_vectors = generate_weight_1_vectors()
    
    if N == 0:
        return weight1_vectors
        
    dots = vectors @ vectors.T
    
    conflict_bits = [0] * N
    for i in range(N):
        idxs = np.where((dots[i] > 2) & (np.arange(N) != i))[0]
        for j in idxs:
            conflict_bits[i] |= (1 << j)
            
    rng = np.random.default_rng(42)
    best_mask = simulated_annealing(conflict_bits, N, deadline - 5, rng)
    
    selected_indices = [i for i in range(N) if (best_mask >> i) & 1]
    
    result_vectors_w4 = vectors[selected_indices].astype(np.int64)
    
    final_vectors = np.vstack([result_vectors_w4, weight1_vectors])
    
    return final_vectors
