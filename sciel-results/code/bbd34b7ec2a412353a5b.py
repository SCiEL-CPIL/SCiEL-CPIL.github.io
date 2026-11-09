"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # cut = sum of A[i,j] where labels[i] != labels[j]
    # We can compute this as: for each vertex i, count neighbors in opposite set
    # Total cut size = sum over i of (neighbors in opposite set) / 2? No.
    # Actually, each edge (i,j) with different labels is counted once.
    # Let's use: cut = sum(A * (labels[:,None] != labels[None,:])) // 2
    # But that's O(n^2). For 100 nodes, it's 10000 ops, very fast.
    # However, for SA we need faster gain computation.
    # For now, let's use the vectorized approach for final evaluation.
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _local_search(adjacency, labels, time_limit=2.0):
    """Greedy local improvement: flip vertices that increase cut size."""
    n = len(adjacency)
    A = adjacency.astype(np.int32)
    labels = labels.copy()
    
    start_time = time.time()
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for v in range(n):
            # Compute gain if we flip vertex v
            # Current: edges from v to opposite set contribute to cut
            # After flip: edges from v to same set (now opposite) contribute to cut
            # Let same_set = neighbors with same label, opp_set = neighbors with opp label
            # Current contribution of v to cut = sum(A[v, opp_set])
            # New contribution of v to cut = sum(A[v, same_set])
            # Gain = sum(A[v, same_set]) - sum(A[v, opp_set])
            
            # Vectorized computation
            same_mask = (labels == labels[v])
            opp_mask = (labels != labels[v])
            
            same_count = A[v, same_mask].sum()
            opp_count = A[v, opp_mask].sum()
            
            gain = same_count - opp_count
            if gain > 0:
                labels[v] = 1 - labels[v]
                improved = True
    
    return labels


def _simulated_annealing(adjacency, labels, time_limit=10.0):
    """Multi-start Simulated Annealing with adaptive cooling."""
    n = len(adjacency)
    A = adjacency.astype(np.int32)
    labels = labels.copy()
    
    start_time = time.time()
    
    # Initial temperature
    T = 5.0
    # Cooling rate
    cooling_rate = 0.995
    # Minimum temperature
    T_min = 0.1
    
    # Compute initial cut size
    diff = (labels[:, None] != labels[None, :])
    current_cut = int((adjacency & diff).sum() // 2)
    
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Count non-improving moves for adaptive cooling
    non_improving_count = 0
    max_non_improving = 1000
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Check time budget
        if (time.time() - start_time) > time_limit - 0.1:
            break
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute gain
        same_mask = (labels == labels[v])
        opp_mask = (labels != labels[v])
        
        same_count = A[v, same_mask].sum()
        opp_count = A[v, opp_mask].sum()
        
        gain = same_count - opp_count
        
        if gain >= 0:
            # Accept improving or neutral move
            labels[v] = 1 - labels[v]
            current_cut += gain
            non_improving_count = 0
        else:
            # Metropolis criterion
            # P(accept) = exp(gain / T)
            prob = np.exp(gain / T)
            if np.random.random() < prob:
                labels[v] = 1 - labels[v]
                current_cut += gain
                non_improving_count += 1
            else:
                non_improving_count += 1
        
        # Track best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Adaptive cooling
        if non_improving_count >= max_non_improving:
            T *= 0.9  # Cool faster if stuck
            non_improving_count = 0
        else:
            T *= cooling_rate
    
    return best_labels


def partition(adjacency):
    n = len(adjacency)
    adjacency = np.array(adjacency, dtype=bool)
    
    # Deadline guard: total time limit is 120s, use 100s for safety
    total_start = time.time()
    time_budget = 100.0
    
    # Multi-start: try several different initializations
    best_labels = None
    best_cut = -1
    
    # Start 1: Alternating
    labels1 = np.arange(n) % 2
    labels1 = _local_search(adjacency, labels1, time_limit=5.0)
    cut1 = _cut_size(adjacency, labels1)
    if cut1 > best_cut:
        best_cut = cut1
        best_labels = labels1.copy()
    
    # Check time
    if (time.time() - total_start) > time_budget * 0.3:
        return best_labels
    
    # Start 2: Random
    labels2 = np.random.randint(0, 2, n)
    labels2 = _local_search(adjacency, labels2, time_limit=5.0)
    cut2 = _cut_size(adjacency, labels2)
    if cut2 > best_cut:
        best_cut = cut2
        best_labels = labels2.copy()
    
    # Check time
    if (time.time() - total_start) > time_budget * 0.5:
        return best_labels
    
    # Start 3: SA from random
    labels3 = np.random.randint(0, 2, n)
    sa_time = time_budget * 0.3
    labels3 = _simulated_annealing(adjacency, labels3, time_limit=sa_time)
    labels3 = _local_search(adjacency, labels3, time_limit=5.0)
    cut3 = _cut_size(adjacency, labels3)
    if cut3 > best_cut:
        best_cut = cut3
        best_labels = labels3.copy()
    
    # Check time
    if (time.time() - total_start) > time_budget * 0.8:
        return best_labels
    
    # Start 4: SA from alternating
    labels4 = np.arange(n) % 2
    remaining_time = time_budget - (time.time() - total_start) - 2.0
    if remaining_time > 5.0:
        labels4 = _simulated_annealing(adjacency, labels4, time_limit=remaining_time)
        labels4 = _local_search(adjacency, labels4, time_limit=2.0)
        cut4 = _cut_size(adjacency, labels4)
        if cut4 > best_cut:
            best_cut = cut4
            best_labels = labels4.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
