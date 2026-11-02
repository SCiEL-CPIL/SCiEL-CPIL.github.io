import numpy as np
import random

def calculate_score_metric(support):
    """
    Calculate the metric V = k / M_count for uniform weights on the support.
    M_count is the maximum number of pairs (i,j) in support such that i+j == k (circular).
    Note: For i != j, there are 2 ordered pairs (i,j) and (j,i).
    The convolution sum is sum h_i h_j.
    If h_i = 1/k, term is 1/k^2.
    Max value M_val = M_count * (1/k^2).
    V = (1/k) / M_val = k / M_count.
    """
    if not support:
        return 0
    k = len(support)
    support_set = set(support)
    max_count = 0
    # Circular convolution sums
    for s in range(50):
        count = 0
        for i in support:
            j = (s - i) % 50
            if j in support_set:
                count += 1
        if count > max_count:
            max_count = count
    return k / max_count if max_count > 0 else 0

def find_max_sidon_set(n):
    # Keep existing function
    pass

def construct():
    best_score = 0
    best_support = []
    
    # 1. Try Sidon set
    sidon = find_max_sidon_set(50)
    score_sidon = calculate_score_metric(sidon)
    if score_sidon > best_score:
        best_score = score_sidon
        best_support = sidon
        
    # 2. Try larger sets with local search
    # We want to maximize k / max_count
    # Let's try k from 8 to 15
    for target_k in range(8, 16):
        # Random restarts
        for restart in range(50):
            # Random initial set
            support = random.sample(range(50), target_k)
            support_set = set(support)
            
            # Local search: swap elements
            improved = True
            while improved:
                improved = False
                current_score = calculate_score_metric(support)
                # Try swapping one element
                for i_idx in range(target_k):
                    old_val = support[i_idx]
                    for new_val in range(50):
                        if new_val in support_set:
                            continue
                        # Test swap
                        support[i_idx] = new_val
                        new_set = set(support)
                        new_score = calculate_score_metric(support)
                        if new_score > current_score:
                            current_score = new_score
                            support_set = new_set
                            improved = True
                            if current_score > best_score:
                                best_score = current_score
                                best_support = list(support)
                        else:
                            support[i_idx] = old_val
                            
    # Construct heights
    k = len(best_support)
    if k == 0:
        return [1.0] + [0.0]*49
    val = 1.0 / k
    h = [0.0] * 50
    for idx in best_support:
        h[idx] = val
    return h
