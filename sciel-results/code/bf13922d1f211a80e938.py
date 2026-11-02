import numpy as np
import time
import sys
from itertools import combinations

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _spike_ratio(indices):
    """
    For a set of spike indices (each with height 1/k), compute the ratio.
    ||h||_2^2 = k * (1/k)^2 = 1/k
    ||h||_1 = 1
    sup|h*h| = max_conv
    Ratio = (1/k) / (1 * max_conv) = 1 / (k * max_conv)
    Note: max_conv for equal spikes is (count_overlap) / k^2.
    At lag 0, count=k, so conv = k/k^2 = 1/k.
    Ratio = (1/k) / (1 * 1/k) = 1.
    If any other lag has count > 1, conv > 1/k, Ratio < 1.
    """
    if len(indices) == 0:
        return 0.0
    k = len(indices)
    h = np.zeros(N)
    for idx in indices:
        h[idx] = 1.0 / k
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return (1.0 / k) / max_conv

def _greedy_add(indices, deadline):
    """
    Greedy: repeatedly add the spike that maximizes the ratio.
    """
    current = list(indices)
    current_score = _spike_ratio(current)
    
    while time.time() < deadline:
        best_add = None
        best_add_score = current_score
        for i in range(N):
            if i in current:
                continue
            trial = current + [i]
            trial_score = _spike_ratio(trial)
            if trial_score > best_add_score:
                best_add_score = trial_score
                best_add = i
        if best_add is None or best_add_score <= current_score:
            break
        current.append(best_add)
        current_score = best_add_score
    
    return current, current_score

def _local_search(indices, deadline):
    """
    Local search: try removing each spike and see if ratio improves.
    Also try swapping.
    """
    current = list(indices)
    current_score = _spike_ratio(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing
        for i in range(len(current)):
            trial = [x for j, x in enumerate(current) if j != i]
            trial_score = _spike_ratio(trial)
            if trial_score > current_score:
                current = trial
                current_score = trial_score
                improved = True
                break
        if improved:
            continue
            
        # Try swapping
        for i in range(len(current)):
            for j in range(N):
                if j in current:
                    continue
                trial = [x for x in current if x != current[i]]
                trial.append(j)
                trial_score = _spike_ratio(trial)
                if trial_score > current_score:
                    current = trial
                    current_score = trial_score
                    improved = True
                    break
            if improved:
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with a small set
    initial = [0]
    
    best_indices = list(initial)
    best_score = _spike_ratio(best_indices)
    
    # Greedy add to expand the set
    if time.time() < deadline - 10:
        greedy_result, greedy_score = _greedy_add(initial, deadline - 10)
        if greedy_score > best_score:
            best_indices = greedy_result
            best_score = greedy_score
    
    # Local search to polish
    if time.time() < deadline - 2:
        ls_result, ls_score = _local_search(best_indices, deadline - 2)
        if ls_score > best_score:
            best_indices = ls_result
            best_score = ls_score

    # Build the final height vector
    h = np.zeros(N)
    for idx in best_indices:
        h[idx] = 1.0 / len(best_indices)
    
    if np.max(h) < 1e-15:
        h[0] = 1.0
        
    return list(h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))
