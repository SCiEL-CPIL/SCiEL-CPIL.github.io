import numpy as np
import time
import sys
from itertools import combinations

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h||_inf * ||h*h||_inf)"""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    l_inf = np.max(h)
    if l1 < 1e-15 or l_inf < 1e-15:
        return 0.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return (l2_sq) / (l1 * l_inf * max_conv)

def _spike_ratio(indices):
    """
    For a set of spike indices (each with height 1), compute the ratio.
    ||h||_2^2 = k, ||h||_1 = k, ||h||_inf = 1
    So ratio = k / max_conv
    """
    if len(indices) == 0:
        return 0.0
    h = np.zeros(N)
    for idx in indices:
        h[idx] = 1.0
    h_conv = np.convolve(h, h, mode='full')
    max_conv = np.max(h_conv)
    if max_conv < 1e-15:
        return 0.0
    return len(indices) / max_conv

def _simulated_annealing(indices, deadline):
    """
    Simulated annealing on the set of spike positions.
    Moves: add a spike, remove a spike, swap a spike position.
    """
    rng = np.random.RandomState(42)
    current = list(indices)
    current_score = _spike_ratio(current)
    best = list(current)
    best_score = current_score
    
    T = 1.0
    T_min = 1e-6
    cooling = 0.995
    
    while time.time() < deadline and T > T_min:
        # Generate a neighbor
        move_type = rng.random()
        if move_type < 0.4:
            # Add a spike
            candidates = [i for i in range(N) if i not in current]
            if not candidates:
                T *= cooling
                continue
            new_idx = rng.choice(candidates)
            neighbor = current + [new_idx]
        elif move_type < 0.7:
            # Remove a spike
            if len(current) <= 2:
                T *= cooling
                continue
            remove_idx = rng.choice(len(current))
            neighbor = [x for j, x in enumerate(current) if j != remove_idx]
        else:
            # Swap: remove one, add another
            if not current:
                T *= cooling
                continue
            remove_idx = rng.choice(len(current))
            removed = current[remove_idx]
            candidates = [i for i in range(N) if i not in current]
            if not candidates:
                T *= cooling
                continue
            new_idx = rng.choice(candidates)
            neighbor = [x for j, x in enumerate(current) if j != remove_idx]
            neighbor.append(new_idx)
        
        neighbor_score = _spike_ratio(neighbor)
        delta = neighbor_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current = neighbor
            current_score = neighbor_score
            if current_score > best_score:
                best = list(current)
                best_score = current_score
        
        T *= cooling
    
    return best, best_score

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
    """
    current = list(indices)
    current_score = _spike_ratio(current)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            trial = [x for j, x in enumerate(current) if j != i]
            trial_score = _spike_ratio(trial)
            if trial_score > current_score:
                current = trial
                current_score = trial_score
                improved = True
                break
    
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial Sidon-like set
    initial = [0, 1, 4, 10, 18, 23, 30]
    
    best_indices = list(initial)
    best_score = _spike_ratio(best_indices)
    
    # Greedy add to expand the set
    if time.time() < deadline - 10:
        greedy_result, greedy_score = _greedy_add(initial, deadline - 10)
        if greedy_score > best_score:
            best_indices = greedy_result
            best_score = greedy_score
    
    # Simulated annealing to optimize
    if time.time() < deadline - 5:
        sa_result, sa_score = _simulated_annealing(best_indices, deadline - 5)
        if sa_score > best_score:
            best_indices = sa_result
            best_score = sa_score
    
    # Local search polish
    if time.time() < deadline - 2:
        ls_result, ls_score = _local_search(best_indices, deadline - 2)
        if ls_score > best_score:
            best_indices = ls_result
            best_score = ls_score
    
    # Build the final height vector
    h = np.zeros(N)
    for idx in best_indices:
        h[idx] = 1.0
    
    if np.max(h) < 1e-15:
        h[0] = 1.0
    
    return list(h)
