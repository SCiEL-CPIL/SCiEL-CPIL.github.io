import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N

def _precompute_inputs():
    mat = np.zeros((N, NUM_INPUTS), dtype=np.uint8)
    for i in range(N):
        half = 1 << i
        period = 2 * half
        pattern = np.concatenate([np.zeros(half, dtype=np.uint8), np.ones(half, dtype=np.uint8)])
        mat[i] = np.tile(pattern, NUM_INPUTS // period)
    return mat

INPUTS = _precompute_inputs()

def _simulate(comparators):
    mat = INPUTS.copy()
    for (i, j) in comparators:
        mi = mat[i] & mat[j]
        mj = mat[i] | mat[j]
        mat[i] = mi
        mat[j] = mj
    return mat

def _is_valid(comparators):
    mat = _simulate(comparators)
    for i in range(N - 1):
        if np.any(mat[i] > mat[i + 1]):
            return False
    return True

def _greedy_remove(net, deadline, order='backward'):
    changed = True
    while changed and time.time() < deadline:
        changed = False
        if order == 'backward':
            indices = range(len(net) - 1, -1, -1)
        elif order == 'forward':
            indices = range(len(net))
        else:  # random
            indices = list(range(len(net)))
            random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                return net
            if idx >= len(net):
                continue
            trial = net[:idx] + net[idx+1:]
            if _is_valid(trial):
                net = trial
                changed = True
                break
    return net

def _swap_and_optimize(net, idx, deadline):
    """Swap adjacent comparators at idx, then try greedy removal."""
    if idx >= len(net) - 1:
        return net
    candidate = list(net)
    candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
    if not _is_valid(candidate):
        return net
    # Try to remove comparators from the swapped network
    candidate = _greedy_remove(candidate, deadline, order='backward')
    if len(candidate) < len(net):
        return candidate
    return net

def construct():
    start_time = time.time()
    # CRITICAL FIX: Set deadline to 50s to ensure we finish before 60s limit
    deadline = start_time + 50
    
    # Initial: odd-even transposition
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    if not _is_valid(net):
        return net
    
    # First pass: greedy removal from the back
    net = _greedy_remove(net, deadline, order='backward')
    best_net = list(net)
    best_len = len(net)
    
    # Multiple passes with different removal orders
    # Increased passes to exploit the corrected time budget
    max_passes = 20
    for pass_num in range(max_passes):
        if time.time() > deadline - 5:
            break
        order = ['forward', 'backward', 'random', 'backward', 'forward'][pass_num % 5]
        net = _greedy_remove(net, deadline, order=order)
        if len(net) < best_len:
            best_len = len(net)
            best_net = list(net)
    
    # Local search: Random Swaps
    # We have ~40 seconds left.
    # Estimate _is_valid cost: ~0.05-0.1s.
    # We can do ~400-800 checks.
    
    attempts = 0
    max_attempts = 800
    
    while attempts < max_attempts and time.time() < deadline - 2:
        if len(best_net) < 2:
            break
        
        # Randomly pick a position to swap
        idx = random.randint(0, len(best_net) - 2)
        candidate = list(best_net)
        candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
        
        # Check validity
        if _is_valid(candidate):
            # If valid, try to remove more
            candidate = _greedy_remove(candidate, deadline, order='backward')
            if len(candidate) < best_len:
                best_net = candidate
                best_len = len(candidate)
        
        attempts += 1
        
    return best_net
