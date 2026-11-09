N = 12
import time
from collections import deque

def _compute_threshold_bitmask(sigma0, sigma1):
    """Compute exact reset threshold via BFS on subset automaton using bitmasks."""
    full = (1 << N) - 1
    start = full
    # Precompute transitions for all subsets is expensive; do on-the-fly
    dist = {start: 0}
    q = deque([start])
    
    def apply_letter(mask, letter):
        if letter == 0:
            return 0
        result = 0
        for i in range(N):
            if mask & (1 << i):
                if letter == 0:
                    result |= 1 << sigma0[i]
                else:
                    result |= 1 << sigma1[i]
        return result
    
    while q:
        mask = q.popleft()
        d = dist[mask]
        for letter in (0, 1):
            new_mask = apply_letter(mask, letter)
            if new_mask not in dist:
                dist[new_mask] = d + 1
                if new_mask & (new_mask - 1) == 0:  # singleton
                    return d + 1
                q.append(new_mask)
    return -1  # not synchronizing

def _pawlik_automaton():
    """Standard Pawlik automaton with threshold (N-1)^2."""
    sigma0 = [(i + 1) % N for i in range(N)]
    sigma1 = [0] * N
    for i in range(1, N):
        sigma1[i] = i - 1
    return [sigma0, sigma1]

def _local_search(initial, max_iters=500, deadline=None):
    """Local search: mutate one transition at a time, accept if threshold improves."""
    best_sigma = [list(initial[0]), list(initial[1])]
    best_thresh = _compute_threshold_bitmask(best_sigma[0], best_sigma[1])
    if best_thresh < 0:
        best_thresh = 0
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        # Pick a random position to mutate
        letter = it % 2
        state = (it // 2) % N
        old_val = best_sigma[letter][state]
        new_val = (old_val + 1) % N
        if new_val == old_val:
            new_val = (old_val + 2) % N
        
        best_sigma[letter][state] = new_val
        thresh = _compute_threshold_bitmask(best_sigma[0], best_sigma[1])
        
        if thresh > best_thresh:
            best_thresh = thresh
        else:
            best_sigma[letter][state] = old_val
    
    return [list(best_sigma[0]), list(best_sigma[1])], best_thresh

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    
    # Start with Pawlik automaton
    best_sigma, best_thresh = _local_search(_pawlik_automaton(), 
                                              max_iters=300, 
                                              deadline=deadline)
    
    return [list(best_sigma[0]), list(best_sigma[1])]
