"""
Optimized search for the Černý problem (N=12).
Uses bitmask BFS for fast threshold calculation and Simulated Annealing for global search.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def compute_threshold(trans_a, trans_b):
    """
    Compute the length of the shortest reset word for the automaton defined by
    trans_a and trans_b.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    
    # Precompute subset transitions
    # next_a[mask] = resulting mask after applying 'a'
    next_a = [0] * size
    next_b = [0] * size
    
    # DP for subset transitions
    # mask is non-zero
    # lsb = mask & -mask
    # i = index of lsb
    # prev = mask ^ lsb
    # next[mask] = next[prev] | (1 << trans[i])
    
    for mask in range(1, size):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev_mask = mask ^ lb
        next_a[mask] = next_a[prev_mask] | (1 << trans_a[i])
        next_b[mask] = next_b[prev_mask] | (1 << trans_b[i])
        
    # BFS from full set
    start = FULL_MASK
    visited = bytearray(size)
    visited[start] = 1
    queue = deque([(start, 0)])
    
    while queue:
        subset, dist = queue.popleft()
        # Check if singleton
        if subset & (subset - 1) == 0:
            return dist
            
        na = next_a[subset]
        if not visited[na]:
            visited[na] = 1
            queue.append((na, dist + 1))
            
        nb = next_b[subset]
        if not visited[nb]:
            visited[nb] = 1
            queue.append((nb, dist + 1))
            
    return -1

def _generate_pinwheel():
    """
    Generate a pinwheel-like automaton.
    a is a cycle. b is a 'comb' that maps most states towards 0, but with a specific pattern.
    """
    a = [(i + 1) % N for i in range(N)]
    # b maps i to i-1, except 0 maps to 0?
    # This creates a "reverse cycle" with a sink.
    b = [0] * N
    for i in range(1, N):
        b[i] = i - 1
    # b[0] = 0
    return a, b

def _generate_random_automaton():
    a = [random.randint(0, N - 1) for _ in range(N)]
    b = [random.randint(0, N - 1) for _ in range(N)]
    return a, b

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # 1. Initialize with a good candidate
    best_a, best_b = _generate_pinwheel()
    best_len = compute_threshold(best_a, best_b)
    
    # Try a few other structured initializations
    candidates = []
    
    # Star
    a_star = [0] * N
    b_star = [(i + 1) % N for i in range(N)]
    candidates.append((a_star, b_star))
    
    # Reverse Star
    a_rev = [(i + 1) % N for i in range(N)]
    b_rev = [0] * N
    candidates.append((a_rev, b_rev))
    
    # Random
    for _ in range(10):
        candidates.append(_generate_random_automaton())
        
    for a, b in candidates:
        if time.time() > deadline - 5:
            break
        l = compute_threshold(a, b)
        if l > best_len:
            best_len = l
            best_a = a[:]
            best_b = b[:]
            
    # 2. Simulated Annealing
    if time.time() < deadline - 2:
        current_a = best_a[:]
        current_b = best_b[:]
        current_len = best_len
        
        T = 5.0
        cooling = 0.999
        min_T = 0.01
        
        while time.time() < deadline - 2:
            # Mutation
            letter = random.choice([0, 1])
            state = random.randint(0, N - 1)
            
            if letter == 0:
                old_val = current_a[state]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_a[state] = new_val
            else:
                old_val = current_b[state]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_b[state] = new_val
            
            new_len = compute_threshold(current_a, current_b)
            
            if new_len < 0:
                # Revert if not synchronizing
                if letter == 0:
                    current_a[state] = old_val
                else:
                    current_b[state] = old_val
                T *= cooling
                continue
                
            delta = new_len - current_len
            if delta > 0 or random.random() < pow(2.71828, delta / max(T, 1e-10)):
                current_len = new_len
                if current_len > best_len:
                    best_len = current_len
                    best_a = current_a[:]
                    best_b = current_b[:]
            else:
                # Revert
                if letter == 0:
                    current_a[state] = old_val
                else:
                    current_b[state] = old_val
                    
            T *= cooling
            if T < min_T:
                T = min_T
                
    return best_a, best_b

