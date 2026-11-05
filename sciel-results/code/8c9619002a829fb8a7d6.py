"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses Simulated Annealing to escape local optima found by hill climbing.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _threshold_fast(a, b):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    Optimized for speed: uses bitmasks and pre-computed bit checks.
    """
    visited = set()
    queue = deque()
    
    visited.add(FULL_MASK)
    queue.append((FULL_MASK, 0))
    
    # Pre-compute bit masks for states for faster checking?
    # Actually, just iterating is fine.
    
    while queue:
        mask, depth = queue.popleft()
        
        # Check if singleton
        # A mask is a singleton if it has exactly one bit set
        # mask & (mask - 1) == 0
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        new_mask_a = 0
        # Unroll for N=12? No, keep generic but efficient.
        for i in range(N):
            if mask & (1 << i):
                new_mask_a |= 1 << a[i]
        
        if new_mask_a not in visited:
            visited.add(new_mask_a)
            queue.append((new_mask_a, depth + 1))
            
        # Apply 'b'
        new_mask_b = 0
        for i in range(N):
            if mask & (1 << i):
                new_mask_b |= 1 << b[i]
            
        if new_mask_b not in visited:
            visited.add(new_mask_b)
            queue.append((new_mask_b, depth + 1))
            
    return -1

def _mutate_automaton(a, b):
    """
    Create a mutated version of the automaton by changing 1 or 2 transitions.
    """
    a_new = a[:]
    b_new = b[:]
    
    # Mutate 1 or 2 positions in total
    num_mutations = random.randint(1, 2)
    
    for _ in range(num_mutations):
        if random.random() < 0.5:
            pos = random.randint(0, N - 1)
            a_new[pos] = random.randint(0, N - 1)
        else:
            pos = random.randint(0, N - 1)
            b_new[pos] = random.randint(0, N - 1)
            
    return a_new, b_new

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initialize with a known good structure (the one from the previous seed)
    # a: cycle, b: 0->0, i->(i+1)%N
    a_init = [(i + 1) % N for i in range(N)]
    b_init = [0] * N
    for i in range(1, N):
        b_init[i] = (i + 1) % N
        
    cur_a = a_init[:]
    cur_b = b_init[:]
    cur_score = _threshold_fast(cur_a, cur_b)
    
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    if best_score < 0:
        # Fallback if initial doesn't sync
        best_score = 0
        # Try a random one
        for _ in range(100):
            a_try = [random.randint(0, N-1) for _ in range(N)]
            b_try = [random.randint(0, N-1) for _ in range(N)]
            s = _threshold_fast(a_try, b_try)
            if s > best_score:
                best_score = s
                best_a = a_try
                best_b = b_try
                cur_a = a_try
                cur_b = b_try
                cur_score = s
            if best_score > 0:
                break

    # Simulated Annealing Parameters
    T = 10.0
    alpha = 0.999
    min_T = 0.01
    
    while time.time() < deadline:
        # Mutate
        new_a, new_b = _mutate_automaton(cur_a, cur_b)
        new_score = _threshold_fast(new_a, new_b)
        
        # Calculate energy difference
        # We want to MAXIMIZE score, so "energy" is -score
        # delta_E = E_new - E_old = (-new_score) - (-cur_score) = cur_score - new_score
        # If new_score > cur_score, delta_E < 0, always accept.
        # If new_score < cur_score, delta_E > 0, accept with prob exp(-delta_E/T)
        
        if new_score < 0:
            # Not synchronizing is very bad
            delta_E = cur_score + 100  # Large penalty
        else:
            delta_E = cur_score - new_score
        
        # Acceptance criterion
        if delta_E < 0:
            # Better solution
            accept = True
        else:
            # Worse or equal
            if T > 0:
                # Avoid overflow in exp
                if delta_E / T < 700: 
                    accept_prob = math.exp(-delta_E / T)
                else:
                    accept_prob = 0.0
                accept = random.random() < accept_prob
            else:
                accept = False
        
        if accept:
            cur_a = new_a
            cur_b = new_b
            cur_score = new_score
            
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
                
        # Cooling
        T *= alpha
        if T < min_T:
            # Reheat if stuck too long or just periodic reheat
            # Simple reheat
            T = 10.0
            
    return [best_a, best_b]

# EVOLVE-BLOCK-END
