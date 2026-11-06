import time
import random
from collections import deque

N = 12
NUM_MASKS = 1 << N
ALL_STATES = NUM_MASKS - 1

def compute_reset_length_fast(trans_a, trans_b):
    # Precompute transition tables for all possible subsets (masks)
    # next_a[m] is the mask resulting from applying 'a' to subset m
    next_a = [0] * NUM_MASKS
    next_b = [0] * NUM_MASKS
    
    # Optimization: Iterate through masks and compute transitions
    # For a mask m, next_mask = OR of (1 << trans[i]) for each i in m
    # We can do this efficiently by iterating set bits.
    for m in range(NUM_MASKS):
        ma = 0
        mb = 0
        # Iterate through set bits in m
        # Since N=12, we can just loop 0..11
        for i in range(N):
            if (m >> i) & 1:
                ma |= (1 << trans_a[i])
                mb |= (1 << trans_b[i])
        next_a[m] = ma
        next_b[m] = mb
        
    # BFS on the subset automaton using the precomputed tables
    start = ALL_STATES
    if start == 0: # Should not happen for N>0
        return 0
        
    visited = [False] * NUM_MASKS
    visited[start] = True
    queue = deque([start])
    dist = 0
    
    while queue:
        size = len(queue)
        for _ in range(size):
            m = queue.popleft()
            if m & (m - 1) == 0: # Check if m is a power of 2 (singleton)
                return dist
                
            # Try 'a'
            nm = next_a[m]
            if not visited[nm]:
                visited[nm] = True
                queue.append(nm)
                
            # Try 'b'
            nm = next_b[m]
            if not visited[nm]:
                visited[nm] = True
                queue.append(nm)
        dist += 1
        
    return -1 # Not synchronizing

def construct():
    deadline = time.time() + 85 # Safety margin (90s limit)
    
    # Initial candidate: A known good structure or random
    # Let's start with a random automaton that is likely synchronizing
    # Or use the seed: Cycle + Reset
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0] * N
    best_len = compute_reset_length_fast(best_a, best_b)
    
    # If seed is bad, try a few random starts immediately
    for _ in range(10):
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        l = compute_reset_length_fast(a, b)
        if l > best_len:
            best_len = l
            best_a = a
            best_b = b
            
    current_a = best_a[:]
    current_b = best_b[:]
    current_len = best_len
    
    # Local Search / Hill Climbing with Random Restarts
    # We will perform hill climbing for a short time, then restart if stuck
    while time.time() < deadline:
        # Hill climbing phase
        stuck_count = 0
        max_stuck = 500 # Adjust based on speed
        
        while time.time() < deadline and stuck_count < max_stuck:
            # Mutate one transition
            if random.random() < 0.5:
                i = random.randint(0, N-1)
                old_val = current_a[i]
                current_a[i] = random.randint(0, N-1)
                if current_a[i] == old_val:
                    continue
                new_len = compute_reset_length_fast(current_a, current_b)
                if new_len > current_len:
                    current_len = new_len
                    if new_len > best_len:
                        best_len = new_len
                        best_a = current_a[:]
                        best_b = current_b[:]
                    stuck_count = 0
                else:
                    # Revert if not improving (Greedy)
                    current_a[i] = old_val
                    stuck_count += 1
            else:
                i = random.randint(0, N-1)
                old_val = current_b[i]
                current_b[i] = random.randint(0, N-1)
                if current_b[i] == old_val:
                    continue
                new_len = compute_reset_length_fast(current_a, current_b)
                if new_len > current_len:
                    current_len = new_len
                    if new_len > best_len:
                        best_len = new_len
                        best_a = current_a[:]
                        best_b = current_b[:]
                    stuck_count = 0
                else:
                    # Revert if not improving
                    current_b[i] = old_val
                    stuck_count += 1
                    
        # If stuck, restart from best found so far or a new random point
        # Restarting from best helps refine, but might get stuck in local max.
        # Restarting from random explores new basins.
        if random.random() < 0.5:
            current_a = best_a[:]
            current_b = best_b[:]
            current_len = best_len
        else:
            # Random restart
            current_a = [random.randint(0, N-1) for _ in range(N)]
            current_b = [random.randint(0, N-1) for _ in range(N)]
            current_len = compute_reset_length_fast(current_a, current_b)
            if current_len > best_len:
                best_len = current_len
                best_a = current_a[:]
                best_b = current_b[:]

    return [best_a, best_b]
