import time
import random
from collections import deque
import numpy as np

N = 12
NUM_MASKS = 1 << N

def compute_threshold(trans_a, trans_b):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    Optimized with precomputed transition tables for all masks.
    """
    full_mask = (1 << N) - 1
    
    # Precompute transitions for all masks
    next_a = [0] * NUM_MASKS
    next_b = [0] * NUM_MASKS
    
    # Optimization: Iterate through masks and bits
    # This is still O(2^N * N) but done once per evaluation
    for m in range(NUM_MASKS):
        ma = 0
        mb = 0
        for i in range(N):
            if (m >> i) & 1:
                ma |= 1 << trans_a[i]
                mb |= 1 << trans_b[i]
        next_a[m] = ma
        next_b[m] = mb
        
    start = full_mask
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        # Letter 0
        next_mask = next_a[current]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Letter 1
        next_mask = next_b[current]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1

def construct():
    start_time = time.time()
    timeout = 85  # Use more of the 90s budget
    
    best_trans = None
    best_threshold = 0
    
    def cerny_automaton():
        a = [0] * N
        b = [0] * N
        for i in range(N):
            if i == 0:
                a[i] = 0
                b[i] = 1
            else:
                a[i] = i - 1
                b[i] = (i + 1) % N
        return a, b
    
    a_init, b_init = cerny_automaton()
    current_a = a_init[:]
    current_b = b_init[:]
    current_threshold = compute_threshold(current_a, current_b)
    
    if current_threshold > best_threshold:
        best_threshold = current_threshold
        best_trans = ([current_a[:], current_b[:]])

    def mutate(a, b):
        new_a = a[:]
        new_b = b[:]
        letter = random.randint(0, 1)
        state = random.randint(0, N-1)
        new_dest = random.randint(0, N-1)
        if letter == 0:
            new_a[state] = new_dest
        else:
            new_b[state] = new_dest
        return new_a, new_b

    T = 10.0
    cooling_rate = 0.995
    min_T = 0.5
    steps_since_improvement = 0
    max_steps_no_improve = 1000 # Increase patience due to faster eval?

    while time.time() - start_time < timeout:
        new_a, new_b = mutate(current_a, current_b)
        new_threshold = compute_threshold(new_a, new_b)
        
        if new_threshold > current_threshold:
            current_a = new_a
            current_b = new_b
            current_threshold = new_threshold
            steps_since_improvement = 0
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_trans = ([current_a[:], current_b[:]])
        else:
            delta = new_threshold - current_threshold
            if delta < 0:
                if T > 0.01:
                    prob = 2.718281828 ** (delta / T)
                else:
                    prob = 0
                if random.random() < prob:
                    current_a = new_a
                    current_b = new_b
                    current_threshold = new_threshold
                    if current_threshold > best_threshold:
                        best_threshold = current_threshold
                        best_trans = ([current_a[:], current_b[:]])
            elif delta == 0:
                if random.random() < 0.5:
                    current_a = new_a
                    current_b = new_b
                    current_threshold = new_threshold
            
            steps_since_improvement += 1
            T *= cooling_rate
            if T < min_T:
                T = min_T
        
        # Restart if stuck
        if steps_since_improvement > max_steps_no_improve:
            # Perturb the BEST solution found so far, not random
            if best_trans is not None:
                new_a, new_b = best_trans[0][:], best_trans[1][:]
                # Apply several random mutations
                for _ in range(5):
                    new_a, new_b = mutate(new_a, new_b)
            else:
                new_a, new_b = cerny_automaton()
            
            current_threshold = compute_threshold(new_a, new_b)
            current_a = new_a
            current_b = new_b
            steps_since_improvement = 0
            T = 10.0
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_trans = ([current_a[:], current_b[:]])

    if best_trans is None:
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
        
    return best_trans
