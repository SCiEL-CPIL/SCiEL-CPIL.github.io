import time
import random
from collections import deque
import numpy as np

N = 12

def compute_threshold(trans_a, trans_b):
    """
    Compute the shortest reset word length using BFS on the subset automaton.
    trans_a, trans_b: lists of length N, destination states in 0..N-1.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    full_mask = (1 << N) - 1
    start = full_mask
    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        if current.bit_count() == 1:
            return dist
            
        # Letter 0
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans_a[i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
        # Letter 1
        next_mask = 0
        for i in range(N):
            if (current >> i) & 1:
                next_mask |= 1 << trans_b[i]
        if next_mask not in visited:
            visited[next_mask] = dist + 1
            queue.append(next_mask)
            
    return -1

def construct():
    start_time = time.time()
    timeout = 55  # Safety margin before the 60s limit
    
    best_trans = None
    best_threshold = 0
    
    # Known high-performing construction: Pinwheel/Cerny variant
    # The classical Cerny automaton for N=12 has reset word length 77.
    # Let's build a Cerny-like automaton.
    
    def cerny_automaton():
        # Cerny automaton:
        # State 0 is a sink for letter a.
        # For i > 0, a(i) = i-1, b(i) = (i+1) % N
        # b(0) = 1
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
    
    # Initialize with Cerny automaton
    a_init, b_init = cerny_automaton()
    current_a = a_init[:]
    current_b = b_init[:]
    current_threshold = compute_threshold(current_a, current_b)
    
    if current_threshold > best_threshold:
        best_threshold = current_threshold
        best_trans = ([current_a[:], current_b[:]])
    
    def random_trans():
        return (
            [random.randint(0, N-1) for _ in range(N)],
            [random.randint(0, N-1) for _ in range(N)]
        )
        
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

    # Simulated Annealing Parameters
    T = 10.0
    cooling_rate = 0.995
    min_T = 0.5
    steps_since_improvement = 0
    max_steps_no_improve = 500
    
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
            new_a, new_b = random_trans()
            current_threshold = compute_threshold(new_a, new_b)
            current_a = new_a
            current_b = new_b
            steps_since_improvement = 0
            T = 10.0
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_trans = ([current_a[:], current_b[:]])

    if best_trans is None:
        # Fallback: known synchronizing automaton
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        return [a, b]
        
    return best_trans
