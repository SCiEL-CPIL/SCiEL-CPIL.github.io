import random
import time
import math
from collections import deque

N = 12

def compute_reset_length(aut):
    """
    Compute the exact shortest reset word length using BFS on subsets.
    Returns 0 if not synchronizing.
    """
    start = (1 << N) - 1
    
    if N == 1:
        return 0

    visited = {start: 0}
    queue = deque([start])
    
    while queue:
        current_subset = queue.popleft()
        dist = visited[current_subset]
        
        if current_subset & (current_subset - 1) == 0:
            return dist

        for l in range(2):
            new_subset = 0
            for i in range(N):
                if current_subset & (1 << i):
                    dest = aut[l][i]
                    new_subset |= (1 << dest)
            
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
                
    return 0

def construct():
    """
    Use Simulated Annealing to find a synchronizing automaton with a long reset word.
    """
    random.seed(42)
    
    # Initialize with a random automaton, ensuring it's not trivially non-synchronizing
    # Or start with a known good structure and perturb it.
    # Let's start with a random automaton.
    current_aut = [[random.randint(0, N-1) for _ in range(N)] for _ in range(2)]
    
    current_len = compute_reset_length(current_aut)
    if current_len == 0:
        # If not synchronizing, try the cycle-reset structure as a fallback start
        current_aut = [
            [(i + 1) % N for i in range(N)],
            [0] + [i - 1 for i in range(1, N)]
        ]
        current_len = compute_reset_length(current_aut)
        
    best_aut = [list(row) for row in current_aut]
    best_len = current_len
    
    start_time = time.time()
    timeout = 55
    
    T = 10.0 # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < timeout:
        # Pick a random transition to modify
        l = random.randint(0, 1)
        i = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        
        if current_aut[l][i] == new_dest:
            continue
            
        # Create a copy
        new_aut = [list(row) for row in current_aut]
        new_aut[l][i] = new_dest
        
        new_len = compute_reset_length(new_aut)
        
        if new_len > current_len:
            current_aut = new_aut
            current_len = new_len
            if current_len > best_len:
                best_len = current_len
                best_aut = [list(row) for row in current_aut]
        else:
            # Accept worse solution with probability exp(-delta/T)
            delta = new_len - current_len # Negative
            if delta < 0:
                prob = math.exp(delta / T)
                if random.random() < prob:
                    current_aut = new_aut
                    current_len = new_len
        
        T *= cooling_rate
        if T < 0.01:
            T = 0.01
            
    return best_aut
