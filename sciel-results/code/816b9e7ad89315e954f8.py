import random
import time
from collections import deque

N = 12

def compute_reset_length(aut):
    """
    Compute the exact shortest reset word length using BFS on subsets.
    Returns 0 if not synchronizing.
    """
    # Subsets are represented as bitmasks
    # Start with all states: (1 << N) - 1
    start = (1 << N) - 1
    
    # If start is a singleton (N=1), length 0
    if N == 1:
        return 0

    # visited[subset] = distance
    visited = {start: 0}
    queue = deque([start])
    
    # Precompute transitions for each letter on each subset?
    # Or compute on the fly.
    # On the fly is fine for N=12.
    
    while queue:
        current_subset = queue.popleft()
        dist = visited[current_subset]
        
        # If this is a singleton, we found the reset word
        # Check if it's a singleton: (x & (x-1)) == 0
        if current_subset & (current_subset - 1) == 0:
            return dist

        # Try both letters
        for l in range(2):
            # Compute image of current_subset under letter l
            new_subset = 0
            for i in range(N):
                if current_subset & (1 << i):
                    dest = aut[l][i]
                    new_subset |= (1 << dest)
            
            if new_subset not in visited:
                visited[new_subset] = dist + 1
                queue.append(new_subset)
                
    return 0 # Not synchronizing

def construct():
    """
    Use Local Search to find a synchronizing automaton with a long reset word.
    Initializes with the standard Černý automaton.
    """
    # Initialize with Černý automaton
    letter_a = [(i + 1) % N for i in range(N)]
    letter_b = [0] + [i - 1 for i in range(1, N)]
    current_aut = [letter_a, letter_b]
    
    current_len = compute_reset_length(current_aut)
    best_aut = current_aut
    best_len = current_len
    
    start_time = time.time()
    timeout = 55 # Leave 5s for overhead
    
    random.seed(42) # For reproducibility in case of ties? Or just random.
    
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
                
    return best_aut
