import time
import random
from collections import deque

N = 12

def _get_reset_length(auto):
    """
    Computes the exact shortest reset word length for a given automaton.
    Returns 0 if the automaton is not synchronizing.
    """
    # BFS on the subset automaton
    # State: frozenset of states
    start = frozenset(range(N))
    if len(start) == 1:
        return 0 # Already synchronized? No, start is all states.
    
    # Queue stores (current_subset, length)
    q = deque()
    q.append((start, 0))
    visited = {start}
    
    # Precompute transitions for speed?
    # auto is [ [t00, t01, ...], [t10, t11, ...] ]
    # auto[letter][state]
    
    while q:
        subset, length = q.popleft()
        
        for letter in range(2):
            # Compute next subset
            next_subset = frozenset(auto[letter][s] for s in subset)
            
            if len(next_subset) == 1:
                return length + 1
            
            if next_subset not in visited:
                visited.add(next_subset)
                q.append((next_subset, length + 1))
                
    return 0 # Not synchronizing

def _random_auto():
    return [[random.randint(0, N-1) for _ in range(N)] for _ in range(2)]

def _mutate(auto):
    a = [row[:] for row in auto]
    letter = random.randint(0, 1)
    state = random.randint(0, N-1)
    a[letter][state] = random.randint(0, N-1)
    return a

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin for 60s limit
    
    best_auto = None
    best_len = 0
    
    # Initialize with some structured candidates
    # 1. Cycle + 0 sink (Length 1)
    seed1 = [[(i + 1) % N for i in range(N)], [0] * N]
    # 2. Cycle + 1 sink (Length N-1 = 11)
    b_map = [(i + 1) % N for i in range(N)]
    b_map[0] = 0
    seed2 = [[(i + 1) % N for i in range(N)], b_map]
    # 3. Random
    seed3 = _random_auto()
    
    candidates = [seed1, seed2, seed3]
    for c in candidates:
        l = _get_reset_length(c)
        if l > best_len:
            best_len = l
            best_auto = c
    
    # Local Search
    current = best_auto
    current_len = best_len
    no_improve_count = 0
    
    # To help escape local minima, we can occasionally jump
    while time.time() < deadline:
        neighbor = _mutate(current)
        n_len = _get_reset_length(neighbor)
        
        if n_len > current_len:
            current = neighbor
            current_len = n_len
            no_improve_count = 0
            if current_len > best_len:
                best_len = current_len
                best_auto = current
        else:
            no_improve_count += 1
            if no_improve_count > 2000:
                # Restart from best found so far
                current = best_auto
                current_len = best_len
                no_improve_count = 0
                # Optionally, randomize current a bit to explore new area
                # But sticking to best is safer to not lose progress
                # Let's try a random jump from best
                if random.random() < 0.5:
                    current = _mutate(best_auto)
                    current_len = _get_reset_length(current)
                    if current_len == 0:
                        current = best_auto
                        current_len = best_len
    
    return best_auto
