import time
import random
from collections import deque

N = 12

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmasks for efficiency.
    """
    start_mask = (1 << N) - 1
    if start_mask == 0:
        return 0
    if start_mask & (start_mask - 1) == 0:
        return 0

    visited = {start_mask: 0}
    q = deque([start_mask])
    
    while q:
        curr_mask = q.popleft()
        curr_dist = visited[curr_mask]
        
        # Check if singleton
        if curr_mask & (curr_mask - 1) == 0:
            return curr_dist

        # Apply 'a'
        next_mask_a = 0
        for i in range(N):
            if curr_mask & (1 << i):
                next_mask_a |= (1 << a_trans[i])
        if next_mask_a not in visited:
            visited[next_mask_a] = curr_dist + 1
            q.append(next_mask_a)
            
        # Apply 'b'
        next_mask_b = 0
        for i in range(N):
            if curr_mask & (1 << i):
                next_mask_b |= (1 << b_trans[i])
        if next_mask_b not in visited:
            visited[next_mask_b] = curr_dist + 1
            q.append(next_mask_b)
            
    return -1  # Not synchronizing

def construct():
    """
    Constructs a binary automaton with 12 states maximizing the reset threshold.
    Uses local search to find a high-threshold automaton.
    """
    best_a = None
    best_b = None
    best_score = -1
    
    deadline = time.time() + 80  # 10s safety margin
    
    # Initialize with a few heuristics
    # Heuristic 1: Random
    # Heuristic 2: Cycle + Sink variant
    # Heuristic 3: Random permutations
    
    def random_automaton():
        a = [random.randint(0, N-1) for _ in range(N)]
        b = [random.randint(0, N-1) for _ in range(N)]
        return a, b

    def cycle_sink_automaton():
        a = [(i + 1) % N for i in range(N)]
        b = [0] * N
        for i in range(1, N):
            b[i] = i - 1
        return a, b

    # Generate initial candidates
    candidates = []
    for _ in range(10):
        candidates.append(random_automaton())
    candidates.append(cycle_sink_automaton())
    
    # Evaluate initial candidates
    for a, b in candidates:
        score = calculate_threshold(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
            
    if time.time() > deadline:
        return [best_a, best_b]

    # Local Search
    # Simulated Annealing or Hill Climbing
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    temperature = 10.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random transition to change
        letter = random.choice([0, 1])
        state = random.randint(0, N-1)
        
        old_val = current_a[state] if letter == 0 else current_b[state]
        new_val = random.randint(0, N-1)
        
        if letter == 0:
            current_a[state] = new_val
        else:
            current_b[state] = new_val
            
        new_score = calculate_threshold(current_a, current_b)
        
        if new_score == -1:
            new_score = -100 # Penalty for non-synchronizing
            
        if new_score > current_score or (new_score < current_score and random.random() < pow(2.718, (new_score - current_score) / temperature)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_a = current_a[:]
                best_b = current_b[:]
        else:
            # Revert
            if letter == 0:
                current_a[state] = old_val
            else:
                current_b[state] = old_val
                
        temperature *= cooling
        if temperature < 0.1:
            temperature = 10.0
            # Restart from best
            current_a = best_a[:]
            current_b = best_b[:]
            current_score = best_score

    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
