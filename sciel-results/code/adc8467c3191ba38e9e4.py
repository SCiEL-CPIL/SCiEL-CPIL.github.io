import time
import random
from collections import deque

N = 12
DEADLINE_MARGIN = 10.0  # seconds

def compute_threshold(trans_a, trans_b):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton
    using bitmasks for efficiency.
    Returns the length, or -1 if not synchronizing.
    """
    start_state = (1 << N) - 1
    visited = {start_state: 0}
    queue = deque([(start_state, 0)])
    
    while queue:
        state, dist = queue.popleft()
        if state & (state - 1) == 0: 
            return dist
        
        # Apply letter 0
        next_state_0 = 0
        temp = state
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            next_state_0 |= (1 << trans_a[s])
            temp ^= lsb
            
        if next_state_0 not in visited:
            visited[next_state_0] = dist + 1
            queue.append((next_state_0, dist + 1))
            
        # Apply letter 1
        next_state_1 = 0
        temp = state
        while temp:
            lsb = temp & -temp
            s = lsb.bit_length() - 1
            next_state_1 |= (1 << trans_b[s])
            temp ^= lsb
            
        if next_state_1 not in visited:
            visited[next_state_1] = dist + 1
            queue.append((next_state_1, dist + 1))
            
    return -1

def construct():
    """
    Returns the transition tables for an optimal 12-state synchronizing automaton.
    Uses optimized BFS and targeted local search.
    """
    start_time = time.time()
    
    # Initial construction: Černý automaton (Cycle + Decrement)
    trans_a = [(i + 1) % N for i in range(N)]
    trans_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    best_threshold = compute_threshold(trans_a, trans_b)
    best_a = trans_a[:]
    best_b = trans_b[:]
    
    if best_threshold >= 121:
        return [best_a, best_b]
        
    random.seed(42)
    
    # Local Search restricted to modifying trans_b, keeping trans_a as a cycle
    # This is a strong heuristic for Černý-type automata.
    current_a = trans_a[:]
    current_b = trans_b[:]
    current_threshold = best_threshold
    
    max_iterations = 1000
    
    for iter_num in range(max_iterations):
        if time.time() - start_time > DEADLINE_MARGIN:
            break
            
        improved = False
        # Try changing each state's destination in trans_b
        for state in range(N):
            old_dest = current_b[state]
            for new_dest in range(N):
                if new_dest == old_dest:
                    continue
                current_b[state] = new_dest
                new_threshold = compute_threshold(current_a, current_b)
                
                if new_threshold > current_threshold:
                    current_threshold = new_threshold
                    improved = True
                    if current_threshold > best_threshold:
                        best_threshold = current_threshold
                        best_a = current_a[:]
                        best_b = current_b[:]
                        if best_threshold >= 121:
                            return [best_a, best_b]
                else:
                    current_b[state] = old_dest
                    
        if not improved:
            # Perturbation: Randomly change a few b transitions
            for _ in range(3):
                state = random.randint(0, N - 1)
                new_dest = random.randint(0, N - 1)
                current_b[state] = new_dest
            current_threshold = compute_threshold(current_a, current_b)
            if current_threshold > best_threshold:
                best_threshold = current_threshold
                best_a = current_a[:]
                best_b = current_b[:]
                if best_threshold >= 121:
                    return [best_a, best_b]
            
            # If stuck, try a different cycle? (Expensive, skip for now)
            # Or restart with random b
            if iter_num % 50 == 0:
                current_b = [random.randint(0, N-1) for _ in range(N)]
                current_threshold = compute_threshold(current_a, current_b)
                if current_threshold > best_threshold:
                    best_threshold = current_threshold
                    best_a = current_a[:]
                    best_b = current_b[:]
                    if best_threshold >= 121:
                        return [best_a, best_b]

    return [best_a, best_b]
