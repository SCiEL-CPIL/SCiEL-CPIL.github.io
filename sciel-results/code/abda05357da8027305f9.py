import time
import random
from collections import deque

def construct():
    N = 12
    deadline = time.time() + 85.0
    
    # Initial Černý automaton
    # a: i -> (i+1)%N
    # b: 0 -> 0, i -> i-1
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] * N
    for i in range(1, N):
        b_map[i] = i - 1
        
    best_a = a_map[:]
    best_b = b_map[:]
    
    def compute_threshold(a, b):
        # BFS on subset automaton
        # Start from all states: {0, 1, ..., N-1}
        start = (1 << N) - 1
        if start == 0:
            return 0
            
        visited = set([start])
        queue = deque([(start, 0)])
        
        while queue:
            state, dist = queue.popleft()
            if state == 1: # Single state {0}
                return dist
                
            # Transition 'a'
            next_a = 0
            for i in range(N):
                if (state >> i) & 1:
                    next_a |= (1 << a[i])
            if next_a not in visited:
                if next_a == 1:
                    return dist + 1
                visited.add(next_a)
                queue.append((next_a, dist + 1))
                
            # Transition 'b'
            next_b = 0
            for i in range(N):
                if (state >> i) & 1:
                    next_b |= (1 << b[i])
            if next_b not in visited:
                if next_b == 1:
                    return dist + 1
                visited.add(next_b)
                queue.append((next_b, dist + 1))
                
        return -1 # Not synchronizing

    current_threshold = compute_threshold(best_a, best_b)
    
    # Local Search
    while time.time() < deadline:
        # Pick random mutation
        state_idx = random.randint(0, N - 1)
        symbol = random.randint(0, 1) # 0 for a, 1 for b
        
        if symbol == 0:
            old_dest = best_a[state_idx]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            best_a[state_idx] = new_dest
            
            new_threshold = compute_threshold(best_a, best_b)
            
            if new_threshold > current_threshold:
                current_threshold = new_threshold
            else:
                best_a[state_idx] = old_dest
        else:
            old_dest = best_b[state_idx]
            new_dest = random.randint(0, N - 1)
            if new_dest == old_dest:
                continue
            best_b[state_idx] = new_dest
            
            new_threshold = compute_threshold(best_a, best_b)
            
            if new_threshold > current_threshold:
                current_threshold = new_threshold
            else:
                best_b[state_idx] = old_dest
                
    return [best_a, best_b]
