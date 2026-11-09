"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_length = 0
    best_a = [(i + 1) % N for i in range(N)]
    best_b = [0 if i == 0 else i - 1 for i in range(N)]
    
    def evaluate(a_map, b_map):
        # BFS on subset automaton to find shortest reset word length
        # States are represented as frozensets of ints
        start_state = frozenset(range(N))
        if len(start_state) == 1:
            return 0
            
        visited = {start_state: 0}
        queue = deque([start_state])
        
        while queue:
            current = queue.popleft()
            dist = visited[current]
            
            # Try 'a'
            next_a = frozenset(a_map[i] for i in current)
            if next_a not in visited:
                if len(next_a) == 1:
                    return dist + 1
                visited[next_a] = dist + 1
                queue.append(next_a)
                
            # Try 'b'
            next_b = frozenset(b_map[i] for i in current)
            if next_b not in visited:
                if len(next_b) == 1:
                    return dist + 1
                visited[next_b] = dist + 1
                queue.append(next_b)
        
        return -1 # Not synchronizing

    # Evaluate current best
    best_length = evaluate(best_a, best_b)
    
    # Random search
    # We generate random automata, biased towards cycles and resets
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate random a_map (permutation)
        a_map = list(range(N))
        random.shuffle(a_map)
        
        # Generate random b_map
        # Bias: 50% chance to be 'reset-like' (many 0s), 50% random
        if random.random() < 0.5:
            b_map = [random.randint(0, N-1) for _ in range(N)]
            # Ensure at least one 0 to help synchronization?
            b_map[random.randint(0, N-1)] = 0
        else:
            b_map = [0 if i == 0 else i - 1 for i in range(N)]
            # Mutate
            for _ in range(random.randint(1, 3)):
                idx = random.randint(0, N-1)
                b_map[idx] = random.randint(0, N-1)
        
        length = evaluate(a_map, b_map)
        if length > best_length:
            best_length = length
            best_a = a_map
            best_b = b_map
            
        # If we hit the theoretical max, stop
        if best_length >= (N-1)**2:
            break
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
