"""Simulated Annealing search for maximum reset threshold in 12-state binary automata."""
import random
import time
from collections import deque

N = 12
DEADLINE = 85  # seconds, safety margin before 90s kill

def _reset_length(a, b):
    """Compute shortest reset word length via BFS on subsets. Returns -1 if not synchronizing."""
    start = tuple(range(N))
    if len(set(a)) == 1 or len(set(b)) == 1:
        return 1 if N > 1 else 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        next_a = tuple(sorted(set(a[i] for i in subset)))
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
        next_b = tuple(sorted(set(b[i] for i in subset)))
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
    return -1

def _energy(a, b):
    rl = _reset_length(a, b)
    if rl < 0:
        return 100000
    return -rl

def _mutate(a, b):
    a_new = list(a)
    b_new = list(b)
    n_mut = random.randint(1, 2)
    for _ in range(n_mut):
        letter = random.choice(['a', 'b'])
        state = random.randint(0, N - 1)
        new_dest = random.randint(0, N - 1)
        if letter == 'a':
            a_new[state] = new_dest
        else:
            b_new[state] = new_dest
    return a_new, b_new

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    best_a, best_b = None, None
    best_energy = float('inf')
    
    for _ in range(30):
        if time.time() - start_time > DEADLINE:
            break
        a = [random.randint(0, N - 1) for _ in range(N)]
        b = [random.randint(0, N - 1) for _ in range(N)]
        e = _energy(a, b)
        if e < best_energy:
            best_energy = e
            best_a, best_b = list(a), list(b)
    
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_energy = _energy(best_a, best_b)
    
    current_a, current_b = list(best_a), list(best_b)
    current_energy = best_energy
    
    T = 50.0
    T_min = 0.01
    cooling = 0.995
    
    while time.time() - start_time < DEADLINE:
        if T < T_min:
            T = T_min
        
        a_new, b_new = _mutate(current_a, current_b)
        e_new = _energy(a_new, b_new)
        delta_e = e_new - current_energy
        
        if delta_e < 0 or random.random() < pow(2.718281828, -delta_e / max(T, 1e-10)):
            current_a, current_b = a_new, b_new
            current_energy = e_new
            if current_energy < best_energy:
                best_energy = current_energy
                best_a, best_b = list(current_a), list(current_b)
        
        T *= cooling
    
    return best_a, best_b
# EVOLVE-BLOCK-END
