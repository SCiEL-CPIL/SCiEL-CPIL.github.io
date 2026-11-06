"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

N = 12

def _compute_reset_length(a, b):
    """Compute the exact shortest reset word length via BFS on subset automaton."""
    full = frozenset(range(N))
    if len(full) == 1:
        return 0
    visited = {full}
    queue = deque([(full, 0)])
    while queue:
        subset, dist = queue.popleft()
        if len(subset) == 1:
            return dist
        for letter in (a, b):
            new_subset = frozenset(letter[s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append((new_subset, dist + 1))
    return -1  # not synchronizing

def _is_synchronizing(a, b):
    """Check if automaton synchronizes using BFS."""
    full = frozenset(range(N))
    visited = {full}
    queue = deque([full])
    while queue:
        subset = queue.popleft()
        if len(subset) == 1:
            return True
        for letter in (a, b):
            new_subset = frozenset(letter[s] for s in subset)
            if new_subset not in visited:
                visited.add(new_subset)
                queue.append(new_subset)
    return False

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # Safety margin under 90s limit
    
    # Start with a known good construction as baseline
    # "Pavlov" style: a is cycle, b is a slow collapse
    best_a = list(range(1, N)) + [0]  # a(i) = (i+1) % N
    best_b = [0] + list(range(N - 1))  # b(i) = i-1 for i>0, b(0)=0
    best_len = _compute_reset_length(best_a, best_b)
    
    # Try several known constructions
    candidates = [
        # Construction 1: cycle + decrement
        (list(range(1, N)) + [0], [0] + list(range(N - 1))),
        # Construction 2: cycle + halving
        (list(range(1, N)) + [0], [i // 2 for i in range(N)]),
        # Construction 3: cycle + b maps 0->0, 1->0, others -> i-1
        (list(range(1, N)) + [0], [0, 0] + list(range(2, N))),
        # Construction 4: cycle + fold
        (list(range(1, N)) + [0], list(range(N // 2)) + [i - N // 2 for i in range(N // 2, N)]),
        # Construction 5: cycle + constant
        (list(range(1, N)) + [0], [0] * N),
        # Construction 6: reverse cycle + constant
        (list(range(N - 1, -1, -1)), [0] * N),
        # Construction 7: cycle + b(0)=0, b(1)=0, b(2)=1, b(i)=i-2 for i>=3
        (list(range(1, N)) + [0], [0, 0, 1] + list(range(2, N - 2))),
    ]
    
    for a, b in candidates:
        if time.time() > deadline:
            break
        if _is_synchronizing(a, b):
            length = _compute_reset_length(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length
    
    # Local search: random perturbations of the best automaton
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        a = list(best_a)
        b = list(best_b)
        
        # Perturb one or two random entries
        num_perturbations = 1 if rng.random() < 0.7 else 2
        for _ in range(num_perturbations):
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                b[idx] = rng.randrange(N)
        
        if _is_synchronizing(a, b):
            length = _compute_reset_length(a, b)
            if length > best_len:
                best_a, best_b, best_len = a, b
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
