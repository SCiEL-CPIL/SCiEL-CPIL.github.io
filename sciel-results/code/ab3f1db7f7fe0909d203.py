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
    deadline = time.time() + 85.0
    
    best_a = None
    best_b = None
    best_len = -1
    
    def update_best(a, b):
        nonlocal best_a, best_b, best_len
        if _is_synchronizing(a, b):
            length = _compute_reset_length(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length
    
    # Try structured candidates
    cycle_a = list(range(1, N)) + [0]
    
    # Family 1: cycle + b with offset and trap
    for offset in range(1, N):
        for trap in range(N):
            if time.time() > deadline:
                return [best_a, best_b]
            b = [(i + offset) % N for i in range(N)]
            b[trap] = 0
            update_best(cycle_a, b)
    
    # Family 2: cycle + b(i) = max(0, i-k)
    for k in range(1, N):
        if time.time() > deadline:
            return [best_a, best_b]
        b = [max(0, i - k) for i in range(N)]
        update_best(cycle_a, b)
    
    # Family 3: cycle + b(i) = min(N-1, i+k) with b(0)=0
    for k in range(1, N):
        if time.time() > deadline:
            return [best_a, best_b]
        b = [min(N-1, i + k) for i in range(N)]
        b[0] = 0
        update_best(cycle_a, b)
    
    # Local search from best
    rng = random.Random(42)
    if best_a is not None:
        while time.time() < deadline:
            a = list(best_a)
            b = list(best_b)
            
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                b[idx] = rng.randrange(N)
            
            if _is_synchronizing(a, b):
                length = _compute_reset_length(a, b)
                if length > best_len:
                    best_a, best_b, best_len = a, b, length
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
