"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque
import numpy as np

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
    deadline = time.time() + 75.0  # safety margin under 90s limit
    
    # Start with known good constructions as baselines
    # Construction 1: cycle + decrement
    a1 = list(range(1, N)) + [0]
    b1 = [0] + list(range(N - 1))
    
    # Construction 2: cycle + halving
    a2 = list(range(1, N)) + [0]
    b2 = [i // 2 for i in range(N)]
    
    # Construction 3: cycle + fold
    a3 = list(range(1, N)) + [0]
    b3 = list(range(N // 2)) + [i - N // 2 for i in range(N // 2, N)]
    
    # Construction 4: cycle + map to 0 for even, i-1 for odd
    a4 = list(range(1, N)) + [0]
    b4 = [i // 2 if i % 2 == 0 else i - 1 for i in range(N)]
    
    # Construction 5: two different structures
    a5 = [(i * 2) % N for i in range(N)]
    b5 = [i // 2 for i in range(N)]
    
    # Construction 6: cycle + b maps pairs to representatives
    a6 = list(range(1, N)) + [0]
    b6 = [i // 2 for i in range(N)]
    
    candidates = [
        (a1, b1),
        (a2, b2),
        (a3, b3),
        (a4, b4),
        (a5, b5),
        (a6, b6),
    ]
    
    best_a = None
    best_b = None
    best_len = -1
    
    for a, b in candidates:
        if time.time() > deadline:
            break
        if _is_synchronizing(a, b):
            length = _compute_reset_length(a, b)
            if length > best_len:
                best_a, best_b, best_len = list(a), list(b), length
    
    # If no synchronizing automaton found, use a guaranteed one
    if best_len < 0:
        best_a = list(range(1, N)) + [0]
        best_b = [0] * N
        best_len = _compute_reset_length(best_a, best_b)
    
    # Local search with multiple strategies
    rng = random.Random(42)
    iterations = 0
    max_iterations = 50000
    
    while iterations < max_iterations:
        if time.time() > deadline:
            break
        iterations += 1
        
        a = list(best_a)
        b = list(best_b)
        
        # Strategy 1: Perturb one random entry in a or b
        strategy = rng.random()
        
        if strategy < 0.4:
            # Perturb one entry in a
            idx = rng.randrange(N)
            a[idx] = rng.randrange(N)
        elif strategy < 0.8:
            # Perturb one entry in b
            idx = rng.randrange(N)
            b[idx] = rng.randrange(N)
        else:
            # Perturb two entries (one in a, one in b)
            idx_a = rng.randrange(N)
            idx_b = rng.randrange(N)
            a[idx_a] = rng.randrange(N)
            b[idx_b] = rng.randrange(N)
        
        if _is_synchronizing(a, b):
            length = _compute_reset_length(a, b)
            if length > best_len:
                best_a, best_b, best_len = a, b, length
    
    # Second phase: more aggressive search around best solution
    # Try systematic variations of the best automaton
    if time.time() < deadline - 10:
        # Try changing b to various structured maps while keeping a as cycle
        a_cycle = list(range(1, N)) + [0]
        
        # Generate various b maps
        b_variants = []
        
        # Halving variants
        for offset in range(3):
            b_variants.append([(i + offset) // 2 for i in range(N)])
        
        # Fold variants
        for k in range(2, 6):
            b_variants.append([i % k for i in range(N)])
        
        # Pair mapping variants
        for shift in range(N):
            b_variants.append([(i + shift) // 2 for i in range(N)])
        
        # Truncation variants
        for threshold in range(1, N):
            b_variants.append([min(i, threshold) for i in range(N)])
        
        for b_var in b_variants:
            if time.time() > deadline:
                break
            if _is_synchronizing(a_cycle, b_var):
                length = _compute_reset_length(a_cycle, b_var)
                if length > best_len:
                    best_a, best_b, best_len = list(a_cycle), list(b_var), length
    
    # Third phase: hill climbing with multiple restarts from different bases
    if time.time() < deadline - 5:
        bases = [
            (list(range(1, N)) + [0], [i // 2 for i in range(N)]),
            (list(range(1, N)) + [0], [0] + list(range(N - 1))),
            (list(range(1, N)) + [0], list(range(N // 2)) + [i - N // 2 for i in range(N // 2, N)]),
        ]
        
        for base_a, base_b in bases:
            if time.time() > deadline:
                break
            
            cur_a = list(base_a)
            cur_b = list(base_b)
            cur_len = _compute_reset_length(cur_a, cur_b)
            
            if cur_len < 0:
                continue
            
            # Hill climb from this base
            for _ in range(5000):
                if time.time() > deadline:
                    break
                
                new_a = list(cur_a)
                new_b = list(cur_b)
                
                # Random perturbation
                if rng.random() < 0.5:
                    idx = rng.randrange(N)
                    new_a[idx] = rng.randrange(N)
                else:
                    idx = rng.randrange(N)
                    new_b[idx] = rng.randrange(N)
                
                if _is_synchronizing(new_a, new_b):
                    new_len = _compute_reset_length(new_a, new_b)
                    if new_len > cur_len:
                        cur_a, cur_b, cur_len = new_a, new_b, new_len
            
            if cur_len > best_len:
                best_a, best_b, best_len = cur_a, cur_b, cur_len
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
