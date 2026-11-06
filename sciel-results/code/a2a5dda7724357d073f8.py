import time
import random
from collections import deque

N = 12

def _compute_reset_length_bitmask(a, b):
    """Compute the exact shortest reset word length via BFS on subset automaton using bitmasks."""
    full_mask = (1 << N) - 1
    if full_mask == 0:
        return 0
    
    # Precompute transitions for each letter as a function on masks
    # trans_a[mask] = mask of destinations
    # This is expensive to precompute for all 2^N masks.
    # Instead, compute on the fly.
    
    def apply_transition(mask, letter):
        new_mask = 0
        for i in range(N):
            if mask & (1 << i):
                dest = letter[i]
                new_mask |= (1 << dest)
        return new_mask

    visited = {full_mask}
    queue = deque([(full_mask, 0)])
    
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0: # Single bit
            return dist
        for letter in (a, b):
            new_mask = apply_transition(mask, letter)
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append((new_mask, dist + 1))
    return -1

def _is_synchronizing_bitmask(a, b):
    """Check if automaton synchronizes using BFS with bitmasks."""
    full_mask = (1 << N) - 1
    visited = {full_mask}
    queue = deque([full_mask])
    
    def apply_transition(mask, letter):
        new_mask = 0
        for i in range(N):
            if mask & (1 << i):
                dest = letter[i]
                new_mask |= (1 << dest)
        return new_mask

    while queue:
        mask = queue.popleft()
        if mask & (mask - 1) == 0:
            return True
        for letter in (a, b):
            new_mask = apply_transition(mask, letter)
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append(new_mask)
    return False

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds deadline
    
    candidates = []
    
    # 1. Černý Automaton
    a_cherny = [(i + 1) % N for i in range(N)]
    b_cherny = [0] + list(range(N - 1))
    candidates.append((a_cherny, b_cherny))
    
    # 2. Pinwheel Automaton (Variant)
    # Sometimes defined differently, but usually same as Černý.
    # Let's try a slight variation if it exists.
    # For now, just keep Černý.
    
    # 3. Random Candidates (if time permits)
    # We can generate a few random automata and check if they are synchronizing.
    # But random automata are rarely optimal.
    
    best_a = a_cherny
    best_b = b_cherny
    best_score = -1
    
    for a, b in candidates:
        if time.time() > deadline:
            break
            
        # Quick check for synchronizing
        if not _is_synchronizing_bitmask(a, b):
            continue
            
        score = _compute_reset_length_bitmask(a, b)
        if score > best_score:
            best_score = score
            best_a = a
            best_b = b
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
