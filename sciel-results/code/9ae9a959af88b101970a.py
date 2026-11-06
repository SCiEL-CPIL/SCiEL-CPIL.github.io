"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import time
import random
from collections import deque

def _compute_reset_length_bitmask(a, b):
    """BFS on subset automaton using bitmasks for speed."""
    full_mask = (1 << N) - 1
    
    # Precompute transition tables for each letter
    # trans_a[mask] = image of mask under a
    # trans_b[mask] = image of mask under b
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if mask & (1 << i):
                sa |= (1 << a[i])
                sb |= (1 << b[i])
        trans_a[mask] = sa
        trans_b[mask] = sb
    
    # BFS from full subset
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        dist = visited[mask]
        
        if mask & (mask - 1) == 0:  # singleton
            return dist
        
        for new_mask in (trans_a[mask], trans_b[mask]):
            if new_mask not in visited:
                visited[new_mask] = dist + 1
                queue.append(new_mask)
    
    return -1  # not synchronizing

def _is_synchronizing(a, b):
    """Quick check if automaton synchronizes using BFS."""
    full_mask = (1 << N) - 1
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        sa = 0
        sb = 0
        for i in range(N):
            if mask & (1 << i):
                sa |= (1 << a[i])
                sb |= (1 << b[i])
        trans_a[mask] = sa
        trans_b[mask] = sb
    
    visited = {full_mask}
    queue = deque([full_mask])
    
    while queue:
        mask = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return True
        
        for new_mask in (trans_a[mask], trans_b[mask]):
            if new_mask not in visited:
                visited.add(new_mask)
                queue.append(new_mask)
    
    return False

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55  # safety margin
    
    # Pinwheel construction
    # a is a cycle: 0->1->...->N-1->0
    # b is a cycle with a fixed point: 0->0, 1->0, 2->1, ..., N-1->N-2
    # This is the standard Pinwheel automaton.
    a0 = [(i + 1) % N for i in range(N)]
    b0 = [0] + [i - 1 for i in range(1, N)]
    
    best_a, best_b = a0[:], b0[:]
    best_score = _compute_reset_length_bitmask(a0, b0)
    
    if best_score <= 0:
        # Fallback to a simple synchronizing automaton
        return [[(i + 1) % N for i in range(N)], [0] * N]
    
    # Local search (Hill Climbing) starting from Pinwheel
    # We try to improve the score by making small random changes.
    # Since Pinwheel is often optimal or near-optimal, we use a strict
    # acceptance criterion and a time limit.
    
    current_a, current_b = a0[:], b0[:]
    current_score = best_score
    
    # To avoid getting stuck in local minima, we can occasionally
    # restart from the original Pinwheel or try different mutation strategies.
    
    while time.time() < deadline:
        improved = False
        # Try several random mutations in one iteration to save overhead
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Pick a random state and change its destination for 'a' or 'b'
            state = random.randint(0, N - 1)
            letter = random.randint(0, 1)
            
            if letter == 0:
                old_val = current_a[state]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_a[state] = new_val
                new_score = _compute_reset_length_bitmask(current_a, current_b)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    # Revert
                    current_a[state] = old_val
            else:
                old_val = current_b[state]
                new_val = random.randint(0, N - 1)
                if new_val == old_val:
                    continue
                current_b[state] = new_val
                new_score = _compute_reset_length_bitmask(current_a, current_b)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    # Revert
                    current_b[state] = old_val
        
        # If no improvement after many tries, restart from Pinwheel to escape local optima
        # But keep the global best.
        if not improved and random.random() < 0.05:
            current_a = a0[:]
            current_b = b0[:]
            current_score = _compute_reset_length_bitmask(current_a, current_b)
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
