import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _build_masks(a_trans, b_trans):
    """Precompute bitmasks for subset transition under a and b."""
    a_mask = [0] * (1 << N)
    b_mask = [0] * (1 << N)
    for s in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if s & (1 << i):
                ma |= 1 << a_trans[i]
                mb |= 1 << b_trans[i]
        a_mask[s] = ma
        b_mask[s] = mb
    return a_mask, b_mask

def calculate_threshold(a_trans, b_trans):
    """
    Computes the exact shortest reset word length (threshold) for a binary automaton.
    Uses BFS on the subset automaton with bitmask representation.
    Returns -1 if not synchronizing.
    """
    start = FULL_MASK
    if start == 0:
        return 0
    
    # A subset is a singleton if it has exactly one bit set
    visited = [0] * (1 << N)  # distance, -1 means unvisited
    visited[start] = 0
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        # Check if singleton (exactly one bit set)
        if curr & (curr - 1) == 0:
            return curr_dist
        
        a_mask, b_mask = a_trans, b_trans  # placeholder, we need precomputed
        
    # This won't work without precomputed masks. Let's use the precomputed version.
    pass

def calculate_threshold_fast(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using precomputed bitmask transitions.
    """
    a_mask = [0] * (1 << N)
    b_mask = [0] * (1 << N)
    for s in range(1 << N):
        ma = 0
        mb = 0
        for i in range(N):
            if s & (1 << i):
                ma |= 1 << a_trans[i]
                mb |= 1 << b_trans[i]
        a_mask[s] = ma
        b_mask[s] = mb
    
    start = FULL_MASK
    if start == 0:
        return 0
    
    visited = [-1] * (1 << N)
    visited[start] = 0
    q = deque([start])
    
    while q:
        curr = q.popleft()
        curr_dist = visited[curr]
        
        if curr & (curr - 1) == 0:
            return curr_dist
        
        na = a_mask[curr]
        if visited[na] == -1:
            visited[na] = curr_dist + 1
            q.append(na)
        
        nb = b_mask[curr]
        if visited[nb] == -1:
            visited[nb] = curr_dist + 1
            q.append(nb)
            
    return -1

def construct():
    """
    Construct a 12-state binary automaton that synchronizes, maximizing the reset threshold.
    Uses bitmask-based BFS for fast threshold computation and local search over both transition functions.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    def try_automaton(a, b):
        nonlocal best_score, best_a, best_b
        score = calculate_threshold_fast(a, b)
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
        return score
    
    # Initialize with a known synchronizing automaton
    # Use a simple one: a = cycle, b = maps all to 0
    init_a = [(i + 1) % N for i in range(N)]
    init_b = [0] * N
    try_automaton(init_a, init_b)
    
    # Local search: mutate one transition at a time, accept if improves or equal (for exploration)
    random.seed(12345)
    
    current_a = best_a[:]
    current_b = best_b[:]
    current_score = best_score
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Decide whether to mutate a or b
        if random.random() < 0.5:
            # Mutate a
            idx = random.randint(0, N - 1)
            old_val = current_a[idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_a[idx] = new_val
            score = calculate_threshold_fast(current_a, current_b)
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif score == -1:
                # Not synchronizing, reject
                current_a[idx] = old_val
            else:
                # Accept with some probability for exploration (simulated annealing-like)
                # Use a simple heuristic: accept if score is not too far from current
                if random.random() < 0.1 and score >= current_score - 2:
                    current_score = score
                else:
                    current_a[idx] = old_val
        else:
            # Mutate b
            idx = random.randint(0, N - 1)
            old_val = current_b[idx]
            new_val = random.randint(0, N - 1)
            if new_val == old_val:
                continue
            current_b[idx] = new_val
            score = calculate_threshold_fast(current_a, current_b)
            if score > current_score:
                current_score = score
                if score > best_score:
                    best_score = score
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif score == -1:
                # Not synchronizing, reject
                current_b[idx] = old_val
            else:
                # Accept with some probability for exploration
                if random.random() < 0.1 and score >= current_score - 2:
                    current_score = score
                else:
                    current_b[idx] = old_val
    
    # Final polish: try all single-transition changes from the best found
    if best_a is not None and time.time() < deadline - 5.0:
        for idx in range(N):
            for new_val in range(N):
                if time.time() > deadline:
                    break
                # Try changing a[idx]
                old_val = best_a[idx]
                best_a[idx] = new_val
                score = calculate_threshold_fast(best_a, best_b)
                if score > best_score:
                    best_score = score
                else:
                    best_a[idx] = old_val
            if time.time() > deadline:
                break
        
        if time.time() < deadline - 5.0:
            for idx in range(N):
                for new_val in range(N):
                    if time.time() > deadline:
                        break
                    # Try changing b[idx]
                    old_val = best_b[idx]
                    best_b[idx] = new_val
                    score = calculate_threshold_fast(best_a, best_b)
                    if score > best_score:
                        best_score = score
                    else:
                        best_b[idx] = old_val
                if time.time() > deadline:
                    break
    
    # Fallback: If no valid automaton found (shouldn't happen), use a known synchronizing one
    if best_a is None:
        best_a = init_a[:]
        best_b = init_b[:]
    
    return [best_a, best_b]

# EVOLVE-BLOCK-START
# The function construct() is defined above.
# EVOLVE-BLOCK-END
