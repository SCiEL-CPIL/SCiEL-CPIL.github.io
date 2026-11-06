"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12

# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def _bfs_reset_length(a_trans, b_trans):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if the automaton is not synchronizing.
    """
    n = len(a_trans)
    full_mask = (1 << n) - 1
    
    # Quick check: if either letter is a constant, reset length is 0
    if len(set(a_trans)) == 1 or len(set(b_trans)) == 1:
        return 0

    # Precompute single-state transitions as bitmasks for speed
    # a_single[i] is the bitmask of the state a(i) maps to
    a_single = [1 << a_trans[i] for i in range(n)]
    b_single = [1 << b_trans[i] for i in range(n)]
    
    def apply_letter(mask, single_masks):
        """Apply a letter (represented by single_masks) to a subset (bitmask)."""
        result = 0
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            result |= single_masks[idx]
            m ^= lsb
        return result

    # BFS
    visited = set()
    visited.add(full_mask)
    queue = deque()
    queue.append((full_mask, 0))
    
    while queue:
        mask, dist = queue.popleft()
        
        # Check if we reached a singleton
        if mask & (mask - 1) == 0:
            return dist
            
        # Expand neighbors
        next_a = apply_letter(mask, a_single)
        if next_a not in visited:
            visited.add(next_a)
            queue.append((next_a, dist + 1))
            
        next_b = apply_letter(mask, b_single)
        if next_b not in visited:
            visited.add(next_b)
            queue.append((next_b, dist + 1))
            
    return -1

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 60s limit
    
    best_score = 0
    best_a = None
    best_b = None
    
    # Cache for BFS results to avoid recomputation of identical automata
    bfs_cache = {}
    
    def evaluate(a_trans, b_trans):
        nonlocal best_score, best_a, best_b
        if time.time() > deadline:
            return
        key = (tuple(a_trans), tuple(b_trans))
        if key in bfs_cache:
            score = bfs_cache[key]
        else:
            score = _bfs_reset_length(a_trans, b_trans)
            bfs_cache[key] = score
        
        if score > best_score:
            best_score = score
            best_a = a_trans[:]
            best_b = b_trans[:]

    # Strategy 1: Systematic Enumeration of Pinwheel/Cycle-Break Families
    # Theoretical max for N=12 is (12-1)^2 = 121.
    # Standard construction: a is a cycle, b is a cycle with one break.
    
    # 1a: a = cycle step 1, b = reverse cycle with one modification
    for step_a in range(1, N):
        a_trans = [(i + step_a) % N for i in range(N)]
        
        # b as reverse cycle with one state modified
        for break_state in range(N):
            for target in range(N):
                b_trans = [(i - 1) % N for i in range(N)]
                b_trans[break_state] = target
                evaluate(a_trans, b_trans)
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break
    
    # 1b: a = cycle step 1, b = forward cycle with one modification
    a_trans = [(i + 1) % N for i in range(N)]
    for break_state in range(N):
        for target in range(N):
            b_trans = [(i + 1) % N for i in range(N)]
            b_trans[break_state] = target
            evaluate(a_trans, b_trans)
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break

    # 1c: Generalized cycle steps for a and b
    # Try a few other step combinations for a
    for step_a in [2, 3, 5, 7]:
        a_trans = [(i + step_a) % N for i in range(N)]
        # b as reverse of a's step? or just reverse 1?
        # Usually b is close to inverse of a to create "slow" convergence
        step_b_inv = (N - step_a) % N
        for break_state in range(N):
            for target in range(N):
                b_trans = [(i + step_b_inv) % N for i in range(N)]
                b_trans[break_state] = target
                evaluate(a_trans, b_trans)
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        if time.time() > deadline - 5.0:
            break

    # Strategy 2: Local Search Polish
    # If we found a good candidate, try to improve it via single-bit mutations.
    # This helps if the optimal solution is a slight variant of the standard pinwheel.
    if best_a is not None and time.time() < deadline - 2.0:
        random.seed(42)
        current_a = best_a[:]
        current_b = best_b[:]
        current_score = best_score
        
        no_improve = 0
        max_iterations = 2000
        
        while time.time() < deadline - 0.5 and no_improve < max_iterations:
            # Pick a random position and letter to mutate
            if random.random() < 0.5:
                pos = random.randint(0, N - 1)
                old_val = current_a[pos]
                current_a[pos] = random.randint(0, N - 1)
                new_score = _bfs_reset_length(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    current_a[pos] = old_val
                    no_improve += 1
            else:
                pos = random.randint(0, N - 1)
                old_val = current_b[pos]
                current_b[pos] = random.randint(0, N - 1)
                new_score = _bfs_reset_length(current_a, current_b)
                if new_score > current_score:
                    current_score = new_score
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_a = current_a[:]
                        best_b = current_b[:]
                else:
                    current_b[pos] = old_val
                    no_improve += 1

    # Fallback: Return a known good pinwheel if search failed
    if best_a is None:
        a_trans = [(i + 1) % N for i in range(N)]
        b_trans = [(i - 1) % N for i in range(N)]
        b_trans[0] = 0
        return [a_trans, b_trans]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
