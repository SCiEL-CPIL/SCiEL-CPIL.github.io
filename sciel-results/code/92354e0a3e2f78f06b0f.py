"""Rewritten program for the Černý problem (N=12).

Uses bitmask-based BFS to efficiently compute reset word lengths,
then searches over structured automaton families to maximize the
shortest reset word length.
"""
N = 12
FULL_MASK = (1 << N) - 1

import time
import itertools
from collections import deque

def _bfs_reset_length(trans_a, trans_b):
    """Compute shortest reset word length using bitmask BFS.
    
    trans_a, trans_b: lists of length N giving destination state for each state.
    Returns the length of the shortest reset word, or -1 if not synchronizing.
    """
    # Precompute bitmask transitions for each letter
    # For a subset S (bitmask), delta_a(S) = union of {trans_a[i] for i in S}
    # We precompute for all 2^N subsets? That's 4096 subsets, feasible.
    
    # Actually, let's precompute delta for single bits, then build up
    # delta_a(1<<i) = 1 << trans_a[i]
    # For a general mask, delta_a(mask) = OR of delta_a(1<<i) for each set bit i
    
    # Better: precompute delta_a[mask] for all masks
    delta_a = [0] * (1 << N)
    delta_b = [0] * (1 << N)
    
    for mask in range(1, 1 << N):
        # Get lowest set bit
        lsb = mask & (-mask)
        bit_idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        delta_a[mask] = delta_a[prev] | (1 << trans_a[bit_idx])
        delta_b[mask] = delta_b[prev] | (1 << trans_b[bit_idx])
    
    # BFS from full set
    start = FULL_MASK
    if start == 1:  # already a singleton
        return 0
    
    visited = set()
    visited.add(start)
    queue = deque([(start, 0)])
    
    while queue:
        state, dist = queue.popleft()
        
        # Try applying 'a'
        next_state = delta_a[state]
        if next_state == 1:
            return dist + 1
        if next_state not in visited:
            visited.add(next_state)
            queue.append((next_state, dist + 1))
        
        # Try applying 'b'
        next_state = delta_b[state]
        if next_state == 1:
            return dist + 1
        if next_state not in visited:
            visited.add(next_state)
            queue.append((next_state, dist + 1))
    
    return -1  # Not synchronizing

def _evaluate(trans_a, trans_b):
    """Return reset word length, or -1 if not synchronizing."""
    return _bfs_reset_length(trans_a, trans_b)

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    
    best_score = -1
    best_a = None
    best_b = None
    
    # Strategy 1: Try Černý-type constructions
    # a = cycle, b = fold toward state 0
    # Various fold patterns
    
    # Standard Černý: a: i -> (i+1)%N, b: 0->0, i->i-1 for i>=1
    a_cycle = [(i + 1) % N for i in range(N)]
    b_fold = [0] + list(range(N - 1))  # b[0]=0, b[1]=0, b[2]=1, ..., b[11]=10
    # Wait: b_fold[i] for i=0 should be 0, for i>=1 should be i-1
    b_fold = [0] * N
    for i in range(1, N):
        b_fold[i] = i - 1
    
    score = _evaluate(a_cycle, b_fold)
    if score > best_score:
        best_score = score
        best_a = a_cycle[:]
        best_b = b_fold[:]
    
    # Try different cycle directions and fold patterns
    # a: i -> (i + k) % N for various k
    for k in range(1, N):
        a_shift = [(i + k) % N for i in range(N)]
        # b: fold toward 0: b[i] = max(0, i - d) for various d
        for d in range(1, N):
            b_fold_d = [0] * N
            for i in range(N):
                b_fold_d[i] = max(0, i - d)
            score = _evaluate(a_shift, b_fold_d)
            if score > best_score:
                best_score = score
                best_a = a_shift[:]
                best_b = b_fold_d[:]
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # Strategy 2: Search over more general structured families
    # a = permutation (cycle), b = arbitrary map with some structure
    
    # Try: a is a single cycle, b maps most states to i-1 but with one "defect"
    base_cycle = [(i + 1) % N for i in range(N)]
    
    # b is mostly fold, but one state maps differently
    for defect_pos in range(N):
        for defect_target in range(N):
            b_defect = [0] * N
            for i in range(1, N):
                b_defect[i] = i - 1
            b_defect[0] = 0
            b_defect[defect_pos] = defect_target
            
            score = _evaluate(base_cycle, b_defect)
            if score > best_score:
                best_score = score
                best_a = base_cycle[:]
                best_b = b_defect[:]
            if time.time() > deadline:
                return best_a, best_b
    
    # Strategy 3: Randomized search with local improvement
    import random
    random.seed(42)
    
    # Start from best found, try random perturbations
    current_a = best_a[:]
    current_b = best_b[:]
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Randomly perturb one entry in a or b
        if random.random() < 0.5:
            pos = random.randrange(N)
            old_val = current_a[pos]
            current_a[pos] = random.randrange(N)
            score = _evaluate(current_a, current_b)
            if score > best_score:
                best_score = score
                best_a = current_a[:]
                best_b = current_b[:]
            else:
                current_a[pos] = old_val
        else:
            pos = random.randrange(N)
            old_val = current_b[pos]
            current_b[pos] = random.randrange(N)
            score = _evaluate(current_a, current_b)
            if score > best_score:
                best_score = score
                best_a = current_a[:]
                best_b = current_b[:]
            else:
                current_b[pos] = old_val
    
    return best_a, best_b
