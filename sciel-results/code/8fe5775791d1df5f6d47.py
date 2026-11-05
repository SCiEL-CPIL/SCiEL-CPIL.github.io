"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses precomputed transition tables for fast BFS evaluation, systematic family search,
and local search refinement.
"""
import time
import random
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
NUM_MASKS = 1 << N

# EVOLVE-BLOCK-START

def _precompute_transitions(a, b):
    """Precompute transition tables for all 2^N masks for letters a and b."""
    trans_a = [0] * NUM_MASKS
    trans_b = [0] * NUM_MASKS
    
    for mask in range(NUM_MASKS):
        ma = 0
        mb = 0
        for i in range(N):
            if mask & (1 << i):
                ma |= 1 << a[i]
                mb |= 1 << b[i]
        trans_a[mask] = ma
        trans_b[mask] = mb
    
    return trans_a, trans_b

def _threshold_from_tables(trans_a, trans_b):
    """
    Compute the exact reset threshold using BFS on the subset automaton.
    Returns -1 if not synchronizing. Uses precomputed transition tables.
    """
    visited = [-1] * NUM_MASKS
    visited[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    
    while queue:
        mask = queue.popleft()
        depth = visited[mask]
        
        # Check if singleton (power of 2)
        if mask & (mask - 1) == 0:
            return depth
        
        # Apply 'a'
        next_a = trans_a[mask]
        if visited[next_a] == -1:
            visited[next_a] = depth + 1
            queue.append(next_a)
            
        # Apply 'b'
        next_b = trans_b[mask]
        if visited[next_b] == -1:
            visited[next_b] = depth + 1
            queue.append(next_b)
            
    return -1

def _threshold_fast(a, b):
    """Compute threshold with precomputed tables."""
    trans_a, trans_b = _precompute_transitions(a, b)
    return _threshold_from_tables(trans_a, trans_b)

def _evaluate_family_cycle_perturbed(deadline):
    """
    Systematic search over families:
    - a = cycle (i+1)%N, b = identity with 1-2 defects
    - a = cycle, b = various structured perturbations
    """
    best_score = -1
    best_a = None
    best_b = None
    
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Family 1: b = identity with single defect
    for k in range(N):
        for j in range(N):
            if j == k:
                continue
            b = list(range(N))
            b[k] = j
            score = _threshold_fast(a_cycle, b)
            if score > best_score:
                best_score = score
                best_a = a_cycle[:]
                best_b = b[:]
                if time.time() > deadline:
                    return best_a, best_b, best_score
    
    # Family 2: b = identity with double defect
    for k1 in range(N):
        for k2 in range(k1 + 1, N):
            for j1 in range(N):
                for j2 in range(N):
                    if j1 == k1 and j2 == k2:
                        continue
                    b = list(range(N))
                    b[k1] = j1
                    b[k2] = j2
                    score = _threshold_fast(a_cycle, b)
                    if score > best_score:
                        best_score = score
                        best_a = a_cycle[:]
                        best_b = b[:]
                    if time.time() > deadline:
                        return best_a, best_b, best_score
    
    # Family 3: a = shifted cycle, b = identity with single defect
    for d in range(1, N):
        a_shift = [(i + d) % N for i in range(N)]
        for k in range(N):
            for j in range(N):
                if j == k:
                    continue
                b = list(range(N))
                b[k] = j
                score = _threshold_fast(a_shift, b)
                if score > best_score:
                    best_score = score
                    best_a = a_shift[:]
                    best_b = b[:]
                if time.time() > deadline:
                    return best_a, best_b, best_score
    
    return best_a, best_b, best_score

def _local_search(a, b, time_budget):
    """
    Steepest ascent local search: explore all single-transition changes.
    Returns best (a, b, score) found within time_budget.
    """
    start_time = time.time()
    current_score = _threshold_fast(a, b)
    best_a = a[:]
    best_b = b[:]
    best_score = current_score
    
    if current_score < 0:
        return best_a, best_b, best_score
    
    improved = True
    while improved and time.time() - start_time < time_budget:
        improved = False
        
        # Try changing each transition in 'a'
        for pos in range(N):
            old_val = a[pos]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                a[pos] = new_val
                score = _threshold_fast(a, b)
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
                    improved = True
                a[pos] = old_val
        
        # Try changing each transition in 'b'
        for pos in range(N):
            old_val = b[pos]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                b[pos] = new_val
                score = _threshold_fast(a, b)
                if score > best_score:
                    best_score = score
                    best_a = a[:]
                    best_b = b[:]
                    improved = True
                b[pos] = old_val
        
        # Update current to best found
        a = best_a[:]
        b = best_b[:]
        current_score = best_score
    
    return best_a, best_b, best_score

def _random_automaton(rng):
    """Generate a random automaton with guaranteed synchronization."""
    a = [rng.randint(0, N - 1) for _ in range(N)]
    b = [rng.randint(0, N - 1) for _ in range(N)]
    # Force a collision to help synchronization
    b[1] = b[0]
    return a, b

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses systematic family search + local search refinement.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s kill
    
    best_a = None
    best_b = None
    best_score = -1
    
    # Phase 1: Systematic search over structured families
    a, b, score = _evaluate_family_cycle_perturbed(deadline)
    if a is not None and score > best_score:
        best_score = score
        best_a = a[:]
        best_b = b[:]
    
    # Fallback if no valid result yet
    if best_a is None:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_score = _threshold_fast(best_a, best_b)
    
    # Phase 2: Local search from best found
    remaining = deadline - time.time()
    if remaining > 10.0:
        local_budget = min(remaining - 5.0, 50.0)
        best_a, best_b, best_score = _local_search(best_a, best_b, local_budget)
    
    # Phase 3: Random restarts with local search if time remains
    rng = random.Random(42)
    while time.time() < deadline - 8.0:
        a, b = _random_automaton(rng)
        score = _threshold_fast(a, b)
        
        if score < 0:
            continue
        
        # Quick local search
        quick_budget = min(3.0, deadline - time.time() - 5.0)
        if quick_budget > 1.0:
            a, b, score = _local_search(a, b, quick_budget)
        
        if score > best_score:
            best_score = score
            best_a = a[:]
            best_b = b[:]
    
    # Ensure we have a valid synchronizing result
    if best_a is None or best_score < 0:
        best_a = [(i + 1) % N for i in range(N)]
        best_b = [0] * N
        best_score = _threshold_fast(best_a, best_b)
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
