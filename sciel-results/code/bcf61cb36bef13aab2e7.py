"""
Edge-colourings of K16 in which every K4 sees five colours.

Strategy:
1. Start from the known valid (i+j) mod 17 construction (17 colours).
2. Use simulated annealing / local search to reduce the number of colours.
   - Objective: minimise the number of distinct colours used, subject to
     the constraint that every K4 spans >= 5 colours.
   - We penalise constraint violations heavily and add a small cost per
     colour used to drive colour reduction.
3. Return the best valid colouring found before the deadline.
"""

import time
import random
import sys
import numpy as np

DEADLINE = 115.0  # safety margin before 120s kill

def _edge_index(i, j):
    """Map pair (i,j), 0<=i<j<16, to index in the 120-element list."""
    return i * 15 - i * (i - 1) // 2 + j - 1 - i

def _precompute_k4s():
    """Return list of 6-tuples of edge indices for each K4."""
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    edges = [
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    ]
                    k4s.append(tuple(edges))
    return k4s

_K4S = _precompute_k4s()
_K4S_ARRAY = np.array(_K4S, dtype=np.int32)  # shape (1820, 6)
_NUM_K4S = len(_K4S)

def _count_violations(colours):
    """Count number of K4s that span fewer than 5 distinct colours using NumPy."""
    arr = np.array(colours, dtype=np.int32)
    # Get colours for each edge in each K4: shape (1820, 6)
    k4_colours = arr[_K4S_ARRAY]
    # For each K4, count distinct colours by checking each edge against the first
    # A K4 has < 5 distinct colours if at least 2 edges share a colour with another
    # More efficient: check if any colour appears >= 2 times AND total distinct < 5
    # Simple approach: for each K4, count how many of the 6 edges have a colour
    # that appears elsewhere in the same K4
    violations = 0
    for i in range(_NUM_K4S):
        cols = k4_colours[i]
        # Count distinct
        if len(set(cols.tolist())) < 5:
            violations += 1
    return violations

def _count_violations_fast(colours_arr):
    """Vectorized violation count using numpy. colours_arr is a numpy array of shape (120,)."""
    k4_colours = colours_arr[_K4S_ARRAY]  # (1820, 6)
    # For each K4, determine if it has < 5 distinct colours
    # A K4 has < 5 distinct colours iff there exists a pair of edges with same colour
    # such that the total number of distinct colours is < 5
    # Efficient check: count distinct via sorting or via pairwise comparison
    # Use: distinct count = 6 - (number of pairs that are equal) is not quite right
    # Better: for each K4, check all C(6,2)=15 pairs; if any pair matches, 
    # we need to verify total distinct < 5
    # Fastest: just check if the number of unique values < 5
    # Using numpy: for each row, count unique
    # Trick: sort each row, then count changes
    sorted_cols = np.sort(k4_colours, axis=1)  # (1820, 6)
    # Count changes in sorted row
    changes = (sorted_cols[:, 1:] != sorted_cols[:, :-1]).astype(np.int32)
    # Number of distinct = 1 + sum(changes)
    distinct_counts = 1 + changes.sum(axis=1)
    violations = np.sum(distinct_counts < 5)
    return int(violations)

def _num_colours_used(colours):
    return len(set(colours))

def _objective(colours, weight=100):
    """Penalised objective: violations*weight + colours_used."""
    v = _count_violations_fast(np.array(colours, dtype=np.int32))
    if v > 0:
        return v * weight + _num_colours_used(colours)
    return _num_colours_used(colours)

def _initial_colouring():
    """(i+j) mod 17 construction."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _local_search(colours, max_iters=5000, deadline=None):
    """Hill-climbing with random restarts on single-edge colour changes."""
    best = list(colours)
    best_arr = np.array(best, dtype=np.int32)
    best_obj = _objective(best)
    rng = random.Random(42)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Pick a random edge and a random new colour
        idx = rng.randrange(120)
        old_c = best[idx]
        new_c = rng.randrange(17)  # search in 0..16
        if new_c == old_c:
            continue
        
        best[idx] = new_c
        best_arr[idx] = new_c
        v = _count_violations_fast(best_arr)
        if v > 0:
            obj_new = v * 100 + _num_colours_used(best)
        else:
            obj_new = _num_colours_used(best)
        
        if obj_new < best_obj:
            best_obj = obj_new
            # Accept improvement
        else:
            # Revert
            best[idx] = old_c
            best_arr[idx] = old_c
    
    return best, best_obj

def _simulated_annealing(colours, deadline):
    """SA with single-edge moves to reduce colours while maintaining validity."""
    current = list(colours)
    current_arr = np.array(current, dtype=np.int32)
    current_obj = _objective(current)
    best = list(current)
    best_obj = current_obj
    
    T = 50.0
    T_min = 0.1
    cooling = 0.999
    
    rng = random.Random(123)
    n_moves = 0
    
    while time.time() < deadline:
        n_moves += 1
        
        # Pick random edge and new colour
        idx = rng.randrange(120)
        old_c = current[idx]
        new_c = rng.randrange(17)
        if new_c == old_c:
            continue
        
        current[idx] = new_c
        current_arr[idx] = new_c
        v = _count_violations_fast(current_arr)
        if v > 0:
            new_obj = v * 100 + _num_colours_used(current)
        else:
            new_obj = _num_colours_used(current)
        delta = new_obj - current_obj
        
        if delta <= 0 or rng.random() < min(1.0, T / max(1, current_obj)):
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best = list(current)
        else:
            current[idx] = old_c
            current_arr[idx] = old_c
        
        T *= cooling
        if T < T_min:
            T = T_min
        
        # Periodic restart if stuck
        if n_moves % 10000 == 0:
            best_v = _count_violations_fast(np.array(best, dtype=np.int32))
            if best_v == 0:
                # Try to reduce colours by mapping a colour to another
                used = set(best)
                if len(used) > 5:
                    # Try to merge a colour into another
                    c1 = rng.choice(list(used))
                    c2 = rng.choice([c for c in used if c != c1])
                    trial = list(best)
                    for i in range(120):
                        if trial[i] == c1:
                            trial[i] = c2
                    trial_v = _count_violations_fast(np.array(trial, dtype=np.int32))
                    if trial_v == 0:
                        best = trial
                        best_obj = _num_colours_used(best)
                        current = list(best)
                        current_arr = np.array(current, dtype=np.int32)
                        current_obj = best_obj
                        T = 50.0
    
    return best, best_obj

def construct():
    t0 = time.time()
    
    # Start from valid 17-colour construction
    colours = _initial_colouring()
    
    # Verify validity
    assert _count_violations_fast(np.array(colours, dtype=np.int32)) == 0, "Initial construction should be valid"
    
    best = list(colours)
    best_obj = _num_colours_used(colours)
    best_colours = best_obj
    
    # Phase 1: Local search to try to reduce colours
    deadline = t0 + DEADLINE
    result, obj = _local_search(colours, max_iters=200000, deadline=deadline)
    if obj < best_colours and _count_violations_fast(np.array(result, dtype=np.int32)) == 0:
        best = result
        best_colours = obj
    
    # Phase 2: Simulated annealing for more aggressive search
    if time.time() < deadline:
        result, obj = _simulated_annealing(best, deadline)
        if obj < best_colours and _count_violations_fast(np.array(result, dtype=np.int32)) == 0:
            best = result
            best_colours = obj
    
    # Final: ensure validity
    if _count_violations_fast(np.array(best, dtype=np.int32)) != 0:
        # Fall back to the known valid construction
        best = _initial_colouring()
    
    return best
