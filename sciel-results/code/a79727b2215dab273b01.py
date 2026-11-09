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
                    edges = (
                        _edge_index(a, b),
                        _edge_index(a, c),
                        _edge_index(a, d),
                        _edge_index(b, c),
                        _edge_index(b, d),
                        _edge_index(c, d),
                    )
                    k4s.append(edges)
    return k4s

_K4S = _precompute_k4s()
_NUM_K4S = len(_K4S)

# Precompute for each edge, which K4s it belongs to
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, edges in enumerate(_K4S):
    for e in edges:
        _EDGE_TO_K4S[e].append(k4_idx)

def _count_violations(colours):
    """Count number of K4s that span fewer than 5 distinct colours."""
    violations = 0
    for edges in _K4S:
        c0, c1, c2, c3, c4, c5 = (colours[e] for e in edges)
        if len({c0, c1, c2, c3, c4, c5}) < 5:
            violations += 1
    return violations

def _num_colours_used(colours):
    return len(set(colours))

def _is_valid(colours):
    return _count_violations(colours) == 0

def _initial_colouring():
    """(i+j) mod 17 construction."""
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

def _try_merge(colours, c1, c2, violations):
    """Try to merge colour c1 into c2. Returns (new_colours, new_violations) if valid, else None."""
    if c1 == c2:
        return None
    
    # Check if c1 exists
    if c1 not in colours:
        return None
    
    # Find edges with colour c1
    edges_to_change = [i for i in range(120) if colours[i] == c1]
    
    # Create a copy
    new_colours = list(colours)
    
    # For each edge being changed, update violations incrementally
    new_violations = violations
    for e in edges_to_change:
        old_c = new_colours[e]
        new_c = c2
        new_colours[e] = new_c
        
        # Update violations for K4s containing edge e
        for k4_idx in _EDGE_TO_K4S[e]:
            edges = _K4S[k4_idx]
            # Count distinct colours before and after
            # Before: use old_c for this edge
            old_colors = set()
            for ee in edges:
                if ee == e:
                    old_colors.add(old_c)
                else:
                    old_colors.add(new_colours[ee])
            
            # After: use new_c for this edge
            new_colors = set()
            for ee in edges:
                if ee == e:
                    new_colors.add(new_c)
                else:
                    new_colors.add(new_colours[ee])
            
            was_violation = len(old_colors) < 5
            is_violation = len(new_colors) < 5
            
            if was_violation and not is_violation:
                new_violations -= 1
            elif not was_violation and is_violation:
                new_violations += 1
    
    if new_violations == 0:
        return (new_colours, new_violations)
    return None

def _reduce_colours_by_merging(colours, violations, deadline):
    """Iteratively try to merge colours to reduce count."""
    current = list(colours)
    current_violations = violations
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used_colours = sorted(set(current))
        
        # Try all pairs of used colours
        for i in range(len(used_colours)):
            for j in range(i + 1, len(used_colours)):
                c1 = used_colours[i]
                c2 = used_colours[j]
                result = _try_merge(current, c1, c2, current_violations)
                if result is not None:
                    current, current_violations = result
                    improved = True
                    break
            if improved:
                break
    return current, current_violations

def _simulated_annealing(colours, violations, deadline):
    """
    SA to escape local minima in the merging process.
    Move: Change colour of a random edge to a random other used colour.
    Objective: violations * 1000 + num_colours
    Uses O(1) incremental violation updates.
    """
    current = list(colours)
    current_violations = violations
    current_colours = _num_colours_used(current)
    current_obj = current_violations * 1000 + current_colours
    
    best = list(current)
    best_obj = current_obj
    
    T = 100.0
    T_min = 1.0
    cooling = 0.999
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Pick a random edge
        idx = rng.randrange(120)
        old_c = current[idx]
        
        # Pick a new colour from the currently used colours (excluding old_c)
        used = set(current)
        if len(used) <= 1:
            break
            
        candidates = [c for c in used if c != old_c]
        if not candidates:
            continue
            
        new_c = rng.choice(candidates)
        
        # Update violations incrementally
        old_violations = current_violations
        new_violations = old_violations
        
        # For each K4 containing this edge, check if violation status changes
        for k4_idx in _EDGE_TO_K4S[idx]:
            edges = _K4S[k4_idx]
            
            # Count distinct colours with old_c
            old_colors = set()
            for ee in edges:
                if ee == idx:
                    old_colors.add(old_c)
                else:
                    old_colors.add(current[ee])
            
            # Count distinct colours with new_c
            new_colors = set()
            for ee in edges:
                if ee == idx:
                    new_colors.add(new_c)
                else:
                    new_colors.add(current[ee])
            
            was_violation = len(old_colors) < 5
            is_violation = len(new_colors) < 5
            
            if was_violation and not is_violation:
                new_violations -= 1
            elif not was_violation and is_violation:
                new_violations += 1
        
        # Check if the colour change affects the number of colours used
        old_num_colours = current_colours
        # If old_c is no longer used, colours decrease by 1
        # If new_c was not used, colours increase by 1
        # But we need to be careful: if old_c == new_c, no change (already handled)
        
        # Count if old_c appears elsewhere
        old_c_still_used = False
        for i in range(120):
            if i != idx and current[i] == old_c:
                old_c_still_used = True
                break
        
        # Count if new_c appears elsewhere
        new_c_already_used = False
        for i in range(120):
            if i != idx and current[i] == new_c:
                new_c_already_used = True
                break
        
        if not old_c_still_used and new_c_already_used:
            new_num_colours = old_num_colours - 1
        elif old_c_still_used and not new_c_already_used:
            new_num_colours = old_num_colours + 1
        else:
            new_num_colours = old_num_colours
        
        current[idx] = new_c
        
        new_obj = new_violations * 1000 + new_num_colours
        
        delta = new_obj - current_obj
        
        # Accept if better, or with probability exp(-delta/T)
        if delta <= 0:
            accept = True
        else:
            accept = rng.random() < (2.718281828 ** (-delta / T))
            
        if accept:
            current_obj = new_obj
            current_violations = new_violations
            current_colours = new_num_colours
            
            if new_obj < best_obj:
                best_obj = new_obj
                best = list(current)
        else:
            current[idx] = old_c
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Periodic merge attempt
        if rng.random() < 0.01:
            if current_violations == 0:
                new_merged, new_viol = _reduce_colours_by_merging(current, current_violations, time.time() + 0.1)
                if new_viol == 0:
                    m_col = _num_colours_used(new_merged)
                    m_obj = m_col
                    if m_obj < best_obj:
                        best_obj = m_obj
                        best = new_merged
                        current = list(best)
                        current_obj = best_obj
                        current_violations = new_viol
                        current_colours = m_col
                        T = 100.0  # Reheat

    return best, best_obj

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    # Start from valid 17-colour construction
    colours = _initial_colouring()
    
    # Verify validity
    initial_violations = _count_violations(colours)
    if initial_violations != 0:
        # Should not happen, but just in case
        pass
        
    best = list(colours)
    best_obj = _num_colours_used(colours)
    
    # Phase 1: Greedy merging
    merged, merged_viol = _reduce_colours_by_merging(colours, initial_violations, deadline)
    if merged_viol == 0:
        m_obj = _num_colours_used(merged)
        if m_obj < best_obj:
            best = merged
            best_obj = m_obj
            
    # Phase 2: Simulated Annealing
    if time.time() < deadline:
        sa_result, sa_obj = _simulated_annealing(best, _count_violations(best), deadline)
        # sa_obj includes violation penalty, so we need to check validity
        if _count_violations(sa_result) == 0:
            sa_colours = _num_colours_used(sa_result)
            if sa_colours < best_obj:
                best = sa_result
                best_obj = sa_colours
                
    # Final check
    if _count_violations(best) != 0:
        best = _initial_colouring()
        
    return best

# Pre-compute K4s at module load time for speed
_K4S = _precompute_k4s()
_NUM_K4S = len(_K4S)
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, edges in enumerate(_K4S):
    for e in edges:
        _EDGE_TO_K4S[e].append(k4_idx)
