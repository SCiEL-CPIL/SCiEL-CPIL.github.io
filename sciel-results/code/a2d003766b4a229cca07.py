"""
Strategy: Direct Integer Linear Programming (ILP) using scipy.optimize.milp,
followed by a post-solve local search with 1-swap and 2-swap moves using
bitmask-based conflict checking to escape local optima.
"""
# EVOLVE-BLOCK-START
import time
from math import gcd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 255  # Numbers 2 to 256
    # Index i in array corresponds to number i + 2
    
    # 1. Precompute conflicts
    conflict_pairs = []
    for i in range(N):
        a = i + 2
        for j in range(i + 1, N):
            b = j + 2
            g = gcd(a, b)
            lcm_val = a * b // g
            if lcm_val <= 256:
                conflict_pairs.append((i, j))
    
    num_vars = N
    num_constraints = len(conflict_pairs)
    
    # Build conflict bitmask for fast compatibility checks
    # conflict_mask[i] is a bitmask of all j that conflict with i
    conflict_mask = [0] * num_vars
    for (i, j) in conflict_pairs:
        conflict_mask[i] |= (1 << j)
        conflict_mask[j] |= (1 << i)
    
    # Objective: Maximize sum(1/i * x_i)
    c = np.zeros(num_vars)
    for i in range(num_vars):
        c[i] = -1.0 / (i + 2)
        
    # Constraints: x_i + x_j <= 1
    A_ub = np.zeros((num_constraints, num_vars))
    for k, (i, j) in enumerate(conflict_pairs):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    b_ub = np.ones(num_constraints)
    
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    integrality = np.ones(num_vars)
    
    if num_constraints == 0:
        return list(range(2, 257))
        
    constraints = LinearConstraint(A_ub, ub=b_ub)
    
    if time.time() > deadline:
        return _greedy_fallback()
        
    try:
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            return _greedy_fallback()
            
        res = milp(
            c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options={"time_limit": remaining_time}
        )
        
        if res.status in [0, 3]:
            if res.x is not None:
                selected_indices = []
                for i in range(num_vars):
                    if res.x[i] > 0.5:
                        selected_indices.append(i + 2)
                
                if selected_indices:
                    if _is_valid(selected_indices):
                        # Apply local search polish
                        remaining = deadline - time.time()
                        if remaining > 1.0:
                            polished = _local_search_polish(
                                selected_indices, conflict_mask, 
                                time_limit=remaining - 0.5
                            )
                            if _is_valid(polished):
                                return polished
                        return selected_indices
                
        return _greedy_fallback()
        
    except Exception as e:
        return _greedy_fallback()

def _is_valid(subset):
    """Verify that the subset satisfies lcm(a,b) > 256 for all distinct pairs."""
    if not subset:
        return False
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            a = subset[i]
            b = subset[j]
            g = gcd(a, b)
            if a * b // g <= 256:
                return False
    return True

def _local_search_polish(subset, conflict_mask, time_limit=10.0):
    """
    Local search with 1-add, 1-swap, and 2-swap moves to improve the ILP solution.
    Uses bitmask-based conflict checking for speed.
    """
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Build index mapping: number -> index (0-based, where index i = number i+2)
    # Convert subset to set of indices
    selected_set = set()
    for num in subset:
        idx = num - 2
        if 0 <= idx < 255:
            selected_set.add(idx)
    
    # Precompute weights
    weights = [1.0 / (i + 2) for i in range(255)]
    
    # Sort unselected candidates by weight descending for early termination
    all_indices = list(range(255))
    all_indices_sorted_by_weight = sorted(all_indices, key=lambda x: -weights[x])
    
    # Current score
    current_score = sum(weights[i] for i in selected_set)
    
    # Build a bitmask representation of selected set for fast conflict checks
    selected_mask = 0
    for i in selected_set:
        selected_mask |= (1 << i)
    
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        
        # 1-add: try adding a single unselected node if compatible
        if time.time() < deadline:
            for u in all_indices_sorted_by_weight:
                if u in selected_set:
                    continue
                if conflict_mask[u] & selected_mask:
                    continue
                # u is compatible with all selected, add it
                selected_set.add(u)
                selected_mask |= (1 << u)
                current_score += weights[u]
                improved = True
                break
        
        # 1-swap: try replacing one selected node with one unselected node
        if not improved and (time.time() < deadline):
            # Sort selected by weight ascending (try to replace low-weight first)
            sel_sorted = sorted(selected_set, key=lambda x: weights[x])
            for sel_idx in sel_sorted:
                if time.time() > deadline:
                    break
                other_selected = selected_mask & ~(1 << sel_idx)
                # Try candidates in descending weight order
                for u in all_indices_sorted_by_weight:
                    if u in selected_set:
                        continue
                    if conflict_mask[u] & other_selected:
                        continue
                    if weights[u] > weights[sel_idx]:
                        # Perform swap
                        selected_set.remove(sel_idx)
                        selected_set.add(u)
                        selected_mask &= ~(1 << sel_idx)
                        selected_mask |= (1 << u)
                        current_score += weights[u] - weights[sel_idx]
                        improved = True
                        break
                if improved:
                    break
        
        # 2-swap: try replacing two selected nodes with two unselected nodes
        if not improved and (time.time() < deadline):
            sel_list = sorted(selected_set, key=lambda x: weights[x])
            n_sel = len(sel_list)
            
            for a in range(n_sel):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n_sel):
                    if time.time() > deadline:
                        break
                    sel_a = sel_list[a]
                    sel_b = sel_list[b]
                    other_selected = selected_mask & ~(1 << sel_a) & ~(1 << sel_b)
                    gain_needed = weights[sel_a] + weights[sel_b]
                    
                    best_pair_gain = -1
                    best_pair = None
                    
                    # Iterate candidates in descending weight order
                    # Once weights[u] + max_possible_v <= gain_needed, we can break
                    for u in all_indices_sorted_by_weight:
                        if u in selected_set:
                            continue
                        # Upper bound: even the best v can't beat gain_needed
                        # The best possible v weight is weights[2] = 0.5
                        if weights[u] + 0.5 <= gain_needed:
                            break
                        if conflict_mask[u] & other_selected:
                            continue
                        for v in all_indices_sorted_by_weight:
                            if v <= u:
                                continue
                            if v in selected_set:
                                continue
                            # Check v compatible with other_selected
                            if conflict_mask[v] & other_selected:
                                continue
                            # Check u and v compatible with each other
                            if (conflict_mask[u] >> v) & 1:
                                continue
                            pair_gain = weights[u] + weights[v]
                            if pair_gain > gain_needed and pair_gain > best_pair_gain:
                                best_pair_gain = pair_gain
                                best_pair = (u, v)
                            # Early termination: no better pair possible from this u
                            # since v is sorted descending, remaining v's have lower weight
                            break
                    
                    if best_pair is not None:
                        u, v = best_pair
                        selected_set.discard(sel_a)
                        selected_set.discard(sel_b)
                        selected_set.add(u)
                        selected_set.add(v)
                        selected_mask &= ~(1 << sel_a)
                        selected_mask &= ~(1 << sel_b)
                        selected_mask |= (1 << u)
                        selected_mask |= (1 << v)
                        current_score += best_pair_gain - gain_needed
                        improved = True
                        break
                if improved:
                    break
    
    # Convert back to numbers
    result = [i + 2 for i in sorted(selected_set)]
    return result if result else [2]

def _greedy_fallback():
    """
    Simple greedy fallback if ILP fails or times out without a good incumbent.
    Sort by weight descending, add if compatible with all current members.
    """
    start_time = time.time()
    deadline = start_time + 5.0
    
    candidates = list(range(2, 257))
    candidates.sort(key=lambda x: -1.0/x)
    
    result = []
    for num in candidates:
        if time.time() > deadline:
            break
        compatible = True
        for existing in result:
            g = gcd(num, existing)
            if num * existing // g <= 256:
                compatible = False
                break
        if compatible:
            result.append(num)
            
    if not result:
        return [2]
        
    return result

# EVOLVE-BLOCK-END
