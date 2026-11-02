"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit
    
    # Generate weight-4 vectors with symmetry breaking:
    # For each set of 4 positions, only keep sign patterns where the first non-zero
    # coordinate is positive. This reduces the pool by roughly half.
    # Actually, let's use a stronger symmetry: fix the first coordinate to +1 if it's non-zero.
    # This gives us C(11,4) * 2^3 = 330 * 8 = 2640 vectors (half of 5280).
    
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        # Fix first position's sign to +1
        first_pos = positions[0]
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            v[first_pos] = 1
            for pos, s in zip(positions[1:], signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_candidates = len(candidates)
    
    # Precompute dot product matrix for fast compatibility checks
    # M[i,j] = candidates[i] · candidates[j]
    # Constraint: M[i,j] <= 2 for all pairs in selected set
    M = candidates @ candidates.T
    np.fill_diagonal(M, 4)  # self-dot is 4 (norm^2)
    
    def greedy_min_degree(order, time_limit):
        """
        Greedy min-degree: at each step, pick the candidate with the fewest
        conflicts (i.e., fewest selected vectors it's incompatible with).
        Ties broken by order.
        """
        selected = []
        selected_set = set()
        remaining = list(order)
        
        for _ in range(len(remaining)):
            if time.time() > time_limit:
                break
            if not remaining:
                break
            
            if not selected:
                idx = remaining.pop(0)
                selected.append(idx)
                selected_set.add(idx)
                continue
            
            # Find the remaining candidate with minimum number of conflicts
            # A conflict exists if M[idx, sel] > 2 for some selected sel
            best_idx = None
            best_conflicts = n_candidates + 1
            
            # Vectorized conflict count
            # For each remaining candidate, count how many selected it conflicts with
            # conflict_matrix[i, j] = 1 if M[i, j] > 2 else 0
            # We want to minimize the sum over selected of conflict_matrix[i, selected]
            
            # Get indices of remaining
            rem_arr = np.array(remaining, dtype=np.int64)
            sel_arr = np.array(selected, dtype=np.int64)
            
            # M[rem_arr, sel_arr] has shape (len(rem), len(sel))
            # Conflict if any entry > 2
            sub = M[np.ix_(rem_arr, sel_arr)]
            conflict_counts = np.sum(sub > 2, axis=1)
            
            # Find minimum conflict count
            min_conf = np.min(conflict_counts)
            
            # Among those with minimum conflicts, pick the first in order
            valid_mask = conflict_counts == min_conf
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                break
            
            # Pick the first valid index (preserving order)
            pick_local = valid_indices[0]
            pick_global = remaining[pick_local]
            
            # Check if it's actually compatible (zero conflicts)
            if conflict_counts[pick_local] == 0:
                selected.append(pick_global)
                selected_set.add(pick_global)
                remaining.pop(pick_local)
            else:
                # If minimum conflicts > 0, we might still add it if we want,
                # but for a valid kissing config, we need zero conflicts.
                # So we break if no candidate has zero conflicts.
                break
        
        return selected
    
    best_selected = []
    np.random.seed(42)
    
    # Strategy 1: Fixed order greedy min-degree
    fixed_order = list(range(n_candidates))
    sel = greedy_min_degree(fixed_order, deadline - 10.0)
    if len(sel) > len(best_selected):
        best_selected = sel
    
    # Strategy 2: Multiple random orderings with greedy min-degree
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 10.0:
            break
        perm = np.random.permutation(n_candidates).tolist()
        sel = greedy_min_degree(perm, deadline - 10.0)
        if len(sel) > len(best_selected):
            best_selected = sel
    
    if not best_selected:
        # Fallback: return a minimal valid set
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
    
    return candidates[best_selected]
# EVOLVE-BLOCK-END
