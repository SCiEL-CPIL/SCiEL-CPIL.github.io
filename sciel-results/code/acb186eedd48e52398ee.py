"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import log, exp

def _compute_score(U):
    """Compute the score for a given set U using vectorized operations."""
    U = sorted(set(U))
    if 0 not in U:
        U = [0] + U
    n = len(U)
    if n == 0 or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    U_arr = np.array(U, dtype=np.int64)
    
    # Compute sumset U+U
    sums = U_arr[:, None] + U_arr[None, :]
    sumset_size = len(np.unique(sums))
    
    # Compute difference set U-U
    diffs = U_arr[:, None] - U_arr[None, :]
    diffset_size = len(np.unique(diffs))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    
    denominator = log(2 * max_u + 1)
    if denominator <= 0:
        return 0.0
    
    return 1.0 + log(ratio) / denominator

def _sa_search(deadline):
    """Simulated annealing search for optimal set U."""
    # Start with known good structure
    best_U = [0, 1, 4, 9]
    best_score = _compute_score(best_U)
    
    current_U = list(best_U)
    current_score = best_score
    
    # Annealing parameters
    T_start = 2.0
    T_end = 0.001
    cooling = 0.998
    
    T = T_start
    step = 0
    max_steps = 100000
    
    while time.time() < deadline and step < max_steps and T > T_end:
        step += 1
        
        # Generate perturbation
        U = list(current_U)
        pert_type = np.random.randint(0, 5)
        
        if pert_type == 0:  # Add an element
            max_u = max(U) if U else 0
            if np.random.random() < 0.5 and len(U) > 1:
                idx = np.random.randint(0, len(U))
                new_val = U[idx] + np.random.randint(1, 5)
            else:
                new_val = np.random.randint(1, max_u + 10)
            if new_val > 0 and new_val not in U:
                U.append(int(new_val))
        elif pert_type == 1:  # Remove an element (not 0)
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                U.pop(idx)
        elif pert_type == 2:  # Shift a non-zero element
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                U[idx] += np.random.randint(-5, 6)
                if U[idx] < 0:
                    U[idx] = 1
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                U = sorted(U)
        elif pert_type == 3:  # Randomize one element
            if len(U) > 1:
                idx = np.random.randint(1, len(U))
                max_u = max(U)
                U[idx] = int(np.random.randint(1, max_u + 10))
                U = list(set(U))
                if 0 not in U:
                    U.append(0)
                U = sorted(U)
        else:  # Swap two non-zero elements' positions
            if len(U) > 2:
                nonzero_indices = [i for i in range(len(U)) if U[i] != 0]
                if len(nonzero_indices) >= 2:
                    idx1, idx2 = np.random.choice(nonzero_indices, 2, replace=False)
                    U[idx1], U[idx2] = U[idx2], U[idx1]
                    U = sorted(set(U))
                    if 0 not in U:
                        U.append(0)
                    U = sorted(U)
        
        if not U or 0 not in U or len(U) < 2:
            continue
        
        new_score = _compute_score(U)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score >= current_score:
            current_U = U
            current_score = new_score
            if new_score > best_score:
                best_U = list(U)
                best_score = new_score
        else:
            delta = current_score - new_score
            if delta < 100 * T:
                if np.random.random() < exp(-delta / T):
                    current_U = U
                    current_score = new_score
        
        T *= cooling
    
    return best_U, best_score

def construct():
    # Deadline: 120s limit, use 100s for safety
    deadline = time.time() + 100
    
    # Try simulated annealing
    best_U, best_score = _sa_search(deadline)
    
    # Verify and return
    U = sorted(set(best_U))
    if 0 not in U:
        U = [0] + U
    if not U or max(U) == 0:
        return [0, 1, 4, 9]
    
    return U
# EVOLVE-BLOCK-END
