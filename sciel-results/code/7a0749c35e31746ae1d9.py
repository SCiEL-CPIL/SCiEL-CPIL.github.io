# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 570  # Safety margin for 600s limit
    
    # Pre-compute grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Pre-compute squared norms for all grid points
    grid_sums = (grid ** 2).sum(axis=1)
    
    # Pre-compute all 4-combinations for various set sizes
    # We'll cache these to avoid regenerating
    combo_cache = {}
    def get_combos(m):
        if m not in combo_cache:
            combo_cache[m] = np.array(list(itertools.combinations(range(m), 4)))
        return combo_cache[m]
    
    def can_add_to_set(set_indices, p_idx):
        """
        Check if adding grid[p_idx] to the set defined by set_indices keeps it valid.
        """
        m = len(set_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        p_sum = grid_sums[p_idx]
        
        pts = grid[set_indices]
        sums = grid_sums[set_indices]
        
        # Build the 4-column matrix W where W[i] = [q_i - p, |q_i|^2 - |p|^2]
        diffs = pts - p
        diff_sums = sums - p_sum
        W = np.hstack([diffs, diff_sums[:, None]])
        
        # Get precomputed combinations
        combos = get_combos(m)
        
        # Extract submatrices and compute determinants
        sub = W[combos]
        
        # Compute determinants in batches for efficiency
        dets = np.linalg.det(sub)
        
        return not np.any(np.abs(dets) < 0.5)
    
    # --- Initial Greedy Construction ---
    np.random.seed(42)
    random.seed(42)
    
    best_set_indices = np.array([], dtype=np.int64)
    
    # Run multiple greedy restarts with different orderings
    for restart in range(20):
        if time.time() > deadline - 30:
            break
            
        order = np.random.permutation(N)
        current_indices = []
        
        for p_idx in order:
            if len(current_indices) < 4:
                current_indices.append(p_idx)
                continue
            
            if can_add_to_set(np.array(current_indices), p_idx):
                current_indices.append(p_idx)
                
        if len(current_indices) > len(best_set_indices):
            best_set_indices = np.array(current_indices, dtype=np.int64)
            
    # --- Simulated Annealing ---
    if len(best_set_indices) == 0:
        return np.array([]).reshape(0, 3).astype(np.int64)
        
    current_indices = best_set_indices.copy()
    current_k = len(current_indices)
    best_indices = current_indices.copy()
    best_k = current_k
    
    T = 5.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    all_indices = np.arange(N)
    
    # Track recent moves to avoid cycling
    recent_moves = []
    
    while time.time() < deadline - 10:
        if T < T_min:
            T = T_min
            
        # Select move type with probabilities
        r = random.random()
        
        if r < 0.4:
            # Add move
            current_set_set = set(current_indices.tolist())
            candidates = np.array([i for i in all_indices if i not in current_set_set])
            
            if len(candidates) == 0:
                continue
                
            p_idx = candidates[random.randint(0, len(candidates)-1)]
            
            if can_add_to_set(current_indices, p_idx):
                new_indices = np.append(current_indices, p_idx)
                new_k = current_k + 1
                current_indices = new_indices
                current_k = new_k
                if current_k > best_k:
                    best_k = current_k
                    best_indices = current_indices.copy()
                recent_moves.append(('add', p_idx))
            else:
                recent_moves.append(('add_fail', p_idx))
                
        elif r < 0.6:
            # Remove move
            if current_k > 10:  # Don't go too small
                remove_idx = random.randint(0, current_k - 1)
                new_indices = np.delete(current_indices, remove_idx)
                new_k = current_k - 1
                
                # Accept with probability based on temperature
                if random.random() < np.exp(-1.0 / T):
                    current_indices = new_indices
                    current_k = new_k
                    recent_moves.append(('remove', current_indices[remove_idx] if remove_idx < len(current_indices) else -1))
        else:
            # Swap move
            if current_k > 5:
                remove_idx = random.randint(0, current_k - 1)
                p_remove = current_indices[remove_idx]
                
                current_set_set = set(current_indices.tolist())
                candidates = np.array([i for i in all_indices if i not in current_set_set])
                
                if len(candidates) == 0:
                    continue
                    
                p_add = candidates[random.randint(0, len(candidates)-1)]
                
                # Check if adding p_add to (set - p_remove) is valid
                mask = current_indices != p_remove
                remaining_indices = current_indices[mask]
                
                if can_add_to_set(remaining_indices, p_add):
                    new_indices = np.append(remaining_indices, p_add)
                    new_k = current_k
                    
                    # Accept with probability 0.3 to allow random walk
                    if random.random() < 0.3:
                        current_indices = new_indices
                        recent_moves.append(('swap', p_remove, p_add))
                else:
                    recent_moves.append(('swap_fail', p_remove, p_add))
        
        # Limit recent moves history
        if len(recent_moves) > 100:
            recent_moves = recent_moves[-50:]
            
        # Cool down
        T *= cooling_rate
    
    # --- Final greedy pass: try to add any remaining valid points ---
    current_set_set = set(best_indices.tolist())
    candidates = np.array([i for i in all_indices if i not in current_set_set])
    
    # Sort candidates by some heuristic (e.g., distance to existing points)
    if len(candidates) > 0:
        # Compute sum of distances to existing points as a heuristic
        existing_pts = grid[best_indices]
        cand_pts = grid[candidates]
        # Sum of squared distances
        dists = np.sum((cand_pts[:, None, :] - existing_pts[None, :, :]) ** 2, axis=2)
        avg_dists = np.mean(dists, axis=1)
        # Prefer points that are "spread out" - higher average distance
        order = np.argsort(-avg_dists)
        
        for p_idx in candidates[order]:
            if time.time() > deadline - 5:
                break
            if can_add_to_set(best_indices, p_idx):
                best_indices = np.append(best_indices, p_idx)
                best_k += 1
    
    # Return the best found set
    best_points = grid[best_indices]
    return best_points.astype(np.int64)

# EVOLVE-BLOCK-END
