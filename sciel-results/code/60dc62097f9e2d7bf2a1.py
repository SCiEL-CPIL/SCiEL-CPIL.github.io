"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def construct():
    deadline = time.time() + 115
    
    # Generate all non-zero vectors with components in {-1, 0, 1}
    all_vectors = []
    all_norms_sq = []
    
    for combo in itertools.product([-1, 0, 1], repeat=D):
        if all(c == 0 for c in combo):
            continue
        v = np.array(combo, dtype=np.int64)
        n_sq = np.dot(v, v)
        all_vectors.append(v)
        all_norms_sq.append(n_sq)
        
    all_vectors = np.array(all_vectors)
    all_norms_sq = np.array(all_norms_sq)
    
    n_candidates = len(all_vectors)
    indices = np.arange(n_candidates)
    
    best_selected = []
    best_size = 0
    
    # Perform multiple restarts
    while time.time() < deadline:
        np.random.shuffle(indices)
        
        selected = []
        selected_norms = []
        max_norm_sq = 0
        
        # Greedy selection
        for idx in indices:
            v = all_vectors[idx]
            n_sq = all_norms_sq[idx]
            new_max_norm_sq = max(max_norm_sq, n_sq)
            
            if not selected:
                selected.append(v)
                selected_norms.append(n_sq)
                max_norm_sq = new_max_norm_sq
            else:
                # Vectorized check
                sel_arr = np.array(selected)
                dots = sel_arr @ v
                dist_sq = selected_norms + n_sq - 2 * dots
                
                if np.all(dist_sq >= new_max_norm_sq):
                    selected.append(v)
                    selected_norms.append(n_sq)
                    max_norm_sq = new_max_norm_sq
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected
            
    return np.array(best_selected, dtype=np.int64)

# EVOLVE-BLOCK-END
