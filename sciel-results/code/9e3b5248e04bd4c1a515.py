"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = 140.0
    
    # Generate candidates of weight 2, 3, 4, 5
    candidates = []
    norms_sq = []
    
    for w in [2, 3, 4, 5]:
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = np.zeros(D, dtype=np.int64)
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
                norms_sq.append(w)
    
    candidates = np.array(candidates)
    norms_sq = np.array(norms_sq)
    n_cands = len(candidates)
    
    def greedy_clique_fast(order):
        selected = []
        selected_matrix = None
        selected_norms = []
        
        for idx in order:
            if time.time() > deadline:
                break
            
            v = candidates[idx]
            v_norm_sq = norms_sq[idx]
            
            if not selected:
                selected.append(v)
                selected_matrix = v.reshape(1, -1)
                selected_norms.append(v_norm_sq)
                continue
            
            dots = selected_matrix @ v
            dist_sqs = np.array(selected_norms) + v_norm_sq - 2 * dots
            u_norms_arr = np.array(selected_norms)
            max_norms = np.maximum(u_norms_arr, v_norm_sq)
            
            if np.all(dist_sqs >= max_norms):
                selected.append(v)
                selected_norms.append(v_norm_sq)
                selected_matrix = np.vstack([selected_matrix, v.reshape(1, -1)])
        
        return selected

    rng = np.random.default_rng(42)
    best_clique = []
    best_size = 0
    
    while time.time() < deadline:
        order = rng.permutation(n_cands)
        clique = greedy_clique_fast(order)
        if len(clique) > best_size:
            best_size = len(clique)
            best_clique = clique
        
    return np.array(best_clique, dtype=np.int64)

# EVOLVE-BLOCK-END
