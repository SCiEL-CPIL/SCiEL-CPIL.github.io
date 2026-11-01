"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill
    
    # Generate candidate vectors: all vectors with entries in {-1, 0, 1}
    # We'll use a smarter approach: generate vectors with small norms first
    # since we need min pairwise distance >= max norm
    
    # Strategy: Use vectors from the D11 lattice shortest vectors
    # These are vectors with exactly two non-zero entries of ±1
    # But we can also use vectors with one non-zero entry (norm 1)
    # and vectors with three non-zero entries (norm sqrt(3))
    
    # Let's build a diverse set of candidate vectors
    candidates = []
    
    # Type 1: unit vectors (norm 1)
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign
            candidates.append(v)
    
    # Type 2: vectors with two non-zero entries (norm sqrt(2))
    for i in range(D):
        for j in range(i+1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Type 3: vectors with three non-zero entries (norm sqrt(3))
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # We need to select a subset S such that for all u, v in S, u != v:
    # ||u - v||^2 >= max_{w in S} ||w||^2
    # 
    # This is a complex constraint. Let's use a greedy + local search approach.
    # 
    # Key insight: if we restrict to vectors of the same norm, the constraint becomes
    # ||u - v||^2 >= ||u||^2 for all u, v in S
    # i.e., <u, v> <= 0 for all u, v in S (when norms are equal)
    # Actually: ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> >= ||u||^2
    # So ||v||^2 - 2<u,v> >= 0, i.e., <u,v> <= ||v||^2/2
    # For equal norms: <u,v> <= ||u||^2/2
    
    # Let's try multiple norm levels and combine them carefully
    
    # First, let's find a large set among vectors of a single norm level
    # then try to add vectors from other levels
    
    def check_validity(selected_indices, candidates, norms_sq):
        """Check if the selected set satisfies the constraint."""
        if len(selected_indices) == 0:
            return True
        sel = candidates[selected_indices]
        sel_norms_sq = norms_sq[selected_indices]
        max_norm_sq = np.max(sel_norms_sq)
        n = len(sel)
        # Compute all pairwise distances squared
        for i in range(n):
            for j in range(i+1, n):
                diff = sel[i] - sel[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < max_norm_sq - 1e-9:
                    return False
        return True
    
    def get_max_norm_sq(selected_indices, norms_sq):
        if len(selected_indices) == 0:
            return 0
        return np.max(norms_sq[selected_indices])
    
    def is_compatible(vec_idx, selected_indices, candidates, norms_sq):
        """Check if adding vec_idx to selected set maintains validity."""
        if len(selected_indices) == 0:
            return True
        new_max_norm_sq = max(norms_sq[vec_idx], get_max_norm_sq(selected_indices, norms_sq))
        v = candidates[vec_idx]
        for si in selected_indices:
            diff = v - candidates[si]
            dist_sq = np.dot(diff, diff)
            if dist_sq < new_max_norm_sq - 1e-9:
                return False
        return True
    
    # Greedy construction with multiple random orders
    best_set = []
    
    # Try different strategies: start with high-norm vectors or low-norm
    # Actually, let's try to maximize count, so lower norms are easier to pack
    
    # Strategy: greedy with random tie-breaking, multiple restarts
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        random.shuffle(candidates)
        selected = []
        selected_set = set()
        
        # Sort by norm squared (ascending) to pack more
        cand_order = sorted(range(n_cand), key=lambda i: norms_sq[i])
        random.shuffle(cand_order)  # randomize ties
        
        for idx in cand_order:
            if time.time() > deadline - 10:
                break
            if is_compatible(idx, selected, candidates, norms_sq):
                selected.append(idx)
                selected_set.add(idx)
        
        if len(selected) > len(best_set):
            best_set = selected[:]
        
        # Local search: try to add more or swap
        if time.time() > deadline - 5:
            break
            
        # Try adding vectors that were skipped
        current_selected = set(best_set)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for idx in range(n_cand):
                if idx in current_selected:
                    continue
                if is_compatible(idx, list(current_selected), candidates, norms_sq):
                    current_selected.add(idx)
                    improved = True
                    break
        
        if len(current_selected) > len(best_set):
            best_set = list(current_selected)
    
    # Convert to array
    if len(best_set) == 0:
        # Fallback: just return a few valid vectors
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    result = candidates[best_set]
    return result
# EVOLVE-BLOCK-END
