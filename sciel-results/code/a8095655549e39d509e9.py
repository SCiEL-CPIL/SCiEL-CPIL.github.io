# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates using a spectral optimization approach on low-weight vectors.
    
    Strategy:
    1. Generate a pool of integer vectors with weights 1, 2, and 3 (entries in {-1, 0, 1}).
    2. Build a compatibility graph where an edge exists if the squared distance between two vectors is >= the maximum of their squared norms.
    3. Use the principal eigenvector of the adjacency matrix to rank nodes.
    4. Use a multi-pass greedy clique construction with multiple starting seeds and local improvement.
    
    Returns:
        points: np.ndarray of shape (num_points, 11)
    """
    d = 11
    
    # 1. Generate pool of low-weight vectors
    candidates = []
    
    # Weight 1
    for i in range(d):
        for sign in [-1, 1]:
            v = np.zeros(d, dtype=np.int64)
            v[i] = sign
            candidates.append(v)
            
    # Weight 2
    for i in range(d):
        for j in range(i + 1, d):
            for si in [-1, 1]:
                for sj in [-1, 1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
                    
    # Weight 3
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        for sk in [-1, 1]:
                            v = np.zeros(d, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
                            
    n_candidates = len(candidates)
    candidates = np.array(candidates, dtype=np.int64)
    
    # 2. Build compatibility matrix
    # dist_sq[i, j] = ||v_i - v_j||^2
    # norm_sq[i] = ||v_i||^2
    # Condition: dist_sq[i, j] >= max(norm_sq[i], norm_sq[j])
    
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # Compute squared distances
    # (a-b)^2 = a^2 + b^2 - 2ab
    # dist_sq[i,j] = norms_sq[i] + norms_sq[j] - 2 * dot(v_i, v_j)
    dots = candidates @ candidates.T
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Max norm squared for each pair
    max_norms_sq = np.maximum(norms_sq[:, None], norms_sq[None, :])
    
    # Compatibility matrix (True if compatible)
    # Diagonal is False (distinct points)
    compatible = dist_sq >= max_norms_sq
    np.fill_diagonal(compatible, False)
    
    # 3. Spectral Optimization
    # Convert boolean to float for eigen decomposition
    A = compatible.astype(np.float64)
    
    try:
        # Compute eigenvalues and eigenvectors
        # We only need the principal eigenvector
        vals, vecs = np.linalg.eigh(A)
        # Principal eigenvector is the one corresponding to the largest eigenvalue
        # eigh returns sorted ascending, so last one is largest
        principal_vec = vecs[:, -1]
    except Exception:
        # Fallback to degree if eigen fails
        principal_vec = np.sum(compatible, axis=1).astype(np.float64)
        
    # Rank candidates by magnitude of principal eigenvector component
    order = np.argsort(-np.abs(principal_vec))
    
    # 4. Multi-pass greedy clique construction
    # Try multiple orderings: spectral order, degree order, and reversed spectral order
    degree_order = np.argsort(-np.sum(compatible, axis=1))
    reversed_spectral = order[::-1]
    
    def greedy_clique(ordering):
        """Greedy clique construction given a fixed ordering."""
        selected = []
        selected_set = set()
        
        for idx in ordering:
            if idx in selected_set:
                continue
            # Check if candidate is compatible with all selected
            is_compatible = True
            for sel_idx in selected:
                if not compatible[idx, sel_idx]:
                    is_compatible = False
                    break
            
            if is_compatible:
                selected.append(idx)
                selected_set.add(idx)
        
        return selected
    
    # Try multiple orderings
    best_clique = []
    
    for ordering in [order, degree_order, reversed_spectral]:
        clique = greedy_clique(ordering)
        if len(clique) > len(best_clique):
            best_clique = clique
    
    # Local improvement: try to add more points
    # For each point not in the clique, check if it can be added
    current_set = set(best_clique)
    improved = True
    while improved:
        improved = False
        for idx in range(n_candidates):
            if idx in current_set:
                continue
            is_compatible = True
            for sel_idx in best_clique:
                if not compatible[idx, sel_idx]:
                    is_compatible = False
                    break
            if is_compatible:
                best_clique.append(idx)
                current_set.add(idx)
                improved = True
    
    selected_indices = best_clique
    
    if not selected_indices:
        # Fallback
        selected_indices = [0, 1]
        
    points = candidates[selected_indices]
    
    # Final validation and rounding (though ints are exact)
    points = np.around(points).astype(np.int64)
    
    return points

# EVOLVE-BLOCK-END
