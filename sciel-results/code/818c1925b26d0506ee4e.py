"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import cKDTree

# Deadline guard
DEADLINE = 115.0  # seconds, safety margin before 120s limit
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _pairwise_min_dist_and_max_norm(vectors):
    """Compute minimum pairwise distance and maximum norm."""
    if len(vectors) < 2:
        norms = np.linalg.norm(vectors, axis=1)
        return 0.0, norms.max() if len(norms) > 0 else 0.0
    
    # Use cKDTree for efficient nearest neighbor search
    tree = cKDTree(vectors)
    dists, _ = tree.query(vectors, k=2)  # k=2 because first is self
    min_pairwise = dists[:, 1].min()
    max_norm = np.linalg.norm(vectors, axis=1).max()
    return min_pairwise, max_norm


def _check_validity(vectors):
    """Check if vectors satisfy min pairwise distance >= max norm."""
    if len(vectors) < 2:
        return True
    min_dist, max_norm = _pairwise_min_dist_and_max_norm(vectors)
    return min_dist >= max_norm - 1e-9


def _generate_candidate_vectors(max_coord=3):
    """Generate candidate vectors with small integer coordinates."""
    candidates = []
    # Start with unit vectors and their negatives
    for i in range(D):
        for sign in (1, -1):
            v = [0] * D
            v[i] = sign
            candidates.append(tuple(v))
    
    # Add vectors with two non-zero entries
    for i in range(D):
        for j in range(i + 1, D):
            for si in range(-max_coord, max_coord + 1):
                if si == 0:
                    continue
                for sj in range(-max_coord, max_coord + 1):
                    if sj == 0:
                        continue
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(tuple(v))
    
    # Add vectors with three non-zero entries (small coords)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        for sk in (-1, 1):
                            v = [0] * D
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(tuple(v))
    
    return np.array(list(set(candidates)), dtype=np.int64)


def _greedy_select(candidates, max_vectors=None):
    """Greedily select vectors maintaining min pairwise distance >= max norm."""
    selected = []
    candidate_list = list(candidates)
    
    # Sort by norm descending to prioritize larger vectors? Or ascending?
    # Let's try: sort by norm, pick one, then filter incompatible ones
    norms = np.linalg.norm(candidate_list, axis=1)
    order = np.argsort(norms)  # ascending norm
    
    for idx in order:
        if _remaining() < 5.0:
            break
        if max_vectors is not None and len(selected) >= max_vectors:
            break
        
        v = candidate_list[idx]
        v_norm = norms[idx]
        
        # Check against all selected vectors
        compatible = True
        for sv in selected:
            dist = np.linalg.norm(v - sv)
            # Need dist >= max(selected_norms, v_norm)
            sv_norm = np.linalg.norm(sv)
            if dist < max(sv_norm, v_norm) - 1e-9:
                compatible = False
                break
        
        if compatible:
            selected.append(v)
    
    return np.array(selected, dtype=np.int64) if selected else np.empty((0, D), dtype=np.int64)


def _local_improve(vectors, candidates, max_iterations=100):
    """Try to add more vectors or swap to improve the set."""
    if len(vectors) == 0:
        return vectors
    
    selected = list(vectors)
    selected_norms = np.linalg.norm(selected, axis=1)
    max_norm = selected_norms.max()
    
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        if _remaining() < 5.0:
            break
        improved = False
        iteration += 1
        
        # Try to add any candidate that is compatible
        for v in candidates:
            v_norm = np.linalg.norm(v)
            new_max_norm = max(max_norm, v_norm)
            
            compatible = True
            for sv in selected:
                dist = np.linalg.norm(v - sv)
                if dist < new_max_norm - 1e-9:
                    compatible = False
                    break
            
            if compatible:
                selected.append(v)
                selected_norms = np.append(selected_norms, v_norm)
                max_norm = new_max_norm
                improved = True
                break
    
    return np.array(selected, dtype=np.int64)


def construct():
    # Generate candidate pool
    candidates = _generate_candidate_vectors(max_coord=2)
    
    if _remaining() < 10.0:
        # Fallback: just return unit vectors
        vectors = []
        for i in range(D):
            for sign in (1, -1):
                v = [0] * D
                v[i] = sign
                vectors.append(v)
        return np.array(vectors, dtype=np.int64)
    
    # Greedy selection
    selected = _greedy_select(candidates)
    
    if _remaining() < 10.0:
        return selected if len(selected) > 0 else np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    # Local improvement
    selected = _local_improve(selected, candidates, max_iterations=50)
    
    # Final validity check
    if len(selected) >= 2:
        if not _check_validity(selected):
            # Fall back to a known valid small set
            vectors = []
            for i in range(D):
                for sign in (1, -1):
                    v = [0] * D
                    v[i] = sign
                    vectors.append(v)
            return np.array(vectors, dtype=np.int64)
    
    if len(selected) == 0:
        return np.array([[1,0,0,0,0,0,0,0,0,0,0]], dtype=np.int64)
    
    return selected
# EVOLVE-BLOCK-END
