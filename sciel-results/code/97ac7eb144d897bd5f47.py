import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    # Generate candidate vectors with different weights
    # Weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
    
    # Weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2 (since norm^2 = 4, distance^2 = 8 - 2*dot, need distance >= 2 => dot <= 2)
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Use a spectral-based greedy independent set algorithm
    # First, compute the adjacency matrix
    adj = conflicts.astype(np.float64)
    
    # Compute eigenvectors of the adjacency matrix
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.zeros(N4), np.eye(N4)
    
    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Use the principal eigenvector to guide selection
    # Vertices with smaller absolute values in the principal eigenvector are "more peripheral"
    principal = np.abs(eigenvectors[:, 0]) if eigenvectors.shape[1] > 0 else np.ones(N4)
    
    # Greedy MIS with spectral guidance
    alive = np.ones(N4, dtype=bool)
    selected_indices = []
    degrees = conflicts.sum(axis=1).astype(np.int64)
    rng = np.random.default_rng(42)
    
    # Precompute neighbor lists
    neighbor_lists = []
    for i in range(N4):
        neighbor_lists.append(np.where(conflicts[i])[0])
    
    # Score function: combine degree and spectral centrality
    # Lower score = better candidate (we want low degree and low spectral centrality)
    
    while np.any(alive) and time.time() < deadline - 5:
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Compute score for each alive node
        alive_degrees = degrees[alive_indices]
        alive_spectral = principal[alive_indices]
        
        # Combine: we want low degree and low spectral value
        # Use a weighted combination
        scores = 0.7 * alive_degrees / max(1, alive_degrees.max()) + 0.3 * alive_spectral / max(1e-10, alive_spectral.max())
        
        # Select the node with minimum score
        min_score_idx = np.argmin(scores)
        chosen = alive_indices[min_score_idx]
        
        selected_indices.append(chosen)
        
        # Collect all nodes to remove: chosen + its alive neighbors
        neighbors = neighbor_lists[chosen]
        to_remove_set = {chosen}
        for n in neighbors:
            if alive[n]:
                to_remove_set.add(n)
        
        to_remove = np.array(list(to_remove_set))
        alive[to_remove] = False
        
        # Update degrees
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            removed_neighbor_counts = conflicts[remaining][:, to_remove].sum(axis=1)
            degrees[remaining] -= removed_neighbor_counts
    
    result_v4 = vectors_v4[selected_indices]
    
    # V1 vectors: all 22 are mutually compatible (as analyzed)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors
