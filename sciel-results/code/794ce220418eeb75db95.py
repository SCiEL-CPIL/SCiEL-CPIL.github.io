import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0  # Safety margin from 330s limit

def _run_aco_fast(D, eta, n_ants=20, n_iter=30, seed=42):
    """Run ACO with given heuristic eta, return best tour length."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)
    
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for it in range(n_iter):
        all_paths = []
        all_lens = []
        
        for ant in range(n_ants):
            current = 0
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            path = [0]
            tour_len = 0.0
            
            for step in range(1, n):
                # Get available nodes
                available = np.where(~visited)[0]
                
                # Compute probabilities: tau * eta
                probs = tau[current, available] * eta[current, available]
                total = probs.sum()
                
                if total <= 0:
                    next_node = available[0]
                else:
                    probs = probs / total
                    r = rng.random()
                    cumsum = np.cumsum(probs)
                    next_node = available[np.searchsorted(cumsum, r)]
                
                tour_len += D[current, next_node]
                current = next_node
                visited[current] = True
                path.append(current)
            
            # Return to start
            tour_len += D[current, 0]
            path.append(0)
            
            if tour_len < best_len:
                best_len = tour_len
            
            all_paths.append(path)
            all_lens.append(tour_len)
        
        # Pheromone update
        tau *= 0.9
        for ant_idx in range(n_ants):
            l = all_lens[ant_idx]
            if l <= 0:
                continue
            reward = 1.0 / l
            path = all_paths[ant_idx]
            for k in range(len(path) - 1):
                i = path[k]
                j = path[k + 1]
                tau[i, j] += reward
    
    return best_len

def _two_opt(tour, dist_matrix, deadline):
    """Improve a tour using 2-opt heuristic. Returns the improved tour."""
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            if time.time() > deadline:
                return tour
            for j in range(i + 1, n):
                # Check if reversing the segment between i and j improves the tour
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                
                if j == i + 1:
                    continue
                    
                current_cost = dist_matrix[a, b] + dist_matrix[c, d]
                new_cost = dist_matrix[a, c] + dist_matrix[b, d]
                
                if new_cost < current_cost - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour

def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]
    
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Return a reasonable default heuristic
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta

    # 1. Compute MST
    try:
        mst = minimum_spanning_tree(D, directed=False)
        mst_dense = mst.toarray()
    except Exception:
        # Fallback to simple distance heuristic
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta

    # 2. Extract coordinates from distance matrix (approximate via embedding or just use indices)
    # Since we only have D, we can't easily get coordinates for angle calculation.
    # However, we can approximate "angular consistency" by looking at the structure of the MST.
    # A better proxy for "good" edges in a tour without coordinates:
    # Edges that are in the MST are likely to be part of the tour (or close to it).
    # Edges that connect nodes with similar "MST degrees" or are part of the "backbone" are good.
    
    # Alternative Strategy: Use the MST to define a "skeleton" and boost edges that are 
    # geometrically consistent with the MST structure.
    # Without coordinates, we can't compute angles. 
    # Let's pivot to a different different strategy: 
    # **MST-Edge Boosting with Distance Decay**
    # The current program uses 1/d^p. 
    # The new strategy: Heuristic = w_mst * (1/d) for MST edges, and w_non * (1/d) for non-MST edges.
    # But we need to be smarter. 
    # Let's use the fact that the optimal tour is a Hamiltonian cycle. 
    # The MST is a subset of the graph. 
    # We can compute a "MST-ness" score for each edge.
    
    # Let's try a different angle: **Geometric Embedding via PCA**
    # We can recover approximate coordinates from the distance matrix using MDS/PCA.
    # Then we can compute actual angles.
    
    try:
        # MDS to get coordinates
        # D is squared distances? No, D is distances. 
        # MDS usually takes squared distances.
        D_sq = D ** 2
        # Double centering
        n = D_sq.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        # Eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Sort in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Take top 2 dimensions
        # Ensure real
        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)
        
        # Only take positive eigenvalues
        pos_idx = eigvals > 1e-10
        eigvals_pos = eigvals[pos_idx]
        eigvecs_pos = eigvecs[:, pos_idx]
        
        if eigvals_pos.size < 2:
            coords = eigvecs_pos @ np.sqrt(np.maximum(eigvals_pos, 0))
        else:
            coords = eigvecs_pos[:, :2] @ np.sqrt(np.maximum(eigvals_pos[:2], 0))
            
    except Exception:
        # Fallback if MDS fails
        eta = 1.0 / (D + 1e-9)
        eta[D == 0] = 1e-9
        return eta

    # 3. Compute Heuristic based on Coordinates
    # For each node i, and each other node j:
    # If edge (i, j) is in MST, boost it.
    # Also, boost edges that form small angles with the MST edges incident to i.
    
    # First, identify MST edges
    mst_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if mst_dense[i, j] > 0:
                mst_edges.add((i, j))
                mst_edges.add((j, i))

    # For each node, find its MST neighbors
    mst_neighbors = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j and mst_dense[i, j] > 0:
                mst_neighbors[i].append(j)

    eta = np.ones((n, n))
    
    # Base heuristic: 1/distance
    mask = D > 0
    eta[mask] = 1.0 / D[mask]
    eta[~mask] = 1e-9
    
    # Boost MST edges
    mst_boost = 2.0
    for i in range(n):
        for j in mst_neighbors[i]:
            eta[i, j] *= mst_boost
    
    # Angular Boost:
    # For each node i, for each candidate j (not in MST neighbors),
    # check if the vector (coords[j] - coords[i]) aligns with any MST neighbor vector.
    # If it aligns well, boost it.
    
    # This is O(N^3) if done naively. N=50, so 125k ops, which is fine.
    for i in range(n):
        if not mst_neighbors[i]:
            continue
        
        # Get MST neighbor vectors
        mst_vecs = []
        for j in mst_neighbors[i]:
            vec = coords[j] - coords[i]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                mst_vecs.append(vec / norm)
        
        if not mst_vecs:
            continue
        
        # For all other nodes j
        for j in range(n):
            if i == j or j in mst_neighbors[i]:
                continue
            
            vec = coords[j] - coords[i]
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                continue
            
            vec_norm = vec / norm
            
            # Max cosine similarity with any MST neighbor vector
            max_cos = 0.0
            for mv in mst_vecs:
                cos_sim = np.dot(vec_norm, mv)
                if cos_sim > max_cos:
                    max_cos = cos_sim
            
            # Boost based on alignment
            # If cos_sim is high (close to 1), it's a good continuation
            # If cos_sim is low, it's a sharp turn
            # We want to encourage smooth turns.
            # Boost factor = 1 + alpha * max_cos
            alpha = 1.0
            boost = 1.0 + alpha * max_cos
            eta[i, j] *= boost

    # Ensure positive
    eta[eta <= 0] = 1e-9
    
    return eta
