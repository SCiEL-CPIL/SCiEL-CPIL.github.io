"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
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
        
    # Compute conflict matrix: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Precompute neighbor lists as arrays
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    rng = np.random.default_rng(42)
    
    best_size = 0
    best_in_set = np.zeros(N4, dtype=bool)
    
    def greedy_min_degree():
        """Greedy independent set with min-degree heuristic."""
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set
    
    def spectral_kmeans_init(n_clusters=5, dim=8):
        """Spectral embedding + k-means clustering to find independent set clusters."""
        # Build adjacency matrix (conflict graph)
        A = conflicts.astype(np.float64)
        
        # Compute Laplacian L = D - A
        D_mat = np.diag(A.sum(axis=1))
        L = D_mat - A
        
        # Eigendecomposition to get low-dimensional embedding
        # Use scipy for efficiency
        try:
            from scipy.sparse.linalg import eigsh
            from scipy.sparse import csr_matrix
            # For dense matrix, use numpy
            evals, evecs = np.linalg.eigh(L)
            # Take the smallest eigenvectors (excluding trivial ones)
            # Sort by eigenvalue
            idx = np.argsort(evals)
            evecs_sorted = evecs[:, idx]
            # Use dim smallest non-trivial eigenvectors
            # Skip the first few (trivial constant vectors)
            start_idx = min(2, dim)
            embedding = evecs_sorted[:, start_idx:start_idx + dim]
        except Exception:
            # Fallback: random projection
            embedding = rng.standard_normal((N4, dim))
        
        # Normalize rows
        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embedding = embedding / norms
        
        # Spherical k-means
        # Initialize centroids by picking spread-out points
        k = min(n_clusters, N4)
        if k < 2:
            k = 2
            
        # Initialize with k-means++ style
        centroids = np.zeros((k, dim))
        # Pick first centroid randomly
        centroids[0] = embedding[rng.integers(N4)]
        
        # Pick remaining centroids far from existing
        for i in range(1, k):
            # Compute distance to nearest existing centroid
            dists = np.min(
                np.linalg.norm(embedding[:, None, :] - centroids[:i][None, :, :], axis=2),
                axis=1
            )
            # Probabilistic selection weighted by distance
            probs = dists / dists.sum() if dists.sum() > 0 else np.ones(N4) / N4
            centroids[i] = embedding[rng.choice(N4, p=probs)]
        
        # K-means iterations
        for iteration in range(50):
            # Assign each point to nearest centroid
            dists = np.linalg.norm(embedding[:, None, :] - centroids[None, :, :], axis=2)
            assignments = np.argmin(dists, axis=1)
            
            # Update centroids
            for c in range(k):
                mask = assignments == c
                if np.sum(mask) > 0:
                    centroid = embedding[mask].mean(axis=0)
                    norm_c = np.linalg.norm(centroid)
                    if norm_c > 0:
                        centroids[c] = centroid / norm_c
        
        # For each cluster, greedily build independent set
        best_cluster_size = 0
        best_cluster_in_set = np.zeros(N4, dtype=bool)
        
        for c in range(k):
            cluster_members = np.where(assignments == c)[0]
            if len(cluster_members) == 0:
                continue
            
            # Greedy: pick vertices from cluster that don't conflict with already chosen
            in_set = np.zeros(N4, dtype=bool)
            chosen = []
            remaining = list(cluster_members)
            
            while remaining and time.time() < deadline - 5:
                # Pick vertex with fewest conflicts among remaining
                best_v = None
                best_conflict_count = np.inf
                for v in remaining:
                    if in_set[v]:
                        continue
                    conflict_count = np.sum(conflicts[v] & in_set)
                    if conflict_count < best_conflict_count:
                        best_conflict_count = conflict_count
                        best_v = v
                
                if best_v is None:
                    break
                
                in_set[best_v] = True
                chosen.append(best_v)
                
                # Remove chosen and its neighbors from remaining
                to_remove = set([best_v] + list(neighbors[best_v]))
                remaining = [v for v in remaining if v not in to_remove]
            
            if len(chosen) > best_cluster_size:
                best_cluster_size = len(chosen)
                best_cluster_in_set = in_set.copy()
        
        return best_cluster_in_set
    
    def sa_local_search(in_set_init, max_time=10.0):
        """Min-conflict simulated annealing local search."""
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 3.0
        cooling = 0.9995
        min_T = 0.05
        
        # count_conflicts[v] = number of neighbors of v that are in in_set
        count_conflicts = np.zeros(N4, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
                
            counts = count_conflicts[not_in_set]
            min_count = np.min(counts)
            
            min_cands = not_in_set[counts == min_count]
            v = min_cands[rng.integers(len(min_cands))]
            
            k = min_count
            delta = 1 - k
            
            if delta > 0:
                accept = True
            elif delta == 0:
                accept = rng.random() < 0.5
            else:
                accept = rng.random() < np.exp(delta / T)
                
            if accept:
                to_remove = np.where(conflicts[v] & in_set)[0]
                
                if len(to_remove) > 0:
                    in_set[to_remove] = False
                    for u in to_remove:
                        count_conflicts[neighbors[u]] -= 1
                
                in_set[v] = True
                current_size += delta
                count_conflicts[neighbors[v]] += 1
                
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_in_set = in_set.copy()
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                stuck_counter += 1
            
            if stuck_counter > 1500:
                T = max(T, 2.0)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    # Multiple restarts with alternating initialization strategies
    n_restarts = 0
    max_restarts = 60
    use_spectral = False
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        # Alternate between greedy and spectral initialization
        if n_restarts % 3 == 1:
            # Spectral k-means init (every 3rd restart)
            in_set = spectral_kmeans_init(n_clusters=6, dim=10)
            current_size = in_set.sum()
        else:
            # Greedy min-degree init
            in_set = greedy_min_degree()
            current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # SA Local Search
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.6, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = sa_local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_in_set = sa_result.copy()
        
        # Second SA pass on best found so far (polish)
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 5 and sa_size > current_size:
            polish_time = min(remaining_time * 0.5, 10.0)
            if polish_time > 3:
                polish_result, polish_size = sa_local_search(sa_result, max_time=polish_time)
                if polish_size > best_size:
                    best_size = polish_size
                    best_in_set = polish_result.copy()
    
    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
