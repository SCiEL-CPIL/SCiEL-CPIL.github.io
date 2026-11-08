"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # safety margin from 150s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    # For weight-4 vectors, dot product is in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    # Condition: min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max norm = sqrt(4) = 2, so max norm^2 = 4
    # We need dist >= max_norm => dist^2 >= max_norm^2 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    # So conflict if u.v > 2, i.e., u.v >= 3
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # We want the largest independent set in this conflict graph.
    
    # Spectral approach: Use eigenvectors of the adjacency matrix to guide selection
    A = conflict.astype(np.float64)
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(A)
    except:
        eigenvalues = np.zeros(n)
        eigenvectors = np.eye(n)
    
    # Sort by eigenvalue (ascending)
    idx_sort = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]
    
    # Use the Fiedler vector (index 1) to create a soft partition
    fiedler = eigenvectors[:, 1] if n > 1 else np.zeros(n)
    
    # Also use the first few eigenvectors to create a feature space
    num_components = min(8, n - 1)
    if num_components > 0:
        features = eigenvectors[:, 1:num_components + 1]
    else:
        features = np.zeros((n, 1))
    
    np.random.seed(42)
    
    best_in_set = None
    best_count = 0
    
    # Strategy 1: Use spectral ordering to guide greedy selection
    spectral_order = np.argsort(fiedler)
    
    # Try multiple spectral-guided greedy approaches
    num_spectral_restarts = 800
    for r in range(num_spectral_restarts):
        if time.time() > deadline - 40:
            break
        
        # Create a modified order based on spectral features
        if r % 4 == 0:
            # Pure spectral order
            order = spectral_order
        elif r % 4 == 1:
            # Reverse spectral order
            order = spectral_order[::-1]
        elif r % 4 == 2:
            # Add noise to spectral values and re-sort
            noisy_fiedler = fiedler + np.random.normal(0, 0.3, size=n)
            order = np.argsort(noisy_fiedler)
        else:
            # Use multiple eigenvector components for ordering
            # Weighted combination of first few eigenvectors
            weights = np.random.dirichlet(np.ones(num_components))
            combined = features @ weights
            order = np.argsort(combined)
        
        in_set = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        
        count = 0
        for idx in order:
            if not remaining[idx]:
                continue
            in_set[idx] = True
            count += 1
            # Remove neighbors from remaining
            neighbors = conflict[idx]
            remaining[neighbors] = False
            remaining[idx] = False
        
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
    
    # Strategy 2: Use feature-based clustering to identify candidate independent sets
    if time.time() < deadline - 40:
        for k_clusters in [3, 4, 5, 6, 8, 10, 12]:
            if time.time() > deadline - 40:
                break
            
            # Simple k-means with multiple restarts
            best_cluster_count = 0
            best_cluster_assignment = None
            best_cluster_centroids = None
            
            for restart in range(5):
                centroids = features[np.random.choice(n, k_clusters, replace=False)]
                
                for _ in range(30):  # k-means iterations
                    # Assign points to nearest centroid
                    dists = np.sum((features[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
                    assignments = np.argmin(dists, axis=1)
                    
                    # Update centroids
                    for c in range(k_clusters):
                        mask = assignments == c
                        if mask.any():
                            new_centroid = features[mask].mean(axis=0)
                            if np.allclose(centroids[c], new_centroid):
                                continue
                            centroids[c] = new_centroid
            
            # For each cluster, try to find a large independent set within it
            for c in range(k_clusters):
                if time.time() > deadline - 40:
                    break
                
                cluster_nodes = np.where(assignments == c)[0]
                if len(cluster_nodes) < 2:
                    continue
                
                # Build subgraph for this cluster
                cluster_conflict = conflict[np.ix_(cluster_nodes, cluster_nodes)]
                
                # Greedy independent set within cluster using minimum degree
                in_set = np.zeros(n, dtype=bool)
                remaining = np.ones(len(cluster_nodes), dtype=bool)
                
                # Use degree-based ordering within cluster
                degrees = cluster_conflict.sum(axis=1)
                order = np.argsort(degrees)
                
                count = 0
                for local_idx in order:
                    if not remaining[local_idx]:
                        continue
                    global_idx = cluster_nodes[local_idx]
                    in_set[global_idx] = True
                    count += 1
                    # Remove neighbors within cluster
                    neighbors_local = cluster_conflict[local_idx]
                    remaining[neighbors_local] = False
                    remaining[local_idx] = False
                
                if count > best_count:
                    best_count = count
                    best_in_set = in_set.copy()
    
    # Strategy 3: Random restarts with spectral bias
    if time.time() < deadline - 30:
        num_random_restarts = 3000
        for r in range(num_random_restarts):
            if time.time() > deadline - 30:
                break
            
            # Create a biased random order using spectral features
            # Nodes with extreme Fiedler values are more likely to be selected early
            bias = np.abs(fiedler - fiedler.mean())
            bias = bias / (bias.sum() + 1e-10)
            
            # Sample without replacement using biased probabilities
            perm = np.random.choice(n, size=n, replace=False, p=bias)
            
            in_set = np.zeros(n, dtype=bool)
            remaining = np.ones(n, dtype=bool)
            
            count = 0
            for idx in perm:
                if not remaining[idx]:
                    continue
                in_set[idx] = True
                count += 1
                neighbors = conflict[idx]
                remaining[neighbors] = False
                remaining[idx] = False
            
            if count > best_count:
                best_count = count
                best_in_set = in_set.copy()
    
    # Strategy 4: Minimum degree greedy (fallback)
    if time.time() < deadline - 20 and best_count < 200:
        initial_degrees = conflict.sum(axis=1)
        num_md_restarts = 200
        for r in range(num_md_restarts):
            if time.time() > deadline - 20:
                break
            
            in_set = np.zeros(n, dtype=bool)
            remaining = np.ones(n, dtype=bool)
            degrees = initial_degrees.copy().astype(np.int32)
            
            count = 0
            while remaining.any():
                masked_degrees = np.where(remaining, degrees, np.inf)
                idx = np.argmin(masked_degrees)
                
                if not remaining[idx]:
                    break
                
                in_set[idx] = True
                count += 1
                remaining[idx] = False
                
                neighbors = conflict[idx] & remaining
                if neighbors.any():
                    degrees[neighbors] -= 1
                    remaining[neighbors] = False
            
            if count > best_count:
                best_count = count
                best_in_set = in_set.copy()
    
    # Phase 5: Local search improvement on the best solution found
    if time.time() < deadline - 5 and best_in_set is not None:
        in_set = best_in_set.copy()
        removed = ~in_set
        
        # Try to add back removed vertices
        # Multiple passes
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            
            # Find removed vertices that have no conflicts with current in_set
            has_conflict = conflict[:, in_set].any(axis=1)
            candidates = removed & ~has_conflict
            
            if not candidates.any():
                break
            
            # Add all candidates that don't conflict with each other
            cand_indices = np.where(candidates)[0]
            added_in_pass = 0
            for v_idx in cand_indices:
                if time.time() > deadline - 5:
                    break
                if not conflict[v_idx, in_set].any():
                    in_set[v_idx] = True
                    removed[v_idx] = False
                    added_in_pass += 1
            
            if added_in_pass == 0:
                break
        
        # Swap-based local search: try removing one vertex and adding multiple non-conflicting ones
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            
            improved = False
            in_set_indices = np.where(in_set)[0]
            
            for remove_idx in in_set_indices:
                if time.time() > deadline - 5:
                    break
                
                # Try removing this vertex
                in_set[remove_idx] = False
                old_count = in_set.sum()
                
                # Find all vertices that can now be added
                has_conflict = conflict[:, in_set].any(axis=1)
                candidates = ~in_set & ~has_conflict
                
                # Add all non-conflicting candidates
                cand_indices = np.where(candidates)[0]
                added = 0
                for v_idx in cand_indices:
                    if not conflict[v_idx, in_set].any():
                        in_set[v_idx] = True
                        added += 1
                
                new_count = in_set.sum()
                
                if new_count > old_count:
                    improved = True
                    # Keep the improvement
                else:
                    # Revert
                    in_set[remove_idx] = True
                    for v_idx in cand_indices:
                        in_set[v_idx] = False
            
            if not improved:
                break
        
        best_count = in_set.sum()
        best_in_set = in_set
    
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
