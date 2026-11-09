import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s kill

    n = distance_matrix.shape[0]
    dm = distance_matrix
    d = demands
    customers = np.arange(1, n)
    n_cust = n - 1
    eps = 1e-9

    # 1. Base Heuristic: Inverse Distance
    base_heur = 1.0 / (dm + eps)
    np.fill_diagonal(base_heur, 0.0)

    # 2. Savings Heuristic
    d_to_depot = dm[:, 0]
    savings = d_to_depot[:, None] + d_to_depot[None, :] - dm
    cust_savings = savings[np.ix_(customers, customers)]
    cust_savings = np.clip(cust_savings, 0, None)
    max_savings = np.max(cust_savings)
    if max_savings > 0:
        norm_savings = cust_savings / max_savings
    else:
        norm_savings = np.zeros_like(cust_savings)

    # 3. Spectral Clustering for Structural Prior
    # This identifies natural clusters in the customer space to encourage
    # routing within clusters rather than across them.
    total_demand = np.sum(demands[1:])
    # Estimate number of routes/vehicles needed
    num_clusters = max(1, int(np.ceil(total_demand / capacity)))
    num_clusters = min(num_clusters, n_cust)
    
    cluster_match = np.zeros((n, n), dtype=float)
    
    if num_clusters > 1 and n_cust > 2:
        try:
            cust_dist = dm[np.ix_(customers, customers)]
            # Affinity based on inverse distance
            affinity = 1.0 / (cust_dist + eps)
            np.fill_diagonal(affinity, 0.0)
            
            # KNN sparsification for stability
            k_knn = max(5, min(15, n_cust // 3))
            
            knn_mask = np.zeros_like(affinity, dtype=bool)
            for i in range(n_cust):
                row = affinity[i]
                idx = np.argpartition(row, -k_knn)[-k_knn:]
                knn_mask[i, idx] = True
            knn_mask = knn_mask | knn_mask.T
            
            W = affinity * knn_mask
            np.fill_diagonal(W, 0.0)
            
            # Normalized Laplacian
            degree = W.sum(axis=1)
            degree_safe = np.where(degree > 0, degree, 1.0)
            d_inv_sqrt = 1.0 / np.sqrt(degree_safe)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L_sym = np.eye(n_cust) - D_inv_sqrt @ W @ D_inv_sqrt
            L_sym = (L_sym + L_sym.T) / 2.0
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
            idx_sort = np.argsort(eigenvalues)
            eigenvectors = eigenvectors[:, idx_sort]
            
            # Take eigenvectors corresponding to the smallest eigenvalues (excluding constant)
            k_vecs = min(num_clusters, n_cust - 1)
            vecs = eigenvectors[:, 1:1+k_vecs]
            
            # Normalize rows for k-means
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs_norm = vecs / norms
            
            # K-Means
            try:
                from scipy.cluster.vq import kmeans2
                centroids, labels = kmeans2(vecs_norm, num_clusters, minit='++', seed=0)
            except ImportError:
                rng = np.random.default_rng(0)
                centroids = vecs_norm[rng.choice(n_cust, num_clusters, replace=False)]
                for _ in range(20):
                    dists = np.linalg.norm(vecs_norm[:, None, :] - centroids[None, :, :], axis=2)
                    labels = np.argmin(dists, axis=1)
                    new_centroids = []
                    for j in range(num_clusters):
                        if np.any(labels == j):
                            new_centroids.append(vecs_norm[labels == j].mean(axis=0))
                        else:
                            new_centroids.append(centroids[j])
                    new_centroids = np.array(new_centroids)
                    if np.allclose(centroids, new_centroids):
                        break
                    centroids = new_centroids
            
            # Create cluster match matrix
            match_matrix = (labels[:, None] == labels[None, :]).astype(float)
            cluster_match[np.ix_(customers, customers)] = match_matrix
            
        except Exception:
            cluster_match = np.zeros((n, n), dtype=float)

    # 4. Combine Heuristics
    # We want to boost edges that are close, have high savings, and are within the same cluster.
    # However, simple multiplication might not be enough. 
    # Let's try a more aggressive weighting for cluster match.
    alpha = 1.0
    beta = 2.0  # Increased weight for cluster coherence
    
    final_heur = base_heur.copy()
    
    full_boost = np.ones((n, n))
    
    # Apply boosts to customer-customer edges
    boost = 1.0 + alpha * norm_savings + beta * cluster_match[np.ix_(customers, customers)]
    full_boost[np.ix_(customers, customers)] = boost
    
    final_heur = final_heur * full_boost
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
        
    return final_heur
