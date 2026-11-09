import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh
from scipy.cluster.vq import kmeans2

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 250.0 # Safety margin
    
    n = distance_matrix.shape[0]
    customers = np.arange(1, n)
    
    # 1. Base Heuristic: Inverse Distance
    # Add small epsilon to avoid division by zero (though diagonal is 0, we don't visit self)
    eps = 1e-9
    base_heur = 1.0 / (distance_matrix + eps)
    
    # Ensure diagonal is 0 or very small, as self-loops are not allowed
    np.fill_diagonal(base_heur, 0.0)
    
    # 2. Savings Heuristic Signal
    # s[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # Only for i, j in customers
    d_to_depot = distance_matrix[:, 0]
    savings = d_to_depot[:, None] + d_to_depot[None, :] - distance_matrix
    
    # Savings for depot links are not typically used in the same way.
    # We will only apply savings boost to customer-customer edges.
    # Normalize savings for customers
    cust_savings = savings[np.ix_(customers, customers)]
    
    # Clip negative savings (triangle inequality usually holds, but just in case)
    cust_savings = np.clip(cust_savings, 0, None)
    
    max_savings = np.max(cust_savings)
    if max_savings > 0:
        norm_savings = cust_savings / max_savings
    else:
        norm_savings = np.zeros_like(cust_savings)
    
    # 3. Spectral Clustering for Structural Prior
    # Estimate number of clusters (vehicles)
    total_demand = np.sum(demands[1:])
    num_clusters = max(1, int(np.ceil(total_demand / capacity)))
    
    # Cap num_clusters to avoid issues if demand is very small or large relative to n
    num_clusters = min(num_clusters, n - 1)
    
    cluster_match = np.zeros((n, n), dtype=float)
    
    if num_clusters > 1 and n > 2:
        # Build affinity matrix for customers
        # Use inverse distance as affinity
        cust_dist = distance_matrix[np.ix_(customers, customers)]
        affinity = 1.0 / (cust_dist + eps)
        np.fill_diagonal(affinity, 0.0)
        
        # Convert to sparse for Laplacian
        aff_sparse = csr_matrix(affinity)
        
        # Compute Normalized Laplacian
        # L_sym = I - D^-1/2 A D^-1/2
        try:
            L = laplacian(aff_sparse, normed=True)
            
            # Get smallest eigenvalues/vectors
            # We want the first K eigenvectors
            k = num_clusters
            # eigsh requires k < N-1
            if k < len(customers) - 1:
                # 'SA' is smallest algebraic
                vals, vecs = eigsh(L, k=k, which='SA')
            else:
                # If k is too large, just use all but last
                vals, vecs = eigsh(L, k=len(customers)-2, which='SA')
                k = len(customers)-2
                
            # vecs are the eigenvectors. Rows correspond to customers.
            # Cluster these points in R^k
            # Normalize rows for k-means
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs_norm = vecs / norms
            
            # Run K-Means
            # scipy.cluster.vq.kmeans2 returns centroids and labels
            centroids, labels = kmeans2(vecs_norm, k, minit='++', seed=0)
            
            # Create cluster match matrix for customers
            for i_idx in range(len(customers)):
                for j_idx in range(i_idx, len(customers)):
                    if labels[i_idx] == labels[j_idx]:
                        c1 = customers[i_idx]
                        c2 = customers[j_idx]
                        cluster_match[c1, c2] = 1.0
                        cluster_match[c2, c1] = 1.0
                        
        except Exception:
            # Fallback to no clustering if spectral fails
            cluster_match = np.zeros((n, n), dtype=float)
            
    # 4. Combine Heuristics
    # H = Base * (1 + alpha * Savings + beta * Cluster)
    # Apply only to customer-customer edges
    alpha = 1.0
    beta = 1.0
    
    final_heur = base_heur.copy()
    
    # Apply boosts to customer-customer edges
    # We create a mask for customer-customer edges
    cust_mask = np.zeros((n, n), dtype=bool)
    cust_mask[np.ix_(customers, customers)] = True
    
    boost = 1.0 + alpha * norm_savings + beta * cluster_match[np.ix_(customers, customers)]
    
    # Map boost back to full matrix
    full_boost = np.ones((n, n))
    full_boost[np.ix_(customers, customers)] = boost
    
    final_heur = final_heur * full_boost
    
    # Ensure positivity and finiteness
    final_heur = np.where(final_heur > 0, final_heur, 1e-9)
    final_heur = np.where(np.isfinite(final_heur), final_heur, 1e-9)
    
    # Diagonal should be 0 or very small, but task says "not positive becomes 1e-9"
    # Usually diagonal is not used in ACO transition, but let's keep it consistent.
    # If diagonal is 0, it becomes 1e-9.
    
    if final_heur.shape != distance_matrix.shape:
        final_heur = 1.0 / (distance_matrix + eps)
        
    return final_heur
