# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit
    
    A = np.array(adjacency, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.array([], dtype=np.float64)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Compute degree
    degree = A.sum(axis=1)
    
    # Compute graph Laplacian L = D - A
    D = np.diag(degree)
    L = D - A
    
    # Compute Fiedler vector (second-smallest eigenvector of L)
    # For n=200, this is fast
    try:
        # Use sparse eigensolver if available, else dense
        eigvals, eigvecs = np.linalg.eigh(L)
        fiedler = eigvecs[:, 1]  # second-smallest eigenvalue's eigenvector
    except Exception:
        # Fallback: use degree-based heuristic
        return -degree.astype(np.float64)
    
    # Normalize Fiedler vector to [0, 1]
    f_min = fiedler.min()
    f_max = fiedler.max()
    if f_max > f_min:
        fiedler_norm = (fiedler - f_min) / (f_max - f_min)
    else:
        fiedler_norm = np.zeros(n)
    
    # The Fiedler vector splits the graph into two communities
    # Vertices with negative Fiedler values are in one partition, positive in another
    # We want to identify which partition is "sparser" (fewer internal edges)
    
    # Partition: split at the median of the Fiedler vector
    median_val = np.median(fiedler)
    partition_A_mask = fiedler <= median_val
    partition_B_mask = ~partition_A_mask
    
    # Count internal edges in each partition
    internal_edges_A = A[np.ix_(partition_A_mask, partition_A_mask)].sum() / 2
    internal_edges_B = A[np.ix_(partition_B_mask, partition_B_mask)].sum() / 2
    
    # Count vertices in each partition
    size_A = np.sum(partition_A_mask)
    size_B = np.sum(partition_B_mask)
    
    # Density of each partition
    if size_A > 1:
        density_A = internal_edges_A / (size_A * (size_A - 1) / 2)
    else:
        density_A = 0.0
    if size_B > 1:
        density_B = internal_edges_B / (size_B * (size_B - 1) / 2)
    else:
        density_B = 0.0
    
    # Vertices in the sparser partition are better candidates for the independent set
    # because they have fewer internal edges to worry about
    if density_A <= density_B:
        # Partition A is sparser; prioritize vertices in A
        partition_score = np.where(partition_A_mask, 1.0, 0.0)
    else:
        partition_score = np.where(partition_B_mask, 1.0, 0.0)
    
    # Combine: prefer vertices that are (1) in the sparser partition, (2) have low degree
    # Use a weighted combination
    # Normalize degree to [0, 1]
    if degree.max() > degree.min():
        degree_norm = (degree - degree.min()) / (degree.max() - degree.min())
    else:
        degree_norm = np.zeros(n)
    
    # Score: high for low degree AND in sparser partition
    # We want to pick vertices that will form a good independent set
    # A vertex is "good" if it has few neighbors AND is in a sparse region
    score = -degree_norm + 0.5 * partition_score
    
    # Check time
    if time_left() < 5.0:
        return score.astype(np.float64)
    
    # Refinement: use the Fiedler vector more directly
    # Vertices far from 0 in the Fiedler vector are "core" of their community
    # Vertices close to 0 are at the boundary
    # For independent set, we want vertices that are NOT at boundaries
    # (boundary vertices have neighbors in both communities)
    
    # Actually, let's think differently:
    # The Fiedler vector gives a continuous embedding
    # Vertices with similar Fiedler values are well-connected
    # For a greedy independent set, we want to pick vertices that are "isolated"
    # in this embedding space
    
    # Compute local connectivity: for each vertex, how many neighbors have
    # similar Fiedler values (i.e., are in the same community)?
    community_similarity = np.zeros(n)
    for i in range(n):
        neighbors = np.where(A[i] > 0)[0]
        if len(neighbors) > 0:
            # Fraction of neighbors in the same partition
            same_partition = np.sum((fiedler[neighbors] - fiedler[i]).astype(np.float64) ** 2)
            community_similarity[i] = same_partition / len(neighbors)
        else:
            community_similarity[i] = 0.0
    
    # Vertices with high community_similarity have many neighbors in the same community
    # These are "core" vertices and harder to include in an independent set
    # Vertices with low community_similarity are more "boundary" or "isolated"
    
    # Final score: combine degree, partition membership, and community similarity
    if community_similarity.max() > community_similarity.min():
        comm_sim_norm = (community_similarity - community_similarity.min()) / (community_similarity.max() - community_similarity.min())
    else:
        comm_sim_norm = np.zeros(n)
    
    # Low degree is good, low community similarity is good, sparser partition is good
    final_score = -degree_norm - 0.3 * comm_sim_norm + 0.2 * partition_score
    
    return final_score.astype(np.float64)
# EVOLVE-BLOCK-END
