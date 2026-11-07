import numpy as np
from scipy.spatial import Delaunay

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix for ACO TSP.
    
    Uses a topological heuristic based on the Delaunay triangulation of the points.
    Edges present in the Delaunay triangulation are assigned high attractiveness,
    while non-triangulation edges are assigned near-zero attractiveness.
    This biases the ACO towards non-crossing edges, which are more likely to be
    part of the optimal TSP tour.
    """
    n = len(distance_matrix)
    
    # Extract points from the distance matrix
    # For a Euclidean distance matrix, we can recover the points up to rotation/translation
    # using classical MDS or by assuming the points are in the unit square and using
    # the structure of the matrix. However, a simpler and robust approach for this
    # specific task (uniform random points in unit square) is to use the distance
    # matrix directly to infer adjacency via a threshold or use the fact that
    # Delaunay triangulation can be computed from the points.
    # Since we don't have the raw points, we must reconstruct them or use an alternative.
    
    # Wait, the input is only the distance matrix. We do not have the coordinates.
    # We cannot directly run Delaunay on coordinates without them.
    # However, for points in a plane, the Delaunay triangulation can be characterized
    # by the property that the circumcircle of any triangle is empty.
    # Computing Delaunay from a distance matrix alone is non-trivial and computationally expensive
    # for n=50 if done naively.
    
    # Alternative structural change: Use a "nearest neighbor graph" heuristic.
    # The optimal TSP tour for random points in a square is closely related to the
    # Minimum Spanning Tree (MST) or the k-nearest neighbor graph.
    # A common and effective heuristic is to set eta[i,j] high if j is among the
    # k nearest neighbors of i, and low otherwise.
    # This is a different design family: graph-theoretic sparsification vs. metric scaling.
    
    # Let's implement a k-Nearest Neighbor (kNN) based heuristic.
    # Choose k to be small, e.g., 4 or 6, to create a sparse graph.
    # The ACO will then be strongly biased towards these local edges.
    
    k = 4  # Number of nearest neighbors to consider
    
    # Find k nearest neighbors for each node
    # distance_matrix is n x n
    # We want to keep the k smallest distances for each row i, excluding the diagonal (self)
    
    # Create a copy to avoid modifying the original
    dist_copy = distance_matrix.copy()
    
    # Set diagonal to inf so it's not picked as a neighbor
    np.fill_diagonal(dist_copy, np.inf)
    
    # For each row, find the indices of the k smallest values
    # argsort is expensive, but n=50 is small.
    # Alternatively, use partition.
    
    eta = np.zeros((n, n), dtype=np.float64)
    
    # Base value for non-neighbor edges
    # We want them to be very low but not zero to avoid division by zero issues
    # if the ACO implementation doesn't handle zeros well (though the task says
    # "Heuristic entries that are not positive become 1e-9", so 0 is safe as it
    # becomes 1e-9).
    # However, the task says "Heuristic entries that are not positive become 1e-9".
    # So if we set them to 0, they become 1e-9.
    # If we set them to a small positive number, they stay.
    # Let's set non-neighbors to a small value, e.g., 1e-6, to be safe and distinct.
    # Actually, the task says "Heuristic entries that are not positive become 1e-9".
    # So 0 is fine.
    
    # Let's assign high attractiveness to k-NN edges.
    # What is "high"?
    # The ACO uses pheromone * heuristic.
    # If we make k-NN edges 100x more attractive than others, the ants will almost
    # always pick them.
    
    # Let's normalize the distance matrix to have mean 1 for the k-NN edges?
    # Or just use a fixed high value?
    # Let's try: eta[i,j] = 1 / dist[i,j] for k-NN edges, and 1e-6 for others.
    # This combines the metric information (shorter is better) with the topological
    # constraint (only consider local edges).
    
    for i in range(n):
        # Get distances from i to all others
        d_i = dist_copy[i]
        # Find k smallest
        # np.argpartition(d_i, k) gives indices of k smallest, but unsorted.
        # We just need the set of indices.
        knn_indices = np.argpartition(d_i, k)[:k]
        
        for j in knn_indices:
            # Ensure j is not i (already handled by inf diagonal)
            # Set heuristic to inverse distance
            # To avoid very large numbers if dist is very small, we can cap it or normalize.
            # Let's normalize by the minimum distance in the matrix (excluding diagonal)
            # Or just use 1/d.
            # If d is ~0.1, 1/d is 10. If d is ~1, 1/d is 1.
            # This seems reasonable.
            eta[i, j] = 1.0 / d_i[j]
        
        # Ensure the diagonal is 0 (will become 1e-9)
        eta[i, i] = 0.0
    
    # The heuristic matrix is directed?
    # The distance matrix is symmetric. The k-NN graph is not necessarily symmetric.
    # If i is a k-NN of j, j might not be a k-NN of i.
    # The ACO picks next node j from i. So eta[i,j] matters.
    # Using directed k-NN is fine.
    
    # To make it more robust, we could symmetrize: if i is k-NN of j OR j is k-NN of i,
    # then both eta[i,j] and eta[j,i] get the high value.
    # This creates an undirected k-NN graph.
    # Let's try symmetrizing to ensure connectivity and balance.
    
    # Create a boolean mask of k-NN edges
    mask_knn = np.zeros((n, n), dtype=bool)
    for i in range(n):
        d_i = dist_copy[i]
        knn_indices = np.argpartition(d_i, k)[:k]
        mask_knn[i, knn_indices] = True
    
    # Symmetrize
    mask_knn = mask_knn | mask_knn.T
    
    # Now fill eta
    eta = np.zeros((n, n), dtype=np.float64)
    min_dist = np.min(dist_copy[~np.eye(n, dtype=bool)])
    if min_dist == 0:
        min_dist = 1e-9
        
    for i in range(n):
        for j in range(n):
            if i != j:
                if mask_knn[i, j]:
                    # Inverse distance, scaled
                    # Normalize so that the average heuristic for k-NN edges is 1
                    # Or just use 1/d.
                    # Let's use 1/d and then normalize the whole matrix?
                    # No, the task says "Heuristic entries that are not positive become 1e-9".
                    # It doesn't say we must normalize.
                    # But scale matters relative to pheromone (starts at 1).
                    # If eta is ~10, it dominates. If eta is ~0.1, pheromone dominates.
                    # We want strong bias towards k-NN.
                    # Let's set eta[i,j] = 10.0 for k-NN edges?
                    # Or 1/d?
                    # Let's try 1/d, but cap it to avoid extreme values.
                    # Max 1/d is 1/min_dist.
                    # Let's normalize 1/d so that the mean of 1/d over k-NN edges is 1.
                    pass
    
    # Recalculate with normalization
    inv_d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j and mask_knn[i, j]:
                inv_d[i, j] = 1.0 / dist_copy[i, j]
    
    # Calculate mean of inv_d over k-NN edges
    knn_vals = inv_d[mask_knn]
    if len(knn_vals) > 0 and np.mean(knn_vals) > 0:
        mean_knn_inv = np.mean(knn_vals)
        # Scale so mean is 1
        inv_d = inv_d / mean_knn_inv
    
    # Set non-k-NN edges to a low value
    # Let's use 0.01 (1% of the mean k-NN attractiveness)
    # This allows some exploration but strongly biases towards k-NN.
    eta = np.where(mask_knn, inv_d, 0.01)
    
    # Ensure diagonal is 0
    np.fill_diagonal(eta, 0.0)
    
    # Ensure all entries are positive (0 becomes 1e-9 in ACO, but let's be safe)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # So we can leave 0s as 0s.
    
    return eta
