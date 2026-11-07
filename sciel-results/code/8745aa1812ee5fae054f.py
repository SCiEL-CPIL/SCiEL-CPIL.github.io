import numpy as np
from scipy.spatial import Delaunay # Not usable without coords

def heuristics(distance_matrix):
    d = distance_matrix
    n = d.shape[0]
    
    # Base heuristic: inverse squared distance
    # Avoid division by zero
    d_safe = d + 1e-9
    H = 1.0 / (d_safe ** 2)
    
    # k-NN boost
    k = 8
    # For each node, find k nearest neighbors
    # We can use np.argpartition for efficiency
    # d[i] is the distance from i to all others. d[i,i] is 0.
    # We want the k smallest values in each row, excluding the diagonal (which is 0).
    
    # Mask out the diagonal for sorting
    d_for_sort = d.copy()
    np.fill_diagonal(d_for_sort, np.inf)
    
    # Get indices of k smallest distances for each row
    # argpartition returns the indices that would sort the array
    # We only need the k smallest, so argpartition(d, k)
    # Note: argpartition is not stable for equal values, but that's fine.
    indices = np.argpartition(d_for_sort, k, axis=1)[:, :k]
    
    # Create a zero matrix for the mask
    knn_mask = np.zeros_like(d, dtype=bool)
    
    # Set the k-NN edges to True
    # indices is shape (n, k)
    rows = np.arange(n)[:, np.newaxis]
    knn_mask[rows, indices] = True
    
    # Symmetrize the mask? 
    # If i is in j's k-NN, is j in i's k-NN? Not necessarily.
    # For TSP, the edge is undirected. If we boost i->j, we should probably boost j->i too?
    # The ACO algorithm uses heuristic[i,j] for step i->j.
    # If we make the mask symmetric, it's consistent with the undirected nature of the graph.
    knn_mask = knn_mask | knn_mask.T
    
    # Apply boost
    boost_factor = 5.0
    H = H * (1.0 + boost_factor * knn_mask)
    
    # Ensure no zeros (though 1/d^2 is always > 0)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Our H is always positive.
    
    return H
