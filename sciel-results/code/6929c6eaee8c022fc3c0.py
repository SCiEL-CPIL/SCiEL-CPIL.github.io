import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def heuristics(distance_matrix):
    # Ensure input is a numpy array
    D = np.asarray(distance_matrix)
    
    # Base heuristic: inverse distance
    # Avoid division by zero
    H = 1.0 / (D + 1e-9)
    
    # Compute MST
    # minimum_spanning_tree returns a sparse matrix
    mst_sparse = minimum_spanning_tree(D)
    mst_dense = mst_sparse.toarray()
    
    # Create a mask for MST edges
    # The MST is undirected, so the matrix is symmetric.
    # We want to boost these edges.
    # Boost factor: Let's try 2.0 (so total multiplier is 1 + 2 = 3 for MST edges, 1 for others)
    # Or just multiply by a factor.
    # Let's use a multiplicative boost.
    boost = 1.0 + 2.0 * (mst_dense > 0)
    
    H = H * boost
    
    # Ensure no non-positive values (though 1/D is always positive, and boost is positive)
    # The prompt says "Heuristic entries that are not positive become 1e-9"
    # Our H is strictly positive.
    
    return H
