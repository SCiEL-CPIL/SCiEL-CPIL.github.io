"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness matrix for ACO TSP.
    
    Strategy: Use a graph-theoretic approach by computing the Minimum Spanning Tree (MST)
    of the complete graph defined by the distance matrix. Edges in the MST are given
    high attractiveness, while other edges are given lower attractiveness based on 
    inverse distance. This encourages the ACO to explore tours that follow the structural
    backbone of the optimal tour.
    
    The heuristic is designed such that:
    - MST edges get a boosted value (high constant + inverse distance contribution)
    - Non-MST edges get only the inverse distance contribution
    
    This creates a sparse guidance structure that helps ants find good tours faster.
    """
    # Ensure the distance matrix is a valid numpy array
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # Handle edge cases
    if n <= 1:
        return np.ones_like(dm, dtype=np.float64)
    
    # Create a sparse matrix from the distance matrix for MST computation
    # We need to handle the diagonal (self-loops) - set them to a large value
    # so they're not chosen in MST
    dm_for_mst = dm.copy()
    np.fill_diagonal(dm_for_mst, np.inf)
    
    # Compute MST using Kruskal's or Prim's algorithm
    try:
        mst_graph = minimum_spanning_tree(csr_matrix(dm_for_mst))
        mst_dense = mst_graph.toarray()
        
        # Create a boolean mask for MST edges
        # An edge (i,j) is in the MST if mst_dense[i,j] is finite and positive
        mst_mask = (mst_dense > 0) & np.isfinite(mst_dense)
        
        # For safety, make sure the diagonal is False
        np.fill_diagonal(mst_mask, False)
    except Exception:
        # If MST computation fails, fall back to inverse distance
        mst_mask = np.zeros((n, n), dtype=bool)
    
    # Compute base heuristic: inverse distance (with small epsilon to avoid division by zero)
    epsilon = 1e-9
    base_heuristic = 1.0 / (dm + epsilon)
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(base_heuristic, 0.0)
    
    # Boost MST edges: add a constant to make them more attractive
    # The constant should be large enough to significantly boost MST edges
    # but not so large that it completely dominates the inverse distance signal
    mst_boost = 5.0
    
    # Create the final heuristic
    heuristic = base_heuristic.copy()
    heuristic[mst_mask] += mst_boost
    
    # Ensure all values are positive (as required by the task)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
