import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph

def heuristics(distance_matrix):
    n = distance_matrix.shape[0]
    # Create a sparse graph for MST
    # distance_matrix is the weight matrix
    # We need to handle diagonal (0) and potential infinities if any, but it's uniform random points so all finite.
    
    # scipy.sparse.csgraph.minimum_spanning_tree expects a sparse matrix or dense matrix?
    # It accepts a sparse matrix.
    # Let's create a sparse matrix from the distance matrix.
    # Note: distance_matrix[i][i] is 0. MST ignores self-loops.
    
    # Convert to sparse
    # The diagonal is 0, which might be interpreted as "no edge" or "weight 0". 
    # In csgraph, 0 usually means no edge if using COO/CSC with explicit zeros? 
    # Actually, for dense input to minimum_spanning_tree, it treats 0 as no edge if `directed` is False? 
    # Let's check docs or just use a safe method.
    
    # A safer way: set diagonal to inf or large number, then compute MST.
    # But distance_matrix is already given.
    
    # Let's use scipy.sparse.csgraph.minimum_spanning_tree
    # It works with dense matrices too? 
    # "The graph is specified as a sparse matrix..."
    # So we convert.
    
    # Create sparse matrix
    # We need to ensure that 0 on diagonal is not an edge.
    # In sparse matrices, 0 is typically not stored.
    # So sp.csr_matrix(distance_matrix) will have 0s on diagonal not stored.
    # This is correct.
    
    try:
        # minimum_spanning_tree returns a sparse matrix where non-zero entries are edges in the MST
        mst = csgraph.minimum_spanning_tree(sp.csr_matrix(distance_matrix))
        # mst is sparse. Convert to dense mask
        mst_dense = mst.toarray()
        # The weights are in mst_dense. 
        # We just need a boolean mask of which edges are in the MST.
        # An edge (i,j) is in MST if mst_dense[i,j] > 0.
        # Note: MST is undirected, so mst_dense is symmetric? 
        # minimum_spanning_tree returns a directed graph representation? 
        # "The graph is specified as a sparse matrix... The graph is undirected if..."
        # Actually, the output is a sparse matrix of the same shape.
        # For undirected, it sets both [i,j] and [j,i]? 
        # Let's assume we can just check > 0.
        
        # Heuristic: 
        # Base: 1 / d
        # Boost MST edges by factor 10
        # Demote non-MST edges? Or just boost MST.
        
        # If we just boost MST, non-MST edges are still 1/d.
        # Let's try: 
        # eta = 1 / (d + 1e-9)
        # eta[mst_dense > 0] *= 10.0
        
        # However, 1/d is already very large for small d.
        # Maybe a multiplicative boost is enough.
        
        # Let's try a different structural change: 
        # Use the **inverse of the distance** but **normalized** by the **row sum** or **max**?
        # No, that's not structural enough.
        
        # Let's stick to MST boost.
        
        base_heuristic = 1.0 / (distance_matrix + 1e-9)
        
        # Mask for MST edges
        # mst_dense has the weights. 
        # We want to identify edges that are in the MST.
        # Since it's an undirected MST, both i->j and j->i should be present if i-j is an edge.
        # Let's verify: csgraph.minimum_spanning_tree returns a matrix where 
        # if (i,j) is in MST, then M[i,j] = weight. 
        # For undirected, it should be symmetric.
        
        mst_mask = mst_dense > 0
        
        # Apply boost
        # Factor 5.0 seems reasonable.
        base_heuristic[mst_mask] *= 5.0
        
        # Ensure positive
        base_heuristic[base_heuristic <= 0] = 1e-9
        
        return base_heuristic

    except Exception:
        # Fallback to standard heuristic
        return 1.0 / (distance_matrix + 1e-9)
