# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 145.0  # safety margin before 150s kill

# Cache for PCA projection and principal axis
_cache = {
    'proj': None,
    'axis': None,
    'dm': None
}


def _time_ok():
    return (time.time() - _T0) < _DEADLINE


def _get_projection(distance_matrix):
    """
    Compute a 1D projection of nodes onto the principal axis of the distance matrix.
    We use the eigenvector of the largest eigenvalue of the distance matrix (or its
    Gram matrix approximation) to define the main axis of the point cloud.
    Since distance_matrix is a distance matrix (not a Gram matrix), we can't directly
    use it as a covariance matrix. However, for a set of points, the principal axis
    can be approximated by the eigenvector of the largest eigenvalue of the matrix
    D * D (which relates to the Gram matrix if centered, but here we just use the
    structure). A more robust and standard way for distance matrices is to use
    Classical MDS or simply project onto the vector of row sums or the first
    principal component of the coordinates if we could recover them.
    
    Given the constraint of only having the distance matrix, a common heuristic
    for the "main axis" is the eigenvector corresponding to the largest eigenvalue
    of the matrix derived from the distances. Let's use the eigenvector of the
    largest eigenvalue of the distance matrix itself as a proxy for the principal
    direction in the embedding space. This is a structural change in representation.
    """
    if _cache['dm'] is distance_matrix:
        return _cache['proj'], _cache['axis']

    dm = np.asarray(distance_matrix)
    n = dm.shape[0]
    
    # To get a spatial projection, we can use the fact that for points in Euclidean space,
    # the principal axis is related to the eigenvectors of the Gram matrix.
    # We can reconstruct a pseudo-Gram matrix or just use the eigenvector of the
    # largest eigenvalue of the distance matrix as a heuristic for the "direction"
    # of the spread.
    
    # A better approach for 1D projection for sweep:
    # Project each node onto the vector formed by the difference between the two
    # farthest nodes? Or use the first principal component of the points.
    # Since we don't have points, let's use the eigenvector of the largest eigenvalue
    # of the distance matrix. This vector v indicates which nodes are "aligned"
    # with the main spread.
    
    # Actually, a simpler and effective geometric proxy for "sweep" without coordinates:
    # Use the row sums of the distance matrix to estimate a 1D coordinate? No.
    # Let's use the eigenvector of the largest eigenvalue of the distance matrix.
    # This vector 'axis' has a value for each node. Nodes with similar 'axis' values
    # are likely in the same "sector" or direction from the center of mass.
    
    try:
        # Symmetrize just in case
        dm_sym = (dm + dm.T) / 2.0
        # Compute eigenvectors
        # For large n, this might be slow, but n=51 is small.
        vals, vecs = np.linalg.eigh(dm_sym)
        # The largest eigenvalue corresponds to the last column in eigh (ascending order)
        # Or we can take the one with the largest absolute value.
        # For a distance matrix, the largest eigenvalue is positive.
        idx = np.argmax(vals)
        axis = vecs[:, idx]
        
        # Ensure the axis is normalized
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 0:
            axis = axis / axis_norm
            
        # The projection of each node onto this axis is just the value in the vector?
        # No, the vector itself represents the coordinates of the principal component.
        # So proj[i] = axis[i] is the coordinate of node i along the principal axis.
        proj = axis
        
        _cache['dm'] = distance_matrix
        _cache['proj'] = proj
        _cache['axis'] = axis
        
    except Exception:
        # Fallback: use row sums as a pseudo-coordinate
        proj = dm.sum(axis=1)
        axis = proj / np.linalg.norm(proj)
        _cache['dm'] = distance_matrix
        _cache['proj'] = proj
        _cache['axis'] = axis

    return _cache['proj'], _cache['axis']


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    n = len(unvisited_nodes)
    if n == 0:
        return int(depot)

    if not _time_ok():
        return int(depot)

    unvisited_arr = np.asarray(unvisited_nodes, dtype=np.intp)
    k = len(unvisited_arr)

    # Feasibility: demand must fit in remaining capacity
    feas_mask = demands[unvisited_arr] <= rest_capacity
    if not np.any(feas_mask):
        return int(depot)

    feas = unvisited_arr[feas_mask]
    m = len(feas)

    # If only one feasible, take it
    if m == 1:
        return int(feas[0])

    # Get projection and axis
    proj, axis = _get_projection(distance_matrix)
    
    # Distances from current node to each feasible candidate
    d_cur = distance_matrix[current_node, feas]
    
    # Strategy: Minimize angular deviation from the current heading.
    # We define the "current heading" as the direction from the previous node to the current node?
    # We don't have the previous node. We only have the current node.
    # Alternative: Define the "desired direction" as the direction towards the "center" of the
    # remaining feasible cluster, or simply use the principal axis direction.
    
    # A better sweep strategy:
    # 1. Project all feasible nodes onto the principal axis (proj).
    # 2. The "sweep" angle for a node j is related to its position in this 1D projection
    #    relative to the current node's projection.
    # 3. We want to visit nodes that are "next in line" along the principal axis.
    
    # Let's refine: 
    # Calculate the "angular" deviation by looking at the 2D embedding if possible?
    # We only have 1D projection.
    # Let's use the 1D projection to define a "sweep order".
    # We want to minimize the difference in projection between the current node and the candidate?
    # That would just be nearest neighbor in projection space.
    
    # Let's try a different geometric interpretation:
    # The "heading" is the vector from the depot to the current node?
    # Or the vector from the current node to the next candidate?
    
    # Let's stick to the prompt's inspiration: "minimizes the angular deviation from the current heading vector".
    # Since we don't have the previous node, we can approximate the "heading" as the direction
    # from the current node towards the centroid of the remaining feasible nodes.
    
    # Calculate centroid of feasible nodes in the projected space?
    # Or use the distance matrix to define a local coordinate system?
    
    # Let's use the principal axis values (proj) as the "angle" proxy.
    # We want to move in the direction that keeps us on the "sweep".
    # If we assume the route is a sweep along the principal axis, we should pick the node
    # that is closest to the current node's position on the axis, but perhaps slightly ahead?
    
    # Actually, a simpler and robust structural change:
    # Use the principal axis to define a "sweep line".
    # Score = d(current, j) + gamma * |proj[j] - proj[current]|
    # This penalizes nodes that are far along the principal axis from the current node,
    # encouraging us to finish the current "sector" before moving to the next.
    
    gamma = 1.0  # Weight for the axis deviation
    
    # proj is indexed by node id
    proj_cur = proj[current_node]
    proj_feas = proj[feas]
    
    # Deviation from current position on the principal axis
    axis_dev = np.abs(proj_feas - proj_cur)
    
    scores = d_cur + gamma * axis_dev
    
    # Pick the candidate with minimum score
    best_idx = np.argmin(scores)
    return int(feas[best_idx])
# EVOLVE-BLOCK-END
