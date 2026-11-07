"""Heuristic matrix for ACO-TSP: Graph-theoretic prior (MST/2-Tree) with Delaunay smoothing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix

_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0

def _remaining():
    return _DEADLINE - time.time()

def _compute_min_2_tree(d, mst):
    """
    Compute a Minimum 2-Tree approximation.
    A 2-tree is a connected graph where every vertex has degree 2.
    It consists of a cycle. The MST is a tree.
    To get a 2-tree, we can add edges to the MST until all degrees are 2.
    However, the standard "Min 2-Tree" heuristic for TSP is:
    Take the MST, and for each vertex v, consider the two shortest edges incident to v.
    The Min 2-Tree is the MST plus the two shortest edges incident to a single vertex v 
    that minimizes the total added length, provided they don't form a triangle with an existing edge?
    Actually, a simpler and effective lower bound is: MST + 2 * min_edge_incident_to_any_node? No.
    
    Let's use the definition: The minimum 2-tree is the minimum weight 2-regular spanning subgraph.
    For TSP, a 2-tree is a Hamiltonian cycle.
    A common approximation is to take the MST and add the two shortest edges incident to each node, 
    then resolve degree conflicts.
    
    Simpler approach for heuristic: 
    We want to boost edges that are likely in a good tour.
    Edges in the MST are very likely in a good tour (or close to it).
    Edges that are short and connect nodes in the MST are also good.
    
    Let's just identify the "critical" edges:
    1. MST edges.
    2. For each node, its two nearest neighbors.
    """
    n = d.shape[0]
    # Find the two nearest neighbors for each node
    # d is symmetric
    # We need to avoid self-loops
    d_safe = d.copy()
    np.fill_diagonal(d_safe, np.inf)
    
    # Get indices of two smallest values for each row
    # np.argpartition returns indices, not sorted, but we only need the set
    # k=2 means the 2 smallest are in the first 2 positions (unsorted)
    nn_indices = np.argpartition(d_safe, kth=1, axis=1)[:, :2]
    
    edges = set()
    for i in range(n):
        for j in nn_indices[i]:
            if i != j:
                edges.add((min(i, j), max(i, j)))
    
    return edges

def heuristics(distance_matrix):
    """
    Return an edge-attractiveness matrix (larger = more attractive).
    
    Strategy:
    1. Base heuristic: Inverse distance squared (standard ACO).
    2. MST Boost: Edges in the Minimum Spanning Tree are boosted.
    3. 2-Tree/NN Boost: Edges connecting nodes to their 2 nearest neighbors are boosted.
    4. Delaunay Smoothing: Edges in the Delaunay triangulation are slightly boosted to encourage local moves.
    5. Long Edge Penalty: Edges that are significantly longer than the median are penalized if not in MST/Delaunay.
    """
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    if n < 3:
        heuristic = 1.0 / (d + 1e-9) ** 2
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Deadline guard
    if _remaining() < 5.0:
        heuristic = 1.0 / (d + 1e-10) ** 2
        np.fill_diagonal(heuristic, 0.0)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # 1. Base Heuristic: 1 / d^2
    # Add small epsilon to avoid division by zero on diagonal (though we set diag to 0 later)
    d_safe = d + 1e-12
    heuristic = 1.0 / (d_safe ** 2)
    
    # Mask diagonal
    diag_mask = np.eye(n, dtype=bool)
    heuristic[diag_mask] = 0.0
    
    # 2. MST Boost
    mst_edges = set()
    try:
        # Convert to sparse matrix for MST
        d_sparse = csr_matrix(d)
        mst = minimum_spanning_tree(d_sparse)
        mst_coo = mst.tocoo()
        for i, j, v in zip(mst_coo.row, mst_coo.col, mst_coo.data):
            mst_edges.add((min(i, j), max(i, j)))
    except Exception:
        pass
    
    # 3. 2-Nearest Neighbor Boost
    nn_edges = _compute_min_2_tree(d, mst_edges)
    
    # 4. Delaunay Triangulation (Geometric Prior)
    delaunay_edges = set()
    coords = None
    try:
        # Recover coordinates using MDS for Delaunay
        # Since d is Euclidean (from uniform random points), MDS should work perfectly
        n = d.shape[0]
        J = np.eye(n) - np.ones((n, n)) / n
        D2 = d ** 2
        B = -0.5 * (J @ D2 @ J)
        B = (B + B.T) / 2.0
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        if w[0] > 1e-9:
            coords = V[:, :2] * np.sqrt(np.maximum(w[:2], 0))
            
            # Verify reconstruction
            rec_D = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
            corr = np.corrcoef(rec_D.flatten(), d.flatten())[0, 1]
            if not np.isnan(corr) and corr > 0.99:
                tri = scipy.spatial.Delaunay(coords)
                for simplex in tri.simplices:
                    i, j, k = simplex
                    delaunay_edges.add((min(i, j), max(i, j)))
                    delaunay_edges.add((min(j, k), max(j, k)))
                    delaunay_edges.add((min(i, k), max(i, k)))
    except Exception:
        pass

    # Apply Boosts
    # Boost factors are tuned to be significant but not dominant
    mst_boost = 5.0
    nn_boost = 2.0
    delaunay_boost = 1.5
    
    # Pre-compute median distance for penalty
    off_diag_d = d[~diag_mask]
    median_d = np.median(off_diag_d)
    
    for i in range(n):
        for j in range(i + 1, n):
            if i == j:
                continue
            
            edge_key = (i, j)
            val = heuristic[i, j]
            
            # MST is strongest
            if edge_key in mst_edges:
                val *= mst_boost
            elif edge_key in nn_edges:
                # If not in MST, check NN
                val *= nn_boost
            
            # Delaunay boost (additive or multiplicative? Multiplicative is standard)
            if edge_key in delaunay_edges:
                val *= delaunay_boost
            
            # Long Edge Penalty
            # If the edge is long (e.g., > 1.5 * median) and NOT in MST/Delaunay, penalize
            if d[i, j] > 1.5 * median_d:
                if edge_key not in mst_edges and edge_key not in delaunay_edges:
                    val *= 0.1
            
            heuristic[i, j] = val
            heuristic[j, i] = val

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic, 0.0)
    
    # Final check: non-finite or non-positive become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
