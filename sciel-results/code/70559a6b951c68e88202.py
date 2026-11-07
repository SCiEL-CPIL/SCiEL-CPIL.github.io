"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import distance as spdist
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO TSP.
    Uses a 2D MDS embedding to compute geometric features:
    1. Base attractiveness: Inverse distance.
    2. Geometric Boost: Edges in the Delaunay triangulation are boosted.
    3. Hull Boost: Edges on the convex hull are further boosted.
    4. Local Alignment: Edges are penalized if they deviate significantly 
       from the local direction of the MST (Minimum Spanning Tree) at the source node.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    
    # Safety check for time, though MDS/Delaunay for N=50 is very fast (<0.1s)
    # If it somehow takes too long, we fall back to simple inverse distance.
    
    try:
        # 1. Compute 2D embedding using Classical MDS
        # We use the distance matrix directly.
        # Classical MDS: 
        # D is distance matrix
        # B = -0.5 * J * D^2 * J
        # where J = I - (1/n) * 11^T
        # Then eigendecompose B to get coordinates.
        
        D_sq = distance_matrix ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ D_sq @ J
        
        # Eigen decomposition
        # We only need the top 2 eigenvalues/vectors for 2D embedding
        eigvals, eigvecs = np.linalg.eigh(B)
        # eigh returns eigenvalues in ascending order, so we take the last 2
        # Ensure we take the largest 2
        idx = np.argsort(eigvals)[::-1][:2]
        eigvals_top = eigvals[idx]
        eigvecs_top = eigvecs[:, idx]
        
        # Coordinates
        # Scale by sqrt(eigenvalues)
        coords = eigvecs_top * np.sqrt(np.maximum(eigvals_top, 0))
        
        # 2. Compute Delaunay Triangulation
        # Delaunay triangulation connects points such that no point is inside the circumcircle of any triangle.
        # Edges in the Delaunay triangulation are "locally non-crossing" and tend to be short.
        tri = Delaunay(coords)
        
        # Create a mask for Delaunay edges
        delaunay_edges = np.zeros((n, n), dtype=bool)
        for simplex in tri.simplices:
            # simplex is (i, j, k)
            i, j, k = simplex
            delaunay_edges[i, j] = True
            delaunay_edges[j, i] = True
            delaunay_edges[i, k] = True
            delaunay_edges[k, i] = True
            delaunay_edges[j, k] = True
            delaunay_edges[k, j] = True
            
        # 3. Compute Convex Hull
        # We can use the Delaunay triangulation's convex hull vertices, 
        # but easier is to use scipy.spatial.ConvexHull
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(coords)
            hull_edges = np.zeros((n, n), dtype=bool)
            for simplex in hull.simplices:
                # In 2D, simplices are edges (i, j)
                i, j = simplex
                hull_edges[i, j] = True
                hull_edges[j, i] = True
        except Exception:
            hull_edges = np.zeros((n, n), dtype=bool)
            
        # 4. Compute MST for local alignment
        # We compute the MST of the complete graph defined by the distance matrix.
        # For each node, we look at its MST neighbors.
        # An edge (i, j) is "aligned" if j is an MST neighbor of i OR i is an MST neighbor of j.
        # Actually, let's be more specific: 
        # Compute the MST using Prim's algorithm or Kruskal's.
        # scipy.sparse.csgraph.minimum_spanning_tree is available.
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        
        # Create a sparse matrix from the distance matrix
        # Note: MST is undirected, so we symmetrize
        dist_sym = (distance_matrix + distance_matrix.T) / 2.0
        # Ensure diagonal is 0 or inf? MST functions usually ignore diagonal or expect 0.
        # Let's set diagonal to 0.
        np.fill_diagonal(dist_sym, 0)
        
        # Convert to sparse
        dist_sparse = csr_matrix(dist_sym)
        
        # Compute MST
        mst = minimum_spanning_tree(dist_sparse)
        mst_dense = mst.toarray()
        
        # Create mask for MST edges
        mst_edges = (mst_dense > 0) & (mst_dense < np.inf)
        # Symmetrize just in case
        mst_edges = mst_edges | mst_edges.T
        
        # 5. Construct Heuristic
        # Base: Inverse distance
        # Add small epsilon to avoid division by zero
        h_base = 1.0 / (distance_matrix + 1e-9)
        
        # Boost factors
        # Delaunay edges are good candidates for TSP (non-crossing, short)
        # Boost by factor 2.0
        # Hull edges are very good (must be in TSP for convex points, often in non-convex too)
        # Boost by factor 3.0
        # MST edges are strong candidates
        # Boost by factor 2.5
        
        # We combine multiplicatively.
        # If an edge is in both Delaunay and Hull, it gets 2.0 * 3.0 = 6.0
        # If it is in MST, it gets 2.5
        # We want to avoid double counting if they overlap? 
        # Actually, MST edges are a subset of Delaunay edges in Euclidean space? 
        # Yes, the Euclidean MST is a subgraph of the Delaunay triangulation.
        # So MST edges will get Delaunay boost AND MST boost.
        # This is fine, it reinforces the strongest local structure.
        
        boost = np.ones((n, n))
        
        # Apply Delaunay boost
        boost[delaunay_edges] *= 2.0
        
        # Apply Hull boost
        boost[hull_edges] *= 3.0
        
        # Apply MST boost
        boost[mst_edges] *= 2.5
        
        # 6. Local Alignment Penalty
        # For each node i, let v_i be the vector to its closest MST neighbor (or average of MST neighbors).
        # For an edge (i, j), compute the angle between vector (j-i) and v_i.
        # If the angle is large, penalize.
        # This encourages the tour to follow the "flow" of the MST.
        
        # Compute MST neighbor vectors for each node
        # For each node i, find its MST neighbors
        # Let's pick the closest MST neighbor to define the local direction.
        
        local_dirs = np.zeros((n, 2))
        for i in range(n):
            neighbors = np.where(mst_edges[i, :])[0]
            if len(neighbors) > 0:
                # Find closest neighbor
                dists = distance_matrix[i, neighbors]
                closest_idx = neighbors[np.argmin(dists)]
                local_dirs[i] = coords[closest_idx] - coords[i]
            else:
                # If no neighbors (isolated, shouldn't happen in connected MST), use zero
                local_dirs[i] = np.array([0.0, 0.0])
        
        # Normalize local directions
        norms = np.linalg.norm(local_dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        local_dirs_norm = local_dirs / norms
        
        # Compute alignment score for each edge (i, j)
        # Vector from i to j: coords[j] - coords[i]
        # We can do this vectorized.
        
        # coords is (n, 2)
        # diff = coords[None, :, :] - coords[:, None, :]  # (n, n, 2) where diff[i, j] = coords[j] - coords[i]
        diff = coords[None, :, :] - coords[:, None, :]
        
        # Normalize diff
        diff_norms = np.linalg.norm(diff, axis=2, keepdims=True)
        diff_norms[diff_norms == 0] = 1.0
        diff_norm = diff / diff_norms
        
        # Dot product with local_dirs_norm
        # local_dirs_norm is (n, 2)
        # We need to broadcast local_dirs_norm[i] to match diff[i, j]
        # local_dirs_norm[:, None, :] is (n, 1, 2)
        # diff_norm is (n, n, 2)
        # dot = sum(local_dirs_norm[i, k] * diff_norm[i, j, k])
        
        dot_products = np.einsum('ik,ijk->ij', local_dirs_norm, diff_norm)
        
        # dot_products[i, j] is cos(angle) between local_dir[i] and vector(i->j)
        # cos(angle) ranges from -1 to 1.
        # 1 means aligned, -1 means opposite.
        # We want to reward alignment.
        # If cos(angle) < 0, the edge goes "against" the local flow.
        # Let's penalize edges where cos(angle) < 0.5 (i.e., angle > 60 degrees).
        
        alignment_factor = np.ones((n, n))
        # If dot product is negative, it's a strong penalty.
        # Let's use a smooth penalty.
        # factor = 1.0 + 0.5 * dot_products
        # If dot = 1, factor = 1.5 (boost)
        # If dot = 0, factor = 1.0 (neutral)
        # If dot = -1, factor = 0.5 (penalty)
        
        # However, we should only apply this if the edge is not self-loop.
        # Also, we should not penalize edges that are very short if they are misaligned?
        # The base heuristic already handles distance.
        
        # Let's refine: Only apply alignment penalty for edges that are not in the MST?
        # MST edges are already boosted. If we penalize them, it might cancel out.
        # Let's apply alignment to all edges, but make it mild.
        
        alignment_factor = 1.0 + 0.3 * dot_products
        
        # Ensure alignment_factor is positive
        alignment_factor = np.maximum(alignment_factor, 0.1)
        
        # Combine all factors
        h_final = h_base * boost * alignment_factor
        
        # Ensure positive and finite
        h_final = np.where(np.isfinite(h_final) & (h_final > 0), h_final, 1e-9)
        
        # Diagonal should be 0 or very small to avoid self-loops if logic fails, 
        # but ACO usually masks visited nodes. 
        # The problem says "picks the next unvisited node", so diagonal doesn't matter much, 
        # but let's set it to 0 to be safe.
        np.fill_diagonal(h_final, 0.0)
        
        return h_final
        
    except Exception as e:
        # Fallback to simple inverse distance if anything goes wrong
        # This ensures the function never crashes.
        h_fallback = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(h_fallback, 0.0)
        return h_fallback
# EVOLVE-BLOCK-END
