"""Distance-based phylogenetic tree reconstruction using optimized Neighbor-Joining with Jukes-Cantor correction."""
import numpy as np
import math


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences.
    
    Args:
        sequences: dict mapping taxon labels (t0..t(n-1)) to DNA strings over ACGT
        
    Returns:
        Newick string representing the unrooted tree topology
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Convert sequences to numpy arrays for faster distance computation
    seq_list = [sequences[label] for label in labels]
    L = len(seq_list[0])
    
    # Pre-convert sequences to integer arrays for faster comparison
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    # Create a 2D array of shape (n, L)
    seq_matrix = np.array([
        [base_to_int[c] for c in s] for s in seq_list
    ], dtype=np.int8)
    
    # Compute pairwise differences using broadcasting
    # diffs[i, j] is the number of positions where seq i and seq j differ
    # (seq_matrix[:, None, :] != seq_matrix[None, :, :]) creates a 3D boolean array (n, n, L)
    # This might be memory heavy for large n and L, but n<=32, L<=2000, so 32*32*2000 ~ 2M booleans, which is fine.
    diffs = (seq_matrix[:, None, :] != seq_matrix[None, :, :]).sum(axis=2)
    
    # Compute Jukes-Cantor corrected distances
    # D[a,b] = -0.75 * ln(1 - 4p/3) where p = diff_count / L
    # Handle p >= 0.75 case to avoid log of non-positive
    p = diffs / L
    # Create mask for valid logs
    valid = p < 0.7499
    dist = np.zeros((n, n), dtype=np.float64)
    dist[valid] = -0.75 * np.log(1.0 - 4.0 * p[valid] / 3.0)
    dist[~valid] = 3.0 # Cap distance
    # Ensure symmetry (should be symmetric already due to broadcasting, but good practice)
    # dist = (dist + dist.T) / 2 
    # Diagonal is 0
    np.fill_diagonal(dist, 0.0)
    
    # Neighbor-Joining algorithm using NumPy arrays
    # current_D: n x n distance matrix for active nodes
    # nodes: list of Newick strings for active nodes, corresponding to rows/cols of current_D
    
    current_D = dist.copy()
    nodes = labels.copy()
    
    while len(nodes) > 3:
        m = len(nodes)
        
        # Compute row sums
        r = current_D.sum(axis=1)
        
        # Compute Q matrix
        # Q[i, j] = (m-2)*D[i, j] - r[i] - r[j]
        Q = (m - 2) * current_D - r[:, None] - r[None, :]
        
        # Set diagonal to infinity to avoid selecting self
        np.fill_diagonal(Q, np.inf)
        
        # Find the pair (i, j) with minimum Q
        # Flatten Q, find min index, convert to 2D indices
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        # Ensure i < j for consistency (not strictly necessary for logic, but good)
        if i > j:
            i, j = j, i
            
        # Compute branch lengths
        delta_i = 0.5 * (current_D[i, j] + (r[i] - r[j]) / (m - 2))
        delta_j = current_D[i, j] - delta_i
        
        # Clamp to non-negative
        if delta_i < 0:
            delta_i = 0.0
        if delta_j < 0:
            delta_j = 0.0
        
        # Create new node string
        new_str = f"({nodes[i]}:{delta_i:.4f},{nodes[j]}:{delta_j:.4f})"
        
        # Compute new distances from the new node to all other active nodes
        # new_d[k] = 0.5 * (D[i, k] + D[j, k] - D[i, j]) for k != i, j
        # We need to construct the new distance matrix
        
        # Indices of remaining nodes
        remaining_indices = [k for k in range(m) if k != i and k != j]
        
        # Get distances to remaining nodes
        # d_i_rem = current_D[i, remaining_indices]
        # d_j_rem = current_D[j, remaining_indices]
        d_i_rem = current_D[i, remaining_indices]
        d_j_rem = current_D[j, remaining_indices]
        
        new_d = 0.5 * (d_i_rem + d_j_rem - current_D[i, j])
        
        # Construct new current_D
        # It will be of size (m-1) x (m-1)
        new_m = m - 1
        new_current_D = np.zeros((new_m, new_m), dtype=np.float64)
        
        # Fill the submatrix for remaining nodes
        # new_current_D[k, l] = current_D[remaining_indices[k], remaining_indices[l]]
        # We can do this with advanced indexing
        new_current_D[:-1, :-1] = current_D[np.ix_(remaining_indices, remaining_indices)]
        
        # Fill the last row and column with new_d
        new_current_D[:-1, -1] = new_d
        new_current_D[-1, :-1] = new_d
        new_current_D[-1, -1] = 0.0
        
        # Update nodes list
        # Remove nodes[i] and nodes[j], add new_str
        new_nodes = [nodes[k] for k in remaining_indices]
        new_nodes.append(new_str)
        
        current_D = new_current_D
        nodes = new_nodes
        
    # Final step: join the remaining 3 nodes into an unrooted tree
    a, b, c = nodes
    
    dab = current_D[0, 1]
    dac = current_D[0, 2]
    dbc = current_D[1, 2]
    
    # For a 3-taxon unrooted tree with topology ((a,b),c):
    # Actually for 3 taxa, the unrooted tree has 3 leaves meeting at one internal node
    # Branch lengths:
    # len_a = (dab + dac - dbc) / 2
    # len_b = (dab + dbc - dac) / 2
    # len_c = (dac + dbc - dab) / 2
    len_a = max(0.0, (dab + dac - dbc) / 2.0)
    len_b = max(0.0, (dab + dbc - dac) / 2.0)
    len_c = max(0.0, (dac + dbc - dab) / 2.0)
    
    # Build final Newick
    result = f"({a}:{len_a:.4f},{b}:{len_b:.4f},{c}:{len_c:.4f})"
    
    return result
