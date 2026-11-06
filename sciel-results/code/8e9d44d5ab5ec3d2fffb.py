"""
Maximum Cut solver using Spectral Initialization + Multi-Flip Best-Improvement Local Search.

Strategy:
1. Spectral Initialization: Use the second eigenvector (Fiedler vector) of the Laplacian.
   This provides a high-quality starting point that is often globally near-optimal for
   random graphs, much better than random starts.
2. Multi-Flip Best-Improvement Local Search (BILS):
   - Standard 1-flip local search often gets stuck in local optima where no single flip
     improves the cut, but a pair of flips does.
   - We vectorize the evaluation of all 2-vertex flips.
   - If the best 2-flip improves the cut, we apply it and repeat.
   - We alternate between 1-flip and 2-flip improvements.
3. Time-bounded execution with a safety margin.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def partition(adjacency):
    """
    Solve Maximum Cut using Spectral Initialization and Multi-Flip Best-Improvement Local Search.
    
    Args:
        adjacency: bool matrix (n x n)
    Returns:
        0/1 label per vertex (numpy array of shape (n,))
    """
    start_time = time.time()
    DEADLINE = 145.0  # seconds, leave 5s safety margin for the 150s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    # Convert to numpy array
    adj = np.array(adjacency, dtype=np.float32)
    
    # Ensure symmetry
    adj = (adj + adj.T) / 2.0
    # Zero out diagonal if not already
    np.fill_diagonal(adj, 0.0)
    
    degrees = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the cut size efficiently."""
        # cut = 0.5 * sum_{i,j} adj[i,j] * (1 - s_i * s_j) where s_i in {-1, 1}
        # Or simpler: for binary labels 0/1:
        # cut = sum_{i,j} adj[i,j] * (l_i != l_j)
        # Vectorized: 
        # Let l be 0/1. 
        # same = l @ adj @ l
        # total_edges = 0.5 * sum(adj)
        # cut = total_edges - 0.5 * (l @ adj @ l + l @ adj @ (1-l)) ... this is getting messy.
        
        # Simplest robust vectorized way:
        # Create a matrix D = diag(l). 
        # Actually, just use the difference matrix.
        # l is (n,). 
        # diff = l[:, None] != l[None, :]
        # cut = np.sum(adj * diff)
        # This creates an n x n boolean matrix. For n=100, this is 10k elements. Very fast.
        diff = (labels[:, None] != labels[None, :]).astype(np.float32)
        return float(np.sum(adj * diff))
    
    def spectral_init():
        """
        Initialize using the Fiedler vector of the Laplacian.
        L = D - A
        """
        # Build Laplacian
        D = np.diag(degrees)
        L = D - adj
        
        # Convert to sparse for eigensolver
        L_sp = sp.csr_matrix(L)
        
        try:
            # Find the smallest eigenvalues. 
            # The smallest is 0 (constant vector). The second smallest is the Fiedler vector.
            # We want the eigenvector corresponding to the second smallest eigenvalue.
            # spla.eigsh can find k smallest.
            k = min(10, n - 1)
            if k < 2:
                return np.zeros(n, dtype=int)
            
            # We need at least 2 smallest.
            # Use which='SM' to find smallest magnitude.
            # Note: For connected graphs, lambda_0 = 0, lambda_1 > 0.
            # For disconnected, there are multiple 0s.
            # We want the vector that best separates the graph.
            
            # Sometimes eigsh fails or returns trivial vectors. 
            # We'll grab the first few non-trivial ones.
            
            # To be safe, let's just find the 2 smallest.
            if n <= 10:
                # Dense eig for small n
                w, v = np.linalg.eigh(L)
                # Sort by eigenvalue
                idx = np.argsort(w)
                # The first is the constant vector. The second is Fiedler.
                # But if disconnected, there might be multiple 0s.
                # We pick the one with the largest variance or just the 2nd index.
                # Let's try index 1.
                fiedler = v[:, idx[1]]
            else:
                # Find 2 smallest eigenvalues
                w, v = spla.eigsh(L_sp, k=2, which='SM')
                # eigsh returns in increasing order of magnitude? No, not necessarily.
                # Sort them.
                idx = np.argsort(w)
                # The smallest is 0. The next is lambda_1.
                # If the graph is disconnected, w[0] and w[1] might both be close to 0.
                # In that case, the eigenvectors are indicators of components.
                # This is actually good for max cut on disconnected graphs (separate components).
                # But we want a single label vector.
                # If we pick the 2nd smallest, we get a vector that separates the two largest components (if 2 components).
                
                # Let's just pick the eigenvector associated with the 2nd smallest eigenvalue.
                fiedler = v[:, idx[1]]
            
            # Map signs to 0/1
            # If fiedler[i] > 0, label 1, else 0
            labels = (fiedler > 0).astype(int)
            return labels
            
        except Exception:
            # Fallback to random if spectral fails
            return np.random.randint(0, 2, size=n)
    
    def one_flip_improve(labels, max_passes=100):
        """
        Standard 1-flip hill climbing.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            # Compute gain for each vertex flip
            # gain[v] = 2 * (neighbors with diff label) - degree[v]
            # Let l be labels. 
            # For v, let same = sum(adj[v, l==l[v]]), diff = sum(adj[v, l!=l[v]])
            # gain = diff - same = diff - (deg[v] - diff) = 2*diff - deg[v]
            
            # Vectorized gain calculation:
            # l is 0/1.
            # We want for each v: sum_{u} adj[v,u] * (l[u] != l[v])
            # Let M = adj * (l[:, None] != l[None, :])
            # Then M.sum(axis=1) gives diff_count for each v.
            # This is O(n^2) per pass. For n=100, 10k ops. Fast.
            
            diff_mask = (labels[:, None] != labels[None, :]).astype(np.float32)
            diff_counts = adj @ diff_mask
            
            gains = 2.0 * diff_counts - degrees
            
            # Find the best flip
            best_v = np.argmax(gains)
            if gains[best_v] > 1e-6:
                labels[best_v] = 1 - labels[best_v]
                current_cut += gains[best_v]
                improved = True
            
            if not improved:
                break
                
        return labels, current_cut
    
    def two_flip_improve(labels, max_passes=50):
        """
        Best-improvement local search considering 2-vertex flips.
        This can escape local optima of 1-flip search.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        for _ in range(max_passes):
            if time.time() > DEADLINE - 5:
                break
            
            # Evaluate all pairs (i, j) with i < j
            # Flipping i and j changes the cut size by:
            # delta = gain_i + gain_j + correction
            # Where correction depends on if i and j are connected and their relative labels.
            
            # Let's derive the exact change in cut size when flipping i and j.
            # Let s be the sign vector (-1, 1). l is 0/1.
            # Cut = 0.5 * sum_{u,v} A_{uv} (1 - s_u s_v)
            # Flipping i changes s_i -> -s_i.
            # Flipping j changes s_j -> -s_j.
            
            # Change in term for (u, v):
            # If neither is i or j: 0
            # If u=i, v=j: 
            #   Old: 1 - s_i s_j
            #   New: 1 - (-s_i)(-s_j) = 1 - s_i s_j
            #   Change: 0
            #   Wait, this logic is for the objective function directly.
            
            # Let's use the 0/1 label formulation.
            # Cut = sum_{u,v} A_{uv} (l_u != l_v)
            
            # Flip i: l_i' = 1 - l_i
            # Flip j: l_j' = 1 - l_j
            
            # Terms involving i:
            # For v != j: A_{iv} (l_i != l_v) -> A_{iv} (1-l_i != l_v)
            # Change for v != j: A_{iv} [ (1-l_i != l_v) - (l_i != l_v) ]
            # If l_i == l_v: (1-l_i != l_v) is True (1), (l_i != l_v) is False (0). Change +1.
            # If l_i != l_v: (1-l_i != l_v) is False (0), (l_i != l_v) is True (1). Change -1.
            # So change is +1 if same, -1 if diff.
            # Sum over v != j: sum_{v != j} A_{iv} * (1 if l_i==l_v else -1)
            # = sum_{v != j} A_{iv} * (2 * (l_i==l_v) - 1)
            
            # Similarly for j: sum_{u != i} A_{ju} * (2 * (l_j==l_u) - 1)
            
            # Terms involving (i, j):
            # Old: A_{ij} (l_i != l_j)
            # New: A_{ij} (1-l_i != 1-l_j) = A_{ij} (l_i != l_j)
            # Change: 0.
            
            # So total change = 
            # Sum_{v != j} A_{iv} (2(l_i==l_v) - 1) + Sum_{u != i} A_{ju} (2(l_j==l_u) - 1)
            
            # Let's compute this vectorized for all pairs.
            # This is complex to vectorize directly for all pairs O(n^2) pairs each O(n) work -> O(n^3).
            # For n=100, n^3 = 1,000,000 ops. This is fast in numpy.
            
            # However, we can be smarter.
            # Let g_i be the gain of flipping i alone.
            # g_i = Sum_{v} A_{iv} (2(l_i==l_v) - 1)
            # Note: The term for v=j is included in g_i.
            # Term for v=j in g_i: A_{ij} (2(l_i==l_j) - 1)
            
            # When we flip both i and j:
            # Change = g_i + g_j - 2 * (Term for v=j in g_i)
            # Why subtract 2 * Term?
            # g_i includes the effect of flipping i on the edge (i,j) assuming j is fixed.
            # g_j includes the effect of flipping j on the edge (j,i) assuming i is fixed.
            # But when both flip, the edge (i,j) change is 0.
            # In g_i, the change for edge (i,j) is C_ij = A_{ij} (2(l_i==l_j) - 1).
            # In g_j, the change for edge (j,i) is C_ji = A_{ji} (2(l_j==l_i) - 1) = C_ij.
            # So g_i + g_j includes C_ij + C_ji = 2 * C_ij.
            # The actual change for edge (i,j) is 0.
            # So we must subtract 2 * C_ij.
            
            # So Delta(i,j) = g_i + g_j - 2 * A_{ij} * (2(l_i==l_j) - 1)
            
            # Let's compute g (gains for 1-flip)
            diff_mask = (labels[:, None] != labels[None, :]).astype(np.float32)
            # same_mask = 1 - diff_mask
            same_mask = 1.0 - diff_mask
            
            # g_i = Sum_v A_{iv} (2 * same_{iv} - 1)
            # Note: 2*same - 1 is 1 if same, -1 if diff.
            # This is equivalent to: 2 * (A @ same_mask) - degrees
            # Or: (A @ (2*same_mask - 1))
            
            signs = 2.0 * same_mask - 1.0  # 1 if same, -1 if diff
            g = adj @ signs
            
            # Now compute Delta(i,j) for all i < j
            # Delta(i,j) = g[i] + g[j] - 2 * adj[i,j] * signs[i,j]
            # Note: signs[i,j] is 1 if same, -1 if diff.
            
            # Create matrix of g[i] + g[j]
            # G_sum = g[:, None] + g[None, :]
            G_sum = g[:, None] + g[None, :]
            
            # Correction term: 2 * adj * signs
            # We only care about i < j.
            # Let's compute the full matrix, then mask.
            Correction = 2.0 * adj * signs
            
            Delta = G_sum - Correction
            
            # Mask out i >= j and i == j
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            Delta_masked = np.where(mask, Delta, -np.inf)
            
            # Find the best pair
            flat_idx = np.argmax(Delta_masked)
            i, j = np.unravel_index(flat_idx, (n, n))
            
            best_gain = Delta_masked[i, j]
            
            if best_gain > 1e-6:
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                current_cut += best_gain
            else:
                break
                
        return labels, current_cut
    
    # Phase 1: Spectral Initialization
    labels = spectral_init()
    cut_val = cut_size(labels)
    
    # Phase 2: 1-Flip Improvement
    labels, cut_val = one_flip_improve(labels, max_passes=200)
    
    # Phase 3: 2-Flip Improvement
    labels, cut_val = two_flip_improve(labels, max_passes=100)
    
    # Phase 4: Alternate 1 and 2 flips until no improvement or time limit
    # We do a few rounds of this to ensure convergence.
    for round_num in range(10):
        if time.time() > DEADLINE - 10:
            break
        
        # 1-flip pass
        new_labels, new_cut = one_flip_improve(labels, max_passes=50)
        if new_cut > cut_val:
            labels = new_labels
            cut_val = new_cut
        else:
            # If 1-flip didn't improve, try 2-flip
            new_labels, new_cut = two_flip_improve(labels, max_passes=50)
            if new_cut > cut_val:
                labels = new_labels
                cut_val = new_cut
            else:
                # No improvement from either, we might be stuck or at optimum.
                # Try a few random restarts of local search if time permits?
                # Or just break.
                # Let's try a small random perturbation to escape local optimum.
                if time.time() < DEADLINE - 20:
                    # Perturb 5% of vertices
                    perturbed = labels.copy()
                    num_perturb = max(1, int(0.05 * n))
                    indices = np.random.choice(n, size=num_perturb, replace=False)
                    perturbed[indices] = 1 - perturbed[indices]
                    
                    p_labels, p_cut = one_flip_improve(perturbed, max_passes=50)
                    if p_cut > cut_val:
                        labels = p_labels
                        cut_val = p_cut
                    else:
                        p_labels, p_cut = two_flip_improve(p_labels, max_passes=50)
                        if p_cut > cut_val:
                            labels = p_labels
                            cut_val = p_cut
                        else:
                            break
                else:
                    break
    
    return labels

# EVOLVE-BLOCK-END
