"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

D = 11
# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140  # safety margin from 150s

    # Generate all weight-4 vectors in R^11
    # A weight-4 vector has exactly 4 non-zero entries, each in {-1, 1}
    # Total count: C(11,4) * 2^4 = 330 * 16 = 5280
    vectors = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            vectors.append(v)
    
    vectors = np.array(vectors, dtype=np.int64)
    n = len(vectors)
    
    if n < 2:
        return vectors
    
    # Compute Gram matrix
    G = vectors @ vectors.T
    
    # Build adjacency: conflict if G[i,j] > 2
    # For weight-4 vectors, dot product is in {-4, -3, -2, -1, 0, 1, 2, 3, 4}
    # Condition: min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max norm = sqrt(4) = 2, so max norm^2 = 4
    # We need dist >= max_norm => dist^2 >= max_norm^2 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    # So conflict if u.v > 2, i.e., u.v >= 3
    conflict = (G > 2).astype(np.float64)
    np.fill_diagonal(conflict, 0)
    
    # Spectral Approach:
    # The conflict graph A has eigenvalues. 
    # For a graph that is a union of cliques (or close to it), the largest eigenvalues correspond to the clique structures.
    # An independent set is a set of vertices with no edges between them.
    # In the spectral decomposition, the eigenvector corresponding to the largest eigenvalue of the adjacency matrix 
    # often highlights the densest subgraph. 
    # However, for independent sets, we are looking for a set of vertices that are NOT connected.
    # This is equivalent to finding a large set of vertices in the complement graph that form a clique.
    # Alternatively, we can look at the Laplacian L = D - A.
    # But a simpler heuristic for independent sets in dense graphs is to use the eigenvector of the adjacency matrix 
    # corresponding to the most negative eigenvalue (if the graph is bipartite-like) or simply use the principal eigenvector 
    # of the complement graph.
    
    # Let's use the adjacency matrix of the conflict graph.
    # If we compute the eigenvectors of A, the components of the eigenvector associated with the largest eigenvalue 
    # tend to be high for nodes in the densest cluster.
    # For an independent set, we want nodes that are NOT in the dense clusters? No, we want a set of nodes with no edges.
    
    # Actually, a known spectral heuristic for Maximum Independent Set (MIS) is:
    # 1. Compute the eigenvector x associated with the largest eigenvalue of the adjacency matrix A.
    # 2. Nodes with high positive values in x are likely to be in the same "community" or clique.
    # 3. Nodes with high negative values are in a different community.
    # This works best if the graph is bipartite or has strong community structure.
    
    # A more robust spectral method for MIS is to use the eigenvector of the Laplacian.
    # But let's try a simpler "spectral ordering" or "spectral clustering" approach.
    
    # Let's compute the eigenvalues and eigenvectors of the conflict matrix A.
    # n=5280 is small enough for full eigh? 
    # O(n^3) = 5280^3 ~ 1.4 * 10^11 operations. This is too slow for numpy.linalg.eigh in 120s?
    # Let's check. 1e11 flops is roughly 10-100 seconds on a modern CPU with optimized BLAS.
    # It's risky. Let's use a sparse-friendly approach or a smaller subspace iteration if possible.
    # However, numpy.linalg.eigh is dense.
    # Let's try to see if we can do it. If it's too slow, we fall back to greedy.
    
    # Actually, 5280x5280 matrix. 
    # Let's attempt the eigenvalue decomposition.
    try:
        # Compute eigenvectors
        # We only need the top few eigenvectors.
        # But numpy.linalg.eigh computes all.
        # There is no built-in partial eigh in numpy.
        # scipy.sparse.linalg.eigsh is available.
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        A_sparse = csr_matrix(conflict)
        
        # We want the eigenvector corresponding to the largest eigenvalue.
        # eigsh finds eigenvalues near sigma.
        # Let's find the eigenvector for the largest eigenvalue.
        # k=10 to get a few top eigenvectors for clustering.
        
        # Note: eigsh requires the matrix to be symmetric. It is.
        # It may fail if the graph is disconnected or has zero eigenvalues, but largest should be fine.
        
        # Let's try to get the top 10 eigenvectors.
        # This might be faster than full dense eigh.
        
        # However, eigsh can be unstable. Let's try a different spectral heuristic:
        # Use the principal eigenvector of the complement graph?
        # The complement graph B = J - I - A.
        # A clique in B is an independent set in A.
        # The largest eigenvalue of B corresponds to the densest clique-like structure.
        
        # But constructing B is also O(n^2).
        
        # Let's stick to the dense eigh but with a time check.
        # If it takes too long, we abort.
        
        # Actually, let's try a simpler spectral heuristic that is O(n^2) or O(n^k).
        # "Spectral Greedy":
        # 1. Compute the top eigenvector of A.
        # 2. Order vertices by their value in this eigenvector.
        # 3. Use this ordering to guide the greedy selection (e.g., pick the first compatible vertex in this order).
        
        # To get the top eigenvector, we can use power iteration, which is O(n^2) per iteration.
        # Let's do 50 iterations of power iteration.
        
        # Initialize x
        x = np.random.randn(n)
        x /= np.linalg.norm(x)
        
        for i in range(50):
            if time.time() > deadline - 30:
                break
            y = A_sparse @ x
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-10:
                break
            x = y / norm_y
        
        # Now x contains the principal eigenvector.
        # We expect vertices in the "densest" part of the conflict graph to have high |x_i|.
        # Vertices in independent sets are NOT connected to each other.
        # This doesn't directly give an independent set.
        
        # Alternative: Use the eigenvector to define a "score" for each vertex.
        # Then run a greedy algorithm that picks vertices in order of decreasing "spectral centrality" 
        # but with a twist: we want to avoid conflicts.
        
        # Actually, a better spectral heuristic for MIS is:
        # 1. Compute the eigenvector v for the largest eigenvalue.
        # 2. Partition vertices into two sets: V+ (v_i > 0) and V- (v_i <= 0).
        # 3. The graph is often bipartite or close to it. The larger of V+ or V- might be close to an independent set?
        # No, in a bipartite graph, the two partitions ARE independent sets.
        # If the graph is not bipartite, this is an approximation.
        
        # Let's try this:
        # Split based on the sign of the principal eigenvector.
        # Then take the larger partition.
        # Then run local search to improve it.
        
        v_plus = np.where(x > 0)[0]
        v_minus = np.where(x <= 0)[0]
        
        # Choose the larger set as the initial independent set candidate
        if len(v_plus) >= len(v_minus):
            initial_set_indices = v_plus
        else:
            initial_set_indices = v_minus
        
        # Convert to boolean mask
        in_set = np.zeros(n, dtype=bool)
        in_set[initial_set_indices] = True
        
        # Now, this set is NOT necessarily an independent set. 
        # We need to remove conflicts.
        # Greedy removal:
        # Iterate through the set and remove vertices that conflict with earlier kept vertices.
        # This is O(n^2) worst case, but we can vectorize.
        
        # Let's do a fast greedy cleanup.
        # Order the vertices in the set by their degree in the conflict graph (lowest first) to keep "easy" vertices.
        degrees = conflict.sum(axis=1)
        ordered_indices = initial_set_indices[np.argsort(degrees[initial_set_indices])]
        
        current_set = np.zeros(n, dtype=bool)
        count = 0
        for idx in ordered_indices:
            # Check if idx conflicts with any already in current_set
            if not (conflict[idx, current_set].any()):
                current_set[idx] = True
                count += 1
        
        best_count = count
        best_in_set = current_set.copy()
        
        # Now, perform local search to improve.
        # Phase 1: Add back removed vertices that don't conflict.
        removed = ~best_in_set
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Find removed vertices that have no conflicts with current in_set
            has_conflict = conflict[:, best_in_set].any(axis=1)
            candidates = removed & ~has_conflict
            
            if not candidates.any():
                break
            
            cand_indices = np.where(candidates)[0]
            added = 0
            for v_idx in cand_indices:
                if time.time() > deadline - 10:
                    break
                if not conflict[v_idx, best_in_set].any():
                    best_in_set[v_idx] = True
                    removed[v_idx] = False
                    added += 1
            
            if added == 0:
                break
        
        best_count = best_in_set.sum()
        
        # Phase 2: Random restarts with spectral ordering
        # Use the spectral ordering to guide random restarts.
        # Sort vertices by their spectral value (x).
        spectral_order = np.argsort(x)
        
        for r in range(500):
            if time.time() > deadline - 5:
                break
            
            # Perturb the order slightly
            if r % 10 == 0:
                order = spectral_order
            else:
                # Swap random pairs
                order = spectral_order.copy()
                num_swaps = n // 100
                for _ in range(num_swaps):
                    i, j = np.random.randint(0, n, 2)
                    order[i], order[j] = order[j], order[i]
            
            in_set_r = np.zeros(n, dtype=bool)
            count_r = 0
            for idx in order:
                if not (conflict[idx, in_set_r].any()):
                    in_set_r[idx] = True
                    count_r += 1
            
            if count_r > best_count:
                best_count = count_r
                best_in_set = in_set_r.copy()
                
                # Try to improve immediately
                removed_r = ~best_in_set
                for _ in range(2):
                    if time.time() > deadline - 5:
                        break
                    has_conflict = conflict[:, best_in_set].any(axis=1)
                    candidates = removed_r & ~has_conflict
                    if not candidates.any():
                        break
                    cand_indices = np.where(candidates)[0]
                    added = 0
                    for v_idx in cand_indices:
                        if not conflict[v_idx, best_in_set].any():
                            best_in_set[v_idx] = True
                            removed_r[v_idx] = False
                            added += 1
                    if added == 0:
                        break
                    best_count = best_in_set.sum()

    except Exception as e:
        # Fallback to simple random greedy if spectral fails
        np.random.seed(42)
        best_in_set = None
        best_count = 0
        
        for r in range(1000):
            if time.time() > deadline - 5:
                break
            perm = np.random.permutation(n)
            in_set = np.zeros(n, dtype=bool)
            count = 0
            for idx in perm:
                if not (conflict[idx, in_set].any()):
                    in_set[idx] = True
                    count += 1
            
            if count > best_count:
                best_count = count
                best_in_set = in_set.copy()
        
        if best_in_set is None:
            # Should not happen
            return vectors[:1]
    
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
