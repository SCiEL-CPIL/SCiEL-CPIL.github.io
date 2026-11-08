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
    # Condition: min pairwise distance >= max norm
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v = 4 + 4 - 2*u.v = 8 - 2*u.v
    # max norm = sqrt(4) = 2, so max norm^2 = 4
    # We need dist >= max_norm => dist^2 >= max_norm^2 => 8 - 2*u.v >= 4 => 2*u.v <= 4 => u.v <= 2
    # So conflict if u.v > 2, i.e., u.v >= 3
    conflict = G > 2
    np.fill_diagonal(conflict, False)
    
    # Convert to boolean matrix for efficient operations
    A = conflict.astype(np.float64)
    
    # Spectral Approach:
    # 1. Compute the Graph Laplacian L = D - A
    # 2. Find the eigenvector corresponding to the second smallest eigenvalue (Fiedler vector)
    #    or a set of eigenvectors for a low-dimensional embedding.
    # 3. The Fiedler vector often separates the graph into clusters or provides a 1D ordering
    #    where independent sets can be approximated by selecting vertices with similar values
    #    or using a thresholding strategy on the embedding.
    
    # However, for Independent Set, we want to avoid edges. 
    # In the Laplacian spectrum, the null space corresponds to connected components.
    # The Fiedler vector (2nd smallest) indicates the "cut" of the graph.
    # Vertices with similar Fiedler values are "close" in the spectral sense.
    # If the graph is bipartite-like or has a strong structure, one side of the bipartition
    # might be an independent set.
    
    # Let's compute the Laplacian
    degrees = A.sum(axis=1)
    D_mat = np.diag(degrees)
    L = D_mat - A
    
    # Compute eigenvalues and eigenvectors
    # n=5280, eigh is O(n^3) which might be too slow for 5280x5280 in pure numpy?
    # 5280^3 ~ 1.4e11. This is likely too slow for 140s in pure python/numpy without specialized libs.
    # Let's check if we can use a faster approximation or smaller subspace.
    # Actually, numpy.linalg.eigh on 5000x5000 matrix takes several seconds to minutes depending on hardware.
    # Let's try to estimate: 5280^3 * constant. It might be borderline.
    # Alternative: Use the fact that the graph is regular or semi-regular?
    # Weight-4 vectors in R^11.
    # Degree of a vertex: How many other weight-4 vectors have dot product > 2?
    # Dot product > 2 means they share at least 3 positions with the same sign, or 4 with mixed signs?
    # If they share 4 positions: dot product is 4 if signs match, less if not.
    # Let's just try the spectral method but with a time guard. If it's too slow, we fall back to greedy.
    
    best_in_set = None
    best_count = 0
    
    # Fallback: Fast Greedy (Minimum Degree) to ensure we have a baseline
    # This is much faster than spectral if spectral is too slow.
    # Let's run a few quick greedy restarts first to have a safe baseline.
    np.random.seed(42)
    
    # Quick baseline: 50 random greedy restarts
    for r in range(50):
        if time.time() > deadline - 60:
            break
        perm = np.random.permutation(n)
        in_set = np.zeros(n, dtype=bool)
        remaining = np.ones(n, dtype=bool)
        count = 0
        for idx in perm:
            if not remaining[idx]:
                continue
            in_set[idx] = True
            count += 1
            neighbors = A[idx]
            remaining[neighbors] = False
            remaining[idx] = False
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
    
    # Now try Spectral Method
    # To speed up eigh, we can use the fact that L is symmetric.
    # But 5280 is large. Let's try to compute only a few eigenpairs using scipy if available?
    # The prompt says NumPy and SciPy are available.
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    # Convert to sparse matrix for faster eigsh
    L_sparse = csr_matrix(L)
    
    # We want the smallest eigenvalues.
    # eigsh finds the k largest or smallest.
    # We want the Fiedler vector (k=2, smallest).
    # Actually, for independent set, we might want to look at the eigenvectors of the adjacency matrix A
    # or the Laplacian.
    # Let's try to find the first few eigenvectors of the Laplacian.
    
    try:
        # Compute 10 smallest eigenvalues and their eigenvectors
        # sigma=0 to find smallest
        # This might be slow. Let's set a timeout or try a smaller k.
        # If this takes > 60s, we skip.
        
        # Let's try a different spectral approach:
        # The adjacency matrix A has eigenvalues.
        # If the graph is bipartite, the spectrum is symmetric.
        # The independent set can be related to the bipartition.
        
        # Let's use eigsh on A to find the largest magnitude eigenvalues?
        # Or stick to L.
        
        # To be safe with time, let's try to compute the embedding using a randomized method
        # or just the top few eigenvectors of A (which are related to L).
        
        # Let's try eigsh on L for the smallest eigenvalues.
        # k=5 smallest eigenvalues.
        # v0 = random vector to avoid null space issues if any.
        v0 = np.random.normal(size=n)
        
        # eigsh returns eigenvalues in ascending order if sigma is not specified?
        # No, eigsh finds eigenvalues closest to sigma.
        # To get smallest, use sigma=0, which finds eigenvalues closest to 0.
        # This requires a shift-invert mode which is expensive.
        # Better: use 'SM' (smallest magnitude) but it's slow for sparse.
        # Alternative: Use the fact that L = D - A.
        # If we want the Fiedler vector, it's the 2nd smallest.
        
        # Let's try a different tactic:
        # Use the eigenvectors of the adjacency matrix A.
        # The largest eigenvalue corresponds to the principal component.
        # For a regular graph, the independent set is related to the bipartite structure.
        
        # Let's just try to get the top 5 eigenvectors of A (largest eigenvalues).
        # This is usually faster than smallest eigenvalues of L.
        # A is sparse.
        A_sparse = csr_matrix(A)
        
        # Find top 5 eigenvectors of A
        # k=5, which='LM' (largest magnitude)
        vals, vecs = eigsh(A_sparse, k=5, which='LM', v0=v0)
        
        # Use the eigenvector corresponding to the largest eigenvalue?
        # Or the second largest?
        # In many graphs, the second eigenvector (Fiedler-like for A) provides a good partition.
        # Let's try sorting vertices by the values of the second eigenvector (index 1 usually, but order depends on vals).
        
        # Sort eigenvalues to be sure
        idx_sort = np.argsort(vals)
        vals = vals[idx_sort]
        vecs = vecs[:, idx_sort]
        
        # Let's try using the eigenvector with the second largest absolute value?
        # Or simply the first few.
        
        # Strategy: 
        # 1. Take the eigenvector corresponding to the largest eigenvalue (principal component).
        # 2. Take the eigenvector corresponding to the second largest.
        # 3. Sort vertices by the second eigenvector.
        # 4. Select a contiguous block? Or select based on sign?
        
        # If the graph is bipartite, the eigenvector for the largest eigenvalue (if it's positive)
        # or the second largest (if the graph is connected and not bipartite) will have positive and negative values.
        # Selecting all vertices with positive values (or negative) might give an independent set if the graph is bipartite.
        # Weight-4 vectors in R^11: Is the conflict graph bipartite?
        # Two vectors conflict if dot > 2.
        # Let v be a vector. -v is also in the set.
        # dot(v, -v) = -4. No conflict.
        # dot(v, w) > 2.
        # This graph is likely not bipartite.
        
        # Let's try a heuristic based on the spectral embedding:
        # Project vertices onto the first 2 eigenvectors.
        # Then use a grid or clustering to find a dense cluster of non-conflicting nodes?
        # That's complex.
        
        # Simpler: 
        # Use the eigenvector of the largest eigenvalue of A.
        # Let x be this eigenvector.
        # Select all i where x[i] > 0.
        # Check if this is an independent set. If not, prune it.
        
        # Let's try multiple eigenvectors.
        for i in range(min(5, vecs.shape[1])):
            if time.time() > deadline - 20:
                break
            x = vecs[:, i]
            # Normalize x
            x = x - np.mean(x)
            
            # Try selecting positive part
            mask = x > 0
            in_set = mask.copy()
            
            # Prune: remove conflicts
            # If in_set has conflicts, we need to remove some.
            # Greedy prune:
            # Order vertices in in_set by x value (descending)
            indices = np.where(in_set)[0]
            indices = indices[np.argsort(x[indices])[::-1]]
            
            final_set = np.zeros(n, dtype=bool)
            remaining = in_set.copy()
            
            for idx in indices:
                if not remaining[idx]:
                    continue
                final_set[idx] = True
                # Remove neighbors
                neighbors = A[idx]
                remaining[neighbors] = False
                remaining[idx] = False
            
            count = final_set.sum()
            if count > best_count:
                best_count = count
                best_in_set = final_set.copy()
                
            # Try selecting negative part
            mask = x < 0
            in_set = mask.copy()
            indices = np.where(in_set)[0]
            indices = indices[np.argsort(x[indices])[::-1]] # Sort by magnitude? Or just order
            
            final_set = np.zeros(n, dtype=bool)
            remaining = in_set.copy()
            
            for idx in indices:
                if not remaining[idx]:
                    continue
                final_set[idx] = True
                neighbors = A[idx]
                remaining[neighbors] = False
                remaining[idx] = False
            
            count = final_set.sum()
            if count > best_count:
                best_count = count
                best_in_set = final_set.copy()
                
    except Exception as e:
        # If spectral fails, rely on the greedy baseline
        pass
    
    # Local Search on the best solution
    if time.time() < deadline - 5 and best_in_set is not None:
        in_set = best_in_set.copy()
        removed = ~in_set
        
        # Try to add back removed vertices
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            
            has_conflict = A[:, in_set].sum(axis=1) > 0
            candidates = removed & ~has_conflict
            
            if not candidates.any():
                break
            
            cand_indices = np.where(candidates)[0]
            added_in_pass = 0
            for v_idx in cand_indices:
                if time.time() > deadline - 5:
                    break
                if not A[v_idx, in_set].any():
                    in_set[v_idx] = True
                    removed[v_idx] = False
                    added_in_pass += 1
            
            if added_in_pass == 0:
                break
        
        best_count = in_set.sum()
        best_in_set = in_set
    
    result = vectors[best_in_set]
    return result
# EVOLVE-BLOCK-END
