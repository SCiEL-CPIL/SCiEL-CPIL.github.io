import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    t_start = time.time()
    time_limit = 140.0

    def elapsed():
        return time.time() - t_start

    def time_exceeded():
        return elapsed() > time_limit

    def make_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
        return U, V, W

    def check_valid(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    U, V, W = make_schoolbook(n, m, p)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]

    assert check_valid(U, V, W, n, m, p), "Schoolbook validation failed"

    # Try to reduce rank by finding a decomposition with fewer terms.
    # Strategy: For target rank r, randomly initialize U, V, W and try to
    # satisfy the tensor equations via iterative methods.
    
    # A more practical approach: try to find if the tensor can be decomposed
    # with rank r by checking if a random linear combination of schoolbook
    # terms can be replaced by a single term.
    
    # Let's try: for each pair of terms (a, b), check if we can write
    # term_a + term_b as a single term. This requires:
    # U_a + U_b = U_new, V_a + V_b = V_new, W_a + W_b = W_new
    # and U_new * V_new * W_new = U_a*V_a*W_a + U_b*V_b*W_b (as tensors)
    
    # This is generally not possible, but let's try a different angle:
    # Use the fact that we can apply invertible linear transformations
    # to U, V, W simultaneously (change of basis in the rank space).
    
    # The key insight: if we can find a rank-r decomposition, we can
    # transform it to have integer/half-integer entries.
    
    # For now, let's try a concrete small search:
    # Try all possible ways to combine 2 terms into 1 (reducing rank by 1)
    
    def try_combine_terms(U, V, W, a, b, n, m, p):
        """Try to combine terms a and b into a single term."""
        # After combining, we have rank-1 terms
        # The new term should satisfy:
        # For all (i,j,k): U_new[i*m+j]*V_new[j*p+k]*W_new[k*n+i] =
        #   U_a[i*m+j]*V_a[j*p+k]*W_a[k*n+i] + U_b[i*m+j]*V_b[j*p+k]*W_b[k*n+i]
        
        # In schoolbook, each term is a "unit" tensor, so their sum is
        # a tensor with two 1s. Can this be written as a product of three vectors?
        # Generally no, unless the two 1s are in positions that allow factorization.
        
        # For term (i1,j1,k1) and (i2,j2,k2), the sum tensor has 1s at
        # (i1,j1,k1) and (i2,j2,k2). This can be factored iff
        # (i1,j1,k1) and (i2,j2,k2) share at least one index value in a way
        # that allows rank-1 factorization.
        
        # Specifically, if i1==i2 and j1==j2 but k1!=k2, then:
        # U_new[i1*m+j1] = 1, V_new[j1*p+k1] = 1, V_new[j1*p+k2] = 1,
        # W_new[k1*n+i1] = 1, W_new[k2*n+i1] = 1
        # But this gives U*V*W = 1 at (i1,j1,k1) AND (i1,j1,k2), which is correct!
        # Wait, no: U_new[i1*m+j1]=1, V_new[j1*p+k1]=1, W_new[k1*n+i1]=1 gives 1 at (i1,j1,k1)
        # U_new[i1*m+j1]=1, V_new[j1*p+k2]=1, W_new[k2*n+i1]=1 gives 1 at (i1,j1,k2)
        # But we also get cross terms: U_new[i1*m+j1]=1, V_new[j1*p+k1]=1, W_new[k2*n+i1]=1
        # gives a spurious 1 at (i1,j1,k2) via the k1 path... 
        # Actually no, the product is U[i,j]*V[j,k]*W[k,i], so for fixed (i,j,k),
        # it's a single value. The issue is that V_new has two non-zero entries,
        # so for (i1,j1,k1): U=1, V[j1*p+k1]=1, W[k1*n+i1]=1 -> product=1 ✓
        # For (i1,j1,k2): U=1, V[j1*p+k2]=1, W[k2*n+i1]=1 -> product=1 ✓
        # But for (i1,j1,k1) via k2 path: U=1, V[j1*p+k1]=1 (not V[j1*p+k2]), so no cross term.
        # Actually the product is well-defined: for each (i,j,k), it's U[i,j]*V[j,k]*W[k,i].
        # So if V[j1*p+k1]=1 and V[j1*p+k2]=1, and W[k1*n+i1]=1 and W[k2*n+i1]=1,
        # then for (i1,j1,k1): U=1, V[j1*p+k1]=1, W[k1*n+i1]=1 -> 1
        # For (i1,j1,k2): U=1, V[j1*p+k2]=1, W[k2*n+i1]=1 -> 1
        # For other (i,j,k): at least one factor is 0.
        # So this works! We can merge two terms that share (i,j) but have different k.
        
        # Similarly, we can merge terms that share (j,k) but have different i,
        # or terms that share (i,k) but have different j.
        
        # Let's check if terms a and b share (i,j) with different k.
        # In schoolbook, term index = i*m*p + j*p + k
        idx_a = a
        i_a, j_a, k_a = idx_a // (m*p), (idx_a % (m*p)) // p, idx_a % p
        idx_b = b
        i_b, j_b, k_b = idx_b // (m*p), (idx_b % (m*p)) // p, idx_b % p
        
        can_merge = False
        merge_type = None
        
        if i_a == i_b and j_a == j_b and k_a != k_b:
            can_merge = True
            merge_type = 'ij'  # merge over k
        elif j_a == j_b and k_a == k_b and i_a != i_b:
            can_merge = True
            merge_type = 'jk'  # merge over i
        elif i_a == i_b and k_a == k_b and j_a != j_b:
            can_merge = True
            merge_type = 'ik'  # merge over j
        
        if not can_merge:
            return None
        
        # Create new U, V, W with rank-1
        new_U = U.copy()
        new_V = V.copy()
        new_W = W.copy()
        
        # Remove term b, modify term a to be the merged term
        # For 'ij' merge: same (i,j), different k
        if merge_type == 'ij':
            # U stays the same (only one non-zero at (i_a, j_a))
            # V: set both V[j_a*p+k_a, a] and V[j_a*p+k_b, a] to 1
            # W: set both W[k_a*n+i_a, a] and W[k_b*n+i_a, a] to 1
            new_V[j_a * p + k_b, a] = 1
            new_W[k_b * n + i_a, a] = 1
            # Zero out term b
            new_U[:, b] = 0
            new_V[:, b] = 0
            new_W[:, b] = 0
        elif merge_type == 'jk':
            # Same (j,k), different i
            # V stays, U and W get two entries
            new_U[i_b * m + j_a, a] = 1
            new_W[k_a * n + i_b, a] = 1
            new_U[:, b] = 0
            new_V[:, b] = 0
            new_W[:, b] = 0
        elif merge_type == 'ik':
            # Same (i,k), different j
            # U stays, V and W get two entries
            new_U[i_a * m + j_b, a] = 1
            new_V[j_b * p + k_a, a] = 1
            new_U[:, b] = 0
            new_V[:, b] = 0
            new_W[:, b] = 0
        
        # Remove the zeroed column
        mask = np.ones(new_U.shape[1], dtype=bool)
        mask[b] = False
        new_U = new_U[:, mask]
        new_V = new_V[:, mask]
        new_W = new_W[:, mask]
        
        return new_U, new_V, new_W

    # Try all pairs of terms to merge
    current_U, current_V, current_W = U.copy(), V.copy(), W.copy()
    current_rank = current_U.shape[1]
    
    # Sort terms to enable merging: group by (i,j), (j,k), (i,k)
    # Try merging greedily
    
    merged_any = True
    while merged_any and not time_exceeded():
        merged_any = False
        for a in range(current_rank - 1):
            if time_exceeded():
                break
            for b in range(a + 1, current_rank):
                result = try_combine_terms(current_U, current_V, current_W, a, b, n, m, p)
                if result is not None:
                    new_U, new_V, new_W = result
                    if check_valid(new_U, new_V, new_W, n, m, p):
                        current_U, current_V, current_W = new_U, new_V, new_W
                        current_rank = new_U.shape[1]
                        merged_any = True
                        if current_rank < best_rank:
                            best_U, best_V, best_W = new_U.copy(), new_V.copy(), new_W.copy()
                            best_rank = current_rank
                        break
            if merged_any:
                break
    
    # Final validation
    if not check_valid(best_U, best_V, best_W, n, m, p):
        best_U, best_V, best_W = make_schoolbook(n, m, p)
        best_rank = best_U.shape[1]

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
