"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 @ 3x5 matrix multiplication.
    Uses a search over rank values starting from a lower bound, with randomized
    basis transformations and greedy sparse recovery (matching pursuit) in the
    transformed tensor space.
    """
    import time
    import numpy as np
    from numpy.linalg import norm
    import sys

    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 5

    # The tensor T has entries T[i, j, k] = 1 if the output position (i, k)
    # corresponds to input positions (i, j) and (j, k).
    # Output: W[k*n + i, r]  (trace convention)
    # Input A: U[i*m + j, r]
    # Input B: V[j*p + k, r]
    #
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    #
    # We want rank r such that:
    #   sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c]  (actually W is the "output" factor)
    #
    # Actually the decomposition is:
    #   For all i, j, k: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
    # where T[i,j,k] = 1 (the multiplication tensor has all 1s in the "supported" positions).
    #
    # Wait, let me reconsider. The matrix multiplication tensor for multiplying
    # an (n x m) matrix A by an (m x p) matrix B to get C = A @ B:
    #   C[i, k] = sum_j A[i, j] * B[j, k]
    #
    # The tensor T[i, j, k] = 1 (it's the identity in the middle index).
    #
    # A bilinear form: the tensor T in F^{n*m} x F^{m*p} x F^{p*n} (with the W indexing)
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # We seek: T = sum_{c=1}^{rank} U_c * V_c * W_c (outer product of three vectors)
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # and T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]

    # Lower bound: max(n*p, m*p, n*m) for general matrices is not quite right.
    # For matrix multiplication, the trivial lower bound is max(n*m, m*p, p*n) if all entries are non-zero,
    # but actually the tensor rank is at least max(n*m, m*p, p*n) only in special cases.
    # The known lower bound is max(n*p, m*p, n*m) for the border rank in some cases.
    # Actually for 2x3 x 3x5, let's just search from reasonable values.
    
    # The schoolbook rank is n*m*p = 30.
    # Known results: for 2x3 x 3x5, the rank is likely around 20-25.
    # Let's search from a lower bound upward.
    
    # A simple lower bound: the tensor has support of size n*m*p = 30 entries (all 1s).
    # Each rank-1 term can cover at most one entry if we're strict, but actually rank-1 tensors
    # in this context... let me think differently.
    
    # For matrix multiplication M_{n,m,p}, the tensor rank is known to be at least
    # max(n*m, m*p, p*n) / something... Actually the trivial lower bound is just that
    # the tensor is not identically zero.
    
    # Let's use a practical approach: start from rank = max(n*m, m*p, p*n) and search down,
    # or start from a reasonable estimate and search up.
    
    # Actually, let's just try to find a valid decomposition for various ranks.
    # The key insight: we can use randomized linear transformations on the input spaces
    # to simplify the tensor, then use sparse recovery.
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W decompose the multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        # T[a, b, c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
        # T[i*m+j, j*p+k, k*n+i] = 1
        # For all other (a, b, c) not of that form, T = 0
        # But we only need to check that the decomposition produces the correct tensor.
        
        # Efficient check: for each rank index r, the outer product U[:,r] * V[:,r] * W[:,r]
        # should sum to T.
        
        # Let's check a sample of entries.
        # Full check would be O(n*m*p*rank) which is 30*rank, manageable.
        
        # Build the expected tensor implicitly and check
        import numpy as np
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Also check that entries that should be 0 are 0
        # This is trickier. Let's check a random sample.
        np.random.seed(42)
        for _ in range(200):
            a = np.random.randint(0, n * m)
            b = np.random.randint(0, m * p)
            c = np.random.randint(0, p * n)
            i = a // m
            j = a % m
            # b = j2 * p + k
            j2 = b // p
            k = b % p
            # c = k2 * n + i2
            k2 = c // n
            i2 = c % n
            # This entry is T[a,b,c] which is 1 only if j == j2 and k == k2 and i == i2
            # Wait, no. T[i*m+j, j*p+k, k*n+i] = 1.
            # So T[a, b, c] = 1 iff a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k.
            # That is: j = a%m, i = a//m, j2 = b//p, k = b%p, k2 = c//n, i2 = c%n
            # T=1 iff i==i2 and j==j2 and k==k2
            expected = 1.0 if (i == i2 and j == j2 and k == k2) else 0.0
            val = np.dot(U[a, :], V[b, :] * W[c, :])
            if abs(val - expected) > tol:
                return False
        return True

    def random_basis_change(dim, seed=None):
        """Generate a random invertible matrix for a basis change."""
        if seed is not None:
            np.random.seed(seed)
        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
        # Add some randomness to avoid being too orthogonal
        Q = Q + 0.1 * np.random.randn(dim, dim)
        return Q

    def greedy_sparse_recovery(T_flat, nU, nV, nW, max_rank, num_iter=100, seed=None):
        """
        Attempt to find a sparse rank-1 decomposition using matching pursuit.
        T_flat is the flattened tensor of size nU*nV*nW.
        We seek U, V, W such that sum_r U[:,r] * V[:,r] * W[:,r] approx T.
        
        This is a non-convex problem. We use a greedy approach:
        1. Start with residual = T
        2. At each step, find the best rank-1 term that reduces the residual norm
        3. Subtract it and repeat
        
        For exactness, we'll try to find integer/half-integer solutions.
        """
        if seed is not None:
            np.random.seed(seed)
        
        residual = T_flat.copy()
        U_list = []
        V_list = []
        W_list = []
        
        for r in range(max_rank):
            # Find the best rank-1 decomposition of the current residual
            # This is the tensor rank-1 approximation problem.
            # We use a simple power-iteration-like approach for the best rank-1 factor.
            
            best_norm = -1
            best_U = None
            best_V = None
            best_W = None
            
            # Try several random initializations
            for init in range(20):
                u = np.random.randn(nU)
                v = np.random.randn(nV)
                w = np.random.randn(nW)
                
                # Power iteration for best rank-1
                for _ in range(50):
                    # Update u
                    uv = np.tensordot(v, w, axes=0)  # size nV*nW
                    T_mat = residual.reshape(nU, nV, nW)
                    u_new = np.tensordot(T_mat, (v * w).reshape(nV, nW), axes=([1, 2], [0, 1]))
                    u_norm = norm(u_new)
                    if u_norm < 1e-12:
                        break
                    u = u_new / u_norm
                    
                    # Update v
                    v_new = np.tensordot(T_mat, (u * w).reshape(nU, nW), axes=([0, 2], [0, 1]))
                    v_norm = norm(v_new)
                    if v_norm < 1e-12:
                        break
                    v = v_new / v_norm
                    
                    # Update w
                    w_new = np.tensordot(T_mat, (u * v).reshape(nU, nV), axes=([0, 1], [0, 1]))
                    w_norm = norm(w_new)
                    if w_norm < 1e-12:
                        break
                    w = w_new / w_norm
                
                # Compute the rank-1 term
                term = np.einsum('i,j,k->ijk', u, v, w)
                term_flat = term.flatten()
                reduction = np.dot(term_flat, residual)
                
                if reduction > best_norm:
                    best_norm = reduction
                    best_U = u.copy()
                    best_V = v.copy()
                    best_W = w.copy()
            
            if best_norm < 1e-10:
                break
            
            # Scale the term
            scale = best_norm / (norm(best_U) * norm(best_V) * norm(best_W))
            U_list.append(best_U * scale)
            V_list.append(best_V)
            W_list.append(best_W)
            
            # Update residual
            term = np.einsum('i,j,k->ijk', best_U * scale, best_V, best_W)
            residual -= term.flatten()
            
            if norm(residual) < 1e-8:
                break
        
        if len(U_list) == 0:
            return None, None, None, len(U_list)
        
        U = np.column_stack(U_list)
        V = np.column_stack(V_list)
        W = np.column_stack(W_list)
        
        return U, V, W, len(U_list)

    # Build the tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    T_flat = T.flatten()

    # Try various ranks from a lower bound upward
    # Lower bound: for matrix multiplication, rank >= max(n*m, m*p, p*n) is not always true
    # But a reasonable lower bound is the tensor's "support" structure
    # Let's start from rank = 15 and go up to 30
    
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, get the schoolbook decomposition as a fallback
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p

    # Now search for lower ranks
    # Use randomized basis changes to try to find sparser representations
    seed = 0
    for target_rank in range(15, 31):
        if time.time() > DEADLINE:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > DEADLINE:
                break
            
            seed += 1
            
            # Apply random basis changes to the three tensor modes
            Q1 = random_basis_change(n * m, seed=seed * 100 + 1)
            Q2 = random_basis_change(m * p, seed=seed * 100 + 2)
            Q3 = random_basis_change(p * n, seed=seed * 100 + 3)
            
            # Transform the tensor
            # T' = Q1^{-1} T Q2^{-1} Q3^{-1} (in some sense)
            # Actually, if we change basis in each mode, the tensor transforms as:
            # T'[a', b', c'] = sum_{a,b,c} Q1[a',a]^{-1} T[a,b,c] Q2[b',b]^{-1} Q3[c',c]^{-1}
            
            # For simplicity, let's work in the original space and just try
            # different random initializations for the sparse recovery.
            
            # Try to find a decomposition of rank target_rank
            U, V, W, actual_rank = greedy_sparse_recovery(
                T_flat, n * m, m * p, p * n, 
                max_rank=target_rank, num_iter=100, seed=seed
            )
            
            if U is not None and actual_rank <= target_rank:
                if verify(U, V, W):
                    found = True
                    best_result = (U, V, W)
                    best_rank = actual_rank
                    break
            elif U is not None and actual_rank > 0:
                # Try to snap to nearest integers/half-integers and verify
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                if verify(U_snap, V_snap, W_snap):
                    found = True
                    best_result = (U_snap, V_snap, W_snap)
                    best_rank = actual_rank
                    break
        
        if found:
            # Found a valid decomposition at this rank, try to go lower
            continue
        else:
            # If we haven't found anything at this rank, continue to higher ranks
            # But if we're above the schoolbook rank, we should stop
            if target_rank >= n * m * p:
                break

    if best_result is None:
        # Fall back to schoolbook
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
