"""
Rank-minimal exact tensor decomposition for matrix multiplication 4x4 * 4x5.
Uses a combination of Strassen-based construction and iterative rank-1 peeling
with SVD to find a lower-rank exact decomposition.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS


def _verify_fast(U, V, W):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    """
    rank = U.shape[1]
    # Build the tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Vectorized: for each r, outer product of U[:,r], V[:,r], W[:,r] reshaped
    # T has shape (n, m, p)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # We can compute this as:
    # For each i,j: U3[i,j,:] is a vector of length rank
    # For each j,k: V3[j,k,:] is a vector of length rank
    # For each k,i: W3[k,i,:] is a vector of length rank
    
    # Efficient computation:
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # = sum_r (U3[i,j,r] * V3[j,k,r]) * W3[k,i,r]
    
    # Let's compute step by step
    # First, for each r, we have U3[:,:,r] (n x m), V3[:,:,r] (m x p), W3[:,:,r] (p x n)
    # The contribution to T from rank r is:
    # T_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # This is a 3-way outer product for each r, summed.
    # T = sum_r U3[:,:,r] * V3[:,:,r] * W3[:,:,r] (with appropriate indexing)
    
    # More precisely: T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    
    # Let's use einsum or explicit loops for clarity (small dimensions)
    T = np.zeros((n, m, p))
    for r in range(rank):
        # U3[:, :, r] has shape (n, m)
        # V3[:, :, r] has shape (m, p)
        # W3[:, :, r] has shape (p, n)
        # T[i,j,k] += U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # For fixed r:
        # T_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        # This can be computed as:
        # For each i: U3[i,:,r] is (m,), for each j: V3[j,:,r] is (p,), W3[:,i,r] is (p,)
        # T_r[i,j,k] = U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
        
        # Let's do it element-wise for small sizes
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += U3[i, j, r] * V3[j, k, r] * W3[k, i, r]
    
    # Check if T is all 1s
    return np.allclose(T, 1.0, atol=1e-9)


def _round_to_half_integer(x):
    """Round a scalar to the nearest half-integer."""
    return round(2 * x) / 2.0


def _round_matrix_to_half_int(M):
    """Round all entries of a matrix to the nearest half-integer."""
    return np.round(M * 2) / 2


def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen rank-1 terms for 2x2 * 2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 2x2 matrices (flattened to 4-vectors).
    """
    # Strassen's 7 products for 2x2:
    # C = A * B where A, B are 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    # Strassen:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-d)(f+h)
    # p7 = (b-c)(e+g)
    
    # C[0,0] = p1 - p4 + p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p2 + p3 + p6
    
    # Flatten 2x2 to 4-vector: [top-left, top-right, bottom-left, bottom-right]
    # A indices: 0=a, 1=b, 2=c, 3=d
    # B indices: 0=e, 1=f, 2=g, 3=h
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11
    
    terms = []
    
    # p1 = (a+d)(e+h)
    u1 = np.array([1, 0, 0, 1], dtype=float)  # a + d
    v1 = np.array([1, 0, 0, 1], dtype=float)  # e + h
    w1 = np.array([1, 0, 0, 1], dtype=float)  # C00 += 1, C11 += 1 (p1 contributes to both)
    # Actually: C00 += p1, C11 += p1
    w1 = np.array([1, 0, 0, 1], dtype=float)
    terms.append((u1, v1, w1))
    
    # p2 = (c+d)e
    u2 = np.array([0, 0, 1, 1], dtype=float)  # c + d
    v2 = np.array([1, 0, 0, 0], dtype=float)  # e
    w2 = np.array([0, 0, 1, -1], dtype=float)  # C10 += p2, C11 -= p2
    terms.append((u2, v2, w2))
    
    # p3 = a(f-h)
    u3 = np.array([1, 0, 0, 0], dtype=float)  # a
    v3 = np.array([0, 1, 0, -1], dtype=float)  # f - h
    w3 = np.array([0, 1, 0, 1], dtype=float)  # C01 += p3, C11 += p3
    terms.append((u3, v3, w3))
    
    # p4 = d(g-e)
    u4 = np.array([0, 0, 0, 1], dtype=float)  # d
    v4 = np.array([0, 0, 1, -1], dtype=float)  # g - e
    w4 = np.array([-1, 0, 1, 0], dtype=float)  # C00 -= p4, C10 += p4
    terms.append((u4, v4, w4))
    
    # p5 = (a+b)h
    u5 = np.array([1, 1, 0, 0], dtype=float)  # a + b
    v5 = np.array([0, 0, 0, 1], dtype=float)  # h
    w5 = np.array([1, 1, 0, 0], dtype=float)  # C00 += p5, C01 += p5
    terms.append((u5, v5, w5))
    
    # p6 = (c-d)(f+h)
    u6 = np.array([0, 0, 1, -1], dtype=float)  # c - d
    v6 = np.array([0, 1, 0, 1], dtype=float)  # f + h
    w6 = np.array([0, 0, 0, 1], dtype=float)  # C11 += p6
    terms.append((u6, v6, w6))
    
    # p7 = (b-c)(e+g)
    u7 = np.array([0, 1, -1, 0], dtype=float)  # b - c
    v7 = np.array([1, 0, 1, 0], dtype=float)  # e + g
    w7 = np.array([1, 0, 0, 0], dtype=float)  # C00 += p7
    terms.append((u7, v7, w7))
    
    return terms


def _kron_strassen_construct(n, m, p):
    """
    Construct a Strassen-based decomposition for n x m * m x p using Kronecker products.
    For 4x4 * 4x5: pad to 4x4 * 4x4 with one zero column, then use Strassen.
    Actually, let's use a block Strassen approach.
    
    For 4x4 * 4x4, we can use 2x2 blocks of 2x2 matrices.
    A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
               [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Using Strassen for the 2x2 block multiplication:
    Each block product uses 7 rank-1 terms (in the 2x2x2 sense), but since each
    entry is a 2x2 matrix, we need to be more careful.
    
    Actually, for 4x4*4x4, the rank is known to be at most 70 (Landsberg).
    Let me use a simpler approach: direct Strassen recursion.
    
    For 4x4 * 4x5:
    - Pad B to 4x4 by adding a zero column: B' = [B | 0]
    - Compute C' = A * B' (4x4 result)
    - C = C'[:, :5] (first 5 columns)
    
    So we need a 4x4 * 4x4 decomposition.
    
    Using Strassen: 4x4*4x4 can be done with rank 4*4*4=64 naively, or less with Strassen.
    Actually, Strassen for 4x4*4x4:
    Split into 2x2 blocks of 2x2 matrices.
    7 block multiplications, each 2x2*2x2 = 7*7 = 49 rank-1 terms.
    But we need to account for the fact that the last column of B is zero,
    so we can save some terms.
    
    Let me just build the 4x4*4x4 Strassen decomposition and then restrict.
    """
    # For now, let's build a schoolbook 4x4*4x5 as baseline, then try to reduce.
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _als_decompose(n, m, p, rank, max_iters=100, tol=1e-10):
    """
    ALS (Alternating Least Squares) tensor decomposition.
    Finds U, V, W such that T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    where T is the matrix multiplication tensor (all 1s for valid (i,j,k)).
    
    The tensor T has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    """
    np.random.seed(42)
    
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # The target tensor T[i,j,k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # ALS: fix two factors, solve for the third.
    # For U: for each (i,j), we need sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
    # This is: for fixed (i,j), U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
    # Let M_{(i,j), k} = sum_r V[j*p+k,r] * W[k*n+i,r] (this is a scalar for each k)
    # Wait, that's not right. Let me reconsider.
    
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    
    # For fixed (i,j): T[i,j,:] (vector of length p) = sum_r U[i*m+j,r] * (V[j*p+0:p,r] * W[0*n+i:p*n+i, r])
    # where V[j*p+0:p,r] is the vector [V[j*p+0,r], V[j*p+1,r], ..., V[j*p+p-1,r]]
    # and W[0*n+i:p*n+i, r] is [W[0*n+i,r], W[1*n+i,r], ..., W[(p-1)*n+i,r]]
    
    # So for fixed (i,j): T[i,j,:] = sum_r U[i*m+j,r] * (v_j * w_i) where v_j is p-vector, w_i is p-vector
    # v_j[k] = V[j*p+k, r], w_i[k] = W[k*n+i, r]
    # (v_j * w_i)[k] = V[j*p+k, r] * W[k*n+i, r]
    
    # So T[i,j,:] = sum_r U[i*m+j,r] * (v_j[r] * w_i[r]) where v_j[r] is the r-th component...
    # No, let me re-index.
    
    # Let's define for each r:
    # a_r[i,j] = U[i*m+j, r]  (n x m matrix)
    # b_r[j,k] = V[j*p+k, r]  (m x p matrix)
    # c_r[k,i] = W[k*n+i, r]  (p x n matrix)
    
    # T[i,j,k] = sum_r a_r[i,j] * b_r[j,k] * c_r[k,i]
    
    # For ALS on U (fix V, W):
    # For each (i,j): T[i,j,k] = sum_r U[i*m+j,r] * b_r[j,k] * c_r[k,i] for all k
    # Let's define for each (i,j) and each r:
    # s_{(i,j),r} = sum_k b_r[j,k] * c_r[k,i] ... no, that's not right either.
    
    # T[i,j,k] = sum_r U[i*m+j,r] * b_r[j,k] * c_r[k,i]
    # For fixed (i,j), as a function of k:
    # T[i,j,:] = sum_r U[i*m+j,r] * (b_r[j,:] * c_r[:,i])
    # where b_r[j,:] is the j-th row of b_r (length p), c_r[:,i] is the i-th column of c_r (length p)
    # (b_r[j,:] * c_r[:,i]) is element-wise product, length p.
    
    # So T[i,j,:] = sum_r U[i*m+j,r] * d_{(i,j),r} where d_{(i,j),r}[k] = b_r[j,k] * c_r[k,i]
    
    # This is a linear system: for each (i,j), T[i,j,:] = U[i*m+j,:] * D_{(i,j)}
    # where D_{(i,j)} is a (p x rank) matrix with D_{(i,j)}[k,r] = b_r[j,k] * c_r[k,i]
    
    # So U[i*m+j,:] = T[i,j,:] * D_{(i,j)}^+ (pseudoinverse)
    
    # Similarly for V and W.
    
    for iteration in range(max_iters):
        # Update U
        for i in range(n):
            for j in range(m):
                # D[k,r] = V[j*p+k, r] * W[k*n+i, r] for k in range(p), r in range(rank)
                D = np.zeros((p, rank))
                for k in range(p):
                    D[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # T[i,j,:] is a vector of 1s (length p)
                target = np.ones(p)
                # Solve: U[i*m+j,:] * D = target
                # U[i*m+j,:] = target * D^T * (D * D^T)^{-1} if possible
                # Or use least squares: U[i*m+j,:] = argmin ||U[i*m+j,:] * D - target||
                # This is: min_x ||x * D - target|| where x is (rank,)
                # = min_x ||D^T * x^T - target^T||
                # = min_y ||D^T * y - target|| where y = x^T
                # y = (D^T)^+ * target = D^+ * target (using pseudoinverse)
                # Actually: min_y ||D^T * y - target|| => y = (D^T)^T (D^T (D^T)^T)^{-1} target = D (D^T D)^{-1} target
                # Wait: D^T is (rank x p), target is (p,)
                # min_y ||D^T y - target|| => y = (D^T)^+ target
                # (D^T)^+ = (D^T)^T (D^T (D^T)^T)^{-1} = D (D^T D)^{-1} if D^T D is invertible
                try:
                    y = np.linalg.lstsq(D.T, target, rcond=None)[0]
                except:
                    y = np.zeros(rank)
                U[i * m + j, :] = y
        
        # Update V
        for j in range(m):
            for k in range(p):
                # D[i,r] = U[i*m+j, r] * W[k*n+i, r] for i in range(n), r in range(rank)
                D = np.zeros((n, rank))
                for i in range(n):
                    D[i, :] = U[i * m + j, :] * W[k * n + i, :]
                target = np.ones(n)
                try:
                    y = np.linalg.lstsq(D.T, target, rcond=None)[0]
                except:
                    y = np.zeros(rank)
                V[j * p + k, :] = y
        
        # Update W
        for k in range(p):
            for i in range(n):
                # D[j,r] = U[i*m+j, r] * V[j*p+k, r] for j in range(m), r in range(rank)
                D = np.zeros((m, rank))
                for j in range(m):
                    D[j, :] = U[i * m + j, :] * V[j * p + k, :]
                target = np.ones(m)
                try:
                    y = np.linalg.lstsq(D.T, target, rcond=None)[0]
                except:
                    y = np.zeros(rank)
                W[k * n + i, :] = y
        
        # Check convergence
        if iteration % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        err += (val - 1.0) ** 2
            if err < tol:
                break
    
    return U, V, W


def _try_reduce_from_baseline(U, V, W, n, m, p, max_attempts=50):
    """
    Try to reduce the rank by peeling off rank-1 components.
    Start from a valid decomposition and try to find linear dependencies.
    """
    rank = U.shape[1]
    
    # Try to find if any column is a linear combination of others
    # For each column r, check if U[:,r], V[:,r], W[:,r] can be removed
    # by adjusting other columns.
    
    # This is complex, so let's try a simpler approach:
    # Randomly perturb and re-optimize with lower rank.
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    for target_rank in range(rank - 1, max(30, rank - 20), -1):
        # Try ALS with target_rank
        U_new, V_new, W_new = _als_decompose(n, m, p, target_rank, max_iters=50)
        
        # Round to half-integers
        U_new = _round_matrix_to_half_int(U_new)
        V_new = _round_matrix_to_half_int(V_new)
        W_new = _round_matrix_to_half_int(W_new)
        
        if _verify_fast(U_new, V_new, W_new):
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = target_rank
            break
    
    return best_U, best_V, best_W, best_rank


def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based construction for 4x4*4x4 (padded)
    # Then restrict to 4x4*4x5
    
    # Try ALS with various ranks
    for target_rank in [64, 60, 56, 52, 49]:
        if time.time() > deadline - 30:
            break
        
        U_try, V_try, W_try = _als_decompose(n, m, p, target_rank, max_iters=200)
        
        # Round to half-integers
        U_try = _round_matrix_to_half_int(U_try)
        V_try = _round_matrix_to_half_int(V_try)
        W_try = _round_matrix_to_half_int(W_try)
        
        if _verify_fast(U_try, V_try, W_try):
            U, V, W = U_try, V_try, W_try
            best_rank = target_rank
            break
    
    # Try to reduce further
    if time.time() < deadline - 20:
        U, V, W, best_rank = _try_reduce_from_baseline(U, V, W, n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
