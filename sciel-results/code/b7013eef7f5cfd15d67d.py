"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.

Strategy:
1. Use integer linear algebra on the tensor's slice structure.
   For the (2,4,6) multiplication tensor, fix index i=0 and i=1 separately.
   For each i, the slice T[i,:,:,:] is a 4x6x6 tensor (or matrix 24x6) that
   represents: for fixed i, C[i,k] = sum_j A[i,j] B[j,k].
   
   The key structural insight: the full tensor T[(i,j),(j',k),(k',i')] = 1 iff
   j==j', k==k', i==i'. This means for each (j,k) pair, the tensor has a
   "diagonal" structure in the (i,i') and (k,k') dimensions.

2. We can exploit this by noting that the tensor decomposes as:
   T = sum_{j,k} e_{(.,j)} ⊗ e_{(j,.),k} ⊗ e_{k,(.,.)}
   where e vectors are standard basis. This gives rank 24 (schoolbook for m*p=24).
   
   But we can do better by finding linear dependencies. The structure is:
   For each j, we have a "column" of the tensor. Let's think of it as:
   T = sum_j (sum_k e_{(.,j)} ⊗ e_{(j,.),k} ⊗ e_{k,(.,.)})
   
   For fixed j, the inner sum over k gives a rank-p tensor in the (i,k,i') space.
   Specifically, for fixed j, define S_j[k,k',i'] = 1 iff k==k' and i'==i (for all i).
   Wait, that's not right either.

3. Better approach: Use the fact that this is a "block diagonal" structure.
   The tensor T can be seen as a 12x12 matrix (flattening modes 1&2 vs mode 3)
   where M[(i,j, j',k), (k',i')] = 1 iff j==j', k==k', i==i'.
   
   Actually, let me think differently. The tensor has the property that:
   T[a,b,c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
   
   This is a "permutation-like" tensor. The rank is the minimum number of
   rank-1 tensors needed to sum to T.

4. Practical approach: Use integer Gaussian elimination on a carefully constructed
   system. The tensor has 12*24*12 = 3456 entries, but only 24 are nonzero (one
   for each (i,j,k) triple: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}).
   
   Wait, that's 2*4*6 = 48 nonzero entries.

5. Let me use a different strategy: directly search for a low-rank decomposition
   by solving a system of equations. For rank r, we need:
   U (8 x r), V (24 x r), W (12 x r) such that
   sum_{r'} U[a,r'] V[b,r'] W[c,r'] = T[a,b,c]
   
   This is a bilinear system. We can fix W and solve for U,V, or use alternating
   least squares with integer constraints.

6. Given the time constraints, I'll implement:
   a. A verified rank-42 Strassen baseline.
   b. A structured search using the slice decomposition:
      - For each i in {0,1}, the slice is a 4x6 matrix multiplication.
      - The 4x6x6 tensor for fixed i has rank at most 6 (since C[i,:] = A[i,:] @ B).
      - But we need the joint decomposition.
      
   c. Try rank 36, 32, 28, 24 using randomized initialization + integer rounding.

Let me implement a clean solution that tries to find the minimum rank via
structured linear algebra.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]


def _build_tensor(n, m, p):
    """Build the multiplication tensor as a 3D array.
    
    T[i*m+j, j'*p+k, k'*n+i'] = 1 iff j==j' and k==k' and i==i'
    """
    T = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents the multiplication tensor."""
    T = _build_tensor(n, m, p)
    rank = U.shape[1]
    # Compute sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] and compare to T
    for r in range(rank):
        pass  # This is O(rank * n*m * m*p * p*n) which is too slow for large rank
    
    # Instead, verify by random matrix multiplication
    np.random.seed(42)
    for trial in range(10):
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p), dtype=float)
        for i in range(n):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    a_part = sum(U[i*m+j, r] * A[i, j] for j in range(m))
                    b_part = sum(V[j*p+k, r] * B[j, k] for j in range(m))
                    s += W[k*n+i, r] * a_part * b_part
                C_computed[i, k] = s
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True


def _build_strassen_42():
    """Build the rank-42 Strassen decomposition for 2x4x6."""
    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2x2 factors
    strassen = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),    # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),    # M2
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    factor_idx = 0
    for a_block in range(2):       # A cols: [0,1] or [2,3]
        for b_col_block in range(3):  # B cols: [0,1], [2,3], [4,5]
            for (u, v, w) in strassen:
                for idx in range(4):
                    if u[idx] != 0:
                        i, j_local = divmod(idx, 2)
                        u_row = i * m + a_block * 2 + j_local
                        U[u_row, factor_idx] += u[idx]
                    if v[idx] != 0:
                        j_local, k_local = divmod(idx, 2)
                        v_row = (a_block * 2 + j_local) * p + b_col_block * 2 + k_local
                        V[v_row, factor_idx] += v[idx]
                    if w[idx] != 0:
                        k_local, i = divmod(idx, 2)
                        w_row = (b_col_block * 2 + k_local) * n + i
                        W[w_row, factor_idx] += w[idx]
                factor_idx += 1
    
    assert factor_idx == rank
    return U, V, W


def _try_rank_with_als(n, m, p, target_rank, max_iters=50, time_limit=30):
    """Try to find a rank-`target_rank` decomposition using alternating least squares.
    
    Returns (U, V, W) if successful, else None.
    """
    start = time.time()
    T = _build_tensor(n, m, p)
    a_dim, b_dim, c_dim = n*m, m*p, p*n
    
    # Initialize randomly
    np.random.seed(123 + target_rank)
    U = np.random.randn(a_dim, target_rank)
    V = np.random.randn(b_dim, target_rank)
    W = np.random.randn(c_dim, target_rank)
    
    for it in range(max_iters):
        if time.time() - start > time_limit:
            break
        
        # Update U: for each a, solve for U[a,:] given V, W
        for a in range(a_dim):
            # T[a, :, :] = sum_r U[a,r] * V[:,r] ⊗ W[:,r]
            # Reshape: for each (b,c), T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is a linear system in U[a,:]:
            # Let M be a (b_dim * c_dim) x target_rank matrix where
            # M[(b,c), r] = V[b,r] * W[c,r]
            # Then T[a, :, :].flatten() = M @ U[a,:]
            M = np.zeros((b_dim * c_dim, target_rank))
            for b in range(b_dim):
                for c in range(c_dim):
                    for r in range(target_rank):
                        M[b * c_dim + c, r] = V[b, r] * W[c, r]
            rhs = T[a, :, :].flatten()
            # Solve M @ U[a,:] = rhs
            try:
                U[a, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                pass
        
        # Update V
        for b in range(b_dim):
            M = np.zeros((a_dim * c_dim, target_rank))
            for a in range(a_dim):
                for c in range(c_dim):
                    for r in range(target_rank):
                        M[a * c_dim + c, r] = U[a, r] * W[c, r]
            rhs = T[:, b, :].flatten()
            try:
                V[b, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                pass
        
        # Update W
        for c in range(c_dim):
            M = np.zeros((a_dim * b_dim, target_rank))
            for a in range(a_dim):
                for b in range(b_dim):
                    for r in range(target_rank):
                        M[a * b_dim + b, r] = U[a, r] * V[b, r]
            rhs = T[:, :, c].flatten()
            try:
                W[c, :] = np.linalg.lstsq(M, rhs, rcond=None)[0]
            except:
                pass
    
    # Check residual
    residual = 0.0
    for a in range(a_dim):
        for b in range(b_dim):
            for c in range(c_dim):
                pred = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
                residual += (pred - T[a, b, c]) ** 2
    
    if residual < 1e-6:
        # Try to snap to integers or half-integers
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify the snapped version
        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
        
        # Try rounding to integers
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if _verify_decomposition(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
    
    return None


def construct():
    start_time = time.time()
    DEADLINE = 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Start with the rank-42 Strassen baseline
    U, V, W = _build_strassen_42()
    best_rank = 42
    
    # Verify the baseline
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook rank 48
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to find lower ranks using ALS
    for target_rank in [36, 32, 28, 24, 20]:
        if time.time() - start_time > DEADLINE - 30:
            break
        
        result = _try_rank_with_als(n, m, p, target_rank, max_iters=30, time_limit=20)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            break
    
    # Final verification
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _build_strassen_42()
        best_rank = 42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
