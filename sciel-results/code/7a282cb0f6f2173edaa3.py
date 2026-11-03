"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_2x2():
    """Return Strassen 2x2x2 decomposition as (U, V, W) numpy arrays, shape (4,7) each.
    U rows: i*2+j (i,j in 0..1); V rows: j*2+k; W rows: k*2+i.
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    # M1 = (a00+a11)(b00+b11) -> C00+=1, C11+=1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    # M2 = (a10+a11)b00 -> C10+=1, C11+=-1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    # M3 = a00(b01-b11) -> C01+=1, C11+=1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    # M4 = a11(b10-b00) -> C00+=1, C10+=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    # M5 = (a00+a01)b11 -> C00+=-1, C01+=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    # M6 = (a10-a00)(b00+b01) -> C11+=1
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    # M7 = (a01-a11)(b10+b11) -> C00+=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1
    return U, V, W

def _strassen_2x2x3():
    """2x2x3 decomposition: A(2x2)*B(2x3)->C(2x3).
    Rank 11: 7 Strassen terms for cols 0-1 + 4 trivial for col 2.
    U rows: i*2+j; V rows: j*3+k (j in 0..1, k in 0..2); W rows: k*2+i (k in 0..2, i in 0..1).
    """
    U1, V1, W1 = _strassen_2x2()
    U = np.zeros((4, 11), dtype=int)
    V = np.zeros((6, 11), dtype=int)
    W = np.zeros((6, 11), dtype=int)
    # Strassen part (cols 0-1), terms 0-6
    for c in range(7):
        for i in range(2):
            for j in range(2):
                U[i*2+j, c] = U1[i*2+j, c]
        # V1 rows j*2+k -> V rows j*3+k
        for j in range(2):
            for k in range(2):
                V[j*3+k, c] = V1[j*2+k, c]
        # W1 rows k*2+i -> W rows k*2+i (k in 0..1)
        for k in range(2):
            for i in range(2):
                W[k*2+i, c] = W1[k*2+i, c]
    # Trivial part (col 2), terms 7-10
    # C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2]
    U[0,7]=1; V[0*3+2,7]=1; W[2*2+0,7]=1
    U[1,8]=1; V[1*3+2,8]=1; W[2*2+0,8]=1
    # C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    U[2,9]=1; V[0*3+2,9]=1; W[2*2+1,9]=1
    U[3,10]=1; V[1*3+2,10]=1; W[2*2+1,10]=1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify the decomposition is correct."""
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    C_computed = np.zeros((n, p))
    for r in range(U.shape[1]):
        alpha_r = np.sum(A_test * U[:, r].reshape(n, m))
        beta_r = np.sum(B_test * V[:, r].reshape(m, p))
        C_computed += alpha_r * beta_r * W[:, r].reshape(p, n).T
    return np.allclose(C_computed, C_expected, atol=1e-10)

def _reduce_rank(U, V, W, n, m, p, deadline):
    """
    Reduce rank by detecting linear dependencies among rank-1 tensor terms.
    A rank-1 term is a tensor T_r = U[:,r] ⊗ V[:,r] ⊗ W[:,r].
    We look for a linear combination sum_r c_r * T_r = 0 (as a tensor),
    which lets us eliminate one term (the one with c_r != 0).
    
    Approach: build the tensor as a 3D array T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r].
    The set of rank-1 terms {T_r} spans a subspace. We find the minimal subset that still
    spans the same tensor T (i.e., the tensor rank). We do this via Gaussian elimination
    over the flattened term vectors, but that gives the rank of the span, not the tensor rank.
    
    Better: we detect zero-sum relations. For each subset, check if sum c_r T_r = 0.
    We use a greedy approach: for each term r, check if T_r is in the span of the others
    (by solving a linear system). If so, eliminate it.
    
    To keep it tractable, we work with the tensor's 3-mode flattening. A term T_r is
    determined by the triple (u_r, v_r, w_r). The condition sum c_r u_r⊗v_r⊗w_r = 0
    is equivalent to: for all (i,j,k), sum c_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 0.
    
    We find a basis of the space of relations by:
    1. Building a matrix M where each column is a "flattened term" in a suitable space.
    2. But the tensor space is (n*m)*(m*p)*(p*n)-dim, too large.
    
    Instead, we use the fact that a zero-sum relation corresponds to a vector c in the
    null space of the "tensor map". We approximate by checking rank deficiency of the
    set of term vectors under multiple random linear functionals.
    
    Practical approach: use the 3-tensor rank via iterative elimination.
    For each term r (in order), test if removing it still allows representing T.
    This is expensive, so we limit iterations.
    """
    n, m, p = n, m, p
    rank = U.shape[1]
    if rank <= 1:
        return U, V, W
    
    # Build the full tensor T (n, p, m) -> but we need (i, k, j) ordering? 
    # T[i,j,k] = (A*B)[i,k] coefficient for A[i,j]*B[j,k]
    # Actually T[i,j,k] = 1 if we want C[i,k] = sum_{j} A[i,j] B[j,k]
    # The tensor for matrix mult: T_{i,j,k} = delta (the 3-tensor with T[i,j,k]=1 for the
    # standard basis). Wait, the matrix multiplication tensor T has T[i,j,k] = 1 for all
    # (i,j,k)? No. The multiplication tensor is sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where
    # (A*B)[i,k] = sum_{j} A[i,j] B[j,k]. So T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k, i.e.,
    # T[i,j,k] = 1 for all i,j,k. Yes! The matrix multiplication tensor is the all-ones
    # tensor in the (i,j,k) space.
    
    # So T[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # A decomposition: T = sum_r U_r ⊗ V_r ⊗ W_r where
    # (U_r ⊗ V_r ⊗ W_r)[i,j,k] = U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    # We need sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all (i,j,k).
    
    # To find rank reduction: we look for c != 0 such that
    # sum_r c_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 0 for all (i,j,k).
    
    # This is a system of n*m*p equations in rank unknowns c_r.
    # Build matrix A_eq of size (n*m*p) x rank:
    # A_eq[(i,j,k), r] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Find null space of A_eq. Each null vector gives a zero-sum relation.
    
    # n*m*p = 4*4*6 = 96 equations, rank unknowns.
    # Build the matrix.
    A_eq = np.zeros((n * m * p, rank), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i * m + j) * p + k  # flatten (i,j,k)
                # A_eq[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                A_eq[idx, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    
    # Find null space of A_eq (rows = equations, cols = terms).
    # Null space vectors c satisfy A_eq @ c = 0.
    # Each such c gives a relation: sum_r c_r (U_r ⊗ V_r ⊗ W_r) = 0.
    # We can use this to eliminate terms.
    
    # Compute SVD to find null space.
    try:
        U_svd, s, Vt = np.linalg.svd(A_eq, full_matrices=True)
    except Exception:
        return U, V, W
    
    # Null space dimension = rank - number of nonzero singular values.
    tol = 1e-10
    null_dim = sum(1 for sv in s if sv < tol)
    
    if null_dim == 0:
        return U, V, W
    
    # Get null space basis vectors (rows of Vt corresponding to small singular values).
    # Vt is (rank, rank). Null space = span of Vt[rank - null_dim:, :].T
    null_basis = Vt[rank - null_dim:, :].T  # (null_dim, rank)
    
    # For each null vector, we can eliminate one term.
    # Strategy: for each null vector c, pick the term r with largest |c_r| and express it
    # in terms of others, then remove it.
    
    # We'll do this greedily: maintain a list of active terms.
    active = list(range(rank))
    
    # For each null vector, try to eliminate a term.
    # A null vector c means sum_r c_r T_r = 0.
    # If c_r != 0 for some r, we can write T_r = -(1/c_r) sum_{r' != r} c_{r'} T_{r'}.
    # But this changes the coefficients of the other terms, which may introduce non-integer
    # values. We need to keep integer/half-integer entries.
    
    # For exactness with integers, we need the null vector to have integer entries (up to scale).
    # SVD gives real null vectors. We need to check if they're rational.
    
    # Alternative: use exact rational arithmetic via sympy-free approach.
    # Since entries are integers, we can use Gaussian elimination with fractions.
    
    # Let's do exact rational null space computation.
    from fractions import Fraction as Fr
    
    # Convert A_eq to Fraction matrix.
    A_frac = [[Fr(int(round(A_eq[i, j]))) for j in range(rank)] for i in range(n * m * p)]
    
    # Gaussian elimination to find null space over Q.
    # We'll do row reduction.
    M = [row[:] for row in A_frac]
    nrows = len(M)
    ncols = len(M[0]) if nrows > 0 else 0
    
    pivot_cols = []
    row_idx = 0
    for col in range(ncols):
        # Find pivot
        pivot = None
        for r in range(row_idx, nrows):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        M[row_idx], M[pivot] = M[pivot], M[row_idx]
        # Normalize
        pv = M[row_idx][col]
        for c in range(col, ncols):
            M[row_idx][c] = M[row_idx][c] / pv
        # Eliminate
        for r in range(nrows):
            if r != row_idx and M[r][col] != 0:
                factor = M[r][col]
                for c in range(col, ncols):
                    M[r][c] = M[r][c] - factor * M[row_idx][c]
        pivot_cols.append(col)
        row_idx += 1
    
    free_cols = [c for c in range(ncols) if c not in pivot_cols]
    
    # Null space basis: for each free column f, set the free var = 1, others = 0,
    # and solve for pivot vars.
    null_vectors = []
    for f in free_cols:
        c_vec = [Fr(0)] * ncols
        c_vec[f] = Fr(1)
        for pi, pc in enumerate(pivot_cols):
            # M[pi][pc] = 1, so c[pc] = -sum_{c' > pc} M[pi][c'] * c[c']
            # Since we've done full elimination, M[pi][pc] = 1 and M[pi][c] for c in pivot_cols, c != pc is 0.
            # c[pc] = -sum_{c' not pivot, c' > pc} M[pi][c'] * c[c']  ... actually all free cols
            c_vec[pc] = Fr(0)
            for c2 in range(ncols):
                if c2 != pc:
                    c_vec[pc] -= M[pi][c2] * c_vec[c2]
        null_vectors.append(c_vec)
    
    if not null_vectors:
        return U, V, W
    
    # Now, for each null vector, we can eliminate a term.
    # We'll try to eliminate terms one by one, checking that the resulting
    # decomposition still has integer/half-integer entries and is valid.
    
    # For a null vector c, to eliminate term r (where c_r != 0):
    # We need to express T_r in terms of others. But this changes U, V, W for other terms.
    # Actually, the relation is on the TENSORS, not on the individual U, V, W.
    # We can't simply eliminate a term without modifying others.
    
    # The correct approach: a zero-sum relation sum c_r T_r = 0 means the set {T_r} is
    # linearly dependent. The tensor rank is the minimum number of rank-1 tensors summing
    # to T. We can reduce the rank by finding a subset that still sums to T.
    
    # Since sum c_r T_r = 0, we have T = sum_{r: c_r=0} T_r + (1/c_s) sum_{r: c_r != 0, r != s} c_r T_r
    # for any s with c_s != 0. But this doesn't reduce the count unless we can combine.
    
    # Actually, the standard way: if T_r = sum_{r' != r} d_{r'} T_{r'}, then we can remove T_r
    # and add d_{r'} to the coefficient of T_{r'}. But our decomposition has coefficient 1
    # for each term (it's a sum of rank-1 tensors, not a weighted sum).
    
    # The decomposition is T = sum_r T_r (each with coefficient 1). A relation sum c_r T_r = 0
    # lets us write T = sum_{r != s} (1 + c_r/c_s) T_r for any s with c_s != 0.
    # This reduces the count by 1 but changes coefficients. The new "terms" are
    # (1 + c_r/c_s) T_r, which are not rank-1 with the same U, V, W.
    
    # This is getting complex. Let me use a different approach: 
    # Find the actual tensor rank by solving for the minimum number of terms.
    # This is NP-hard in general, but for small cases we can use iterative methods.
    
    # Practical approach: use the null space to find a smaller spanning set.
    # The terms {T_r} span a subspace of dimension = rank(A_eq) = ncols - null_dim.
    # But the tensor rank is not the dimension of the span; it's the minimum number
    # of rank-1 tensors needed.
    
    # For now, let's just check if we can eliminate terms by finding a basis of the
    # span and seeing if fewer terms suffice. This is approximate.
    
    # Given the complexity, let's return the original if we can't cleanly reduce.
    # The main value is in the Kronecker construction itself.
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    # Kronecker decomposition: 4x4x6 = (2x2x2) ⊗ (2x2x3)
    U1, V1, W1 = _strassen_2x2()  # 2x2x2, rank 7
    U2, V2, W2 = _strassen_2x2x3()  # 2x2x3, rank 11
    
    rank = 7 * 11  # 77
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((24, rank), dtype=int)
    W = np.zeros((24, rank), dtype=int)
    
    # Index mapping:
    # A[i,j] where i = i1*2+i2, j = j1*2+j2
    # B[j,k] where j = j1*2+j2, k = k1*3+k2
    # C[i,k] where i = i1*2+i2, k = k1*3+k2
    #
    # U row: i*4+j = (i1*2+i2)*4 + (j1*2+j2) = i1*8 + i2*4 + j1*2 + j2
    # V row: j*6+k = (j1*2+j2)*6 + (k1*3+k2) = j1*12 + j2*6 + k1*3 + k2
    # W row: k*4+i = (k1*3+k2)*4 + (i1*2+i2) = k1*12 + k2*4 + i1*2 + i2
    #
    # Term index: c1*11 + c2 where c1 in 0..6, c2 in 0..10
    
    for c1 in range(7):
        for c2 in range(11):
            c = c1 * 11 + c2
            for i1 in range(2):
                for i2 in range(2):
                    for j1 in range(2):
                        for j2 in range(2):
                            u1 = U1[i1*2+j1, c1]
                            u2 = U2[i2*2+j2, c2]
                            if u1 != 0 and u2 != 0:
                                row = i1*8 + i2*4 + j1*2 + j2
                                U[row, c] = u1 * u2
            for j1 in range(2):
                for j2 in range(2):
                    for k1 in range(2):
                        for k2 in range(3):
                            v1 = V1[j1*2+k1, c1]
                            v2 = V2[j2*3+k2, c2]
                            if v1 != 0 and v2 != 0:
                                row = j1*12 + j2*6 + k1*3 + k2
                                V[row, c] = v1 * v2
            for k1 in range(2):
                for k2 in range(3):
                    for i1 in range(2):
                        for i2 in range(2):
                            w1 = W1[k1*2+i1, c1]
                            w2 = W2[k2*2+i2, c2]
                            if w1 != 0 and w2 != 0:
                                row = k1*12 + k2*4 + i1*2 + i2
                                W[row, c] = w1 * w2
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Attempt rank reduction (best-effort, with deadline guard)
    if time.time() < deadline - 20.0:
        U_red, V_red, W_red = _reduce_rank(U, V, W, n, m, p, deadline)
        if _verify(U_red, V_red, W_red, n, m, p) and U_red.shape[1] < rank:
            U, V, W = U_red, V_red, W_red
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
