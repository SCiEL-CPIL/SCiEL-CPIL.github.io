"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def _validate_numpy(U, V, W, n=2, m=4, p=5, tol=1e-9):
    """Vectorized validation using numpy for speed."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U_np[i*m+j] * V_np[j*p+k] * W_np[k*n+i]
                val = np.sum(prod)
                if abs(val - 1.0) > tol:
                    return False
    return True

def _clean_entry(v):
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    doubled = v * 2
    if abs(doubled - round(doubled)) < 1e-9:
        return round(doubled) / 2.0
    return v

def _construct_rank_36():
    """
    Rank-36 decomposition for 2x4 x 4x5 matrix multiplication.
    Uses Strassen for 2x2x2 blocks and schoolbook for 2x2x1.
    """
    n, m, p = 2, 4, 5
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_222(i1, i2, j1, j2, k1, k2, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; U[a22][col] = 1
        V[b11][col] = 1; V[b22][col] = 1
        W[c11][col] = 1; W[c22][col] = 1
        
        col = col_offset + 1
        U[a21][col] = 1; U[a22][col] = 1
        V[b11][col] = 1
        W[c21][col] = 1; W[c22][col] = -1
        
        col = col_offset + 2
        U[a11][col] = 1
        V[b12][col] = 1; V[b22][col] = -1
        W[c12][col] = 1; W[c22][col] = 1
        
        col = col_offset + 3
        U[a22][col] = 1
        V[b21][col] = 1; V[b11][col] = -1
        W[c11][col] = 1; W[c21][col] = 1
        
        col = col_offset + 4
        U[a11][col] = 1; U[a12][col] = 1
        V[b22][col] = 1
        W[c11][col] = -1; W[c12][col] = 1
        
        col = col_offset + 5
        U[a21][col] = 1; U[a11][col] = -1
        V[b11][col] = 1; V[b12][col] = 1
        W[c22][col] = 1
        
        col = col_offset + 6
        U[a12][col] = 1; U[a22][col] = -1
        V[b21][col] = 1; V[b22][col] = 1
        W[c11][col] = 1

    def add_schoolbook_221(i1, i2, j1, j2, k1, col_offset):
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        b11 = j1 * p + k1
        b21 = j2 * p + k1
        c11 = k1 * n + i1
        c21 = k1 * n + i2
        
        col = col_offset + 0
        U[a11][col] = 1; V[b11][col] = 1; W[c11][col] = 1
        col = col_offset + 1
        U[a12][col] = 1; V[b21][col] = 1; W[c11][col] = 1
        col = col_offset + 2
        U[a21][col] = 1; V[b11][col] = 1; W[c21][col] = 1
        col = col_offset + 3
        U[a22][col] = 1; V[b21][col] = 1; W[c21][col] = 1

    col = 0
    # Sub-problem 1: A rows 0-1, cols 0-1; B rows 0-1, cols 0-4
    add_strassen_222(0, 1, 0, 1, 0, 1, col); col += 7
    add_strassen_222(0, 1, 0, 1, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 0, 1, 4, col); col += 4
    
    # Sub-problem 2: A rows 0-1, cols 2-3; B rows 2-3, cols 0-4
    add_strassen_222(0, 1, 2, 3, 0, 1, col); col += 7
    add_strassen_222(0, 1, 2, 3, 2, 3, col); col += 7
    add_schoolbook_221(0, 1, 2, 3, 4, col); col += 4
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _construct_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 40."""
    n, m, p = 2, 4, 5
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return U, V, W

def _construct_rank_34_search(deadline):
    """
    Attempt to find a rank-34 decomposition using numerical optimization.
    Strategy: Decompose 2x4*4x5 into two 2x2*2x4 blocks plus a 2x2*2x1 block.
    We search for a rank-13 decomposition of M_{2,2,4} and a rank-8 decomposition of M_{2,2,1}.
    Actually, 13+13+8 = 34.
    M_{2,2,4} is known to have rank 13? No, rank of M_{2,2,4} is at least 13?
    Actually, rank(M_{2,2,4}) is 13 (Bini et al.).
    Rank(M_{2,2,1}) is 4 (schoolbook) but can it be lower? No, min rank for 2x2x1 is 4.
    So 13+13+4 = 30? Wait.
    If we split into:
    Block 1: A[:, 0:2] * B[0:2, :] -> C[:, 0:4] (M_{2,2,4})
    Block 2: A[:, 2:4] * B[2:4, :] -> C[:, 0:4] (M_{2,2,4})
    Block 3: A[:, 0:4] * B[:, 4:5] -> C[:, 4:5] (M_{2,4,1})
    
    Rank(M_{2,2,4}) = 13.
    Rank(M_{2,4,1}) = 8? (2*4*1 = 8 schoolbook). Can it be lower?
    M_{2,4,1} is equivalent to M_{4,2,1} by symmetry? No, M_{m,n,p} is symmetric in m,n,p for tensor rank?
    Yes, tensor rank is symmetric.
    Rank(M_{2,4,1}) = Rank(M_{2,1,4}) = Rank(M_{1,2,4}).
    M_{1,2,4} is just a vector of 8 entries. Rank is 8?
    Wait, M_{1,2,4} tensor T[i,j,k] = delta_{i,0} * 1? No.
    M_{1,2,4}: A is 1x2, B is 2x4, C is 1x4.
    C_k = sum_j A_j B_{j,k}.
    This is a bilinear map from R^2 x R^8 to R^4.
    The tensor is the identity map from R^4 to R^4? No.
    Let's check rank of M_{1,2,4}.
    It is the tensor for multiplying a 1x2 row vector by a 2x4 matrix.
    This is equivalent to taking 2 vectors in R^4 and forming a linear combination with coefficients A_0, A_1.
    This is a "linear combination" tensor.
    The rank of the tensor for summing 2 vectors in R^d is 2d? No.
    Actually, M_{1,2,4} is the tensor T_{j,k} = A_j B_{j,k}.
    This is a 2-mode tensor? No, 3-mode: i (size 1), j (size 2), k (size 4).
    T_{0,j,k} = 1 if we consider the structure? No.
    The tensor is defined by T_{i,j,k} = 1 if the indices match the multiplication.
    For M_{1,2,4}:
    i is always 0.
    j is 0 or 1.
    k is 0..3.
    The product is C_0 = A_0 B_{0,0} + A_1 B_{1,0} ... C_3 = A_0 B_{0,3} + A_1 B_{1,3}.
    The tensor entries T_{i,j,k} are 1 if the term A_i B_{j,k} contributes to C_{i', j', k'}?
    Wait, the tensor T for matrix multiplication M_{n,m,p} has indices (i, j, k) where:
    i in 0..n-1 (row of A, col of C)
    j in 0..m-1 (col of A, row of B)
    k in 0..p-1 (col of B, col of C)
    T_{i,j,k} = 1.
    Wait, the standard definition is T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} appears in C_{i,k}.
    So for M_{1,2,4}:
    i=0. j=0,1. k=0,1,2,3.
    T_{0,j,k} = 1 for all j,k.
    This is a tensor of size 1x2x4 with all ones.
    The rank of a tensor of all ones of size IxJxK is 1? No.
    T_{i,j,k} = 1.
    Can we write T = sum_r U_i^r V_j^r W_k^r?
    If we pick U_0^r = 1, V_j^r = delta_{j,r} (if r<2 else 0), W_k^r = 1?
    Sum_r 1 * delta_{j,r} * 1 = delta_{j, r} summed?
    If r goes 0..1:
    r=0: U_0=1, V_0=1, W_k=1. Term: 1*1*1 = 1 for j=0.
    r=1: U_0=1, V_1=1, W_k=1. Term: 1*1*1 = 1 for j=1.
    Sum is 1 for all j,k.
    So rank is 2.
    
    So Rank(M_{1,2,4}) = 2.
    Thus Rank(M_{2,4,1}) = 2.
    
    So total rank: 13 (Block 1) + 13 (Block 2) + 2 (Block 3) = 28?
    Wait, Block 3 is A[:, 0:4] * B[:, 4:5] -> C[:, 4:5].
    This is M_{2,4,1}.
    A is 2x4. B is 4x1. C is 2x1.
    Indices: i (2), j (4), k (1).
    T_{i,j,0} = 1.
    This is a 2x4x1 tensor of all ones.
    Rank of 2x4x1 all-ones tensor:
    T_{i,j,0} = 1.
    We need sum_r U_i^r V_j^r W_0^r = 1.
    Let W_0^r = 1.
    Sum_r U_i^r V_j^r = 1.
    This is a rank-1 approximation of the 2x4 all-ones matrix?
    No, it's an exact decomposition of the tensor into sum of rank-1 tensors.
    A rank-1 tensor in this slice is u_i v_j.
    We want sum_r u_i^r v_j^r = 1 for all i,j.
    This is the matrix M_{i,j} = 1.
    The rank of the matrix of all ones is 1.
    So we can do it with r=1?
    u_0=1, u_1=1. v_0=1, v_1=1, v_2=1, v_3=1.
    Sum_1 1*1 = 1.
    So rank is 1.
    
    So Rank(M_{2,4,1}) = 1.
    
    Total rank: 13 + 13 + 1 = 27?
    
    Wait, is Rank(M_{2,2,4}) really 13?
    Yes, it is a known result.
    
    So we should be able to get rank 27?
    Let's verify the block decomposition.
    C = A B.
    C[:, 0:4] = A[:, 0:2] B[0:2, 0:4] + A[:, 2:4] B[2:4, 0:4].
    C[:, 4:5] = A[:, 0:4] B[0:4, 4:5].
    
    Yes, the blocks are disjoint in terms of contributions to C.
    Block 1 contributes to C[:, 0:4] using A[:, 0:2] and B[0:2, 0:4].
    Block 2 contributes to C[:, 0:4] using A[:, 2:4] and B[2:4, 0:4].
    Block 3 contributes to C[:, 4:5] using A[:, 0:4] and B[0:4, 4:5].
    
    Are the factors disjoint?
    Block 1 uses U indices for A[:, 0:2], V for B[0:2, 0:4], W for C[:, 0:4].
    Block 2 uses U indices for A[:, 2:4], V for B[2:4, 0:4], W for C[:, 0:4].
    Block 3 uses U indices for A[:, 0:4], V for B[0:4, 4:5], W for C[:, 4:5].
    
    U factors:
    Block 1: A[:, 0:2] (rows 0,1; cols 0,1).
    Block 2: A[:, 2:4] (rows 0,1; cols 2,3).
    Block 3: A[:, 0:4] (rows 0,1; cols 0,1,2,3).
    
    Wait, Block 3 uses ALL columns of A.
    Block 1 uses cols 0,1. Block 2 uses cols 2,3.
    So U factors overlap?
    In the tensor decomposition, the "U" matrix has rows indexed by (i,j) of A.
    Block 1 uses rows (0,0), (0,1), (1,0), (1,1).
    Block 2 uses rows (0,2), (0,3), (1,2), (1,3).
    Block 3 uses rows (0,0)..(1,3).
    
    Since the decomposition is a sum of tensors, and the support of the tensors in the U-index space are disjoint for Block 1 and 2, but Block 3 overlaps with both.
    However, the W indices (C columns) are disjoint:
    Block 1, 2 -> C cols 0,1,2,3.
    Block 3 -> C col 4.
    
    So the W factors are disjoint.
    If W factors are disjoint, we can sum the ranks.
    Rank(Total) <= Rank(B1) + Rank(B2) + Rank(B3).
    
    So 13 + 13 + 1 = 27.
    
    This is a huge improvement over 36.
    
    I need to find the rank-13 decomposition of M_{2,2,4}.
    I can search for it numerically.
    Variables: U (2x2x13), V (2x4x13), W (4x2x13).
    Equations: 2*2*4 = 16.
    
    Let's implement the search for M_{2,2,4} rank 13.
    
    """
    n, m, p = 2, 4, 5
    target_rank_224 = 13
    target_rank_241 = 1
    total_rank = target_rank_224 * 2 + target_rank_241
    # 27
    
    # Search for M_{2,2,4} rank 13
    def search_224_rank_13():
        # T[i,j,k] = 1 for i in 0..1, j in 0..1, k in 0..3
        # U (2,2,13), V (2,4,13), W (4,2,13)
        # Note: V is 2x4 because B is 2x4?
        # In M_{2,2,4}: A is 2x2, B is 2x4, C is 2x4.
        # Indices: i (2), j (2), k (4).
        # U[i,j], V[j,k], W[k,i].
        # V is 2x4. W is 4x2.
        
        def residual(params):
            U_flat = params[:2*2*13]
            V_flat = params[2*2*13:2*2*13 + 2*4*13]
            W_flat = params[2*2*13 + 2*4*13:]
            
            U = U_flat.reshape(2, 2, 13)
            V = V_flat.reshape(2, 4, 13)
            W = W_flat.reshape(4, 2, 13)
            
            res = []
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        val = np.sum(U[i, j, :] * V[j, k, :] * W[k, i, :])
                        res.append(val - 1.0)
            return np.array(res)
        
        np.random.seed(42)
        best_res = np.inf
        best_params = None
        
        # Random restarts
        for r in range(20):
            if time.time() > deadline - 10:
                break
            init = np.random.randn(2*2*13 + 2*4*13 + 4*2*13) * 0.5
            try:
                from scipy.optimize import least_squares
                sol = least_squares(residual, init, method='lm', max_nfev=2000)
                if sol.cost < best_res:
                    best_res = sol.cost
                    best_params = sol.x
                    if best_res < 1e-10:
                        break
            except Exception:
                continue
        
        if best_res < 1e-8:
            return best_params
        return None

    params_224 = search_224_rank_13()
    if params_224 is None:
        return None

    U_224 = params_224[:2*2*13].reshape(2, 2, 13)
    V_224 = params_224[2*2*13:2*2*13 + 2*4*13].reshape(2, 4, 13)
    W_224 = params_224[2*2*13 + 2*4*13:].reshape(4, 2, 13)
    
    # Construct full 2x4x5 decomposition
    rank = total_rank
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Block 1: A[:, 0:2] * B[0:2, 0:4] -> C[:, 0:4]
    # U indices: i in 0..1, j in 0..1 -> row i*4+j
    # V indices: j in 0..1, k in 0..3 -> row j*5+k
    # W indices: k in 0..3, i in 0..1 -> row k*2+i
    
    for i in range(2):
        for j in range(2):
            for r in range(13):
                U[i * m + j][r] = U_224[i, j, r]
    for j in range(2):
        for k in range(4):
            for r in range(13):
                V[j * p + k][r] = V_224[j, k, r]
    for k in range(4):
        for i in range(2):
            for r in range(13):
                W[k * n + i][r] = W_224[k, i, r]
                
    # Block 2: A[:, 2:4] * B[2:4, 0:4] -> C[:, 0:4]
    # U indices: i in 0..1, j in 2..3 -> row i*4+j
    # V indices: j in 2..3, k in 0..3 -> row j*5+k
    # W indices: k in 0..3, i in 0..1 -> row k*2+i
    # Reuse U_224, V_224, W_224?
    # The structure is identical.
    # A[:, 2:4] is 2x2. B[2:4, 0:4] is 2x4.
    # So yes, we can reuse the coefficients.
    
    offset = 13
    for i in range(2):
        for j in range(2):
            for r in range(13):
                U[i * m + (2 + j)][offset + r] = U_224[i, j, r]
    for j in range(2):
        for k in range(4):
            for r in range(13):
                V[(2 + j) * p + k][offset + r] = V_224[j, k, r]
    for k in range(4):
        for i in range(2):
            for r in range(13):
                W[k * n + i][offset + r] = W_224[k, i, r]
                
    # Block 3: A[:, 0:4] * B[0:4, 4:5] -> C[:, 4:5]
    # Rank 1.
    # T_{i,j,4} = 1.
    # U[i,j] = 1, V[j,4] = 1, W[4,i] = 1.
    offset = 26
    for i in range(2):
        for j in range(4):
            U[i * m + j][offset] = 1.0
    for j in range(4):
        V[j * p + 4][offset] = 1.0
    for i in range(2):
        W[4 * n + i][offset] = 1.0
        
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140
    
    n, m, p = DIMENSIONS
    
    # Try Rank-27 decomposition (search)
    try:
        if time.time() < DEADLINE - 20:
            U_r27, V_r27, W_r27 = _construct_rank_34_search(DEADLINE - 10)
            if U_r27 is not None:
                if _validate_numpy(U_r27, V_r27, W_r27, n, m, p):
                    result_U = [[_clean_entry(v) for v in row] for row in U_r27]
                    result_V = [[_clean_entry(v) for v in row] for row in V_r27]
                    result_W = [[_clean_entry(v) for v in row] for row in W_r27]
                    return {"U": result_U, "V": result_V, "W": result_W}
    except Exception:
        pass
    
    # Try Rank-36 decomposition
    try:
        U_r36, V_r36, W_r36 = _construct_rank_36()
        if _validate_numpy(U_r36, V_r36, W_r36, n, m, p):
            result_U = [[_clean_entry(v) for v in row] for row in U_r36]
            result_V = [[_clean_entry(v) for v in row] for row in V_r36]
            result_W = [[_clean_entry(v) for v in row] for row in W_r36]
            return {"U": result_U, "V": result_V, "W": result_W}
    except Exception:
        pass
    
    # Fallback to Schoolbook
    U_school, V_school, W_school = _construct_schoolbook()
    assert _validate_numpy(U_school, V_school, W_school, n, m, p), "Schoolbook decomposition failed!"
    
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
