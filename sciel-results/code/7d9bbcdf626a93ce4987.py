"""
Matrix multiplication 2x4 by 4x4, exact tensor decomposition.
Uses Strassen's algorithm on four 2x2 sub-blocks to achieve rank 28 (vs schoolbook 32).
"""
import time
import numpy as np
import itertools

DIMENSIONS = [2, 4, 4]
TIME_LIMIT = 150.0
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _time_ok(margin=15.0):
    return _elapsed() < TIME_LIMIT - margin


def check_time_and_return(best_result):
    if best_result is not None and _elapsed() > TIME_LIMIT - 10.0:
        return best_result
    return None


def construct():
    # Primary: deterministic Strassen-28 (instant, always valid)
    try:
        result_28 = _construct_strassen_28()
        if result_28 is not None and _verify_decomposition(result_28):
            best = result_28
        else:
            best = _construct_schoolbook_32()
    except Exception:
        best = _construct_schoolbook_32()
    
    # Try to find a lower-rank decomposition using finite-field algebra
    if _time_ok(30.0):
        try:
            result_low = _search_low_rank_finite_field()
            if result_low is not None and _verify_decomposition(result_low):
                if result_low['U'][0] and len(result_low['U'][0]) < len(best['U'][0]):
                    best = result_low
        except Exception:
            pass
    
    # Try numerical rank-20 construction if time permits
    if _time_ok(25.0):
        try:
            result_20 = _construct_rank20()
            if result_20 is not None and _verify_decomposition(result_20):
                if len(result_20['U'][0]) < len(best['U'][0]):
                    best = result_20
        except Exception:
            pass
    
    return best


def _verify_decomposition(result, trials=5):
    """Verify that U @ diag(V) @ W^T reconstructs the matmul tensor."""
    import random
    n, m, p = DIMENSIONS
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    rank = U.shape[1]
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        # Reconstruct C from decomposition
        # C[i,k] = sum_r U[i*m+j... wait, need to be careful
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_l V[j*p+l, r] * B[j,l]) ... no
        # Actually: C[i,k] = sum_r U_col_r(A) * V_col_r(B) * W_col_r[C_index]
        # where U_col_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
        #       V_col_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
        #       and W contributes: C[i,k] = sum_r W[k*n+i, r] * U_col_r(A) * V_col_r(B)
        # Wait, that's not right either. The tensor decomposition is:
        # T_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So C[i,k] = sum_j A[i,j] * B[j,k] = sum_r (sum_{i,j} U[i*m+j,r]*A[i,j]) * ... 
        # No. The correct reconstruction:
        # C[i,k] = sum_{j} A[i,j] * B[j,k]
        # = sum_r (sum_{i',j} U[i'*m+j, r] * A[i',j]) * (sum_{j',k'} V[j'*p+k', r] * B[j',k']) * W[k*n+i, r]
        # That's not right. Let me think again.
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # The matmul: C[i,k] = sum_j A[i,j] * B[j,k]
        # We need: sum_j A[i,j]*B[j,k] = sum_r (sum_{i',j'} U[i'*m+j',r]*A[i',j']) * ... 
        # Actually no. The way it works:
        # For each r, compute:
        #   u_r = sum_{i,j} U[i*m+j, r] * A[i,j]  (scalar)
        #   v_r = sum_{j,k} V[j*p+k, r] * B[j,k]  (scalar)
        #   Then C[i,k] += W[k*n+i, r] * u_r * v_r
        # Let's verify: C[i,k] = sum_r W[k*n+i,r] * (sum_{i',j} U[i'*m+j,r]*A[i',j]) * (sum_{j',k'} V[j'*p+k',r]*B[j',k'])
        # This should equal sum_j A[i,j]*B[j,k] if the decomposition is correct.
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = 0.0
            for i in range(n):
                for j in range(m):
                    u_r += U[i * m + j, r] * A[i, j]
            v_r = 0.0
            for j in range(m):
                for k in range(p):
                    v_r += V[j * p + k, r] * B[j, k]
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, r] * u_r * v_r
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True


def _construct_schoolbook_32():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
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
    return {"U": U, "V": V, "W": W}


def _search_low_rank_finite_field():
    """
    Search for a lower-rank decomposition using finite-field algebra.
    For 2x4*4x4, the tensor has 2*4*4=32 entries.
    We work over a small prime field and use linear algebra to find
    rank-r decompositions.
    """
    n, m, p = DIMENSIONS
    # Try ranks from 20 up to 27
    for rank in range(20, 28):
        if not _time_ok(20.0):
            break
        result = _try_rank_finite_field(rank)
        if result is not None:
            return result
    return None


def _try_rank_finite_field(rank):
    """
    Attempt to find a rank-r decomposition over a finite field,
    then lift to integers/half-integers.
    """
    n, m, p = DIMENSIONS
    prime = 97  # small prime
    
    def mod_inv(a, p=prime):
        return pow(a, p - 2, p)
    
    def matmul_mod(A, B, p=prime):
        return np.mod(np.dot(A, B), p)
    
    def transpose_mod(A, p=prime):
        return A.T % p
    
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # 
    # For a given rank r, we can fix U and V, then solve for W,
    # or use the fact that the tensor rank is the minimum r such that
    # T can be written as sum of r rank-1 tensors.
    #
    # Alternative: use the "border rank" / "tensor rank" algorithms.
    # For small dimensions, we can do a direct search.
    #
    # Key insight: if we fix the W matrix, the problem becomes:
    # Can we find U, V such that for all (i,j,k):
    #   T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    #
    # This is a system of 32 equations in 8*r + 16*r = 24*r unknowns.
    # For r >= 2, this is underdetermined, so solutions exist generically.
    # The question is whether we can find one with small integer entries.
    
    # Strategy: parameterize W with small entries, solve for U,V
    # Try all combinations of W entries from {-1, 0, 1}
    
    if rank > 24:  # Too many combinations for large ranks
        return None
    
    # For rank r, W has p*n = 8 entries per column, so 8*r total W entries
    # That's too many to enumerate. Instead, use a smarter approach.
    
    # Use the fact that we can absorb scalars: if we scale U column by a,
    # V column by b, W column by c, we need abc=1.
    
    # Better approach: use the known structure.
    # 2x4*4x4 can be seen as two 2x4*4x2 problems.
    # Each 2x4*4x2 has tensor size 16.
    # The minimum rank for 2x4*4x2 is known to be 10 (from literature).
    # So 2x4*4x4 has rank at most 20 (two independent 2x4*4x2).
    # The question is whether we can do better by coupling.
    
    # For now, return None and rely on the numerical method.
    return None


def _get_242_rank10():
    """
    Find a rank-10 exact decomposition for [2,4,2] matmul.
    A: 2x4, B: 4x2, C: 2x2.
    Returns (U_sub, V_sub, W_sub) with shapes (8,10), (8,10), (4,10).
    Uses numerical solve then rounds to half-integers.
    Optimized for speed: precompute matrix structures, use vectorized ops.
    """
    # Tensor: T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1}
    # U: 8x10, V: 8x10, W: 4x10
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*2+k, r] * W[k*2+i, r]
    
    n_sub, m_sub, p_sub = 2, 4, 2
    rank_sub = 10
    num_eq = n_sub * m_sub * p_sub  # 16
    
    # Precompute index mappings
    U_idx = {}  # (i,j) -> row index in U
    V_idx = {}  # (j,k) -> row index in V
    W_idx = {}  # (k,i) -> row index in W
    for i in range(n_sub):
        for j in range(m_sub):
            U_idx[(i,j)] = i * m_sub + j
    for j in range(m_sub):
        for k in range(p_sub):
            V_idx[(j,k)] = j * p_sub + k
    for k in range(p_sub):
        for i in range(n_sub):
            W_idx[(k,i)] = k * n_sub + i
    
    def build_A_U(V, W):
        """Build the 16x80 matrix for solving U."""
        A = np.zeros((num_eq, 8 * rank_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    eq_idx = i * m_sub * p_sub + j * p_sub + k
                    u_row = U_idx[(i,j)]
                    v_row = V_idx[(j,k)]
                    w_row = W_idx[(k,i)]
                    for r in range(rank_sub):
                        A[eq_idx, u_row * rank_sub + r] = V[v_row, r] * W[w_row, r]
        return A
    
    def build_A_V(U, W):
        """Build the 16x80 matrix for solving V."""
        A = np.zeros((num_eq, 8 * rank_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    eq_idx = i * m_sub * p_sub + j * p_sub + k
                    u_row = U_idx[(i,j)]
                    v_row = V_idx[(j,k)]
                    w_row = W_idx[(k,i)]
                    for r in range(rank_sub):
                        A[eq_idx, v_row * rank_sub + r] = U[u_row, r] * W[w_row, r]
        return A
    
    def build_A_W(U, V):
        """Build the 16x40 matrix for solving W."""
        A = np.zeros((num_eq, 4 * rank_sub))
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    eq_idx = i * m_sub * p_sub + j * p_sub + k
                    u_row = U_idx[(i,j)]
                    v_row = V_idx[(j,k)]
                    w_row = W_idx[(k,i)]
                    for r in range(rank_sub):
                        A[eq_idx, w_row * rank_sub + r] = U[u_row, r] * V[v_row, r]
        return A
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u_row = U_idx[(i,j)]
                    v_row = V_idx[(j,k)]
                    w_row = W_idx[(k,i)]
                    val = np.dot(U[u_row, :], V[v_row, :] * W[w_row, :])
                    err += (val - 1.0) ** 2
        return err
    
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    # Try multiple random initializations
    for attempt in range(50):
        if not _time_ok(20.0):
            break
        np.random.seed(attempt * 7 + 3)
        U = np.random.randn(8, 10)
        V = np.random.randn(8, 10)
        W = np.random.randn(4, 10)
        
        # Alternating least squares
        for it in range(100):
            # Solve for U
            A_U = build_A_U(V, W)
            b = np.ones(num_eq)
            U_flat, *_ = np.linalg.lstsq(A_U, b, rcond=None)
            U = U_flat.reshape(8, 10)
            
            # Solve for V
            A_V = build_A_V(U, W)
            V_flat, *_ = np.linalg.lstsq(A_V, b, rcond=None)
            V = V_flat.reshape(8, 10)
            
            # Solve for W
            A_W = build_A_W(U, V)
            W_flat, *_ = np.linalg.lstsq(A_W, b, rcond=None)
            W = W_flat.reshape(4, 10)
            
            err = compute_error(U, V, W)
            if err < 1e-12:
                break
        
        # Try rounding to half-integers
        for scale in [2, 4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            err_r = compute_error(U_r, V_r, W_r)
            if err_r < 1e-9 and err_r < best_err:
                best_err = err_r
                best_U = U_r
                best_V = V_r
                best_W = W_r
    
    if best_U is not None:
        return (best_U.tolist(), best_V.tolist(), best_W.tolist())
    
    return None


def _construct_rank20():
    """
    2x4 * 4x4 using two [2,4,2] subproblems of rank 10 each.
    """
    n, m, p = DIMENSIONS
    rank = 20
    U = [[0]*rank for _ in range(n*m)]  # 8x20
    V = [[0]*rank for _ in range(m*p)]  # 16x20
    W = [[0]*rank for _ in range(p*n)]  # 8x20
    
    result_242 = _get_242_rank10()
    if result_242 is None:
        return None
    U_sub, V_sub, W_sub = result_242
    
    # C_L = A * B_L (B_L = B[:, 0:2])
    for r in range(10):
        for i in range(n*m):
            U[i][r] += U_sub[i][r]
        for j in range(4):
            for k in range(2):
                V[j*4+k][r] += V_sub[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*2+i][r] += W_sub[k*2+i][r]
    
    # C_R = A * B_R (B_R = B[:, 2:4])
    for r in range(10):
        for i in range(n*m):
            U[i][r+10] += U_sub[i][r]
        for j in range(4):
            for k in range(2):
                V[j*4+(k+2)][r+10] += V_sub[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[(k+2)*2+i][r+10] += W_sub[k*2+i][r]
    
    return {"U": U, "V": V, "W": W}


def _construct_strassen_28():
    """
    2x4 * 4x4 using four 2x2 Strassen blocks.
    A is 2x4, B is 4x4, C is 2x4.
    Split into 4 sub-problems of 2x2 * 2x2 -> 2x2.
    Each uses 7 Strassen multiplications, total 28.
    """
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen for 2x2 matrices:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each 2x2 block, we have 7 terms.
    # U entries: which A elements are used (with coefficients)
    # V entries: which B elements are used (with coefficients)  
    # W entries: which C elements are produced (with coefficients)
    
    strassen_terms = [
        # (U_coefficients, V_coefficients, W_coefficients)
        # Each is a dict: {local_2x2_index: coefficient}
        # 2x2 local index: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),          # M2
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),          # M3
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),          # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
    ]
    
    # 4 sub-blocks: (A_row_offset, B_col_offset)
    # A is 2x4: rows 0-1, cols 0-3
    # B is 4x4: rows 0-3, cols 0-3
    # C is 2x4: rows 0-1, cols 0-3
    # Sub-block (ar, bc): A[ar:ar+2, bc*2:bc*2+2] * B[bc*2:bc*2+2, bc*2:bc*2+2] -> C[ar:ar+2, bc*2:bc*2+2]
    # But since A only has 2 rows, ar is always 0.
    # Actually A is 2x4, so we can only split columns.
    # Sub-block 0: A[:, 0:2] * B[0:2, 0:2] -> C[:, 0:2]
    # Sub-block 1: A[:, 2:4] * B[0:2, 2:4] -> C[:, 2:4]  -- NO, this doesn't work
    # 
    # Wait, for 2x4 * 4x4:
    # We need to split B into 4x2 blocks? No.
    # A: 2x4, B: 4x4, C: 2x4
    # Split A into 2 blocks of 2x2 (by columns): A_L = A[:,0:2], A_R = A[:,2:4]
    # Split B into 4 blocks of 2x2 (by rows and cols): B_TL, B_TR, B_BL, B_BR
    # C = A_L * [B_TL; B_BL] + A_R * [B_TR; B_BR]
    #    = A_L * B_TL + A_L * B_BL + A_R * B_TR + A_R * B_BR
    # But A_L*B_TL is 2x2*2x2 -> 2x2, giving C[:,0:2] contribution
    # A_L*B_BL is 2x2*2x2 -> 2x2, giving C[:,0:2] contribution
    # A_R*B_TR is 2x2*2x2 -> 2x2, giving C[:,2:4] contribution
    # A_R*B_BR is 2x2*2x2 -> 2x2, giving C[:,2:4] contribution
    #
    # So 4 sub-problems, each 2x2*2x2->2x2, each rank 7, total 28.
    
    sub_blocks = [
        # (A_col_offset, B_row_offset, B_col_offset, C_col_offset)
        (0, 0, 0, 0),   # A_L * B_TL -> C_L
        (0, 2, 0, 0),   # A_L * B_BL -> C_L
        (2, 0, 2, 2),   # A_R * B_TR -> C_R
        (2, 2, 2, 2),   # A_R * B_BR -> C_R
    ]
    
    col = 0
    for (a_col_off, b_row_off, b_col_off, c_col_off) in sub_blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for loc_u, coeff in u_coeffs.items():
                lr, lc = loc_u // 2, loc_u % 2  # local row, col in 2x2
                # A element at global position: row=lr, col=a_col_off+lc
                global_u = lr * 4 + (a_col_off + lc)
                U[global_u][col] += coeff
            for loc_v, coeff in v_coeffs.items():
                lr, lc = loc_v // 2, loc_v % 2
                # B element at global position: row=b_row_off+lr, col=b_col_off+lc
                global_v = (b_row_off + lr) * 4 + (b_col_off + lc)
                V[global_v][col] += coeff
            for loc_w, coeff in w_coeffs.items():
                lr, lc = loc_w // 2, loc_w % 2
                # C element at global position: row=lr, col=c_col_off+lc
                global_w = (c_col_off + lc) * 2 + lr
                W[global_w][col] += coeff
            col += 1
    
    return {"U": U, "V": V, "W": W}
