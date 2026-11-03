import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

def _build_tensor():
    """Build the matrix multiplication tensor T[n*m*p] as a flat list."""
    n, m, p = DIMENSIONS
    size = n * m * p
    T = [0.0] * size
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m * p + j * p + k] = 1.0
    return T, n, m, p

def _apply_factors(U, V, W, n, m, p):
    """Apply factor matrices to produce the reconstructed tensor."""
    rank = len(U[0]) if U else 0
    nmp = n * m * p
    result = [0.0] * nmp
    for r in range(rank):
        u_col = [U[i][r] for i in range(n * m)]
        v_col = [V[j][r] for j in range(m * p)]
        w_col = [W[k][r] for k in range(p * n)]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_col[j * p + k]
                    if v_val == 0:
                        continue
                    w_val = w_col[k * n + i]
                    result[i * m * p + j * p + k] += u_val * v_val * w_val
    return result

def _verify(U, V, W, n, m, p, T):
    """Verify that the decomposition is exact using NumPy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
        
    U_np = np.array(U, dtype=float).reshape(n*m, rank)
    V_np = np.array(V, dtype=float).reshape(m*p, rank)
    W_np = np.array(W, dtype=float).reshape(p*n, rank)
    
    # Reshape to tensor-like structures
    # U: (n, m, rank)
    # V: (m, p, rank)
    # W: (p, n, rank)
    
    U_t = U_np.reshape(n, m, rank)
    V_t = V_np.reshape(m, p, rank)
    W_t = W_np.reshape(p, n, rank)
    
    # Contract over rank r
    # Result[i, j, k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Use einsum
    try:
        result = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        T_np = np.array(T, dtype=float).reshape(n, m, p)
        if np.allclose(result, T_np, atol=1e-9):
            return True
    except Exception:
        pass
        
    # Fallback to manual check if einsum fails
    result = _apply_factors(U, V, W, n, m, p)
    for i in range(len(T)):
        if abs(result[i] - T[i]) > 1e-9:
            return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return Strassen decomposition for 2x2x2 matrix multiplication.
    Rank 7.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10 + A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B10 - B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01 - B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00 + A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01 - A11)(B10 + B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, 7

def _verify_strassen():
    """Verify Strassen decomposition is correct for 2x2x2."""
    sU, sV, sW, srank = _strassen_2x2x2()
    n, m, p = 2, 2, 2
    T, _, _, _ = _build_tensor()
    # Build 2x2x2 tensor
    T222 = [0.0] * 8
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i * 4 + j * 2 + k] = 1.0
    result = _apply_factors(sU, sV, sW, n, m, p)
    for idx in range(8):
        if abs(result[idx] - T222[idx]) > 1e-9:
            return False
    return True

def _block_decomposition(n, m, p, T):
    """
    Block decomposition for 2x3x7 using Strassen for 2x2x2 sub-blocks.
    Returns (U, V, W, rank) or None if verification fails.
    """
    total_rank = 39  # 14 + 25
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Block 2: j=1,2, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 2a: j=1,2; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2b: j=1,2; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2c: j=1,2; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + (1 + j)][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[(1 + j) * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 2d: j=1,2; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + (1 + j)][col] = 1
            V[(1 + j) * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _block_decomposition_alt(n, m, p, T):
    """
    Alternative block split: m=3 as {0,1}|{2}, p=7 as (0,1),(2,3),(4,5),(6)
    Same total rank 39, different arrangement.
    """
    total_rank = 39
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Block 1: j=0,1, m_block=2, p=7
    sU, sV, sW, srank = _strassen_2x2x2()
    
    # Sub-block 1a: j=0,1; k=0,1. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + k][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[k * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1b: j=0,1; k=2,3. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (2 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(2 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1c: j=0,1; k=4,5. Strassen 2x2x2.
    for r in range(srank):
        for i in range(2):
            for j in range(2):
                U[i * m + j][col] += sU[i * 2 + j][r]
        for j in range(2):
            for k in range(2):
                V[j * p + (4 + k)][col] += sV[j * 2 + k][r]
        for k in range(2):
            for i in range(2):
                W[(4 + k) * n + i][col] += sW[k * 2 + i][r]
        col += 1
    
    # Sub-block 1d: j=0,1; k=6. Schoolbook 2x2x1.
    for i in range(2):
        for j in range(2):
            U[i * m + j][col] = 1
            V[j * p + 6][col] = 1
            W[6 * n + i][col] = 1
            col += 1
    
    # Block 2: j=2, m_block=1, p=7. Schoolbook 2x1x7.
    for i in range(n):
        for k in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    if col != total_rank:
        return None
    
    if not _verify(U, V, W, n, m, p, T):
        return None
    
    return U, V, W, total_rank

def _is_half_integer(x, tol=1e-9):
    """Check if x is a half-integer (k/2 for integer k)."""
    val = 2.0 * x
    nearest = round(val)
    return abs(val - nearest) < tol

def _project_half_integer(vec, tol=1e-9):
    """Project vector to nearest half-integer grid."""
    return [round(2.0 * v) / 2.0 for v in vec]

def _greedy_peel(T, n, m, p, max_rank, deadline):
    """
    Greedy rank-1 peeling: iteratively extract the best rank-1 term
    via SVD on the tensor matricization, project to half-integers,
    and subtract. Returns (U, V, W, rank) or None if fails.
    """
    nmp = n * m * p
    T_np = np.array(T, dtype=float).reshape(n, m, p)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for r in range(max_rank):
        if time.time() > deadline:
            break
        
        # Build matricization T_1: (n) x (m*p)
        # T_1[i, j*p+k] = T[i,j,k]
        T_1 = T_np.reshape(n, m * p)
        
        # SVD
        try:
            U_mat, s, Vt = np.linalg.svd(T_1, full_matrices=False)
        except:
            break
        
        if len(s) == 0 or s[0] < 1e-12:
            break
        
        # Take first singular vector
        u_vec = U_mat[:, 0]  # length n
        v_vec = Vt[0, :]     # length m*p
        
        # Reconstruct rank-1 term: u_vec[i] * v_vec[j*p+k]
        # This gives us T[i,j,k] = u[i] * v[j*p+k] * w[k*n+i]
        # For the matricization T_1[i, j*p+k] = sum_r u_r[i] * v_r[j*p+k] * w_r[k*n+i]
        # The third factor w must satisfy: for each (i,j,k), 
        #   T[i,j,k] = u[i] * v[j*p+k] * w[k*n+i]
        # So w[k*n+i] = T[i,j,k] / (u[i] * v[j*p+k]) when both are nonzero
        
        # Build w vector
        w_vec = [0.0] * (p * n)
        valid = True
        for i in range(n):
            if abs(u_vec[i]) < 1e-12:
                continue
            for j in range(m):
                for k in range(p):
                    idx_v = j * p + k
                    if abs(v_vec[idx_v]) < 1e-12:
                        continue
                    expected = T_np[i, j, k]
                    if abs(expected) < 1e-12:
                        continue
                    w_val = expected / (u_vec[i] * v_vec[idx_v])
                    w_idx = k * n + i
                    if w_vec[w_idx] != 0.0 and abs(w_vec[w_idx] - w_val) > 1e-9:
                        valid = False
                        break
                    w_vec[w_idx] = w_val
            if not valid:
                break
        if not valid:
            # Try next singular vector or give up
            # For now, just use the raw SVD result
            # Project to half-integers
            u_proj = _project_half_integer(u_vec.tolist())
            v_proj = _project_half_integer(v_vec.tolist())
            w_proj = _project_half_integer(w_vec)
            
            # Verify this rank-1 term
            term = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term[i, j, k] = u_proj[i] * v_proj[j * p + k] * w_proj[k * n + i]
            
            # Subtract from remaining tensor
            T_np -= term
            U_cols.append(u_proj)
            V_cols.append(v_proj)
            W_cols.append(w_proj)
        else:
            # Valid rank-1 term found
            u_proj = _project_half_integer(u_vec.tolist())
            v_proj = _project_half_integer(v_vec.tolist())
            w_proj = _project_half_integer(w_vec)
            
            # Verify
            term = np.zeros((n, m, p), dtype=float)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term[i, j, k] = u_proj[i] * v_proj[j * p + k] * w_proj[k * n + i]
            
            T_np -= term
            U_cols.append(u_proj)
            V_cols.append(v_proj)
            W_cols.append(w_proj)
    
    if not U_cols:
        return None
    
    rank = len(U_cols)
    
    # Check if tensor is fully decomposed
    if np.max(np.abs(T_np)) > 1e-9:
        return None
    
    # Build final matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = U_cols[r][i]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = V_cols[r][j * p + k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = W_cols[r][k * n + i]
    
    return U, V, W, rank

def _row_sum_reduction(n, m, p, T):
    """
    Try to reduce rank by exploiting row-sum identities.
    For 2x3x7: express row 2 of intermediate as combo of rows 0,1.
    Returns (U, V, W, rank) or None.
    """
    # Strategy: Use Strassen-like tricks on the 3x7 part
    # Split m=3 into {0,1} and {2}, but use the identity that
    # row_2 = row_0 + row_1 - (row_0 - row_1) to share computations
    
    # Actually, let's try a different approach: 
    # For the 3x7 multiplication, we can use the fact that
    # C[i,k] = sum_j A[i,j] B[j,k]
    # We can compute C[0,:], C[1,:], C[2,:] using shared products
    
    # Let's try: compute D0 = A[0,:]*B, D1 = A[1,:]*B, D2 = A[2,:]*B
    # But also compute E = (A[0,:]+A[1,:])*B = D0 + D1
    # And F = (A[0,:]-A[1,:])*B = D0 - D1
    # Then D2 = A[2,:]*B can be computed independently
    
    # This doesn't directly help. Let's try a more systematic search.
    
    # Try greedy peeling with various max_rank values
    for max_rank in range(30, 40):
        result = _greedy_peel(T, n, m, p, max_rank, time.time() + 30)
        if result is not None:
            U, V, W, rank = result
            if _verify(U, V, W, n, m, p, T):
                return U, V, W, rank
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    T, n, m, p = _build_tensor()
    
    # Verify Strassen is correct
    if not _verify_strassen():
        # Strassen is broken, fall back to schoolbook
        U, V, W, rank = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Start with schoolbook as fallback
    U_school, V_school, W_school, rank_school = _schoolbook(n, m, p)
    
    # Best result so far
    best_result = (U_school, V_school, W_school, rank_school)
    
    # Try primary block decomposition
    result = _block_decomposition(n, m, p, T)
    if result is not None:
        U, V, W, rank = result
        if rank < best_result[3]:
            best_result = result
    
    # Try alternative block decomposition
    if time.time() < deadline:
        result = _block_decomposition_alt(n, m, p, T)
        if result is not None:
            U, V, W, rank = result
            if rank < best_result[3]:
                best_result = result
    
    # Try row-sum reduction / greedy peeling
    if time.time() < deadline - 20:
        result = _row_sum_reduction(n, m, p, T)
        if result is not None:
            U, V, W, rank = result
            if rank < best_result[3]:
                best_result = result
    
    U, V, W, rank = best_result
    return {"U": U, "V": V, "W": W}
