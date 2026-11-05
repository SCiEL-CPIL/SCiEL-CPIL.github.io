import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS

def _build_target_tensor():
    """Build the matrix multiplication tensor T with shape (n*m, m*p, p*n)."""
    T = np.zeros((N*M, M*P, P*N), dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T=None, tol=1e-8):
    """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = T[i*m+j, j*p+k, k*n+i]."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(N, M, rank)
    V3 = V.reshape(M, P, rank)
    W3 = W.reshape(P, N, rank)
    T_approx = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    # Map back to flat indices
    T_approx_flat = np.zeros_like(T)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                T_approx_flat[i*M+j, j*P+k, k*N+i] = T_approx[i, j, k]
    return np.max(np.abs(T_approx_flat - T)) < tol

def _schoolbook():
    """Return schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = np.zeros((N*M, rank))
    V = np.zeros((M*P, rank))
    W = np.zeros((P*N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i*M+j, col] = 1.0
                V[j*P+k, col] = 1.0
                W[k*N+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Strassen 2x2x2 decomposition.
    Returns U (4x7), V (4x7), W (4x7) for local indices.
    U index = i*2+j, V index = j*2+k, W index = k*2+i
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1 = (A11 + A22) * (B11 + B22)
    # C11 += M1, C22 += M1
    U[0, 0] = 1; U[3, 0] = 1  # A[0,0], A[1,1]
    V[0, 0] = 1; V[3, 0] = 1  # B[0,0], B[1,1]
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0], C[1,1]
    
    # M2 = (A21 + A22) * B11
    # C21 += M2
    U[2, 1] = 1; U[3, 1] = 1  # A[1,0], A[1,1]
    V[0, 1] = 1  # B[0,0]
    W[1, 1] = 1  # C[1,0] (k=0, i=1 -> index 0*2+1=1)
    
    # M3 = A11 * (B21 - B22)
    # C10 += M3, C11 -= M3
    U[0, 2] = 1  # A[0,0]
    V[2, 2] = 1; V[3, 2] = -1  # B[1,0], B[1,1]
    W[0, 2] = 1  # C[0,0] (k=0,i=0 -> 0)
    W[2, 2] = -1  # C[0,1] (k=1,i=0 -> 2)
    
    # M4 = A22 * (B11 - B12)
    # C20 += M4, C21 -= M4
    U[3, 3] = 1  # A[1,1]
    V[0, 3] = 1; V[1, 3] = -1  # B[0,0], B[0,1]
    W[1, 3] = 1  # C[1,0] (k=0,i=1 -> 1)
    W[3, 3] = -1  # C[1,1] (k=1,i=1 -> 3)
    
    # M5 = (A11 + A12) * B22
    # C11 += M5
    U[0, 4] = 1; U[1, 4] = 1  # A[0,0], A[0,1]
    V[3, 4] = 1  # B[1,1]
    W[2, 4] = 1  # C[0,1] (k=1,i=0 -> 2)
    
    # M6 = (A21 - A11) * (B11 + B12)
    # C20 += M6, C21 += M6, C10 -= M6, C11 -= M6
    U[2, 5] = 1; U[0, 5] = -1  # A[1,0], A[0,0]
    V[0, 5] = 1; V[1, 5] = 1  # B[0,0], B[0,1]
    W[1, 5] = 1  # C[1,0]
    W[3, 5] = 1  # C[1,1]
    W[0, 5] = -1  # C[0,0]
    W[2, 5] = -1  # C[0,1]
    
    # M7 = (A12 - A22) * (B21 + B22)
    # C10 += M7, C11 += M7, C20 -= M7, C21 -= M7
    U[1, 6] = 1; U[3, 6] = -1  # A[0,1], A[1,1]
    V[2, 6] = 1; V[3, 6] = 1  # B[1,0], B[1,1]
    W[0, 6] = 1  # C[0,0]
    W[2, 6] = 1  # C[0,1]
    W[1, 6] = -1  # C[1,0]
    W[3, 6] = -1  # C[1,1]
    
    return U, V, W

def _embed_strassen(U_s, V_s, W_s, i_off, j_off, k_off):
    """Embed a 2x2x2 Strassen block into the global 3x4x5 problem with offsets."""
    cols_U = []
    cols_V = []
    cols_W = []
    for c in range(7):
        U_col = np.zeros(N*M)
        V_col = np.zeros(M*P)
        W_col = np.zeros(P*N)
        for li in range(2):
            for lj in range(2):
                val = U_s[li*2+lj, c]
                if val != 0:
                    gi = li + i_off
                    gj = lj + j_off
                    U_col[gi*M + gj] = val
        for lj in range(2):
            for lk in range(2):
                val = V_s[lj*2+lk, c]
                if val != 0:
                    gj = lj + j_off
                    gk = lk + k_off
                    V_col[gj*P + gk] = val
        for li in range(2):
            for lk in range(2):
                val = W_s[lk*2+li, c]
                if val != 0:
                    gi = li + i_off
                    gk = lk + k_off
                    W_col[gk*N + gi] = val
        cols_U.append(U_col)
        cols_V.append(V_col)
        cols_W.append(W_col)
    return np.column_stack(cols_U), np.column_stack(cols_V), np.column_stack(cols_W)

def _build_strassen_blocks():
    """
    Build decomposition using Strassen for 2x2x2 sub-blocks.
    
    Strategy:
    - Split A (3x4) into row blocks: rows 0-1 (2 rows) and row 2 (1 row)
    - Split B (4x5) into column blocks: cols 0-3 (4 cols) and col 4 (1 col)
    - Split B rows into blocks: rows 0-1 (2 rows) and rows 2-3 (2 rows)
    
    Blocks:
    1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] : 2x2x2 Strassen (rank 7)
    2. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] : 2x2x2 Strassen (rank 7)
    3. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] : 2x2x2 Strassen (rank 7)
    4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] : 2x2x2 Strassen (rank 7)
    5. A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5] : 2x2x1 schoolbook (rank 4)
    6. A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5] : 2x2x1 schoolbook (rank 4)
    7. A[2:3, 0:2] * B[0:2, 0:5] -> C[2:3, 0:5] : 1x2x5 schoolbook (rank 10)
    8. A[2:3, 2:4] * B[2:4, 0:5] -> C[2:3, 0:5] : 1x2x5 schoolbook (rank 10)
    
    Total: 4*7 + 2*4 + 2*10 = 28 + 8 + 20 = 56
    """
    U_s, V_s, W_s = _strassen_2x2x2()
    
    all_U = []
    all_V = []
    all_W = []
    
    # Block 1: i_off=0, j_off=0, k_off=0
    U1, V1, W1 = _embed_strassen(U_s, V_s, W_s, 0, 0, 0)
    all_U.append(U1); all_V.append(V1); all_W.append(W1)
    
    # Block 2: i_off=0, j_off=0, k_off=2
    U2, V2, W2 = _embed_strassen(U_s, V_s, W_s, 0, 0, 2)
    all_U.append(U2); all_V.append(V2); all_W.append(W2)
    
    # Block 3: i_off=0, j_off=2, k_off=0
    U3, V3, W3 = _embed_strassen(U_s, V_s, W_s, 0, 2, 0)
    all_U.append(U3); all_V.append(V3); all_W.append(W3)
    
    # Block 4: i_off=0, j_off=2, k_off=2
    U4, V4, W4 = _embed_strassen(U_s, V_s, W_s, 0, 2, 2)
    all_U.append(U4); all_V.append(V4); all_W.append(W4)
    
    # Block 5: 2x2x1 schoolbook, i_off=0, j_off=0, k_off=4
    for lj in range(2):
        U_col = np.zeros(N*M)
        V_col = np.zeros(M*P)
        W_col = np.zeros(P*N)
        for li in range(2):
            U_col[(li+0)*M + (lj+0)] = 1.0
        V_col[(lj+0)*P + 4] = 1.0
        for li in range(2):
            W_col[4*N + (li+0)] = 1.0
        all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    # Block 6: 2x2x1 schoolbook, i_off=0, j_off=2, k_off=4
    for lj in range(2):
        U_col = np.zeros(N*M)
        V_col = np.zeros(M*P)
        W_col = np.zeros(P*N)
        for li in range(2):
            U_col[(li+0)*M + (lj+2)] = 1.0
        V_col[(lj+2)*P + 4] = 1.0
        for li in range(2):
            W_col[4*N + (li+0)] = 1.0
        all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    # Block 7: 1x2x5 schoolbook, i=2, j in 0,1, k in 0..4
    for lj in range(2):
        for gk in range(5):
            U_col = np.zeros(N*M)
            V_col = np.zeros(M*P)
            W_col = np.zeros(P*N)
            U_col[2*M + (lj+0)] = 1.0
            V_col[(lj+0)*P + gk] = 1.0
            W_col[gk*N + 2] = 1.0
            all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    # Block 8: 1x2x5 schoolbook, i=2, j in 2,3, k in 0..4
    for lj in range(2):
        for gk in range(5):
            U_col = np.zeros(N*M)
            V_col = np.zeros(M*P)
            W_col = np.zeros(P*N)
            U_col[2*M + (lj+2)] = 1.0
            V_col[(lj+2)*P + gk] = 1.0
            W_col[gk*N + 2] = 1.0
            all_U.append(U_col.reshape(-1,1)); all_V.append(V_col.reshape(-1,1)); all_W.append(W_col.reshape(-1,1))
    
    U_final = np.hstack(all_U)
    V_final = np.hstack(all_V)
    W_final = np.hstack(all_W)
    
    return U_final, V_final, W_final, U_final.shape[1]

def _round_to_halfint(mat):
    """Round to nearest half-integer."""
    return np.round(2.0 * mat) / 2.0

def _optimize_rank(U_init, V_init, W_init, T, max_iters=200, tol=1e-8):
    """Try to optimize a decomposition to reduce rank or verify exactness."""
    n, m, p = N, M, P
    rank = U_init.shape[1]
    U = U_init.copy()
    V = V_init.copy()
    W = W_init.copy()
    
    # Compute current error
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T_approx = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Map to flat
    T_approx_flat = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_approx_flat[i*m+j, j*p+k, k*n+i] = T_approx[i, j, k]
    
    error = np.max(np.abs(T_approx_flat - T))
    if error < tol:
        return U, V, W, True
    
    # Gradient-based optimization
    # Objective: sum of squared errors
    def compute_error(U, V, W):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_approx = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        T_approx_flat = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_approx_flat[i*m+j, j*p+k, k*n+i] = T_approx[i, j, k]
        return np.sum((T_approx_flat - T)**2)
    
    # Simple gradient descent
    lr = 0.01
    for it in range(max_iters):
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T_approx = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        T_approx_flat = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_approx_flat[i*m+j, j*p+k, k*n+i] = T_approx[i, j, k]
        
        grad_T = 2 * (T_approx_flat - T)
        
        # Compute gradients for U, V, W
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for r in range(rank):
            # grad_U[i*m+j, r] = sum over (j',k') where T[i*m+j, j'*p+k', k'*n+i] != 0
            # Actually, T_approx[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # dT_approx[a,b,c]/dU[a,r] = V[b,r]*W[c,r]
            # So grad_U[a,r] = sum_{b,c} grad_T[a,b,c] * V[b,r] * W[c,r]
            # This is expensive. Let's use einsum.
            pass
        
        # Use a simpler approach: just check if rounding helps
        break
    
    return U, V, W, False

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # First, try the structured Strassen-based decomposition
    U_struct, V_struct, W_struct, rank_struct = _build_strassen_blocks()
    
    T = _build_target_tensor()
    
    # Verify the structured decomposition
    if _verify_decomposition(U_struct, V_struct, W_struct, T, tol=1e-8):
        # Round to half-integers
        U_rounded = _round_to_halfint(U_struct)
        V_rounded = _round_to_halfint(V_struct)
        W_rounded = _round_to_halfint(W_struct)
        
        if _verify_decomposition(U_rounded, V_rounded, W_rounded, T, tol=1e-8):
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            # Use the exact values (they should already be integers/half-integers)
            return {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
    
    # Fallback: schoolbook
    U_sb, V_sb, W_sb, rank_sb = _schoolbook()
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

# Test the Strassen embedding
if __name__ == "__main__":
    T = _build_target_tensor()
    U, V, W, rank = _build_strassen_blocks()
    print(f"Structured rank: {rank}")
    valid = _verify_decomposition(U, V, W, T, tol=1e-8)
    print(f"Valid: {valid}")
    
    if valid:
        Ur = _round_to_halfint(U)
        Vr = _round_to_halfint(V)
        Wr = _round_to_halfint(W)
        valid_r = _verify_decomposition(Ur, Vr, Wr, T, tol=1e-8)
        print(f"Rounded valid: {valid_r}")
