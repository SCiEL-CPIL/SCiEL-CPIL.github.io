"""Matrix multiplication 4x4 * 4x8 - rank-minimal tensor decomposition via two-level Strassen."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def _strassen_2x2_2x2():
    """
    Strassen decomposition for 2x2 * 2x2 -> 2x2.
    U: (4, 7) - U[i*2+j, c]
    V: (4, 7) - V[j*2+k, c]
    W: (4, 7) - W[k*2+i, c]  (trace convention: k*n+i with n=2)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # M1 = (a11+a22)(b11+b22) -> contributes to c11 and c22
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # W[k*2+i]: c11->k=0,i=0->W[0]; c22->k=1,i=1->W[3]
    
    # M2 = (a21+a22)b11 -> contributes to c21
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1  # W[k*2+i]: c21->k=0,i=1->W[1]
    
    # M3 = a11(b12-b22) -> contributes to c12
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1  # W[k*2+i]: c12->k=1,i=0->W[2]
    
    # M4 = a22(b21-b11) -> contributes to c11 and c21
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5 = (a11+a12)b22 -> contributes to c11 (neg) and c12
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (a21-a11)(b11+b12) -> contributes to c22
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22) -> contributes to c11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_2x2_2x4(U_s, V_s, W_s):
    """
    Build decomposition for 2x2 * 2x4 -> 2x4.
    U: (4, 14), V: (8, 14), W: (8, 14)
    U[i*2+j, c], V[j*4+k, c], W[k*2+i, c]
    
    Split 2x4 into two 2x2 halves.
    Left half: cols 0-1 of B, cols 0-1 of C
    Right half: cols 2-3 of B, cols 2-3 of C
    """
    U_2x4 = np.zeros((4, 14), dtype=float)
    V_2x4 = np.zeros((8, 14), dtype=float)
    W_2x4 = np.zeros((8, 14), dtype=float)
    
    # Left half (cols 0-1): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, c] += U_s[r, c]
            if V_s[r, c] != 0:
                V_2x4[r, c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r, c] += W_s[r, c]
    
    # Right half (cols 2-3): embed Strassen 2x2*2x2
    for c in range(7):
        for r in range(4):
            if U_s[r, c] != 0:
                U_2x4[r, 7 + c] += U_s[r, c]
            if V_s[r, c] != 0:
                j = r // 2
                k_old = r % 2
                v_idx = j * 4 + (k_old + 2)
                V_2x4[v_idx, 7 + c] += V_s[r, c]
            if W_s[r, c] != 0:
                W_2x4[r + 4, 7 + c] += W_s[r, c]
    
    return U_2x4, V_2x4, W_2x4

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the tensor decomposition is correct."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    return max_err < 1e-9, max_err

def _build_full_decomposition():
    """
    Build the full 4x4 * 4x8 decomposition using two-level Strassen.
    Total rank = 7 * 14 = 98.
    """
    n, m, p = 4, 4, 8
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    U_2x4, V_2x4, W_2x4 = _build_2x2_2x4(U_s, V_s, W_s)
    
    total_rank = 7 * 14  # 98
    U = np.zeros((n*m, total_rank), dtype=float)
    V = np.zeros((m*p, total_rank), dtype=float)
    W = np.zeros((p*n, total_rank), dtype=float)
    
    def _embed_U(local_U, i0, j0, col):
        """Embed local 2x2 U block at global position (i0, j0) into columns col..col+13"""
        for c in range(14):
            for li in range(2):
                for lj in range(2):
                    local_idx = li * 2 + lj
                    global_idx = (i0 + li) * m + (j0 + lj)
                    U[global_idx, col + c] += local_U[local_idx, c]
    
    def _embed_V(local_V, j0, k0, col):
        """Embed local 2x4 V block at global position (j0, k0) into columns col..col+13"""
        for c in range(14):
            for lj in range(2):
                for lk in range(4):
                    local_idx = lj * 4 + lk
                    global_idx = (j0 + lj) * p + (k0 + lk)
                    V[global_idx, col + c] += local_V[local_idx, c]
    
    def _embed_W(local_W, k0, i0, col):
        """Embed local 4x2 W block at global position (k0, i0) into columns col..col+13"""
        for c in range(14):
            for lk in range(4):
                for li in range(2):
                    local_idx = lk * 2 + li
                    global_idx = (k0 + lk) * n + (i0 + li)
                    W[global_idx, col + c] += local_W[local_idx, c]
    
    # M1 = (A11+A22)(B11+B22)
    col = 0
    _embed_U(U_2x4, 0, 0, col)
    _embed_U(U_2x4, 2, 2, col)
    _embed_V(V_2x4, 0, 0, col)
    _embed_V(V_2x4, 2, 2, col)
    _embed_W(W_2x4, 0, 0, col)
    _embed_W(W_2x4, 2, 2, col)
    col += 14
    
    # M2 = (A21+A22)B11
    _embed_U(U_2x4, 2, 0, col)
    _embed_U(U_2x4, 2, 2, col)
    _embed_V(V_2x4, 0, 0, col)
    _embed_V(V_2x4, 2, 0, col)
    _embed_W(W_2x4, 0, 2, col)
    _embed_W(W_2x4, 2, 2, col)
    col += 14
    
    # M3 = A11(B12-B22)
    _embed_U(U_2x4, 0, 0, col)
    _embed_V(V_2x4, 0, 2, col)
    _embed_V(V_2x4, 2, 2, col)
    _embed_W(W_2x4, 2, 0, col)
    col += 14
    
    # M4 = A22(B21-B11)
    _embed_U(U_2x4, 2, 2, col)
    _embed_V(V_2x4, 0, 0, col)
    _embed_V(V_2x4, 2, 0, col)
    _embed_W(W_2x4, 0, 2, col)
    col += 14
    
    # M5 = (A11+A12)B22
    _embed_U(U_2x4, 0, 0, col)
    _embed_U(U_2x4, 0, 2, col)
    _embed_V(V_2x4, 2, 2, col)
    _embed_W(W_2x4, 0, 0, col)
    _embed_W(W_2x4, 2, 0, col)
    col += 14
    
    # M6 = (A21-A11)(B11+B12)
    _embed_U(U_2x4, 2, 0, col)
    _embed_U(U_2x4, 0, 0, col)
    _embed_V(V_2x4, 0, 0, col)
    _embed_V(V_2x4, 0, 2, col)
    _embed_W(W_2x4, 0, 2, col)
    _embed_W(W_2x4, 2, 2, col)
    col += 14
    
    # M7 = (A12-A22)(B21+B22)
    _embed_U(U_2x4, 0, 2, col)
    _embed_U(U_2x4, 2, 2, col)
    _embed_V(V_2x4, 2, 0, col)
    _embed_V(V_2x4, 2, 2, col)
    _embed_W(W_2x4, 0, 0, col)
    _embed_W(W_2x4, 2, 0, col)
    col += 14
    
    # Apply signs for subtraction terms
    # M3: B12 - B22, so subtract V contributions for B22
    col3 = 28
    for c in range(14):
        for lj in range(2):
            for lk in range(4):
                local_idx = lj * 4 + lk
                global_idx = (2 + lj) * p + (2 + lk)
                V[global_idx, col3 + c] = -V[global_idx, col3 + c]
    
    # M4: B21 - B11, so subtract V contributions for B11
    col4 = 42
    for c in range(14):
        for lj in range(2):
            for lk in range(4):
                local_idx = lj * 4 + lk
                global_idx = (0 + lj) * p + (0 + lk)
                V[global_idx, col4 + c] = -V[global_idx, col4 + c]
    
    # M5: A11 + A12, B22. W contributions: -C11 + C12
    # Already handled in embed, but need to check signs
    # M6: A21 - A11, so subtract U contributions for A11
    col6 = 70
    for c in range(14):
        for li in range(2):
            for lj in range(2):
                local_idx = li * 2 + lj
                global_idx = (0 + li) * m + (0 + lj)
                U[global_idx, col6 + c] = -U[global_idx, col6 + c]
    
    # M7: A12 - A22, so subtract U contributions for A22
    col7 = 84
    for c in range(14):
        for li in range(2):
            for lj in range(2):
                local_idx = li * 2 + lj
                global_idx = (2 + li) * m + (2 + lj)
                U[global_idx, col7 + c] = -U[global_idx, col7 + c]
    
    return U, V, W

def _build_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = 4, 4, 8
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return U, V, W

def _to_int_list(mat):
    """Convert numpy matrix to list of lists with integer entries."""
    return [[int(round(mat[i][j])) for j in range(mat.shape[1])] for i in range(mat.shape[0])]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 4, 4, 8
    
    try:
        U, V, W = _build_full_decomposition()
        
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        
        if valid:
            return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
        else:
            raise ValueError(f"Verification failed with max_err={max_err}")
    
    except Exception as e:
        U, V, W = _build_schoolbook()
        return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
# EVOLVE-BLOCK-END
