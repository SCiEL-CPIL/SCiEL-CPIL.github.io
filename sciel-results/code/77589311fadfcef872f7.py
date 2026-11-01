"""Matrix multiplication 2x5x5 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p, num_tests=20):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rng = np.random.default_rng(42)
    for _ in range(num_tests):
        A = rng.standard_normal((n, m))
        B = rng.standard_normal((m, p))
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U_np[:, r].reshape((n, m))
            v_r = V_np[:, r].reshape((m, p))
            w_r = W_np[:, r].reshape((n, p)).T  # W indexed by k*n+i, so reshape to (n,p) gives [i,k]
            
            a_proj = np.sum(A * u_r)
            b_proj = np.sum(B * v_r)
            C_computed += a_proj * b_proj * w_r
        
        if not np.allclose(C_computed, C_expected, atol=1e-8):
            return False
    return True

def _build_strassen_2x2x2(U, V, W, col_offset, a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start, n, m, p):
    """
    Build Strassen 2x2x2 block.
    A_sub[i,j] = A[a_row_start+i, a_col_start+j] for i,j in 0,1
    B_sub[j,k] = B[b_row_start+j, b_col_start+k] for j,k in 0,1
    C_sub[i,k] = C[c_row_start+i, c_col_start+k] for i,k in 0,1
    
    Strassen:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B10-B11)
    M4 = A11*(B01-B10)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B10)
    M7 = (A01-A11)*(B01+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    # A_sub indices: (i, j) -> A[a_row_start+i, a_col_start+j]
    # U index: i*m + j where i is local row, j is local col
    # But we need global indices: global_row = a_row_start + i, global_col = a_col_start + j
    # U[global_row * m + global_col, col]
    
    # B_sub indices: (j, k) -> B[b_row_start+j, b_col_start+k]
    # V index: j*p + k where j is local row, k is local col
    # Global: global_row = b_row_start + j, global_col = b_col_start + k
    # V[global_row * p + global_col, col]
    
    # C_sub indices: (i, k) -> C[c_row_start+i, c_col_start+k]
    # W index: k*n + i (trace convention)
    # Global: global_k = c_col_start + k, global_i = c_row_start + i
    # W[global_k * n + global_i, col]
    
    # Define the 7 Strassen multiplications
    # Each: U_combo, V_combo, W_contributions
    # U_combo: dict from (global_row, global_col) to coefficient
    # V_combo: dict from (global_row, global_col) to coefficient
    # W_contributions: dict from (global_i, global_k) to coefficient
    
    strassen_terms = []
    
    # M1 = (A00+A11)*(B00+B11)
    u1 = {(0,0): 1.0, (1,1): 1.0}
    v1 = {(0,0): 1.0, (1,1): 1.0}
    w1 = {(0,0): 1.0, (1,1): 1.0}
    strassen_terms.append((u1, v1, w1))
    
    # M2 = (A10+A11)*B00
    u2 = {(1,0): 1.0, (1,1): 1.0}
    v2 = {(0,0): 1.0}
    w2 = {(1,0): 1.0}
    strassen_terms.append((u2, v2, w2))
    
    # M3 = A00*(B10-B11)
    u3 = {(0,0): 1.0}
    v3 = {(1,0): 1.0, (1,1): -1.0}
    w3 = {(0,1): 1.0}
    strassen_terms.append((u3, v3, w3))
    
    # M4 = A11*(B01-B10)
    u4 = {(1,1): 1.0}
    v4 = {(0,1): 1.0, (1,0): -1.0}
    w4 = {(0,0): 1.0, (1,0): 1.0}
    strassen_terms.append((u4, v4, w4))
    
    # M5 = (A00+A01)*B11
    u5 = {(0,0): 1.0, (0,1): 1.0}
    v5 = {(1,1): 1.0}
    w5 = {(0,1): 1.0, (0,0): -1.0}
    strassen_terms.append((u5, v5, w5))
    
    # M6 = (A10-A00)*(B00+B10)
    u6 = {(1,0): 1.0, (0,0): -1.0}
    v6 = {(0,0): 1.0, (1,0): 1.0}
    w6 = {(1,1): 1.0}
    strassen_terms.append((u6, v6, w6))
    
    # M7 = (A01-A11)*(B01+B11)
    u7 = {(0,1): 1.0, (1,1): -1.0}
    v7 = {(0,1): 1.0, (1,1): 1.0}
    w7 = {(0,0): 1.0}
    strassen_terms.append((u7, v7, w7))
    
    for u_combo, v_combo, w_combo in strassen_terms:
        col = col_offset
        # Fill U
        for (li, lj), coeff in u_combo.items():
            gr = a_row_start + li
            gc = a_col_start + lj
            U[gr * m + gc][col] += coeff
        # Fill V
        for (lj, lk), coeff in v_combo.items():
            gr = b_row_start + lj
            gc = b_col_start + lk
            V[gr * p + gc][col] += coeff
        # Fill W (trace convention: W[k*n + i])
        for (li, lk), coeff in w_combo.items():
            gi = c_row_start + li
            gk = c_col_start + lk
            W[gk * n + gi][col] += coeff
        col_offset += 1
    
    return col_offset

def _build_schoo1_2x2x1(U, V, W, col_offset, a_row_start, a_cols, b_row_start, b_col, c_row_start, c_col, n, m, p):
    """
    Schoolbook for A_sub (2x2) * B_sub (2x1) = C_sub (2x1)
    C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    Two rank-1 terms.
    """
    for j in range(2):
        col = col_offset
        # U: A[a_row_start+i, a_cols[j]]
        for i in range(2):
            gr = a_row_start + i
            gc = a_cols[j]
            U[gr * m + gc][col] += 1.0
        # V: B[b_row_start+j, b_col]
        gr = b_row_start + j
        gc = b_col
        V[gr * p + gc][col] += 1.0
        # W: C[c_row_start+i, c_col]
        for i in range(2):
            gi = c_row_start + i
            gk = c_col
            W[gk * n + gi][col] += 1.0
        col_offset += 1
    return col_offset

def _build_schoo1_2x1x2(U, V, W, col_offset, a_row_start, a_col, b_rows, b_col_start, c_row_start, c_cols, n, m, p):
    """
    Schoolbook for A_sub (2x1) * B_sub (1x2) = C_sub (2x2)
    C[i,k] = A[i,0]*B[0,k]
    Two rank-1 terms.
    """
    for k in range(2):
        col = col_offset
        # U: A[a_row_start+i, a_col]
        for i in range(2):
            gr = a_row_start + i
            gc = a_col
            U[gr * m + gc][col] += 1.0
        # V: B[b_rows[0], b_col_start+k]
        gr = b_rows[0]
        gc = b_col_start + k
        V[gr * p + gc][col] += 1.0
        # W: C[c_row_start+i, c_cols[k]]
        for i in range(2):
            gi = c_row_start + i
            gk = c_cols[k]
            W[gk * n + gi][col] += 1.0
        col_offset += 1
    return col_offset

def _build_schoo1_2x1x1(U, V, W, col_offset, a_row_start, a_col, b_row, b_col, c_row_start, c_col, n, m, p):
    """
    Schoolbook for A_sub (2x1) * B_sub (1x1) = C_sub (2x1)
    C[i,0] = A[i,0]*B[0,0]
    One rank-1 term.
    """
    col = col_offset
    # U: A[a_row_start+i, a_col]
    for i in range(2):
        gr = a_row_start + i
        gc = a_col
        U[gr * m + gc][col] += 1.0
    # V: B[b_row, b_col]
    gr = b_row
    gc = b_col
    V[gr * p + gc][col] += 1.0
    # W: C[c_row_start+i, c_col]
    for i in range(2):
        gi = c_row_start + i
        gk = c_col
        W[gk * n + gi][col] += 1.0
    return col_offset + 1

def construct():
    """
    Find a low-rank exact decomposition for 2x5x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    # Total elements
    n_m = n * m  # 10
    m_p = m * p  # 25
    p_n = p * n  # 10
    
    # Initialize with zeros
    U = [[0.0] * 1 for _ in range(n_m)]  # We'll determine rank first
    V = [[0.0] * 1 for _ in range(m_p)]
    W = [[0.0] * 1 for _ in range(p_n)]
    
    # Determine total rank needed
    # 4 Strassen 2x2x2 blocks: 4 * 7 = 28
    # 2 schoolbook 2x2x1 blocks: 2 * 2 = 4
    # 2 schoolbook 2x1x2 blocks: 2 * 2 = 4
    # 1 schoolbook 2x1x1 block: 1 * 1 = 1
    # Total: 37
    total_rank = 37
    
    U = [[0.0] * total_rank for _ in range(n_m)]
    V = [[0.0] * total_rank for _ in range(m_p)]
    W = [[0.0] * total_rank for _ in range(p_n)]
    
    col_offset = 0
    
    # Split j-dimension: {0,1}, {2,3}, {4}
    # Split k-dimension: {0,1}, {2,3}, {4}
    
    # Four 2x2x2 Strassen blocks:
    # (j in {0,1}, k in {0,1}), (j in {2,3}, k in {0,1}), (j in {0,1}, k in {2,3}), (j in {2,3}, k in {2,3})
    
    j_groups = [(0, 1), (2, 3)]
    k_groups = [(0, 1), (2, 3)]
    
    for j_start in j_groups:
        for k_start in k_groups:
            col_offset = _build_strassen_2x2x2(
                U, V, W, col_offset,
                a_row_start=0, a_col_start=j_start[0],
                b_row_start=j_start[0], b_col_start=k_start[0],
                c_row_start=0, c_col_start=k_start[0],
                n=n, m=m, p=p
            )
    
    # Schoolbook 2x2x1 blocks:
    # (j in {0,1}, k=4), (j in {2,3}, k=4)
    for j_start in j_groups:
        col_offset = _build_schoo1_2x2x1(
            U, V, W, col_offset,
            a_row_start=0, a_cols=[j_start[0], j_start[1]],
            b_row_start=j_start[0], b_col=4,
            c_row_start=0, c_col=4,
            n=n, m=m, p=p
        )
    
    # Schoolbook 2x1x2 blocks:
    # (j=4, k in {0,1}), (j=4, k in {2,3})
    for k_start in k_groups:
        col_offset = _build_schoo1_2x1x2(
            U, V, W, col_offset,
            a_row_start=0, a_col=4,
            b_rows=[4], b_col_start=k_start[0],
            c_row_start=0, c_cols=[k_start[0], k_start[1]],
            n=n, m=m, p=p
        )
    
    # Schoolbook 2x1x1 block:
    # (j=4, k=4)
    col_offset = _build_schoo1_2x1x1(
        U, V, W, col_offset,
        a_row_start=0, a_col=4,
        b_row=4, b_col=4,
        c_row_start=0, c_col=4,
        n=n, m=m, p=p
    )
    
    assert col_offset == total_rank, f"Expected rank {total_rank}, got {col_offset}"
    
    # Verify
    if _verify_decomposition(U, V, W, n, m, p, num_tests=10):
        result = {"U": U, "V": V, "W": W}
        return result
    
    # Fallback: schoolbook rank 50
    rank50 = n * m * p
    U50 = [[0.0] * rank50 for _ in range(n_m)]
    V50 = [[0.0] * rank50 for _ in range(m_p)]
    W50 = [[0.0] * rank50 for _ in range(p_n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U50[i * m + j][col] = 1.0
                V50[j * p + k][col] = 1.0
                W50[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U50, "V": V50, "W": W50}
# EVOLVE-BLOCK-END

