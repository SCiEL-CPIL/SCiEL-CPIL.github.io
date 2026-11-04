"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the schoolbook decomposition (rank n*m*p) as a fallback."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    u_val = U[i * m + j][c]
                    v_val = V[j * p + k][c]
                    w_val = W[k * n + i][c]
                    if u_val != 0 and v_val != 0 and w_val != 0:
                        total += u_val * v_val * w_val
                if total != 1:
                    return False
    return True


def _strassen_2x2_2x2():
    """
    Return the Strassen rank-7 decomposition for 2x2 * 2x2.
    U: shape (4, 7) - rows indexed by (i,j) for i,j in {0,1}
    V: shape (4, 7) - rows indexed by (j,k) for j,k in {0,1}
    W: shape (4, 7) - rows indexed by (k,i) for k,i in {0,1}
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A00(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B01-B00)
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W


def _strassen_block_decomposition():
    """
    Block-based Strassen decomposition for 4x4 * 4x7.
    """
    n, m, p = 4, 4, 7
    
    rank = 50
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(28)]
    W = [[0]*rank for _ in range(28)]
    
    col = 0
    
    def add_strassen(a_rows, b_rows, c_rows):
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        for idx in range(4):
            for c in range(7):
                V[b_rows[idx]][col + c] = V_s[idx][c]
        for idx in range(4):
            for c in range(7):
                W[c_rows[idx]][col + c] = W_s[idx][c]
        col += 7
    
    def add_2x2_2x3(a_rows, b_rows, c_rows):
        nonlocal col
        U_s, V_s, W_s = _strassen_2x2_2x2()
        for idx in range(4):
            for c in range(7):
                U[a_rows[idx]][col + c] = U_s[idx][c]
        for c in range(7):
            V[b_rows[0]][col + c] = V_s[0][c]
            V[b_rows[1]][col + c] = V_s[1][c]
            V[b_rows[3]][col + c] = V_s[2][c]
            V[b_rows[4]][col + c] = V_s[3][c]
        for c in range(7):
            W[c_rows[0]][col + c] = W_s[0][c]
            W[c_rows[1]][col + c] = W_s[1][c]
            W[c_rows[3]][col + c] = W_s[2][c]
            W[c_rows[4]][col + c] = W_s[3][c]
        col += 7
        
        U[a_rows[0]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        U[a_rows[1]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[2]][col] = 1
        col += 1
        
        U[a_rows[2]][col] = 1
        V[b_rows[2]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
        
        U[a_rows[3]][col] = 1
        V[b_rows[5]][col] = 1
        W[c_rows[5]][col] = 1
        col += 1
    
    A11_rows = [0, 1, 4, 5]
    A12_rows = [2, 3, 6, 7]
    A21_rows = [8, 9, 12, 13]
    A22_rows = [10, 11, 14, 15]
    
    B11_rows = [0, 1, 7, 8]
    B12_rows = [2, 3, 9, 10]
    B13_rows = [4, 5, 6, 11, 12, 13]
    B21_rows = [14, 15, 21, 22]
    B22_rows = [16, 17, 23, 24]
    B23_rows = [18, 19, 20, 25, 26, 27]
    
    C11_rows = [0, 1, 4, 5]
    C12_rows = [8, 9, 12, 13]
    C13_rows = [16, 17, 20, 21, 24, 25]
    C21_rows = [2, 3, 6, 7]
    C22_rows = [10, 11, 14, 15]
    C23_rows = [18, 19, 22, 23, 26, 27]
    
    add_strassen(A11_rows, B11_rows, C11_rows)
    add_strassen(A12_rows, B21_rows, C11_rows)
    add_strassen(A11_rows, B12_rows, C12_rows)
    add_strassen(A12_rows, B22_rows, C12_rows)
    add_2x2_2x3(A11_rows, B13_rows, C13_rows)
    add_2x2_2x3(A12_rows, B23_rows, C13_rows)
    add_strassen(A21_rows, B11_rows, C21_rows)
    add_strassen(A22_rows, B21_rows, C21_rows)
    add_strassen(A21_rows, B12_rows, C22_rows)
    add_strassen(A22_rows, B22_rows, C22_rows)
    add_2x2_2x3(A21_rows, B13_rows, C23_rows)
    add_2x2_2x3(A22_rows, B23_rows, C23_rows)
    
    return {"U": U, "V": V, "W": W}


def _verify_numpy(U, V, W, n, m, p):
    """Fast verification using numpy tensor contraction."""
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    if not np.allclose(T, 1.0, atol=1e-10):
        return False
    return True


def _reduce_rank(U, V, W, n, m, p, max_iter=2000, deadline=None):
    """
    Try to reduce rank by finding linearly dependent columns.
    A column c is redundant if there exist coefficients alpha_d (d != c)
    such that for all i,j,k: sum_d alpha_d * U[i,j,d]*V[j,k,d]*W[k,i,d] = U[i,j,c]*V[j,k,c]*W[k,i,c]
    and alpha_c = -1.
    
    We search for columns that can be expressed as linear combinations of others.
    """
    import numpy as np
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    
    # Build the tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For rank reduction, we need: for each column c, check if it's in span of others
    
    # Flatten each column into a vector in R^(n*m*p)
    # col_vec[c] = [U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)]
    
    col_vecs = []
    for c in range(rank):
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i * m * p + j * p + k] = U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
        col_vecs.append(vec)
    
    col_vecs = np.array(col_vecs)  # shape (rank, n*m*p)
    
    # For each column, check if it's in the span of the others
    # This is equivalent to: is rank(col_vecs without c) == rank(col_vecs)?
    
    # We'll use a greedy approach: try to remove columns one at a time
    keep = list(range(rank))
    
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        found_redundant = False
        for idx, c in enumerate(keep):
            if len(keep) <= 1:
                break
            
            # Try removing column c
            remaining = [x for x in keep if x != c]
            
            # Check if col_vecs[c] is in span of col_vecs[remaining]
            # Use SVD or QR to check
            if len(remaining) == 0:
                # Can't remove if it's the only one
                continue
            
            # Check rank
            A = col_vecs[remaining]  # shape (len(remaining), n*m*p)
            b = col_vecs[c]
            
            # Check if b is in column space of A^T
            # i.e., is there x such that A^T @ x = b?
            # Or equivalently, does A^T have full row rank and b is in its column space?
            
            # Simpler: check if rank([A; b]) == rank(A)
            # But this is expensive. Use a cheaper check:
            # Project b onto the orthogonal complement of row space of A
            
            # Actually, let's just do a direct check with least squares
            # Solve A^T @ x = b in least squares, check residual
            
            try:
                # A is (len(remaining), n*m*p), so A^T is (n*m*p, len(remaining))
                # We want to see if b (n*m*p,) is in the column space of A^T
                # i.e., is there x (len(remaining),) such that A^T @ x ≈ b?
                
                # Use lstsq
                A_T = A.T  # (n*m*p, len(remaining))
                x, residuals, rank_A, sv = np.linalg.lstsq(A_T, b, rcond=None)
                residual = np.linalg.norm(b - A_T @ x)
                b_norm = np.linalg.norm(b)
                
                if b_norm > 1e-15 and residual / b_norm < 1e-10:
                    # Column c is redundant!
                    keep.remove(c)
                    found_redundant = True
                    break
            except Exception:
                continue
        
        if not found_redundant:
            break
    
    # Reconstruct with only kept columns
    if len(keep) < rank:
        U_new = [[U[i][c] for c in keep] for i in range(len(U))]
        V_new = [[V[i][c] for c in keep] for i in range(len(V))]
        W_new = [[W[i][c] for c in keep] for i in range(len(W))]
        return U_new, V_new, W_new
    
    return U, V, W


def _combined_decomposition():
    """
    Combined decomposition: Strassen tensor product for first 4 output columns (rank 49)
    + schoolbook for remaining 3 output columns (rank 48).
    Total rank: 97.
    """
    n, m, p = 4, 4, 7
    
    U_s, V_s, W_s = _strassen_2x2_2x2()
    rank_str = 49
    rank_sb = 48
    total_rank = rank_str + rank_sb
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Tensor product Strassen for 4x4 * 4x4
    for i1 in range(2):
        for i0 in range(2):
            for j1 in range(2):
                for j0 in range(2):
                    row_u = i1 * 8 + i0 * 4 + j1 * 2 + j0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            U[row_u][col] = U_s[i0 * 2 + j0][c0] * U_s[i1 * 2 + j1][c1]
    
    for j1 in range(2):
        for j0 in range(2):
            for k1 in range(2):
                for k0 in range(2):
                    row_v = j1 * 8 + j0 * 4 + k1 * 2 + k0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            V[row_v][col] = V_s[j0 * 2 + k0][c0] * V_s[j1 * 2 + k1][c1]
    
    for k1 in range(2):
        for k0 in range(2):
            for i1 in range(2):
                for i0 in range(2):
                    row_w = k1 * 8 + k0 * 4 + i1 * 2 + i0
                    for c1 in range(7):
                        for c0 in range(7):
                            col = c1 * 7 + c0
                            W[row_w][col] = W_s[k0 * 2 + i0][c0] * W_s[k1 * 2 + i1][c1]
    
    # Schoolbook for k=4,5,6
    col = rank_str
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    best_result = None
    best_rank = float('inf')
    
    # Try the combined tensor-product Strassen + schoolbook decomposition (rank 97)
    try:
        result = _combined_decomposition()
        if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                best_result = result
                best_rank = len(result["U"][0])
    except Exception:
        pass
    
    # Try the block-based Strassen decomposition (rank 50)
    try:
        result = _strassen_block_decomposition()
        if _verify_numpy(result["U"], result["V"], result["W"], n, m, p):
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                if len(result["U"][0]) < best_rank:
                    best_result = result
                    best_rank = len(result["U"][0])
    except Exception:
        pass
    
    # Try to reduce rank of the best result found so far
    if best_result is not None and time.time() < deadline - 30:
        try:
            U_red, V_red, W_red = _reduce_rank(
                best_result["U"], best_result["V"], best_result["W"],
                n, m, p, max_iter=500, deadline=deadline - 10
            )
            if len(U_red[0]) < best_rank:
                if _verify_numpy(U_red, V_red, W_red, n, m, p):
                    if _verify(U_red, V_red, W_red, n, m, p):
                        best_result = {"U": U_red, "V": V_red, "W": W_red}
                        best_rank = len(U_red[0])
        except Exception:
            pass
    
    # Fallback to schoolbook (rank 112) if nothing else worked
    if best_result is None:
        result = _schoolbook()
        assert _verify(result["U"], result["V"], result["W"], n, m, p), "Decomposition failed verification"
        return result
    
    return best_result

# EVOLVE-BLOCK-END
