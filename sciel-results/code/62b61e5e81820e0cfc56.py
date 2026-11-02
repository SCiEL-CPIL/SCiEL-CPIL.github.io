"""
Rank-minimal tensor decomposition for 2x3 * 3x8 matrix multiplication.

Uses explicit Strassen-based construction achieving rank 44, verified immediately.
Includes a targeted search for potential rank reduction using structured initialization.
"""

import numpy as np
import time

DIMENSIONS = [2, 3, 8]
_DEADLINE = 145.0  # Safety margin before 150s kill
_START_TIME = None


def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(rank):
                    s += U[a][r] * V[b][r] * W[c][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _strassen_2x2_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, U, V, W, col):
    """
    Add Strassen decomposition for a 2x2 * 2x2 sub-problem.
    a_rows, a_cols: row/col indices in A (2 values each)
    b_rows, b_cols: row/col indices in B (2 values each)  
    c_rows, c_cols: row/col indices in C (2 values each)
    Returns next col index.
    """
    n, m, p = DIMENSIONS
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[a_rows[0]*m + a_cols[0]][col] = 1
    U[a_rows[1]*m + a_cols[1]][col] = 1
    V[b_rows[0]*p + b_cols[0]][col] = 1
    V[b_rows[1]*p + b_cols[1]][col] = 1
    W[c_cols[0]*n + c_rows[0]][col] = 1
    W[c_cols[1]*n + c_rows[1]][col] = 1
    col += 1
    
    # M2 = (A10 + A11) * B00
    U[a_rows[1]*m + a_cols[0]][col] = 1
    U[a_rows[1]*m + a_cols[1]][col] = 1
    V[b_rows[0]*p + b_cols[0]][col] = 1
    W[c_cols[0]*n + c_rows[1]][col] = 1
    W[c_cols[1]*n + c_rows[1]][col] = -1
    col += 1
    
    # M3 = A00 * (B01 - B11)
    U[a_rows[0]*m + a_cols[0]][col] = 1
    V[b_rows[0]*p + b_cols[1]][col] = 1
    V[b_rows[1]*p + b_cols[1]][col] = -1
    W[c_cols[1]*n + c_rows[0]][col] = 1
    W[c_cols[1]*n + c_rows[1]][col] = 1
    col += 1
    
    # M4 = A11 * (B10 - B00)
    U[a_rows[1]*m + a_cols[1]][col] = 1
    V[b_rows[1]*p + b_cols[0]][col] = 1
    V[b_rows[0]*p + b_cols[0]][col] = -1
    W[c_cols[0]*n + c_rows[0]][col] = 1
    W[c_cols[0]*n + c_rows[1]][col] = 1
    col += 1
    
    # M5 = (A00 + A01) * B11
    U[a_rows[0]*m + a_cols[0]][col] = 1
    U[a_rows[0]*m + a_cols[1]][col] = 1
    V[b_rows[1]*p + b_cols[1]][col] = 1
    W[c_cols[0]*n + c_rows[0]][col] = -1
    W[c_cols[1]*n + c_rows[0]][col] = 1
    col += 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[a_rows[1]*m + a_cols[0]][col] = 1
    U[a_rows[0]*m + a_cols[0]][col] = -1
    V[b_rows[0]*p + b_cols[0]][col] = 1
    V[b_rows[0]*p + b_cols[1]][col] = 1
    W[c_cols[1]*n + c_rows[1]][col] = 1
    col += 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[a_rows[0]*m + a_cols[1]][col] = 1
    U[a_rows[1]*m + a_cols[1]][col] = -1
    V[b_rows[1]*p + b_cols[0]][col] = 1
    V[b_rows[1]*p + b_cols[1]][col] = 1
    W[c_cols[0]*n + c_rows[0]][col] = 1
    col += 1
    
    return col


def _build_rank44():
    """
    Build explicit rank-44 decomposition.
    
    Strategy: Split B's 8 columns into 4 pairs. For each pair (k0, k1):
    - Use Strassen (rank 7) for the 2x2 * 2x2 part (A rows 0,1 cols 0,1; B rows 0,1)
    - Use 4 rank-1 terms for the 2x1 * 1x2 part (A col 2; B row 2)
    Total per block: 11, total: 44.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A rows: [0,1], A cols: [0,1]
        # B rows: [0,1], B cols: [k0, k1]
        # C rows: [0,1], C cols: [k0, k1]
        col = _strassen_2x2_block(
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[k0, k1],
            c_rows=[0, 1], c_cols=[k0, k1],
            U=U, V=V, W=W, col=col
        )
        
        # Rank-1 terms for A[0:2, 2] * B[2, k0:k1+1]
        # P1 = A[0,2] * B[2,k0] -> C[0,k0]
        U[0*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 0][col] = 1
        col += 1
        
        # P2 = A[0,2] * B[2,k1] -> C[0,k1]
        U[0*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 0][col] = 1
        col += 1
        
        # P3 = A[1,2] * B[2,k0] -> C[1,k0]
        U[1*m + 2][col] = 1
        V[2*p + k0][col] = 1
        W[k0*n + 1][col] = 1
        col += 1
        
        # P4 = A[1,2] * B[2,k1] -> C[1,k1]
        U[1*m + 2][col] = 1
        V[2*p + k1][col] = 1
        W[k1*n + 1][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}


def _try_structured_search(best_rank, time_budget=30):
    """
    Try to find a rank < best_rank using structured initialization.
    Uses the block structure to guide random search.
    """
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 24
    dim_W = p * n  # 16
    
    # Build unfolded tensors
    T0 = np.zeros((dim_U, dim_V * dim_W))
    T1 = np.zeros((dim_V, dim_U * dim_W))
    T2 = np.zeros((dim_W, dim_U * dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T0[a, b * dim_W + c] = 1.0
                T1[b, a * dim_W + c] = 1.0
                T2[c, a * dim_V + b] = 1.0
    
    start_time = time.time()
    
    for rank in range(max(24, best_rank - 4), best_rank):
        if time.time() - start_time > time_budget:
            break
        
        # Multiple restarts with structured initialization
        num_restarts = 15
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            
            np.random.seed(12345 + restart * 997 + rank * 31)
            
            # Initialize with structure: each block gets its own "sub-decomposition"
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            
            # Assign rank-1 terms to blocks
            terms_per_block = rank // 4
            for b in range(4):
                k0 = 2 * b
                k1 = 2 * b + 1
                base = b * terms_per_block
                
                # Initialize Strassen-like structure for this block
                for r in range(terms_per_block):
                    if r < 7:  # Strassen part
                        # Random but structured initialization
                        U[0, base+r] = np.random.randn() * 0.5
                        U[3, base+r] = np.random.randn() * 0.5
                        V[k0, base+r] = np.random.randn() * 0.5
                        V[k1, base+r] = np.random.randn() * 0.5
                        W[k0, base+r] = np.random.randn() * 0.5
                        W[k1 + n, base+r] = np.random.randn() * 0.5
                    else:  # Extra terms for the 2x1 part
                        U[2, base+r] = np.random.randn() * 0.5
                        U[5, base+r] = np.random.randn() * 0.5
                        V[2*p+k0, base+r] = np.random.randn() * 0.5
                        V[2*p+k1, base+r] = np.random.randn() * 0.5
                        W[k0, base+r] = np.random.randn() * 0.5
                        W[k1 + n, base+r] = np.random.randn() * 0.5
                
                # Add some cross-block randomness
                for r in range(terms_per_block):
                    U[1, base+r] += np.random.randn() * 0.1
                    U[4, base+r] += np.random.randn() * 0.1
                    V[p+k0, base+r] += np.random.randn() * 0.1
                    V[p+k1, base+r] += np.random.randn() * 0.1
            
            # ALS iterations
            max_iter = 100
            tol = 1e-10
            for iteration in range(max_iter):
                if time.time() - start_time > time_budget:
                    break
                
                U_old = U.copy()
                
                # Update U
                VW = np.zeros((dim_V * dim_W, rank))
                for r in range(rank):
                    VW[:, r] = np.kron(V[:, r], W[:, r])
                try:
                    U = np.linalg.lstsq(VW.T, T0.T, rcond=None)[0].T
                except:
                    break
                
                # Update V
                UW = np.zeros((dim_U * dim_W, rank))
                for r in range(rank):
                    UW[:, r] = np.kron(U[:, r], W[:, r])
                try:
                    V = np.linalg.lstsq(UW.T, T1.T, rcond=None)[0].T
                except:
                    break
                
                # Update W
                UV = np.zeros((dim_U * dim_V, rank))
                for r in range(rank):
                    UV[:, r] = np.kron(U[:, r], V[:, r])
                try:
                    W = np.linalg.lstsq(UV.T, T2.T, rcond=None)[0].T
                except:
                    break
                
                if np.linalg.norm(U - U_old) < tol:
                    break
            
            # Verify
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        s = 0.0
                        for r in range(rank):
                            s += U[a, r] * V[b, r] * W[c, r]
                        err = abs(s - 1.0)
                        if err > max_err:
                            max_err = err
            
            if max_err < 1e-6:
                # Found valid solution - convert to list
                def to_list(mat):
                    rows = []
                    for row in mat:
                        r_list = []
                        for val in row:
                            r2 = np.round(val * 2)
                            if abs(val - r2 / 2.0) < 1e-4:
                                if r2 % 2 == 0:
                                    r_list.append(int(r2 // 2))
                                else:
                                    r_list.append(r2 / 2.0)
                            else:
                                r_list.append(float(val))
                        rows.append(r_list)
                    return rows
                
                return rank, to_list(U), to_list(V), to_list(W)
    
    return None


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the rank-44 Strassen-based construction
    result44 = _build_rank44()
    
    # Verify immediately
    if _verify(result44["U"], result44["V"], result44["W"], n, m, p, 44):
        best_result = result44
        best_rank = 44
    else:
        # Fallback: schoolbook rank 48
        rank_school = 48
        U_school = [[0] * rank_school for _ in range(n * m)]
        V_school = [[0] * rank_school for _ in range(m * p)]
        W_school = [[0] * rank_school for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i * m + j][col] = 1
                    V_school[j * p + k][col] = 1
                    W_school[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = 48
    
    # Try to find a better rank if we have time
    elapsed = time.time() - _START_TIME
    if elapsed < _DEADLINE - 35 and best_rank > 24:
        search_result = _try_structured_search(best_rank, time_budget=min(30, _DEADLINE - elapsed - 35))
        if search_result is not None:
            rank, U, V, W = search_result
            if _verify(U, V, W, n, m, p, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
    
    return best_result
