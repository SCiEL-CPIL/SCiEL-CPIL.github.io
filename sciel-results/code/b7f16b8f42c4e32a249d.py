# EVOLVE-BLOCK-START
import time
import numpy as np
import random

_START_TIME = time.time()
_TIME_LIMIT = 140.0

def _build_tensor(n, m, p):
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    T = _build_tensor(n, m, p)
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    T_recon = np.zeros_like(T, dtype=float)
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        T_recon += np.einsum('a,b,c->abc', u, v, w)
    if not np.allclose(T_recon, T, atol=tol):
        return False
    return True

def _search_low_rank(n, m, p, target_rank, time_budget=10.0):
    """
    Attempt to find a decomposition of the given rank using random initialization and local search.
    This is a heuristic approach.
    """
    nm, mp, pn = n * m, m * p, p * n
    T = _build_tensor(n, m, p)
    
    start = time.time()
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Initialize with random sparse matrices
    # Each row of U, V, W should be sparse.
    for _ in range(50): # 50 restarts
        if time.time() - start > time_budget:
            break
            
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        # Randomly assign non-zeros
        # We want each term to contribute to the tensor.
        # For each r, pick random a, b, c and set U[a,r]=1, V[b,r]=1, W[c,r]=1?
        # This is too sparse. 
        # Better: initialize with the rank-32 solution and merge?
        # Or use a structured initialization.
        
        # Let's try a structured initialization for rank 20.
        # We know the tensor has 40 non-zeros.
        # We can try to cover them with 20 terms.
        
        # Simple random init
        for r in range(target_rank):
            # Pick 2 indices for U, 2 for V, 2 for W?
            # This creates a rank-1 tensor with 8 non-zeros.
            # 20 * 8 = 160 non-zeros capacity, we need 40.
            u_idx = np.random.choice(nm, 2, replace=False)
            v_idx = np.random.choice(mp, 2, replace=False)
            w_idx = np.random.choice(pn, 2, replace=False)
            
            U[u_idx, r] = 1
            V[v_idx, r] = 1
            W[w_idx, r] = 1
            
        # Now we have a fixed sparse structure for U, V.
        # We can solve for W linearly?
        # T_recon[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # For each (a,b,c) where T[a,b,c]=1:
        # 1 = sum_{r: U[a,r]!=0, V[b,r]!=0} U[a,r] V[b,r] W[c,r]
        
        # This is a linear system for W.
        # Let's construct the system.
        # Variables: W[c, r] for all c, r.
        # Equations: for each (a,b,c) with T[a,b,c]=1.
        
        # However, the system might be underdetermined or inconsistent.
        # We can use least squares.
        
        # Build matrix A and vector b for the linear system: A x = b
        # x is the flattened W matrix.
        # A has rows for each non-zero in T.
        # A[row, (c, r)] = U[a, r] * V[b, r] if T[a,b,c]=1, else 0.
        # b[row] = 1.
        
        # Note: This assumes U and V are fixed.
        
        num_equations = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if T[a,b,c] == 1:
                        num_equations += 1
        
        A = np.zeros((num_equations, pn * target_rank))
        b_vec = np.ones(num_equations)
        
        row = 0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if T[a,b,c] == 1:
                        for r in range(target_rank):
                            val = U[a, r] * V[b, r]
                            if val != 0:
                                A[row, c * target_rank + r] = val
                        row += 1
        
        # Solve least squares
        try:
            x, residuals, rank, sv = np.linalg.lstsq(A, b_vec, rcond=None)
            W_sol = x.reshape(pn, target_rank)
            
            # Verify
            T_recon = np.zeros((nm, mp, pn), dtype=float)
            for r in range(target_rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W_sol[:, r])
                
            error = np.max(np.abs(T_recon - T))
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W_sol.copy()
                
            if error < 1e-6:
                # Found a solution!
                return {
                    "U": best_U.tolist(),
                    "V": best_V.tolist(),
                    "W": best_W.tolist()
                }
                
        except Exception:
            pass
            
    # If we found a good solution, return it
    if best_U is not None and best_error < 1e-6:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
        
    return None

def _construct_rank32(n, m, p):
    """
    Construct a rank-32 decomposition.
    """
    rank = 32
    nm, mp, pn = n * m, m * p, p * n
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    def a_idx(i, j):
        return i * m + j
        
    def b_idx(i, j):
        return i * p + j
        
    def c_idx(i, j):
        return j * n + i
        
    def add_strassen_2x2_2x2(A_row_offset, A_col_offset, B_row_offset, B_col_offset, C_row_offset, C_col_offset, start_col):
        A11 = a_idx(A_row_offset, A_col_offset)
        A12 = a_idx(A_row_offset, A_col_offset + 1)
        A21 = a_idx(A_row_offset + 1, A_col_offset)
        A22 = a_idx(A_row_offset + 1, A_col_offset + 1)
        
        B11 = b_idx(B_row_offset, B_col_offset)
        B12 = b_idx(B_row_offset, B_col_offset + 1)
        B21 = b_idx(B_row_offset + 1, B_col_offset)
        B22 = b_idx(B_row_offset + 1, B_col_offset + 1)
        
        C11 = c_idx(C_row_offset, C_col_offset)
        C12 = c_idx(C_row_offset, C_col_offset + 1)
        C21 = c_idx(C_row_offset + 1, C_col_offset)
        C22 = c_idx(C_row_offset + 1, C_col_offset + 1)
        
        # M1
        U[A11][start_col] = 1
        U[A22][start_col] = 1
        V[B11][start_col] = 1
        V[B22][start_col] = 1
        W[C11][start_col] = 1
        W[C22][start_col] = 1
        col = start_col + 1
        
        # M2
        U[A21][col] = 1
        U[A22][col] = 1
        V[B11][col] = 1
        W[C21][col] = 1
        col += 1
        
        # M3
        U[A11][col] = 1
        V[B12][col] = 1
        V[B22][col] = -1
        W[C12][col] = 1
        col += 1
        
        # M4
        U[A22][col] = 1
        V[B21][col] = 1
        V[B11][col] = -1
        W[C11][col] = 1
        W[C21][col] = 1
        col += 1
        
        # M5
        U[A11][col] = 1
        U[A12][col] = 1
        V[B22][col] = 1
        W[C11][col] = -1
        W[C12][col] = 1
        col += 1
        
        # M6
        U[A21][col] = 1
        U[A11][col] = -1
        V[B11][col] = 1
        V[B12][col] = 1
        W[C22][col] = 1
        col += 1
        
        # M7
        U[A12][col] = 1
        U[A22][col] = -1
        V[B21][col] = 1
        V[B22][col] = 1
        W[C11][col] = 1
        W[C22][col] = 1
        col += 1
        
        return col
    
    def add_schoolbook_2x2_2x1(A_row_offset, A_col_offset, B_row_offset, B_col, C_row_offset, C_col, start_col):
        for i in range(2):
            for j in range(2):
                U[a_idx(A_row_offset + i, A_col_offset + j)][start_col] = 1
                V[b_idx(B_row_offset + j, B_col)][start_col] = 1
                W[c_idx(C_row_offset + i, C_col)][start_col] = 1
                start_col += 1
        return start_col
    
    col = add_strassen_2x2_2x2(0, 0, 0, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 0, 0, 2, 0, 2, col)
    col = add_schoolbook_2x2_2x1(0, 0, 0, 4, 0, 4, col)
    
    col = add_strassen_2x2_2x2(0, 2, 2, 0, 0, 0, col)
    col = add_strassen_2x2_2x2(0, 2, 2, 2, 0, 2, col)
    col = add_schoolbook_2x2_2x1(0, 2, 2, 4, 0, 4, col)
    
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}, actual_rank

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Try to find a lower rank decomposition (e.g., rank 20)
    # Time budget for search: 10 seconds
    try:
        result_20 = _search_low_rank(n, m, p, 20, time_budget=10.0)
        if result_20:
            if _verify_decomposition(result_20["U"], result_20["V"], result_20["W"], n, m, p):
                return result_20
    except Exception:
        pass
    
    # Fallback to rank 32
    try:
        result, rank = _construct_rank32(n, m, p)
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
