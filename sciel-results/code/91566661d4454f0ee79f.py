"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_decomposition():
    """
    Construct a Strassen-based decomposition for 2x4x8 matrix multiplication.
    
    A (2x4) * B (4x8) = C (2x8)
    
    Partition: A = [A_left A_right] where A_left, A_right are 2x2
               B = [B_top; B_bottom] where B_top, B_bottom are 2x8
    
    C = A_left * B_top + A_right * B_bottom
    
    We partition the 4 columns of A into two 2-column blocks, and the 8 columns 
    of B into four 2-column blocks. For each of the 2 A-blocks and 4 B-blocks,
    we perform a 2x2 by 2x2 matrix multiplication using Strassen's algorithm (7 mults).
    
    Total rank = 2 * 4 * 7 = 56.
    
    Strassen's 7 products for C = A * B where A, B are 2x2:
    M1 = (a00 + a11)(b00 + b11)
    M2 = (a10 + a11)b00
    M3 = a00(b01 - b11)
    M4 = a11(b10 - b00)
    M5 = (a00 + a01)b11
    M6 = (a10 - a00)(b00 + b01)
    M7 = (a01 - a11)(b10 + b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: dict mapping (i, j) -> coeff for A[i,j]
    # B_coeffs: dict mapping (i, j) -> coeff for B[i,j]  
    # C_coeffs: dict mapping (i, j) -> coeff for C[i,j]
    
    strassen_terms = [
        # M1 = (a00 + a11)(b00 + b11)
        (
            {(0,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1, (1,1): 1}    # C coeffs
        ),
        # M2 = (a10 + a11)b00
        (
            {(1,0): 1, (1,1): 1},   # A coeffs
            {(0,0): 1},             # B coeffs
            {(1,0): 1, (1,1): -1}   # C coeffs
        ),
        # M3 = a00(b01 - b11)
        (
            {(0,0): 1},             # A coeffs
            {(0,1): 1, (1,1): -1},  # B coeffs
            {(0,1): 1, (1,1): 1}    # C coeffs
        ),
        # M4 = a11(b10 - b00)
        (
            {(1,1): 1},             # A coeffs
            {(1,0): 1, (0,0): -1},  # B coeffs
            {(0,0): 1, (1,0): 1}    # C coeffs
        ),
        # M5 = (a00 + a01)b11
        (
            {(0,0): 1, (0,1): 1},   # A coeffs
            {(1,1): 1},             # B coeffs
            {(0,0): -1, (0,1): 1}   # C coeffs
        ),
        # M6 = (a10 - a00)(b00 + b01)
        (
            {(1,0): 1, (0,0): -1},  # A coeffs
            {(0,0): 1, (0,1): 1},   # B coeffs
            {(1,1): 1}              # C coeffs
        ),
        # M7 = (a01 - a11)(b10 + b11)
        (
            {(0,1): 1, (1,1): -1},  # A coeffs
            {(1,0): 1, (1,1): 1},   # B coeffs
            {(0,0): 1}              # C coeffs
        ),
    ]
    
    col = 0
    # For each A-block (2 of them) and B-block (4 of them)
    for a_block in range(2):  # a_block = 0 or 1, corresponds to j in {2*a_block, 2*a_block+1}
        for b_block in range(4):  # b_block = 0..3, corresponds to k in {2*b_block, 2*b_block+1}
            for term in strassen_terms:
                A_coeffs, B_coeffs, C_coeffs = term
                
                # Fill U: U[i*m + j, col]
                for (li, lj), coeff in A_coeffs.items():
                    global_i = li
                    global_j = 2 * a_block + lj
                    U[global_i * m + global_j, col] += coeff
                
                # Fill V: V[j*p + k, col]
                for (lj, lk), coeff in B_coeffs.items():
                    global_j = 2 * a_block + lj
                    global_k = 2 * b_block + lk
                    V[global_j * p + global_k, col] += coeff
                
                # Fill W: W[k*n + i, col]
                for (li, lk), coeff in C_coeffs.items():
                    global_i = li
                    global_k = 2 * b_block + lk
                    W[global_k * n + global_i, col] += coeff
                
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition with rank n*m*p = 64."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """Try to reduce the rank by identifying and removing linearly dependent terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    # Precompute the "contribution" of each column to each (i,j,k) entry
    # For column c, contribution to (i,j,k) is U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # We need sum over c of these = 1 for all (i,j,k)
    
    # Fast approach: represent each column's contribution as a vector in R^(n*m*p)
    # and check if removing a column can be compensated by others
    # Even faster: use the structure. For fixed (i,j), the contribution across k is:
    #   U[i*m+j,c] * W[k*n+i,c] * V[j*p+k,c]
    # This is a rank-1 update pattern.
    
    # Most efficient: build the full tensor contribution matrix
    # M[c, (i,j,k)] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need M @ ones = ones_vector (all ones)
    # Removing column c is valid if the remaining columns still sum to 1 everywhere.
    
    # Build M efficiently
    nm = n * m
    mp = m * p
    pn = p * n
    total_entries = n * m * p  # 64
    
    # M has shape (rank, total_entries)
    # M[c, idx(i,j,k)] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    M = np.zeros((rank, total_entries), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[:, idx] = U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :]
                idx += 1
    
    # Now M @ ones(rank) should equal ones(total_entries)
    # Removing column c: check if M without col c still sums to 1
    # i.e., M[:, active] @ ones(len(active)) = ones(total_entries)
    
    active_cols = list(range(rank))
    current_rank = rank
    
    # Precompute column contributions
    # col_contrib[c] = M[c, :] (shape total_entries)
    # sum_contrib = sum of col_contrib over active cols
    sum_contrib = np.sum(M[:, active_cols], axis=0)
    target = np.ones(total_entries)
    
    for iteration in range(max_iter):
        if time.time() > deadline_global:
            break
        removed_any = False
        for idx_remove in range(len(active_cols) - 1, -1, -1):
            col = active_cols[idx_remove]
            new_sum = sum_contrib - M[col, :]
            if np.allclose(new_sum, target, atol=tol):
                active_cols.pop(idx_remove)
                sum_contrib = new_sum
                current_rank = len(active_cols)
                removed_any = True
                break
        
        if not removed_any:
            break
    
    if current_rank < rank:
        U_reduced = U_arr[:, active_cols].tolist()
        V_reduced = V_arr[:, active_cols].tolist()
        W_reduced = W_arr[:, active_cols].tolist()
        return U_reduced, V_reduced, W_reduced, current_rank
    
    return U, V, W, rank

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        result = _strassen_decomposition()
        U, V, W = result["U"], result["V"], result["W"]
        if _verify(U, V, W, n, m, p, tol=1e-9):
            # Try to reduce rank using fast linear-algebra-based method
            U_red, V_red, W_red, new_rank = _try_reduce_rank(U, V, W, n, m, p)
            if new_rank < len(U[0]):
                if _verify(U_red, V_red, W_red, n, m, p, tol=1e-9):
                    return {"U": U_red, "V": V_red, "W": W_red}
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
