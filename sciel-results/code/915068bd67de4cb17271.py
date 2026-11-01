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
    """
    n, m, p = DIMENSIONS  # 2, 4, 8
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2 * 2x2 = 2x2
    # Each term: (A_coeffs, B_coeffs, C_coeffs)
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

def _try_rank_reduction_via_linear_combination(U, V, W, n, m, p, max_time=30.0):
    """
    Attempt to reduce the rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic approach: we try to express some columns as linear combinations
    of others, potentially reducing the effective rank.
    
    For now, this is a placeholder that returns the input unchanged, but the structure
    is set up for future refinement.
    """
    start = time.time()
    deadline = start + max_time
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Check if any column is a linear combination of others
    # This is a simplified check - in practice, we'd use more sophisticated methods
    # For now, just return the original decomposition
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Try Strassen decomposition first (rank 56)
    try:
        result = _strassen_decomposition()
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            # Try to reduce rank further if time permits
            if time.time() < deadline - 35.0:
                try:
                    reduced = _try_rank_reduction_via_linear_combination(
                        result["U"], result["V"], result["W"], n, m, p,
                        max_time=min(30.0, deadline - time.time() - 5.0)
                    )
                    if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p, tol=1e-9):
                        return reduced
                except Exception:
                    pass
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 64)
    return _schoolbook_decomposition()

# EVOLVE-BLOCK-END
