"""Structured search for matrix multiplication tensor decomposition 4x4 * 4x5."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS
NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20

def _verify(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the all-ones multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    r = U_np.shape[1]
    
    tensor = np.zeros((n, m, p), dtype=np.float64)
    for c in range(r):
        u_c = U_np[:, c]
        v_c = V_np[:, c]
        w_c = W_np[:, c]
        
        for i in range(n):
            for j in range(m):
                u_val = u_c[i * m + j]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = v_c[j * p + k]
                    w_val = w_c[k * n + i]
                    tensor[i, j, k] += u_val * v_val * w_val
    
    return np.allclose(tensor, np.ones((n, m, p)), atol=tol)

def _build_schoolbook():
    """Build the schoolbook (rank 80) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2 as (U, V, W) with rank 7.
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications for 2x2 matrices
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # Each column c corresponds to one Strassen multiplication
    # U encodes coefficients on A elements, V on B elements, W on C output
    
    # M1 = (A11+A22)*(B11+B22)
    U[0][0] = 1; U[3][0] = 1  # A11, A22
    V[0][0] = 1; V[1][0] = 1  # B11, B22 (B11 is at [0,0], B22 is at [1,1])
    # Wait, V[j*2+k, c]: B11 is j=0,k=0 -> idx 0; B22 is j=1,k=1 -> idx 3
    V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C11, C22 (W[k*2+i, c]: C11 is k=0,i=0 -> idx 0; C22 is k=1,i=1 -> idx 3)
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1; U[3][1] = 1  # A21 (idx 2), A22 (idx 3)
    V[0][1] = 1  # B11
    W[2][1] = 1; W[3][1] = 1  # C21 (k=1,i=0 -> idx 2), C22 (k=1,i=1 -> idx 3)
    # Wait, C21 is row 2, col 1: k=1, i=0 -> W[1*2+0] = W[2]
    # C22 is row 2, col 2: k=1, i=1 -> W[1*2+1] = W[3]
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1; V[3][2] = -1  # B12 (j=0,k=1 -> idx 1), B22 (j=1,k=1 -> idx 3)
    W[1][2] = 1; W[3][2] = 1  # C12 (k=1,i=0 -> idx 1), C22 (k=1,i=1 -> idx 3)
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1; V[0][3] = -1  # B21 (j=1,k=0 -> idx 2), B11 (j=0,k=0 -> idx 0)
    W[0][3] = 1; W[2][3] = 1  # C11 (k=0,i=0 -> idx 0), C21 (k=1,i=0 -> idx 2)
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1; U[1][4] = 1  # A11 (idx 0), A12 (idx 1)
    V[3][4] = 1  # B22 (idx 3)
    W[0][4] = -1; W[1][4] = 1  # C11 (idx 0), C12 (idx 1)
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1; U[0][5] = -1  # A21 (idx 2), A11 (idx 0)
    V[0][5] = 1; V[1][5] = 1  # B11 (idx 0), B12 (idx 1)
    W[3][5] = 1  # C22 (idx 3)
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1; U[3][6] = -1  # A12 (idx 1), A22 (idx 3)
    V[2][6] = 1; V[3][6] = 1  # B21 (idx 2), B22 (idx 3)
    W[0][6] = 1  # C11 (idx 0)
    
    return U, V, W

def _strassen_2x2_2x3():
    """
    Return a decomposition for 2x2 * 2x3.
    This is trickier. Let's use a simple approach:
    C (2x3) = A (2x2) * B (2x3)
    C[0,0] = A00*B00 + A01*B10
    C[0,1] = A00*B01 + A01*B11
    C[0,2] = A00*B02 + A01*B12
    C[1,0] = A10*B00 + A11*B10
    C[1,1] = A10*B01 + A11*B11
    C[1,2] = A10*B02 + A11*B12
    
    Schoolbook: 12 multiplications.
    We can try to reduce. One approach: group by A columns.
    
    Actually, for 2x2 * 2x3, the rank is known to be at most 10 or so.
    Let me try a structured approach with rank 10.
    
    Alternatively, I can pad B to 2x4 and use Strassen-like, but that's wasteful.
    
    Let me just use schoolbook for now (rank 12) and combine with Strassen for 2x2*2x2.
    """
    rank = 12
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(6)]
    W = [[0] * rank for _ in range(6)]
    
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i * 2 + j][col] = 1
                V[j * 3 + k][col] = 1
                W[k * 2 + i][col] = 1
                col += 1
    
    return U, V, W

def _combine_decompositions():
    """
    Combine Strassen 2x2*2x2 (rank 7) with schoolbook 2x2*2x3 (rank 12)
    to get a decomposition for 4x4 * 4x5.
    
    Block structure:
    A = [[A11, A12], [A21, A22]] (each 2x2)
    B = [[B11, B12], [B21, B22]] where B11,B21 are 2x2, B12,B22 are 2x3
    
    C11 (2x2) = A11*B11 + A12*B21  -> use Strassen (rank 7)
    C21 (2x2) = A21*B11 + A22*B21  -> use Strassen (rank 7)
    C12 (2x3) = A11*B12 + A12*B22  -> use schoolbook (rank 12)
    C22 (2x3) = A21*B12 + A22*B22  -> use schoolbook (rank 12)
    
    Total rank: 7 + 7 + 12 + 12 = 38
    """
    # Get the sub-decompositions
    U_s, V_s, W_s = _strassen_2x2_2x2()
    U_b, V_b, W_b = _strassen_2x2_2x3()
    
    total_rank = 7 + 7 + 12 + 12  # 38
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(20)]
    W = [[0] * total_rank for _ in range(20)]
    
    # Column offset for each block
    # Block 1: C11 (rows 0-1, cols 0-1 of C) -> A11, A12 (rows 0-1 of A), B11, B21 (cols 0-1 of B)
    # Block 2: C21 (rows 2-3, cols 0-1 of C) -> A21, A22 (rows 2-3 of A), B11, B21 (cols 0-1 of B)
    # Block 3: C12 (rows 0-1, cols 2-4 of C) -> A11, A12 (rows 0-1 of A), B12, B22 (cols 2-4 of B)
    # Block 4: C22 (rows 2-3, cols 2-4 of C) -> A21, A22 (rows 2-3 of A), B12, B22 (cols 2-4 of B)
    
    def _place_strassen(U_out, V_out, W_out, U_sub, V_sub, W_sub, 
                        a_row_offset, b_col_offset, c_row_offset, c_col_offset, col_offset):
        """Place a 2x2*2x2 Strassen decomposition into the larger matrices."""
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    val = U_sub[i * 2 + j][c]
                    if val != 0:
                        U_out[a_row_offset + i][col_offset + c] = val
            for j in range(2):
                for k in range(2):
                    val = V_sub[j * 2 + k][c]
                    if val != 0:
                        V_out[b_col_offset + j][col_offset + c] = val
            for k in range(2):
                for i in range(2):
                    val = W_sub[k * 2 + i][c]
                    if val != 0:
                        W_out[c_row_offset + k][col_offset + c] = val
    
    def _place_schoolbook_2x2_2x3(U_out, V_out, W_out, U_sub, V_sub, W_sub,
                                   a_row_offset, b_col_offset, c_row_offset, c_col_offset, col_offset):
        """Place a 2x2*2x3 schoolbook decomposition into the larger matrices."""
        for c in range(12):
            for i in range(2):
                for j in range(2):
                    val = U_sub[i * 2 + j][c]
                    if val != 0:
                        U_out[a_row_offset + i][col_offset + c] = val
            for j in range(2):
                for k in range(3):
                    val = V_sub[j * 3 + k][c]
                    if val != 0:
                        V_out[b_col_offset + j][col_offset + c] = val
            for k in range(3):
                for i in range(2):
                    val = W_sub[k * 2 + i][c]
                    if val != 0:
                        W_out[c_row_offset + k][col_offset + c] = val
    
    # Block 1: C11 = A11*B11 + A12*B21
    # A rows 0-1, B cols 0-1, C rows 0-1, C cols 0-1
    _place_strassen(U, V, W, U_s, V_s, W_s, 0, 0, 0, 0, 0)
    
    # Block 2: C21 = A21*B11 + A22*B21
    # A rows 2-3, B cols 0-1, C rows 2-3, C cols 0-1
    _place_strassen(U, V, W, U_s, V_s, W_s, 2, 0, 2, 0, 7)
    
    # Block 3: C12 = A11*B12 + A12*B22
    # A rows 0-1, B cols 2-4, C rows 0-1, C cols 2-4
    _place_schoolbook_2x2_2x3(U, V, W, U_b, V_b, W_b, 0, 2, 0, 2, 14)
    
    # Block 4: C22 = A21*B12 + A22*B22
    # A rows 2-3, B cols 2-4, C rows 2-3, C cols 2-4
    _place_schoolbook_2x2_2x3(U, V, W, U_b, V_b, W_b, 2, 2, 2, 2, 26)
    
    return U, V, W

def _verify_combined():
    """Verify the combined decomposition."""
    U, V, W = _combine_decompositions()
    return _verify(U, V, W)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Start with schoolbook as guaranteed valid solution
    U_sb, V_sb, W_sb = _build_schoolbook()
    best_rank = n * m * p  # 80
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try the structured combined decomposition
    U_comb, V_comb, W_comb = _combine_decompositions()
    if _verify(U_comb, V_comb, W_comb):
        rank_comb = len(U_comb[0])
        if rank_comb < best_rank:
            best_rank = rank_comb
            best_solution = {"U": U_comb, "V": V_comb, "W": W_comb}
    
    # If we have time, try to improve with random search on lower ranks
    if time.time() < deadline - 10:
        for r in range(max(10, best_rank - 10), best_rank):
            if time.time() > deadline - 5:
                break
            
            remaining = deadline - time.time() - 5
            if remaining <= 0:
                break
            
            num_trials = min(500, int(remaining * 100))
            num_trials = max(num_trials, 20)
            
            found = False
            for trial in range(num_trials):
                if time.time() > deadline - 5:
                    break
                
                entries = (-1, 0, 1)
                U = [[random.choice(entries) for _ in range(r)] for _ in range(NM)]
                V = [[random.choice(entries) for _ in range(r)] for _ in range(MP)]
                W = [[random.choice(entries) for _ in range(r)] for _ in range(PN)]
                
                if _verify(U, V, W, tol=1e-9):
                    best_rank = r
                    best_solution = {"U": U, "V": V, "W": W}
                    found = True
                    break
            
            if found:
                continue
    
    return best_solution
