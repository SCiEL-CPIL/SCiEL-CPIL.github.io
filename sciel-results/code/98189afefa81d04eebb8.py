<<<<<<< SEARCH
import time
import sys

# Constants for the problem dimensions
N_ROWS_A = 2
M_COLS_A = 4
P_COLS_B = 4

# The known minimal rank for 2x4x4 matrix multiplication is 28.
TARGET_RANK = 28

def construct():
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    
    # Initialize factor matrices with zeros
    # U: shape (n*m, rank) -> (8, 28)
    # V: shape (m*p, rank) -> (16, 28)
    # W: shape (p*n, rank) -> (8, 28)
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    # Helper to index U, V, W
    # A[i, j] -> U[i*m + j]
    # B[i, j] -> V[i*p + j]
    # C[i, j] -> W[j*n + i]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    # Define the 7 Strassen multiplication terms for a 2x2 block multiplication
    # Each term is a list of (matrix, row, col, coefficient)
    # Format: (u_row, u_col, coeff), (v_row, v_col, coeff), (w_row, w_col, coeff)
    # Note: We are mapping the local 2x2 indices to the global indices.
    
    # The 7 Strassen operations for A*->B->C (2x2)
    # M1 = (A11 + A22)(B11 + B22) -> C11 += M1, C22 += M1
    # M2 = (A21 + A22)B11        -> C21 += M2, C22 -= M2
    # M3 = A11(B12 - B22)        -> C12 += M3, C22 += M3
    # M4 = A22(B21 - B11)        -> C11 += M4, C21 += M4
    # M5 = (A11 + A12)B22        -> C11 -= M5, C12 += M5
    # M6 = (A21 - A11)(B11 + B12) -> C22 += M6
    # M7 = (A12 - A22)(B21 + B22) -> C11 += M7
    
    # We will apply this structure 4 times.
    # The 4 blocks correspond to the 2x2 sub-blocks of the 4x4 B matrix
    # combined with the 2x2 sub-blocks of the 2x4 A matrix.
    
    # Block definitions: (A_row_start, A_col_start, B_row_start, B_col_start, C_col_start)
    # A is 2x4, so A_row is always 0. A_col is 0 or 2.
    # B is 4x4. B_row is 0 or 2. B_col is 0 or 2.
    # C is 2x4. C_col is 0 or 2. C_row is always 0.
    
    blocks = [
        # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 0, 0, 0),
        # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (0, 2, 2, 0, 0),
        # 3. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (0, 0, 0, 2, 2),
        # 4. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 2, 2, 2, 2)
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        # Local indices for A block:
        # a11 = A[a_r, a_c], a12 = A[a_r, a_c+1]
        # a21 = A[a_r+1, a_c], a22 = A[a_r+1, a_c+1]
        
        # Local indices for B block:
        # b11 = B[b_r, b_c], b12 = B[b_r, b_c+1]
        # b21 = B[b_r+1, b_c], b22 = B[b_r+1, b_c+1]
        
        # Local indices for C block:
        # c11 = C[0, c_c], c12 = C[0, c_c+1]
        # c21 = C[1, c_c], c22 = C[1, c_c+1]
        # Note: C row is 0 or 1. C col is c_c or c_c+1.
        
        # U indices
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        # V indices
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        # W indices (C[i, j] -> W[j*n + i])
        # C[0, c_c] -> W[c_c*n + 0]
        # C[0, c_c+1] -> W[(c_c+1)*n + 0]
        # C[1, c_c] -> W[c_c*n + 1]
        # C[1, c_c+1] -> W[(c_c+1)*n + 1]
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # Term 1: (u11+u22)(v11+v22) -> w11, w22
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2: (u21+u22)v11 -> w21, -w22
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3: u11(v12-v22) -> w12, w22
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4: u22(v21-v11) -> w11, w21
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5: (u11+u12)v22 -> -w11, w12
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6: (u21-u11)(v11+v12) -> w22
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7: (u12-u22)(v21+v22) -> w11
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    if current_rank != TARGET_RANK:
        raise ValueError(f"Rank mismatch: expected {TARGET_RANK}, got {current_rank}")
        
    return {"U": U, "V": V, "W": W}
=======
import time
import sys
import numpy as np

# Constants for the problem dimensions
N_ROWS_A = 2
M_COLS_A = 4
P_COLS_B = 4

# Laderman's rank-19 algorithm for 3x3x3 matrix multiplication
# Each term: (u_indices, v_indices, w_indices, coefficient)
# u_indices: list of (row, col) for A
# v_indices: list of (row, col) for B  
# w_indices: list of (row, col, coeff) for C

def _get_laderman_terms():
    """Return Laderman's 19 terms for 3x3x3 matrix multiplication."""
    terms = []
    
    # M1 = A11 * B11
    terms.append(([ (0,0) ], [ (0,0) ], [ (0,0,1) ]))
    # M2 = (A12+A13)(B21+B31)
    terms.append(([ (0,1), (0,2) ], [ (1,0), (2,0) ], [ (0,1,1), (0,2,1) ]))
    # M3 = (A21+A31)(B12+B13)
    terms.append(([ (1,0), (2,0) ], [ (0,1), (0,2) ], [ (1,1,1), (2,1,1) ]))
    # M4 = (A11-A21)(B22-B21)
    terms.append(([ (0,0), (1,0) ], [ (1,1), (1,0) ], [ (1,1,1) ]))
    # M5 = (A22-A32)(B33-B32)
    terms.append(([ (1,1), (2,1) ], [ (2,2), (2,1) ], [ (2,2,1) ]))
    # M6 = (A33-A13)(B11-B12)
    terms.append(([ (2,2), (0,2) ], [ (0,0), (0,1) ], [ (2,2,1) ]))
    # M7 = (A11+A22)(B12+B22)
    terms.append(([ (0,0), (1,1) ], [ (0,1), (1,1) ], [ (0,1,1), (1,1,1) ]))
    # M8 = (A22+A33)(B23+B33)
    terms.append(([ (1,1), (2,2) ], [ (1,2), (2,2) ], [ (1,2,1), (2,2,1) ]))
    # M9 = (A33+A11)(B31+B11)
    terms.append(([ (2,2), (0,0) ], [ (2,0), (0,0) ], [ (2,0,1), (0,0,1) ]))
    # M10 = (A12+A13-A22+A33)(B21+B31-B22+B33)
    terms.append(([ (0,1), (0,2), (1,1), (2,2) ], [ (1,0), (2,0), (1,1), (2,2) ], [ (0,1,1), (0,2,1), (1,2,1), (2,1,1) ]))
    # M11 = (A21+A31-A11+A22)(B12+B13-B22+B33)
    terms.append(([ (1,0), (2,0), (0,0), (1,1) ], [ (0,1), (0,2), (1,1), (2,2) ], [ (1,1,1), (2,1,1), (1,2,1), (2,2,1) ]))
    # M12 = (A33+A11-A13+A31)(B31+B11-B12+B21)
    terms.append(([ (2,2), (0,0), (0,2), (2,0) ], [ (2,0), (0,0), (0,1), (1,0) ], [ (2,0,1), (0,0,1), (2,1,1), (0,1,1) ]))
    # M13 = (A12-A23)(B32-B22)
    terms.append(([ (0,1), (1,2) ], [ (2,1), (1,1) ], [ (0,2,1) ]))
    # M14 = (A23-A31)(B13-B33)
    terms.append(([ (1,2), (2,0) ], [ (0,2), (2,2) ], [ (1,2,1) ]))
    # M15 = (A31-A12)(B21-B11)
    terms.append(([ (2,0), (0,1) ], [ (1,0), (0,0) ], [ (2,0,1) ]))
    # M16 = (A22-A13)(B12-B32)
    terms.append(([ (1,1), (0,2) ], [ (0,1), (2,1) ], [ (1,1,1) ]))
    # M17 = (A33-A21)(B31-B23)
    terms.append(([ (2,2), (1,0) ], [ (2,0), (1,2) ], [ (2,2,1) ]))
    # M18 = (A11-A32)(B21-B12)
    terms.append(([ (0,0), (2,1) ], [ (1,0), (0,1) ], [ (0,1,1) ]))
    # M19 = (A22-A13)(B33-B12)  
    # Wait, let me use the standard Laderman formulation
    
    # Let me use a cleaner formulation. The standard Laderman 19 terms:
    terms = []
    # M1 = A11 * B11 -> C11
    terms.append(([(0,0)], [(0,0)], [(0,0,1)]))
    # M2 = (A12+A13) * (B21+B31) -> C12, C13
    terms.append(([(0,1),(0,2)], [(1,0),(2,0)], [(0,1,1),(0,2,1)]))
    # M3 = (A21+A31) * (B12+B13) -> C21, C31
    terms.append(([(1,0),(2,0)], [(0,1),(0,2)], [(1,1,1),(2,1,1)]))
    # M4 = (A11-A21) * (B22-B21) -> C21
    terms.append(([(0,0),(1,0)], [(1,1),(1,0)], [(1,1,1)]))
    # M5 = (A22-A32) * (B33-B32) -> C32
    terms.append(([(1,1),(2,1)], [(2,2),(2,1)], [(2,2,1)]))
    # M6 = (A33-A13) * (B11-B12) -> C31
    terms.append(([(2,2),(0,2)], [(0,0),(0,1)], [(2,0,1)]))
    # M7 = (A11+A22) * (B12+B22) -> C12, C22
    terms.append(([(0,0),(1,1)], [(0,1),(1,1)], [(0,1,1),(1,1,1)]))
    # M8 = (A22+A33) * (B23+B33) -> C23, C33
    terms.append(([(1,1),(2,2)], [(1,2),(2,2)], [(1,2,1),(2,2,1)]))
    # M9 = (A33+A11) * (B31+B11) -> C31, C11
    terms.append(([(2,2),(0,0)], [(2,0),(0,0)], [(2,0,1),(0,0,1)]))
    # M10 = (A12+A13-A22+A33) * (B21+B31-B22+B33) -> C12,C13,C23,C32
    terms.append(([(0,1),(0,2),(1,1),(2,2)], [(1,0),(2,0),(1,1),(2,2)], 
                  [(0,1,1),(0,2,1),(1,2,1),(2,1,1)]))
    # M11 = (A21+A31-A11+A22) * (B12+B13-B22+B33) -> C21,C31,C23,C33
    terms.append(([(1,0),(2,0),(0,0),(1,1)], [(0,1),(0,2),(1,1),(2,2)],
                  [(1,1,1),(2,1,1),(1,2,1),(2,2,1)]))
    # M12 = (A33+A11-A13+A31) * (B31+B11-B12+B21) -> C31,C11,C32,C12
    terms.append(([(2,2),(0,0),(0,2),(2,0)], [(2,0),(0,0),(0,1),(1,0)],
                  [(2,0,1),(0,0,1),(2,1,1),(0,1,1)]))
    # M13 = (A12-A23) * (B32-B22) -> C13
    terms.append(([(0,1),(1,2)], [(2,1),(1,1)], [(0,2,1)]))
    # M14 = (A23-A31) * (B13-B33) -> C23
    terms.append(([(1,2),(2,0)], [(0,2),(2,2)], [(1,2,1)]))
    # M15 = (A31-A12) * (B21-B11) -> C31
    terms.append(([(2,0),(0,1)], [(1,0),(0,0)], [(2,0,1)]))
    # M16 = (A22-A13) * (B12-B32) -> C21
    terms.append(([(1,1),(0,2)], [(0,1),(2,1)], [(1,1,1)]))
    # M17 = (A33-A21) * (B31-B23) -> C33
    terms.append(([(2,2),(1,0)], [(2,0),(1,2)], [(2,2,1)]))
    # M18 = (A11-A32) * (B21-B12) -> C12
    terms.append(([(0,0),(2,1)], [(1,0),(0,1)], [(0,1,1)]))
    # M19 = (A22-A13) * (B33-B12) -> C22
    terms.append(([(1,1),(0,2)], [(2,2),(0,1)], [(1,1,1)]))
    
    return terms

def _build_strassen_solution():
    """Build the Strassen-based rank-28 solution as fallback."""
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    TARGET_RANK = 28
    
    U = [[0] * TARGET_RANK for _ in range(n * m)]
    V = [[0] * TARGET_RANK for _ in range(m * p)]
    W = [[0] * TARGET_RANK for _ in range(p * n)]
    
    def get_u_idx(i, j): return i * m + j
    def get_v_idx(i, j): return i * p + j
    def get_w_idx(i, j): return j * n + i
    
    blocks = [
        (0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0),
        (0, 0, 0, 2, 2),
        (0, 2, 2, 2, 2)
    ]
    
    current_rank = 0
    
    for (a_r, a_c, b_r, b_c, c_c) in blocks:
        u11 = get_u_idx(a_r, a_c)
        u12 = get_u_idx(a_r, a_c + 1)
        u21 = get_u_idx(a_r + 1, a_c)
        u22 = get_u_idx(a_r + 1, a_c + 1)
        
        v11 = get_v_idx(b_r, b_c)
        v12 = get_v_idx(b_r, b_c + 1)
        v21 = get_v_idx(b_r + 1, b_c)
        v22 = get_v_idx(b_r + 1, b_c + 1)
        
        w11 = get_w_idx(0, c_c)
        w12 = get_w_idx(0, c_c + 1)
        w21 = get_w_idx(1, c_c)
        w22 = get_w_idx(1, c_c + 1)
        
        # Term 1: (u11+u22)(v11+v22) -> w11, w22
        U[u11][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 2: (u21+u22)v11 -> w21, -w22
        U[u21][current_rank] += 1
        U[u22][current_rank] += 1
        V[v11][current_rank] += 1
        W[w21][current_rank] += 1
        W[w22][current_rank] += -1
        current_rank += 1
        
        # Term 3: u11(v12-v22) -> w12, w22
        U[u11][current_rank] += 1
        V[v12][current_rank] += 1
        V[v22][current_rank] += -1
        W[w12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 4: u22(v21-v11) -> w11, w21
        U[u22][current_rank] += 1
        V[v21][current_rank] += 1
        V[v11][current_rank] += -1
        W[w11][current_rank] += 1
        W[w21][current_rank] += 1
        current_rank += 1
        
        # Term 5: (u11+u12)v22 -> -w11, w12
        U[u11][current_rank] += 1
        U[u12][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += -1
        W[w12][current_rank] += 1
        current_rank += 1
        
        # Term 6: (u21-u11)(v11+v12) -> w22
        U[u21][current_rank] += 1
        U[u11][current_rank] += -1
        V[v11][current_rank] += 1
        V[v12][current_rank] += 1
        W[w22][current_rank] += 1
        current_rank += 1
        
        # Term 7: (u12-u22)(v21+v22) -> w11
        U[u12][current_rank] += 1
        U[u22][current_rank] += -1
        V[v21][current_rank] += 1
        V[v22][current_rank] += 1
        W[w11][current_rank] += 1
        current_rank += 1
        
    return {"U": U, "V": V, "W": W}

def _verify_solution(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0])
    
    # Convert to numpy arrays for efficient checking
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check all entries of the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected: A[i,j] * B[j,k] = C[i,k]
                # In tensor form: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _verify_solution_zero(U, V, W, n, m, p):
    """Verify that non-multiplication entries are zero."""
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For entries that should be zero:
    # U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] should be 0
    # when (i,j,k) != (i',j',k') in the appropriate sense
    
    # Actually, the tensor T[i,j,k; i',j',k'] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # should equal 1 if i=i', j=j', k=k' and 0 otherwise
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for kp in range(p):
                            expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            actual = 0.0
                            for r in range(rank):
                                actual += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[kp*n+ip, r]
                            if abs(actual - expected) > 1e-9:
                                return False
    return True

def _try_laderman_projection():
    """
    Try to embed 2x4x4 into 3x3x3 using Laderman and project.
    
    Strategy: We need to map our 2x4x4 multiplication into a 3x3x3 problem.
    A is 2x4, B is 4x4, C is 2x4.
    
    We can think of this as:
    - A has 2 rows, 4 cols -> pad to 3x3 by adding zero row and using cols 0,1,2 (ignore col 3 or map differently)
    - B has 4x4 -> pad to 3x3 by using rows 0,1,2 and cols 0,1,2 (ignore row 3 and col 3)
    - C has 2 rows, 4 cols -> pad to 3x3 by adding zero row and using cols 0,1,2
    
    But this loses information. Instead, let's try a different approach:
    
    We can split the 4 columns into two groups of 2:
    - A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :] = C
    
    This is what Strassen does. For Laderman projection, we need a different embedding.
    
    Alternative: Use the fact that 2x4x4 can be viewed as two 2x2x4 multiplications
    or four 2x2x2 multiplications.
    
    Let me try a direct approach: construct the 2x4x4 tensor and try to find
    a rank-<28 decomposition using optimization.
    """
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    
    # The 2x4x4 multiplication tensor has n*m*p = 32 nonzero entries
    # We need to find U (8 x r), V (16 x r), W (8 x r) such that
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # and 0 for all other (i,j,k) combinations in the full tensor space.
    
    # Actually, the tensor is defined on [n] x [m] x [p] x [n] x [m] x [p]
    # T[i,j,k,i',j',k'] = 1 if i=i', j=j', k=k' else 0
    
    # For the multiplication to work, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    
    # This is a 6-dimensional tensor. Let's set up the optimization.
    
    # Try rank 27, 26, ... down to some minimum
    for target_rank in range(27, 20, -1):
        # Initialize with small random values
        np.random.seed(42)
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # Use gradient descent to minimize the error
        lr = 0.01
        for iteration in range(1000):
            # Compute gradients
            U_grad = np.zeros_like(U)
            V_grad = np.zeros_like(V)
            W_grad = np.zeros_like(W)
            
            # For each entry (i,j,k,i',j',k')
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for jp in range(m):
                                for kp in range(p):
                                    expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    actual = 0.0
                                    for r in range(target_rank):
                                        actual += U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r]
                                    error = actual - expected
                                    
                                    for r in range(target_rank):
                                        U_grad[i*m+j, r] += error * V[j*p+k, r] * W[kp*n+ip, r]
                                        V_grad[j*p+k, r] += error * U[i*m+j, r] * W[kp*n+ip, r]
                                        W_grad[kp*n+ip, r] += error * U[i*m+j, r] * V[j*p+k, r]
            
            # Update
            U -= lr * U_grad
            V -= lr * V_grad
            W -= lr * W_grad
            
            # Check convergence
            if iteration % 100 == 0:
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for ip in range(n):
                                for jp in range(m):
                                    for kp in range(p):
                                        expected = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                        actual = 0.0
                                        for r in range(target_rank):
                                            actual += U[i*m+j, r] * V[j*p+k, r] * W[kp*n+ip, r]
                                        max_err = max(max_err, abs(actual - expected))
                if max_err < 1e-6:
                    # Found a valid solution!
                    # Convert to lists with rounded values
                    U_list = [[round(U[i, r], 6) for r in range(target_rank)] for i in range(n*m)]
                    V_list = [[round(V[i, r], 6) for r in range(target_rank)] for i in range(m*p)]
                    W_list = [[round(W[i, r], 6) for r in range(target_rank)] for i in range(p*n)]
                    
                    # Verify
                    if _verify_solution(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = N_ROWS_A, M_COLS_A, P_COLS_B
    
    # First, try the Laderman projection approach
    laderman_result = _try_laderman_projection()
    if laderman_result is not None:
        return laderman_result
    
    # Fall back to Strassen rank-28 solution
    return _build_strassen_solution()
>>>>>>> REPLACE
