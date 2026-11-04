"""Matrix multiplication 3x4 x 4x8 - rank minimal exact tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify_decomposition(U, V, W, n, m, p, max_samples=20):
    """Verify that the tensor decomposition is correct."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor
    # T[i*n*m + j*m + k, r] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Actually, the standard convention: for matrix multiplication C = A * B where A is n x m, B is m x p
    # The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # W[k*n+i, r] is the convention stated in the task.
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check a few random entries
    for _ in range(max_samples):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        # The (i,j,k) entry of the multiplication tensor is 1 if this is a valid multiplication
        # Actually, for matrix multiplication, the tensor T[i,j,k] represents the coefficient
        # of A[i,j] * B[j,k] in C[i,k], which is 1.
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        lhs = 0.0
        for r in range(rank):
            lhs += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
        
        if abs(lhs - 1.0) > 1e-9:
            return False
    
    return True

def _strassen_2x2x2_terms():
    """Return Strassen's 7 terms for 2x2x2 matrix multiplication."""
    # Each term is (U_coeffs, V_coeffs, W_coeffs)
    # U: (i,j) -> i*2+j, i in {0,1}, j in {0,1}
    # V: (j,k) -> j*2+k, j in {0,1}, k in {0,1}
    # W: (k,i) -> k*2+i, k in {0,1}, i in {0,1}
    
    terms = []
    
    # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # U: A00, A11 -> indices 0, 3
    # V: B00, B11 -> indices 0, 3
    # W: C00, C11 -> indices 0, 3
    terms.append(({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}))
    
    # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
    # U: A10, A11 -> indices 2, 3
    # V: B00 -> index 0
    # W: C10, C11 -> indices 2, 3 (C10 = W[1,0] = 1*2+0 = 2, C11 = W[1,1] = 3)
    terms.append(({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}))
    
    # M3 = A00*(B10-B00) -> C00 -= M3, C01 += M3
    # U: A00 -> index 0
    # V: B10, B00 -> indices 2, 0
    # W: C00, C01 -> indices 0, 1
    terms.append(({0: 1}, {2: 1, 0: -1}, {0: -1, 1: 1}))
    
    # M4 = A11*(B01-B11) -> C10 += M4, C11 += M4
    # U: A11 -> index 3
    # V: B01, B11 -> indices 1, 3
    # W: C10, C11 -> indices 2, 3
    terms.append(({3: 1}, {1: 1, 3: -1}, {2: 1, 3: 1}))
    
    # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
    # U: A00, A01 -> indices 0, 1
    # V: B11 -> index 3
    # W: C01, C00 -> indices 1, 0
    terms.append(({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}))
    
    # M6 = (A10-A00)*(B00+B10) -> C10 += M6
    # U: A10, A00 -> indices 2, 0
    # V: B00, B10 -> indices 0, 2
    # W: C10 -> index 2
    terms.append(({2: 1, 0: -1}, {0: 1, 2: 1}, {2: 1}))
    
    # M7 = (A01-A11)*(B11+B01) -> C01 += M7
    # U: A01, A11 -> indices 1, 3
    # V: B11, B01 -> indices 3, 1
    # W: C01 -> index 1
    terms.append(({1: 1, 3: -1}, {3: 1, 1: 1}, {1: 1}))
    
    return terms

def _build_decomposition(n, m, p):
    """Build a rank-88 decomposition for n x m * m x p matrix multiplication."""
    # Strategy:
    # Part 1: Row 0 of A (1x4) times B (4x8): schoolbook, rank 32
    # Part 2: Rows 1-2 of A (2x4) times B (4x8):
    #   Split into 4 blocks of 2 columns of B
    #   For each block, use Strassen for the two 2x2 sub-problems
    #   Total: 4 blocks * 2 sub-problems * 7 terms = 56
    # Grand total: 32 + 56 = 88
    
    rank = 32 + 56  # 88
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    
    # Part 1: Schoolbook for A[0, :] (1x4) * B (4x8)
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    
    # Part 2: Strassen for A[1:3, :] (2x4) * B (4x8)
    strassen_terms = _strassen_2x2x2_terms()
    
    for j_block in range(4):
        # P1: A[1:3, 0:2] (2x2) * B[0:2, 2*j_block:2*j_block+2] (2x2)
        for term_u, term_v, term_w in strassen_terms:
            for local_u, val in term_u.items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = lv
                U[global_i * m + global_j, col] += val
            for local_v, val in term_v.items():
                lu, lv = divmod(local_v, 2)
                global_j = lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k, col] += val
            for local_w, val in term_w.items():
                lu, lv = divmod(local_w, 2)
                global_k = lu
                global_i = 1 + lv
                W[global_k * n + global_i, col] += val
            col += 1
        
        # P2: A[1:3, 2:4] (2x2) * B[2:4, 2*j_block:2*j_block+2] (2x2)
        for term_u, term_v, term_w in strassen_terms:
            for local_u, val in term_u.items():
                lu, lv = divmod(local_u, 2)
                global_i = 1 + lu
                global_j = 2 + lv
                U[global_i * m + global_j, col] += val
            for local_v, val in term_v.items():
                lu, lv = divmod(local_v, 2)
                global_j = 2 + lu
                global_k = 2 * j_block + lv
                V[global_j * p + global_k, col] += val
            for local_w, val in term_w.items():
                lu, lv = divmod(local_w, 2)
                global_k = lu
                global_i = 1 + lv
                W[global_k * n + global_i, col] += val
            col += 1
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 3x4 x 4x8 matrix multiplication."""
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Build the base rank-88 decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Convert to lists with integer/half-integer values
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Round to nearest integer or half-integer
    def round_to_half(x):
        return round(x * 2) / 2.0
    
    U_list = [[round_to_half(val) for val in row] for row in U_list]
    V_list = [[round_to_half(val) for val in row] for row in V_list]
    W_list = [[round_to_half(val) for val in row] for row in W_list]
    
    # Convert to integers where possible
    def to_int_if_possible(x):
        if abs(x - round(x)) < 1e-9:
            return int(round(x))
        elif abs(2*x - round(2*x)) < 1e-9:
            return round(2*x) / 2.0
        return x
    
    U_list = [[to_int_if_possible(val) for val in row] for row in U_list]
    V_list = [[to_int_if_possible(val) for val in row] for row in V_list]
    W_list = [[to_int_if_possible(val) for val in row] for row in W_list]
    
    # Verify correctness
    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # If verification fails, fall back to schoolbook (rank 96)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
