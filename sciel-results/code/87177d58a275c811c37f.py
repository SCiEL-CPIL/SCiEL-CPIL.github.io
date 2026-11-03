"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the rank-7 Strassen decomposition for 2x2 * 2x2.
    Returns U, V, W matrices (4x7 each) with integer entries.
    
    Indexing convention:
    - Input A: 2x2, flattened row-major: A[0,0], A[0,1], A[1,0], A[1,1] -> indices 0,1,2,3
    - Input B: 2x2, flattened row-major: B[0,0], B[0,1], B[1,0], B[1,1] -> indices 0,1,2,3
    - Output C: 2x2, flattened row-major: C[0,0], C[0,1], C[1,0], C[1,1] -> indices 0,1,2,3
    
    U[i, r] = coefficient of A[i] in rank-r term
    V[j, r] = coefficient of B[j] in rank-r term
    W[k, r] = coefficient of C[k] in rank-r term (trace convention: C[k] = sum_r W[k,r] * (U^T A) * (V^T B))
    """
    # Strassen's 7 products:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U: 4x7, rows correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    U = np.array([
        # A00 (row 0)
        [1, 0, 1, 0, 1, -1, 0],
        # A01 (row 1)
        [0, 0, 0, 0, 1, 0, -1],
        # A10 (row 2)
        [0, 1, 0, 0, 0, 1, 0],
        # A11 (row 3)
        [1, 1, 0, 1, 0, 0, -1],
    ], dtype=object)
    
    # V: 4x7, rows correspond to B[0,0], B[0,1], B[1,0], B[1,1]
    V = np.array([
        # B00 (row 0)
        [1, 1, 0, -1, 0, 1, 0],
        # B01 (row 1)
        [0, 0, 1, 0, 0, 1, 0],
        # B10 (row 2)
        [0, 0, 0, 1, 0, 0, 1],
        # B11 (row 3)
        [1, 0, -1, 0, 1, 0, 1],
    ], dtype=object)
    
    # W: 4x7, rows correspond to C[0,0], C[0,1], C[1,0], C[1,1]
    W = np.array([
        # C00
        [1, 0, 0, 1, -1, 0, 1],
        # C01
        [0, 0, 1, 0, 1, 0, 0],
        # C10
        [0, 1, 0, 1, 0, 0, 0],
        # C11
        [1, -1, 1, 0, 0, 1, 0],
    ], dtype=object)
    
    return U, V, W


def _build_block_decomposition():
    """
    Build the rank-56 decomposition by partitioning 2x4 * 4x8 into 
    8 independent 2x2 * 2x2 sub-problems and applying Strassen to each.
    
    The 2x4 matrix A has indices (i, j) with i in {0,1}, j in {0,1,2,3}.
    The 4x8 matrix B has indices (j, k) with j in {0,1,2,3}, k in {0,...,7}.
    The output C has indices (i, k) with i in {0,1}, k in {0,...,7}.
    
    We group by j-pairs: j-group 0 = {j=0, j=1}, j-group 1 = {j=2, j=3}
    And k-groups: k-group 0 = {k=0,1}, k-group 1 = {k=2,3}, k-group 2 = {k=4,5}, k-group 3 = {k=6,7}
    
    For each (j-group, k-group) pair, we have a 2x2 * 2x2 multiplication:
    A_block[i, j_local] * B_block[j_local, k_local] = C_block[i, k_local]
    
    Total: 2 j-groups * 4 k-groups = 8 sub-problems, each rank 7, total rank 56.
    """
    n, m, p = 2, 4, 8
    U_str, V_str, W_str = _build_strassen_2x2()
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=object)  # 8 x 56
    V = np.zeros((m * p, rank), dtype=object)  # 32 x 56
    W = np.zeros((p * n, rank), dtype=object)  # 16 x 56
    
    col = 0
    for jg in range(2):  # j-group: 0 or 1
        j_base = jg * 2  # j indices: j_base, j_base+1
        for kg in range(4):  # k-group: 0, 1, 2, 3
            k_base = kg * 2  # k indices: k_base, k_base+1
            
            for r in range(7):
                # U: A has index (i, j), flattened as i*m + j
                # For this sub-problem, A_block[i, j_local] where j_local in {0,1}
                # Global j = j_base + j_local
                for i in range(2):
                    for jl in range(2):
                        global_j = j_base + jl
                        U[i * m + global_j, col] = U_str[i, jl, r] if False else U_str[i, r] if jl == 0 else 0
                        # Wait, U_str is 4x7 where row index is the A element index
                        # A element index: i*2 + jl (for 2x2)
                        # But U_str[i, r] where i is the row index of A (0 or 1)
                        # Actually U_str is indexed by A's flat index: 0=A00, 1=A01, 2=A10, 3=A11
                        # In our 2x2 sub-problem, A[i, jl] has flat index i*2 + jl
                        # But U_str's row index corresponds to the flat index of the 2x2 A
                        # So U_str[i*2+jl, r] is the coefficient
                        U[i * m + global_j, col] = U_str[i * 2 + jl, r]
                
                # V: B has index (j, k), flattened as j*p + k
                # B_block[jl, kl] where jl in {0,1}, kl in {0,1}
                # Global j = j_base + jl, global k = k_base + kl
                for jl in range(2):
                    for kl in range(2):
                        global_j = j_base + jl
                        global_k = k_base + kl
                        V[global_j * p + global_k, col] = V_str[jl * 2 + kl, r]
                
                # W: C has index (i, k), flattened as k*n + i (trace convention W[k*n+i, r])
                # C_block[i, kl] where i in {0,1}, kl in {0,1}
                # Global i = i, global k = k_base + kl
                for i in range(2):
                    for kl in range(2):
                        global_k = k_base + kl
                        W[global_k * n + i, col] = W_str[i * 2 + kl, r]
                
                col += 1
    
    return U, V, W


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    try:
        U, V, W = _build_block_decomposition()
        
        # Convert to lists of fractions/floats
        U_list = [[int(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
        V_list = [[int(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
        W_list = [[int(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    
    except Exception:
        # Fallback to schoolbook if something goes wrong
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

# EVOLVE-BLOCK-END
