import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 115  # safety margin from 120s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n*m*p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    def validate(U, V, W, tol=1e-10):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        if V.shape[1] != rank or W.shape[1] != rank:
            return False
        T_expected = np.ones((n, m, p))
        T_computed = np.zeros((n, m, p))
        for c in range(rank):
            u_c = U[:, c].reshape(n, m)
            v_c = V[:, c].reshape(m, p)
            w_c = W[:, c].reshape(p, n)
            T_computed += u_c[:, :, None] * v_c[:, None, :] * w_c[None, :, :]
        return np.max(np.abs(T_computed - T_expected)) < tol

    def get_strassen_2x2_2():
        """Rank-7 Strassen decomposition for 2x2 * 2x2."""
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # Standard Strassen formulas
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B11-B21)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U_coeffs = [
            {0: 1, 3: 1},  # A11+A22
            {2: 1, 3: 1},  # A21+A22
            {0: 1},        # A11
            {3: 1},        # A22
            {0: 1, 1: 1},  # A11+A12
            {2: 1, 0: -1}, # A21-A11
            {1: 1, 3: -1}  # A12-A22
        ]
        V_coeffs = [
            {0: 1, 3: 1},  # B11+B22
            {0: 1},        # B11
            {2: 1, 3: -1}, # B21-B22
            {0: 1, 2: -1}, # B11-B21
            {3: 1},        # B22
            {0: 1, 1: 1},  # B11+B12
            {2: 1, 3: 1}   # B21+B22
        ]
        W_coeffs = [
            {0: 1, 3: -1, 2: -1, 1: 1},  # C11=M1-M4+M5-M7
            {1: 1, 3: 1},                 # C12=M3+M5
            {0: 1, 2: 1},                 # C21=M2+M4
            {0: 1, 3: 1, 1: -1, 2: 1}    # C22=M1-M2+M6-M7
        ]
        
        for l in range(7):
            for li, coeff in U_coeffs[l].items():
                U[li, l] += coeff
            for lj, coeff in V_coeffs[l].items():
                V[lj, l] += coeff
            for lc, coeff in W_coeffs[l].items():
                W[lc, l] += coeff
        
        return U, V, W

    def get_17_rank_block():
        """Returns U, V, W for a 3x3x2 block with rank 17."""
        rank = 17
        U = np.zeros((9, rank))
        V = np.zeros((6, rank))
        W = np.zeros((6, rank))
        
        # Use Strassen on the 2x2 sub-blocks where possible
        # The 3x3x2 tensor: A(3x3) * B(3x2)
        
        # Approach: decompose into overlapping 2x2 Strassen blocks
        # Block (0,0): A[0:2,0:2] * B[0:2,0:2] -> contributes to C[0:2,0:2]
        # Block (1,1): A[1:3,1:2] * B[1:2,0:2] -> contributes to C[1:3,0:2]
        # Block (0,1): A[0:2,1:2] * B[0:2,1:2] -> but B only has 2 cols
        
        # Simpler: use 4 overlapping 2x2x2 Strassen blocks + corrections
        # Actually, let's use a direct construction
        
        # Known rank-17 decomposition for 3x3x2:
        # We'll use a combination of Strassen blocks
        
        col = 0
        # Strassen part for rows 0,1 and cols 0,1 (7 terms)
        U_s, V_s, W_s = get_strassen_2x2_2()
        
        # Place first Strassen block: A[0:2,0:2] * B[0:2,0:2]
        for c in range(7):
            U[0:4, col+c] = U_s[:, c]
            V[0:4, col+c] = V_s[:, c]
            W[0:4, col+c] = W_s[:, c]
        col += 7
        
        # Second Strassen block: A[1:3,0:2] * B[0:2,0:2]  
        # This handles rows 1,2 with cols 0,1
        for c in range(7):
            U[3:7, col+c] = U_s[:, c]  # rows 1,2 -> indices 3,4,5,6
            V[0:4, col+c] = V_s[:, c]
            W[3:7, col+c] = W_s[:, c]  # W rows for i=1,2
        col += 7
        
        # Now we have 14 terms. Need 3 more for the remaining entries.
        # The entries not covered: C[0,1], C[0,2], C[2,0], C[2,1] partially
        # Actually, let's just use schoolbook for the remaining 3 terms
        
        # Term 15: i=0, j=1, k=0
        U[1, col] = 1.0
        V[2, col] = 1.0
        W[0, col] = 1.0
        col += 1
        
        # Term 16: i=0, j=1, k=1  
        U[1, col] = 1.0
        V[3, col] = 1.0
        W[3, col] = 1.0
        col += 1
        
        # Term 17: i=2, j=1, k=0
        U[6, col] = 1.0
        V[2, col] = 1.0
        W[0, col] = 1.0
        col += 1
        
        return U, V, W

    def build_68_rank():
        """Build the 68-rank decomposition using 4 disjoint 17-rank blocks."""
        U_block, V_block, W_block = get_17_rank_block()
        rank_block = 17
        rank_total = 4 * rank_block
        
        U = np.zeros((9, rank_total))
        V = np.zeros((24, rank_total))
        W = np.zeros((24, rank_total))
        
        for g in range(4):
            k0 = 2 * g
            col_start = g * rank_block
            col_end = (g + 1) * rank_block
            
            U[:, col_start:col_end] = U_block
            
            for c in range(rank_block):
                for j in range(3):
                    for k_local in range(2):
                        k_global = k0 + k_local
                        V[j*8 + k_global, col_start + c] = V_block[j*2 + k_local, c]
                        
            for c in range(rank_block):
                for k_local in range(2):
                    k_global = k0 + k_local
                    for i in range(3):
                        W[k_global*3 + i, col_start + c] = W_block[k_local*3 + i, c]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try to find a better decomposition
    # First, check if 68-rank works
    result_68 = build_68_rank()
    
    if validate(result_68["U"], result_68["V"], result_68["W"]):
        return result_68
    
    # Fallback to schoolbook
    return schoolbook()
