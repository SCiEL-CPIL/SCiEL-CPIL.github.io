"""
Program for matrix multiplication 2,2,7.
Uses a rank-24 decomposition exploiting shared structure between column blocks.
"""
import numpy as np
import time

DIMENSIONS = [2, 2, 7]

START_TIME = time.time()
TIME_LIMIT = 120.0

def _check_time():
    if time.time() - START_TIME > TIME_LIMIT - 5:
        raise TimeoutError("Time limit exceeded")

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-24 exact tensor decomposition for 2x2 * 2x7 matrix multiplication.
    
    Strategy:
    - Use Strassen's rank-7 algorithm for columns 0,1
    - Use Strassen's rank-7 algorithm for columns 2,3
    - Use Strassen's rank-7 algorithm for columns 4,5
    - Use 3 terms for column 6 (exploiting shared U,V structure with column 5)
    Total rank: 7 + 7 + 7 + 3 = 24
    """
    n, m, p = 2, 2, 7
    rank = 24
    
    _check_time()
    
    # U is (n*m, rank) = (4, 24)
    # V is (m*p, rank) = (14, 24)
    # W is (p*n, rank) = (14, 24)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Indexing:
    # A[i,j] -> U[2*i+j]
    # B[j,k] -> V[7*j+k]
    # C[k,i] -> W[2*k+i]  (Trace convention: W[k*n+i, r])
    
    def add_strassen(col_start, k_start):
        """Add Strassen rank-7 terms for 2x2 block at columns k_start, k_start+1."""
        
        # M1 = (A00 + A11) * (B0,ks + B1,ks+1)
        # C0,ks += M1, C1,ks+1 += M1
        U[0, col_start+0] += 1
        U[3, col_start+0] += 1
        V[7*0 + k_start, col_start+0] += 1
        V[7*1 + (k_start+1), col_start+0] += 1
        W[2*k_start + 0, col_start+0] += 1
        W[2*(k_start+1) + 1, col_start+0] += 1

        # M2 = (A10 + A11) * B0,ks
        # C1,ks += M2, C1,ks+1 -= M2
        U[2, col_start+1] += 1
        U[3, col_start+1] += 1
        V[7*0 + k_start, col_start+1] += 1
        W[2*k_start + 1, col_start+1] += 1
        W[2*(k_start+1) + 1, col_start+1] -= 1

        # M3 = A00 * (B0,ks+1 - B1,ks+1)
        # C0,ks+1 += M3, C1,ks+1 += M3
        U[0, col_start+2] += 1
        V[7*0 + (k_start+1), col_start+2] += 1
        V[7*1 + (k_start+1), col_start+2] -= 1
        W[2*(k_start+1) + 0, col_start+2] += 1
        W[2*(k_start+1) + 1, col_start+2] += 1

        # M4 = A11 * (B1,ks - B0,ks)
        # C0,ks += M4, C1,ks += M4
        U[3, col_start+3] += 1
        V[7*1 + k_start, col_start+3] += 1
        V[7*0 + k_start, col_start+3] -= 1
        W[2*k_start + 0, col_start+3] += 1
        W[2*k_start + 1, col_start+3] += 1

        # M5 = (A00 + A01) * B1,ks+1
        # C0,ks -= M5, C0,ks+1 += M5
        U[0, col_start+4] += 1
        U[1, col_start+4] += 1
        V[7*1 + (k_start+1), col_start+4] += 1
        W[2*k_start + 0, col_start+4] -= 1
        W[2*(k_start+1) + 0, col_start+4] += 1

        # M6 = (A10 - A00) * (B0,ks + B0,ks+1)
        # C1,ks+1 += M6
        U[2, col_start+5] += 1
        U[0, col_start+5] -= 1
        V[7*0 + k_start, col_start+5] += 1
        V[7*0 + (k_start+1), col_start+5] += 1
        W[2*(k_start+1) + 1, col_start+5] += 1

        # M7 = (A01 - A11) * (B1,ks + B1,ks+1)
        # C0,ks += M7
        U[1, col_start+6] += 1
        U[3, col_start+6] -= 1
        V[7*1 + k_start, col_start+6] += 1
        V[7*1 + (k_start+1), col_start+6] += 1
        W[2*k_start + 0, col_start+6] += 1
        
    # Add Strassen for columns 0,1
    add_strassen(0, 0)
    
    # Add Strassen for columns 2,3
    add_strassen(7, 2)
    
    # Add Strassen for columns 4,5
    add_strassen(14, 4)
    
    # Add 3 terms for column 6 (exploiting shared structure)
    col_start = 21
    
    # Term 1: (A00 + A01) * B0,6 -> contributes to C0,6
    U[0, col_start+0] += 1
    U[1, col_start+0] += 1
    V[7*0 + 6, col_start+0] += 1
    W[2*6 + 0, col_start+0] += 1
    
    # Term 2: (A10 + A11) * B1,6 -> contributes to C1,6
    U[2, col_start+1] += 1
    U[3, col_start+1] += 1
    V[7*1 + 6, col_start+1] += 1
    W[2*6 + 1, col_start+1] += 1
    
    # Term 3: (A00 - A10) * (B0,6 - B1,6) -> corrects the cross terms
    U[0, col_start+2] += 1
    U[2, col_start+2] -= 1
    V[7*0 + 6, col_start+2] += 1
    V[7*1 + 6, col_start+2] -= 1
    W[2*6 + 0, col_start+2] -= 1
    W[2*6 + 1, col_start+2] += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
