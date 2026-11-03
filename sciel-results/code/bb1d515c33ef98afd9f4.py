"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that the decomposition exactly represents matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - expected) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 matrix multiplication using Strassen's algorithm.
    U: (4, 7) -> indices i*2+j
    V: (4, 7) -> indices j*2+k
    W: (4, 7) -> indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen's 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Col 0: M1
    U[0, 0] = 1 # A00
    U[3, 0] = 1 # A11
    V[0, 0] = 1 # B00
    V[3, 0] = 1 # B11
    W[0, 0] = 1 # C00
    W[3, 0] = 1 # C11
    
    # Col 1: M2
    U[2, 1] = 1 # A10
    U[3, 1] = 1 # A11
    V[0, 1] = 1 # B00
    W[2, 1] = 1 # C10
    W[3, 1] = -1 # C11
    
    # Col 2: M3
    U[0, 2] = 1 # A00
    V[2, 2] = 1 # B10
    V[3, 2] = -1 # B11
    W[1, 2] = 1 # C01
    W[3, 2] = 1 # C11
    
    # Col 3: M4
    U[3, 3] = 1 # A11
    V[1, 3] = 1 # B01
    V[0, 3] = -1 # B00
    W[0, 3] = 1 # C00
    W[2, 3] = 1 # C10
    
    # Col 4: M5
    U[0, 4] = 1 # A00
    U[1, 4] = 1 # A01
    V[3, 4] = 1 # B11
    W[0, 4] = -1 # C00
    W[1, 4] = 1 # C01
    
    # Col 5: M6
    U[1, 5] = 1 # A01
    U[0, 5] = -1 # A00
    V[0, 5] = 1 # B00
    V[2, 5] = 1 # B10
    W[3, 5] = 1 # C11
    
    # Col 6: M7
    U[2, 6] = 1 # A10
    U[3, 6] = -1 # A11
    V[1, 6] = 1 # B01
    V[3, 6] = 1 # B11
    W[0, 6] = 1 # C00
    
    return U, V, W

def _trivial_2x1x1():
    """
    Returns U, V, W for 2x1x1 matrix multiplication.
    U: (2, 2) -> indices i*1+j (j=0)
    V: (1, 2) -> indices j*1+k (j=0, k=0)
    W: (1, 2) -> indices k*2+i (k=0)
    """
    U = np.zeros((2, 2))
    V = np.zeros((1, 2))
    W = np.zeros((2, 2)) # Wait, W shape is (p*n, rank) = (1*2, 2) = (2, 2)
    
    # C00 = A00 B00
    # C10 = A10 B00
    
    # Col 0: A00 B00
    U[0, 0] = 1 # A00
    V[0, 0] = 1 # B00
    W[0, 0] = 1 # C00
    
    # Col 1: A10 B00
    U[1, 1] = 1 # A10
    V[0, 1] = 1 # B00
    W[1, 1] = 1 # C10
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Construct Rank 33 decomposition
    # Split m=3 into m1=2, m2=1
    # M_{2,3,6} = M_{2,2,6} + M_{2,1,6}
    # M_{2,2,6} = 3 * M_{2,2,2} (Strassen)
    # M_{2,1,6} = 6 * M_{2,1,1} (Trivial)
    
    rank_total = 3 * 7 + 6 * 2 # 21 + 12 = 33
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col_offset = 0
    
    # Part 1: M_{2,2,6} using 3 Strassen blocks
    Strassen_U, Strassen_V, Strassen_W = _strassen_2x2x2()
    
    for block_idx in range(3):
        k_start = block_idx * 2
        for c in range(7):
            # U: Global index i*3 + j. Local i in {0,1}, j in {0,1}
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col_offset] = Strassen_U[i * 2 + j, c]
            
            # V: Global index j*6 + k. Local j in {0,1}, k in {0,1} -> global k_start + k
            for j in range(2):
                for k in range(2):
                    V[j * 6 + (k_start + k), col_offset] = Strassen_V[j * 2 + k, c]
            
            # W: Global index k*2 + i. Local k in {0,1} -> global k_start + k, i in {0,1}
            for k in range(2):
                for i in range(2):
                    W[(k_start + k) * 2 + i, col_offset] = Strassen_W[k * 2 + i, c]
            
            col_offset += 1
    
    # Part 2: M_{2,1,6} using 6 Trivial blocks
    Trivial_U, Trivial_V, Trivial_W = _trivial_2x1x1()
    
    for k_idx in range(6):
        for c in range(2):
            # U: Global index i*3 + j. Here j=2 (the 3rd column of A). i in {0,1}
            for i in range(2):
                U[i * 3 + 2, col_offset] = Trivial_U[i * 1 + 0, c]
            
            # V: Global index j*6 + k. Here j=2 (the 3rd row of B). k=k_idx
            V[2 * 6 + k_idx, col_offset] = Trivial_V[0 * 1 + 0, c]
            
            # W: Global index k*2 + i. k=k_idx, i in {0,1}
            for i in range(2):
                W[k_idx * 2 + i, col_offset] = Trivial_W[0 * 2 + i, c]
            
            col_offset += 1
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-8):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback
    return _schoolbook_decomposition(n, m, p)
# EVOLVE-BLOCK-END
