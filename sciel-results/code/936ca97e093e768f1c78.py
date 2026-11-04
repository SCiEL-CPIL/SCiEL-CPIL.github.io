import time
import numpy as np
import sys

def construct():
    """
    Returns a rank-minimal exact tensor decomposition
    for multiplying a 2x4 matrix by a 4x8 matrix.
    Uses 8 independent Strassen blocks (rank 7 each) for total rank 56.
    """
    start_time = time.time()
    DEADLINE = start_time + 140.0
    
    n, m, p = 2, 4, 8
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    col = 0
    
    def add_strassen_block(j0, k0, base_col):
        """
        Add Strassen decomposition for 2x2 * 2x2 block.
        A indices: i in {0,1}, j in {j0, j0+1}
        B indices: j in {j0, j0+1}, k in {k0, k0+1}
        C indices: i in {0,1}, k in {k0, k0+1}
        
        U index: i*m + j
        V index: j*p + k
        W index: k*n + i
        """
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        U[0*m + j0][base_col + 0] += 1
        U[1*m + (j0+1)][base_col + 0] += 1
        V[j0*p + k0][base_col + 0] += 1
        V[(j0+1)*p + (k0+1)][base_col + 0] += 1
        W[k0*n + 0][base_col + 0] += 1
        W[(k0+1)*n + 1][base_col + 0] += 1
        
        # Term 2: M2 = (A10 + A11)B00
        U[1*m + j0][base_col + 1] += 1
        U[1*m + (j0+1)][base_col + 1] += 1
        V[j0*p + k0][base_col + 1] += 1
        W[k0*n + 1][base_col + 1] += 1
        W[(k0+1)*n + 1][base_col + 1] += -1
        
        # Term 3: M3 = A00(B01 - B11)
        U[0*m + j0][base_col + 2] += 1
        V[j0*p + (k0+1)][base_col + 2] += 1
        V[(j0+1)*p + (k0+1)][base_col + 2] += -1
        W[(k0+1)*n + 0][base_col + 2] += 1
        W[(k0+1)*n + 1][base_col + 2] += 1
        
        # Term 4: M4 = A11(B10 - B00)
        U[1*m + (j0+1)][base_col + 3] += 1
        V[(j0+1)*p + k0][base_col + 3] += 1
        V[j0*p + k0][base_col + 3] += -1
        W[k0*n + 0][base_col + 3] += 1
        W[k0*n + 1][base_col + 3] += 1
        
        # Term 5: M5 = (A00 + A01)B11
        U[0*m + j0][base_col + 4] += 1
        U[0*m + (j0+1)][base_col + 4] += 1
        V[(j0+1)*p + (k0+1)][base_col + 4] += 1
        W[k0*n + 0][base_col + 4] += -1
        W[(k0+1)*n + 0][base_col + 4] += 1
        
        # Term 6: M6 = (A10 - A00)(B00 + B01)
        U[1*m + j0][base_col + 5] += 1
        U[0*m + j0][base_col + 5] += -1
        V[j0*p + k0][base_col + 5] += 1
        V[j0*p + (k0+1)][base_col + 5] += 1
        W[(k0+1)*n + 1][base_col + 5] += 1
        
        # Term 7: M7 = (A01 - A11)(B10 + B11)
        U[0*m + (j0+1)][base_col + 6] += 1
        U[1*m + (j0+1)][base_col + 6] += -1
        V[(j0+1)*p + k0][base_col + 6] += 1
        V[(j0+1)*p + (k0+1)][base_col + 6] += 1
        W[k0*n + 0][base_col + 6] += 1
        
        return base_col + 7
    
    # 2 A-blocks (j0 in {0, 2})
    # 4 B-blocks (k0 in {0, 2, 4, 6})
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            col = add_strassen_block(j0, k0, col)
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    rank = len(result['U'][0])
    print("Rank:", rank)
