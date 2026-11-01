import numpy as np
import time

DIMENSIONS = [2, 3, 5]
_START_TIME = time.time()
DEADLINE = 140.0  # Safety margin

def _elapsed():
    return time.time() - _START_TIME

def _verify(U, V, W, n, m, p, num_tests=5):
    """Verify the decomposition computes correct matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape[1] != V_arr.shape[1] or V_arr.shape[1] != W_arr.shape[1]:
        return False
        
    rank = U_arr.shape[1]
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.einsum('ijr,jkr,kir->ik', U_3d, V_3d, W_3d)
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    
    # Rank 28 decomposition
    # Structure:
    # 1. j=0 (1 col of B): 2x1 by 1x5. Rank 10.
    # 2. j=1,2 (2 cols of B): 2x2 by 2x5.
    #    Split k into 0,1 (2x2x2), 2,3 (2x2x2), 4 (2x2x1).
    #    Rank = 7 (Strassen) + 7 (Strassen) + 4 (Schoolbook) = 18.
    # Total Rank = 10 + 18 = 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M2: A22(B21-B11)
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = -1
        col += 1
        
        # M3: (A12-A22)B22
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M4: A11(B12-B22)
        U[(1 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (0 + k_off)][col] = -1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        W[(0 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M5: (A11+A12)B12
        U[(0 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = -1
        W[(1 + k_off) * n + (0 + i_off)][col] = 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(1 + i_off) * m + (0 + j_off)][col] = 1
        U[(0 + i_off) * m + (0 + j_off)][col] = -1
        V[(0 + j_off) * p + (0 + k_off)][col] = 1
        V[(0 + j_off) * p + (1 + k_off)][col] = 1
        W[(1 + k_off) * n + (1 + i_off)][col] = 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[(0 + i_off) * m + (1 + j_off)][col] = 1
        U[(1 + i_off) * m + (1 + j_off)][col] = -1
        V[(1 + j_off) * p + (0 + k_off)][col] = 1
        V[(1 + j_off) * p + (1 + k_off)][col] = 1
        W[(0 + k_off) * n + (0 + i_off)][col] = 1
        col += 1

    # Block 1: j=0 (Rank 10)
    # A is 2x1, B is 1x5.
    # C[i,k] = A[i,0] * B[0,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: j=1,2 (Rank 18)
    # A is 2x2 (cols 1,2), B is 2x5 (cols 0..4 of the sub-block? No, rows 1,2 of B)
    # Wait, the indices for B in the sub-block are rows 1 and 2 of the original B.
    # The columns of B are 0..4.
    
    # Sub-block k=0,1 (cols 0,1 of B): 2x2x2 Strassen
    add_strassen(0, 1, 0)
    
    # Sub-block k=2,3 (cols 2,3 of B): 2x2x2 Strassen
    add_strassen(0, 1, 2)
    
    # Sub-block k=4 (col 4 of B): 2x2x1 Schoolbook
    # C[i,4] = A[i,1]*B[1,4] + A[i,2]*B[2,4]
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify if time permits
    if _elapsed() < DEADLINE - 5:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook if verification fails (should not happen)
            rank_school = n * m * p
            U_s = [[0] * rank_school for _ in range(n * m)]
            V_s = [[0] * rank_school for _ in range(m * p)]
            W_s = [[0] * rank_school for _ in range(p * n)]
            col_s = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j][col_s] = 1
                        V_s[j * p + k][col_s] = 1
                        W_s[k * n + i][col_s] = 1
                        col_s += 1
            result = {"U": U_s, "V": V_s, "W": W_s}
            
    return result

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
