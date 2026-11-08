import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # Rank 71 solution:
    # Four Strassen blocks (rank 7 each) + Schoolbook for the rest (43 terms)
    # Blocks are defined by (i0, j0, k0) with size 2x2x2
    # We choose blocks that cover the densest part of the tensor to maximize Strassen usage.
    # Partition:
    # i in {0,1}, j in {0,1}, k in {0,1}
    # i in {0,1}, j in {2,3}, k in {0,1}
    # i in {0,1}, j in {0,1}, k in {2,3}
    # i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    num_strassen_blocks = len(blocks)
    total_strassen_rank = num_strassen_blocks * 7
    
    # Calculate number of schoolbook terms
    total_terms = n * m * p
    strassen_terms_count = num_strassen_blocks * 8
    schoolbook_terms_count = total_terms - strassen_terms_count
    
    total_rank = total_strassen_rank + schoolbook_terms_count
    
    # Initialize matrices with integers to avoid floating point issues
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        # Local indices 0,1 map to global i0, i0+1 etc.
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add Strassen blocks
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Schoolbook for the rest
    # Identify which (i,j,k) triples were covered by Strassen
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
                    
    # Verify correctness in debug mode (optional, but good for safety if time permits)
    # if time.time() - start_time < timeout - 10:
    #     U_np = np.array(U)
    #     V_np = np.array(V)
    #     W_np = np.array(W)
    #     # Check T_ijk = sum_r U_ijk_r V_jk_r W_ki_r == 1
    #     # This is O(n*m*p*rank) = 75*71 ~ 5000 ops, very fast.
    #     for i in range(n):
    #         for j in range(m):
    #             for k in range(p):
    #                 val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
    #                 if val != 1:
    #                     # Fallback to pure schoolbook if error (rank 75)
    #                     U_fallback = [[0]*75 for _ in range(n*m)]
    #                     V_fallback = [[0]*75 for _ in range(m*p)]
    #                     W_fallback = [[0]*75 for _ in range(p*n)]
    #                     c = 0
    #                     for i in range(n):
    #                         for j in range(m):
    #                             for k in range(p):
    #                                 U_fallback[i*m+j][c] = 1
    #                                 V_fallback[j*p+k][c] = 1
    #                                 W_fallback[k*n+i][c] = 1
    #                                 c += 1
    #                     return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
