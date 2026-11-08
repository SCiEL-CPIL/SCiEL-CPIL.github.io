"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Try the Strassen-based rank-14 decomposition
    # Split the 4-dimensional inner product into two 2-dimensional blocks
    # Top block: j in {0,1}, Bottom block: j in {2,3}
    # Each block is a 2x2 x 6 multiplication, which has rank 7
    
    # Define the 7 Strassen terms for 2x2x2
    # Each term is (u, v, w) where u, v, w are 2x2 matrices
    # u[i,j] = coefficient of A[i,j], v[j,k] = coefficient of B[j,k], w[i,k] = output coefficient
    
    strassen_terms = [
        # (u, v, w) as flattened 2x2 matrices (row-major)
        # Term 1: M1 = (A00+A11)(B00+B11)
        ([1,0, 0,1],  # u: A00+A11
         [1,1, 1,1],  # v: B00+B11
         [1,0, 0,1]), # w: C00+C11
        # Term 2: M2 = (A10+A11)B00
        ([0,0, 1,1],  # u: A10+A11
         [1,0, 0,0],  # v: B00
         [0,0, 1,-1]), # w: C10-C11
        # Term 3: M3 = A00(B10-B11)
        ([1,0, 0,0],  # u: A00
         [0,1, 1,0],  # v: B10-B11  (B[1,0]-B[1,1] → v[1,0]=1, v[1,1]=-1)
         [0,1, 0,1]), # w: C01+C11
        # Term 4: M4 = A11(B01-B00)
        ([0,0, 0,1],  # u: A11
         [1,0, 0,1],  # v: B01-B00  (B[0,1]-B[0,0] → v[0,1]=1, v[0,0]=-1)
         [-1,0, 1,0]), # w: -C00+C10
        # Term 5: M5 = (A00+A01)B11
        ([1,1, 0,0],  # u: A00+A01
         [0,0, 0,1],  # v: B11
         [1,0, -1,0]), # w: C00-C11
        # Term 6: M6 = (A10-A00)(B00+B10)
        ([-1,0, 1,0],  # u: -A00+A10
         [1,1, 0,1],  # v: B00+B10  (B[0,0]+B[1,0] → v[0,0]=1, v[1,0]=1)
         [0,0, 0,1]), # w: C11
        # Term 7: M7 = (A01-A11)(B01+B11)
        ([0,1, -1,1],  # u: A01-A11
         [1,0, 1,1],  # v: B01+B11  (B[0,1]+B[1,1] → v[0,1]=1, v[1,1]=1)
         [0,-1, 0,1]), # w: -C00+C01
    ]
    
    # Verify the 2x2x2 decomposition
    def verify_2x2x2(terms):
        T = np.zeros((2,2,2))
        for u_flat, v_flat, w_flat in terms:
            u = np.array(u_flat).reshape(2,2)
            v = np.array(v_flat).reshape(2,2)
            w = np.array(w_flat).reshape(2,2)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        T[i,j,k] += u[i,j] * v[j,k] * w[i,k]
        return np.allclose(T, 1.0)
    
    if not verify_2x2x2(strassen_terms):
        # Fall back to schoolbook
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
    
    # Now construct the 2x4x6 decomposition using 14 terms (7 for top, 7 for bottom)
    rank = 14
    U = [[0] * rank for _ in range(n * m)]  # 8 x 14
    V = [[0] * rank for _ in range(m * p)]  # 24 x 14
    W = [[0] * rank for _ in range(p * n)]  # 12 x 14
    
    # Top block: j in {0,1}
    # A indices: i*4+j for i in {0,1}, j in {0,1} → rows 0,1,4,5
    # B indices: j*6+k for j in {0,1}, k in {0..5} → rows 0-5, 6-11
    # C indices: k*2+i for k in {0..5}, i in {0,1} → rows 0-11
    
    for t_idx in range(7):
        u_flat, v_flat, w_flat = strassen_terms[t_idx]
        u = np.array(u_flat).reshape(2,2)
        v = np.array(v_flat).reshape(2,2)
        w = np.array(w_flat).reshape(2,2)
        
        # Map to top block indices
        for i in range(2):
            for j in range(2):
                if u[i,j] != 0:
                    U[i*4 + j][t_idx] = u[i,j]
        
        for j in range(2):
            for k in range(2):
                if v[j,k] != 0:
                    # For 2x2x6, v[j,k] is the same for all 6 columns
                    for kk in range(6):
                        V[j*6 + kk][t_idx] = v[j,k]
        
        for i in range(2):
            for k in range(2):
                if w[i,k] != 0:
                    # For 2x2x6, w[i,k] is the same for all 6 columns
                    for kk in range(6):
                        W[kk*2 + i][t_idx] = w[i,k]
    
    # Bottom block: j in {2,3}
    # A indices: i*4+j for i in {0,1}, j in {2,3} → rows 2,3,6,7
    # B indices: j*6+k for j in {2,3}, k in {0..5} → rows 12-17, 18-23
    # C indices: k*2+i for k in {0..5}, i in {0,1} → rows 0-11
    
    for t_idx in range(7):
        u_flat, v_flat, w_flat = strassen_terms[t_idx]
        u = np.array(u_flat).reshape(2,2)
        v = np.array(v_flat).reshape(2,2)
        w = np.array(w_flat).reshape(2,2)
        
        col = t_idx + 7
        
        # Map to bottom block indices
        for i in range(2):
            for j in range(2):
                if u[i,j] != 0:
                    U[i*4 + (j+2)][col] = u[i,j]
        
        for j in range(2):
            for k in range(2):
                if v[j,k] != 0:
                    for kk in range(6):
                        V[(j+2)*6 + kk][col] = v[j,k]
        
        for i in range(2):
            for k in range(2):
                if w[i,k] != 0:
                    for kk in range(6):
                        W[kk*2 + i][col] = w[i,k]
    
    # Verify the full 2x4x6 decomposition
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
    
    if not np.allclose(T, 1.0):
        # Fall back to schoolbook
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
