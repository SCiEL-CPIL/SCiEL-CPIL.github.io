"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Strategy: Partition m=5 as 2+2+1 and p=6 as 2+2+2
    # For each (j-block, k-block) pair, use optimal rank for 2 x (j_size) x (k_size)
    # - 2x2x2: Strassen rank 7
    # - 2x2x1: schoolbook rank 4
    # - 2x1x2: schoolbook rank 4
    # - 2x1x1: schoolbook rank 2
    # Total: 4*7 + 2*4 + 2*4 + 1*2 = 46
    
    # j-blocks: indices 0-1, 2-3, 4
    # k-blocks: indices 0-1, 2-3, 4-5
    
    j_blocks = [(0, 1), (2, 3), (4,)]  # sizes 2, 2, 1
    k_blocks = [(0, 1), (2, 3), (4, 5)]  # sizes 2, 2, 2
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    for j_start, j_end in j_blocks:
        for k_start, k_end in k_blocks:
            j_size = len(j_blocks[j_blocks.index((j_start, j_end))]) if False else (j_end - j_start + 1)
            k_size = (k_end - k_start + 1)
            
            # Get actual indices
            j_indices = list(range(j_start, j_end + 1))
            k_indices = list(range(k_start, k_end + 1))
            
            if j_size == 2 and k_size == 2:
                # Strassen 2x2x2: rank 7
                # Strassen's algorithm for multiplying two 2x2 matrices
                # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
                # C = A*B
                # m1 = (a11+a22)(b11+b22)
                # m2 = (a21+a22)b11
                # m3 = a11(b12-b22)
                # m4 = a22(b21-b11)
                # m5 = (a11+a12)b22
                # m6 = (a21-a11)(b11+b12)
                # m7 = (a12-a22)(b21+b22)
                # C11 = m1+m4-m5+m7
                # C12 = m3+m5
                # C21 = m2+m4
                # C22 = m1-m2+m3+m6
                
                # For our case: i in {0,1}, j in j_indices (2 elements), k in k_indices (2 elements)
                # A[i, j], B[j, k] -> C[i, k]
                # This is a 2x2x2 tensor: A is 2x2 (i x j), B is 2x2 (j x k), C is 2x2 (i x k)
                
                # The 7 Strassen terms:
                # m1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
                # m2: (A[1,0]+A[1,1]) * B[0,0]
                # m3: A[0,0] * (B[0,1]-B[1,1])
                # m4: A[1,1] * (B[1,0]-B[0,0])
                # m5: (A[0,0]+A[0,1]) * B[1,1]
                # m6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
                # m7: (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
                
                # U vectors (for A[i,j]):
                # m1: A[0,0]+A[1,1] -> [1, 0, 1, 0] for (i,j) in (0,0),(0,1),(1,0),(1,1)
                # m2: A[1,0]+A[1,1] -> [0, 0, 1, 1]
                # m3: A[0,0] -> [1, 0, 0, 0]
                # m4: A[1,1] -> [0, 0, 0, 1]
                # m5: A[0,0]+A[0,1] -> [1, 1, 0, 0]
                # m6: A[1,0]-A[0,0] -> [-1, 0, 1, 0]
                # m7: A[0,1]-A[1,1] -> [0, 1, 0, -1]
                
                # V vectors (for B[j,k]):
                # m1: B[0,0]+B[1,1] -> [1, 0, 0, 1] for (j,k) in (0,0),(0,1),(1,0),(1,1)
                # m2: B[0,0] -> [1, 0, 0, 0]
                # m3: B[0,1]-B[1,1] -> [0, 1, 0, -1]
                # m4: B[1,0]-B[0,0] -> [0, 0, 1, -1]
                # m5: B[1,1] -> [0, 0, 0, 1]
                # m6: B[0,0]+B[0,1] -> [1, 1, 0, 0]
                # m7: B[1,0]+B[1,1] -> [0, 0, 1, 1]
                
                # W vectors (for C[i,k]):
                # C11 = m1+m4-m5+m7
                # C12 = m3+m5
                # C21 = m2+m4
                # C22 = m1-m2+m3+m6
                
                # W for each term t:
                # t=0 (m1): C11+=1, C22+=1 -> [1, 0, 0, 1]
                # t=1 (m2): C21+=1, C22+=-1 -> [0, 0, 1, -1]
                # t=2 (m3): C12+=1, C22+=1 -> [0, 1, 0, 1]
                # t=3 (m4): C11+=1, C21+=1 -> [1, 0, 1, 0]
                # t=4 (m5): C11+=-1, C12+=1 -> [-1, 1, 0, 0]
                # t=5 (m6): C22+=1 -> [0, 0, 0, 1]
                # t=6 (m7): C11+=1 -> [1, 0, 0, 0]
                
                U_222 = [
                    [1, 0, 1, 0],   # m1
                    [0, 0, 1, 1],   # m2
                    [1, 0, 0, 0],   # m3
                    [0, 0, 0, 1],   # m4
                    [1, 1, 0, 0],   # m5
                    [-1, 0, 1, 0],  # m6
                    [0, 1, 0, -1]   # m7
                ]
                V_222 = [
                    [1, 0, 0, 1],   # m1
                    [1, 0, 0, 0],   # m2
                    [0, 1, 0, -1],  # m3
                    [0, 0, 1, -1],  # m4
                    [0, 0, 0, 1],   # m5
                    [1, 1, 0, 0],   # m6
                    [0, 0, 1, 1]    # m7
                ]
                W_222 = [
                    [1, 0, 0, 1],   # m1: C11+=1, C22+=1
                    [0, 0, 1, -1],  # m2: C21+=1, C22+=-1
                    [0, 1, 0, 1],   # m3: C12+=1, C22+=1
                    [1, 0, 1, 0],   # m4: C11+=1, C21+=1
                    [-1, 1, 0, 0],  # m5: C11+=-1, C12+=1
                    [0, 0, 0, 1],   # m6: C22+=1
                    [1, 0, 0, 0]    # m7: C11+=1
                ]
                
                for t in range(7):
                    # U vector: 2x2 = 4 entries, indexed as [i*m+j] = [i*2+j]
                    u_vec = [0] * (n * m)  # 10 entries
                    for idx, val in enumerate(U_222[t]):
                        ii = idx // 2
                        jj = idx % 2
                        u_vec[ii * m + j_indices[jj]] = val
                    U_cols.append(u_vec)
                    
                    # V vector: 2x2 = 4 entries, indexed as [j*p+k]
                    v_vec = [0] * (m * p)  # 30 entries
                    for idx, val in enumerate(V_222[t]):
                        jj = idx // 2
                        kk = idx % 2
                        v_vec[j_indices[jj] * p + k_indices[kk]] = val
                    V_cols.append(v_vec)
                    
                    # W vector: 2x2 = 4 entries, indexed as [k*n+i]
                    w_vec = [0] * (p * n)  # 12 entries
                    for idx, val in enumerate(W_222[t]):
                        ii = idx // 2
                        kk = idx % 2
                        w_vec[k_indices[kk] * n + ii] = val
                    W_cols.append(w_vec)
                    
            elif j_size == 2 and k_size == 1:
                # 2x2x1: schoolbook, rank 4
                # C[i, k] = sum_j A[i,j] * B[j,k]
                # 4 terms: (i,j) pairs for fixed k
                for ii in range(2):
                    for jj in range(2):
                        u_vec = [0] * (n * m)
                        u_vec[ii * m + j_indices[jj]] = 1
                        U_cols.append(u_vec)
                        
                        v_vec = [0] * (m * p)
                        v_vec[j_indices[jj] * p + k_indices[0]] = 1
                        V_cols.append(v_vec)
                        
                        w_vec = [0] * (p * n)
                        w_vec[k_indices[0] * n + ii] = 1
                        W_cols.append(w_vec)
                        
            elif j_size == 1 and k_size == 2:
                # 2x1x2: schoolbook, rank 4
                for ii in range(2):
                    for kk in range(2):
                        u_vec = [0] * (n * m)
                        u_vec[ii * m + j_indices[0]] = 1
                        U_cols.append(u_vec)
                        
                        v_vec = [0] * (m * p)
                        v_vec[j_indices[0] * p + k_indices[kk]] = 1
                        V_cols.append(v_vec)
                        
                        w_vec = [0] * (p * n)
                        w_vec[k_indices[kk] * n + ii] = 1
                        W_cols.append(w_vec)
                        
            elif j_size == 1 and k_size == 1:
                # 2x1x1: schoolbook, rank 2
                for ii in range(2):
                    u_vec = [0] * (n * m)
                    u_vec[ii * m + j_indices[0]] = 1
                    U_cols.append(u_vec)
                    
                    v_vec = [0] * (m * p)
                    v_vec[j_indices[0] * p + k_indices[0]] = 1
                    V_cols.append(v_vec)
                    
                    w_vec = [0] * (p * n)
                    w_vec[k_indices[0] * n + ii] = 1
                    W_cols.append(w_vec)
                    
    rank = len(U_cols)
    
    # Convert to proper format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    # Verify the decomposition
    # C[i,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] ... wait, no.
    # The tensor decomposition is: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[k,i,t]
    # But that's not right either. Let me think about the convention.
    # 
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T_{i,j,k} = delta_{i,j,k} such that C = A*B means
    # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k] where T[i,j,k] = 1 if consistent
    # 
    # Actually, the standard tensor decomposition for matrix multiplication:
    # The multiplication tensor has entries T[i,j,k] = 1 for all valid (i,j,k)
    # A rank-r decomposition is: T[i,j,k] = sum_{t=1}^r U[i,j,t] * V[j,k,t] * W[k,i,t]
    # where W uses the trace convention W[k*n+i, t]
    #
    # So we need: for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    #
    # Let me verify our construction...
    
    # Quick verification with random matrices
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    
    C_computed = np.zeros((n, p))
    for t in range(rank):
        # Extract U_t, V_t, W_t
        U_t = np.array([U[i][t] for i in range(n * m)]).reshape(n, m)
        V_t = np.array([V[j][t] for j in range(m * p)]).reshape(m, p)
        W_t = np.array([W[k][t] for k in range(p * n)]).reshape(p, n)
        
        # Contribution to C: sum over i,j,k of U_t[i,j] * V_t[j,k] * W_t[k,i] * A[i,j] * B[j,k]
        # But actually the decomposition is:
        # C[i,k] = sum_j A[i,j] B[j,k] = sum_t (sum_j U_t[i,j] A[i,j]) (sum_k' V_t[j,k'] B[j,k']) ... 
        # No, that's not right.
        #
        # The correct interpretation: 
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # And C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # = sum_j sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] * A[i,j] * B[j,k]
        # = sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * A[i,j]) * (sum_j V[j*p+k,t] * B[j,k])
        #
        # Wait, that's not quite right either because U and V are indexed differently.
        # Let me re-read the problem statement.
        #
        # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "output factor follows the official trace convention W[k*n+i,r]"
        #
        # So U has shape (n*m, rank), indexed as U[i*m+j, t]
        # V has shape (m*p, rank), indexed as V[j*p+k, t]  
        # W has shape (p*n, rank), indexed as W[k*n+i, t]
        #
        # The tensor entry T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        #
        # And matrix multiplication: C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
        # = sum_j sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] * A[i,j] * B[j,k]
        
    # Let me just directly compute C[i,k] using the tensor definition
    for i in range(n):
        for k in range(p):
            total = 0
            for j in range(m):
                T_ijk = 0
                for t in range(rank):
                    T_ijk += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                total += T_ijk * A_test[i, j] * B_test[j, k]
            C_computed[i, k] = total
    
    error = np.max(np.abs(C_computed - C_expected))
    if error > 1e-10:
        # Fallback to schoolbook if verification fails
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
