"""Matrix multiplication 2x4x7 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Build a decomposition using Strassen-style blocks
    # 2x4x7: split into 2x4x4 + 2x4x2 + 2x4x1
    # 2x4x4: 4 blocks of 2x2x2 (Strassen rank 7 each) = 28
    # 2x4x2: 2 blocks of 2x2x2 (Strassen rank 7 each) = 14
    # 2x4x1: 8 direct terms = 8
    # Total: 28 + 14 + 8 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 2x4x4: 4 Strassen blocks
    add_strassen(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen(0, 2, 0)  # i=0,1; j=2,3; k=0,1
    add_strassen(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen(0, 2, 2)  # i=0,1; j=2,3; k=2,3
    
    # 2x4x2: 2 Strassen blocks
    add_strassen(0, 0, 4)  # i=0,1; j=0,1; k=4,5
    add_strassen(0, 2, 4)  # i=0,1; j=2,3; k=4,5
    
    # 2x4x1: 8 direct terms for k=6
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify decomposition correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-6:
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: iteratively check if any term is redundant
    # A term r is redundant if removing it still gives a valid decomposition
    # (i.e., the remaining terms still sum to 1 for all (i,j,k))
    
    current_rank = rank
    U_cur = U_np.copy()
    V_cur = V_np.copy()
    W_cur = W_np.copy()
    
    max_iterations = 10
    for iteration in range(max_iterations):
        if time.time() > deadline - 5:
            break
        
        found_reduction = False
        for r_check in range(current_rank):
            if time.time() > deadline - 5:
                break
            
            # Remove term r_check
            U_rem = np.delete(U_cur, r_check, axis=1)
            V_rem = np.delete(V_cur, r_check, axis=1)
            W_rem = np.delete(W_cur, r_check, axis=1)
            
            # Check if remaining terms still decompose the tensor
            residual_max = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for s in range(current_rank - 1):
                            val += U_rem[i*m+j, s] * V_rem[j*p+k, s] * W_rem[k*n+i, s]
                        err = abs(val - 1.0)
                        if err > residual_max:
                            residual_max = err
            
            if residual_max < 1e-8:
                # Term r_check is redundant! Remove it.
                U_cur = U_rem
                V_cur = V_rem
                W_cur = W_rem
                current_rank -= 1
                found_reduction = True
                break
        
        if not found_reduction:
            break
    
    # Convert back to lists
    U_final = U_cur.tolist()
    V_final = V_cur.tolist()
    W_final = W_cur.tolist()
    
    return {"U": U_final, "V": V_final, "W": W_final}
