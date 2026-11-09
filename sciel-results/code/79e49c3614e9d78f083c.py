import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    # Try rank 45 first, fall back to 46 if verification fails
    for target_rank in [45, 46]:
        rank = target_rank
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        col = 0
        
        def add_strassen_block(ja, jb, ka, kb):
            nonlocal col
            # M1 = (A00+A11)(B00+B11)
            U[0*m+ja][col] = 1
            U[1*m+jb][col] = 1
            V[ja*p+ka][col] = 1
            V[jb*p+kb][col] = 1
            W[ka*n+0][col] = 1
            W[kb*n+1][col] = 1
            col += 1
            
            # M2 = (A10+A11)B00
            U[1*m+ja][col] = 1
            U[1*m+jb][col] = 1
            V[ja*p+ka][col] = 1
            W[ka*n+1][col] = 1
            W[kb*n+1][col] = -1
            col += 1
            
            # M3 = A00(B01-B11)
            U[0*m+ja][col] = 1
            V[ja*p+kb][col] = 1
            V[jb*p+kb][col] = -1
            W[kb*n+0][col] = 1
            W[kb*n+1][col] = 1
            col += 1
            
            # M4 = A11(B10-B00)
            U[1*m+jb][col] = 1
            V[jb*p+ka][col] = 1
            V[ja*p+ka][col] = -1
            W[ka*n+0][col] = 1
            W[ka*n+1][col] = 1
            col += 1
            
            # M5 = (A00+A01)B11
            U[0*m+ja][col] = 1
            U[0*m+jb][col] = 1
            V[jb*p+kb][col] = 1
            W[ka*n+0][col] = -1
            W[kb*n+0][col] = 1
            col += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[1*m+ja][col] = 1
            U[0*m+ja][col] = -1
            V[ja*p+ka][col] = 1
            V[ja*p+kb][col] = 1
            W[kb*n+1][col] = 1
            col += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[0*m+jb][col] = 1
            U[1*m+jb][col] = -1
            V[jb*p+ka][col] = 1
            V[jb*p+kb][col] = 1
            W[ka*n+0][col] = 1
            col += 1

        # Apply Strassen to 4 blocks
        add_strassen_block(0, 1, 0, 1)
        add_strassen_block(2, 3, 0, 1)
        add_strassen_block(0, 1, 2, 3)
        add_strassen_block(2, 3, 2, 3)
        
        # For rank 46: 18 remaining schoolbook terms
        # For rank 45: need to compress 18 terms into 17
        # Try to merge one pair of terms using a linear combination
        
        if target_rank == 46:
            # Schoolbook for remaining terms
            for i in [0, 1]:
                for k in [0, 1, 2, 3]:
                    U[i*m+4][col] = 1
                    V[4*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
            for i in [0, 1]:
                for j in [0, 1, 2, 3]:
                    U[i*m+j][col] = 1
                    V[j*p+4][col] = 1
                    W[4*n+i][col] = 1
                    col += 1
                    
            for i in [0, 1]:
                U[i*m+4][col] = 1
                V[4*p+4][col] = 1
                W[4*n+i][col] = 1
                col += 1
        else:
            # Try to compress: combine two terms that share structure
            # Terms for J=4, K=0 and J=4, K=1 can potentially be combined
            # C[k=0, i] += A[i,4]*B[4,0] and C[k=1, i] += A[i,4]*B[4,1]
            # These are independent, so we can't simply merge them.
            
            # Alternative: use a different block partition
            # Let's try: use Strassen on (0,1,2) x (0,1) with a 3x2x2 trick
            # Or accept that 45 may not work and fall through
            
            # For now, fill remaining 17 slots with best effort
            # This will likely fail verification, triggering fallback to 46
            for i in [0, 1]:
                for k in [0, 1, 2, 3]:
                    U[i*m+4][col] = 1
                    V[4*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
                    
            for i in [0, 1]:
                for j in [0, 1, 2, 3]:
                    U[i*m+j][col] = 1
                    V[j*p+4][col] = 1
                    W[4*n+i][col] = 1
                    col += 1
                    
            # Only 1 slot left for the (4,4,both_i) terms - can't do both
            # This will fail, so we'll fall back to 46
            U[0*m+4][col] = 1
            V[4*p+4][col] = 1
            W[4*n+0][col] = 1
            col += 1
        
        # Verify
        if col != rank:
            continue
            
        U_arr = U
        V_arr = V
        W_arr = W
        
        T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
        
        T_target = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        
        if np.allclose(T_recon, T_target, atol=1e-9, rtol=0):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Time check
        if time.time() > deadline:
            break
    
    # Fallback: rank 50 schoolbook
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][c] = 1
                V_school[j*p+k][c] = 1
                W_school[k*n+i][c] = 1
                c += 1
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
