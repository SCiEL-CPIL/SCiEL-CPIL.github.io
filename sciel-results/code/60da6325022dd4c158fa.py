import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    def build_rank55():
        """
        Construct a rank-55 exact tensor decomposition using Strassen's algorithm
        for the 2x2x2 sub-blocks.
        
        The matrix multiplication C = A * B where A is 2x3 and B is 3x10.
        We can think of this as:
        - For j=0: C[:,k] = A[:,0] * B[0,k] (20 terms, one per (i,k) pair)
        - For j=1,2: Use Strassen on 5 blocks of 2x2 sub-matrices
        
        Actually, let me reconsider. The tensor T[i,j,k] = 1 for all i,j,k.
        We need: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        Rank-55 construction:
        - Part A (j=0): 20 terms, each with U[i*3+0,r]=1, V[0*10+k,r]=1, W[k*2+i,r]=1
        - Part B (j=1,2): 35 terms using Strassen on 5 blocks of 2x2
        """
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (2 x 10 = 20 terms)
        for i in range(n):
            for k in range(p):
                U[i * m + 0, col] = 1
                V[0 * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks of 2x2
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1 = (a11 + a22)(b11 + b22)
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def compute_error(U, V, W, r):
        """Fast vectorized error computation"""
        err_max = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U[i*m + j, :]
                    v_row = V[j*p + k, :]
                    w_row = W[k*n + i, :]
                    val = np.dot(u_row, v_row * w_row)
                    err_max = max(err_max, abs(val - 1.0))
        return err_max
    
    def verify(U, V, W, r, tol=1e-8):
        err = compute_error(U, V, W, r)
        return err < tol, err
    
    # Start with known rank-55 solution
    U_best, V_best, W_best, rank_best = build_rank55()
    valid, err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to reduce rank using targeted search
    # Key insight: The tensor T[i,j,k] = 1 for all i,j,k has a simple structure.
    # For fixed j, the slice T[:,j,:] is a 2x10 all-ones matrix, which has rank 1.
    # This suggests we might be able to do better than 55.
    
    # Try rank-10 construction:
    # U[i*m+j, r] = 1 for all i,j,r (all ones)
    # V[j*p+k, r] = 1 if k == r else 0 (selector for k)
    # W[k*n+i, r] = 1 if k == r else 0 (selector for k)
    # Then sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] 
    # = sum_r 1 * (1 if k==r else 0) * (1 if k==r else 0)
    # = 1 if k < 10 else 0
    # This works for k=0..9!
    
    def build_rank10():
        """
        Rank-10 construction:
        U[i*m+j, r] = 1 for all i, j, r
        V[j*p+k, r] = 1 if k == r else 0
        W[k*n+i, r] = 1 if k == r else 0
        
        Verification:
        sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        = sum_r 1 * (1 if k==r else 0) * (1 if k==r else 0)
        = 1 (since exactly one term where r=k contributes 1*1*1)
        """
        rank = 10
        U = np.ones((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for r in range(rank):
            # V[j*p+k, r] = 1 if k == r
            for j in range(m):
                V[j*p + r, r] = 1
            
            # W[k*n+i, r] = 1 if k == r
            for i in range(n):
                W[r*n + i, r] = 1
        
        return U, V, W, rank
    
    # Try rank-10 first
    if time.time() < deadline - 10:
        U10, V10, W10, r10 = build_rank10()
        valid10, err10 = verify(U10, V10, W10, r10)
        
        if valid10:
            U_best, V_best, W_best, rank_best = U10, V10, W10, r10
    
    # Final verification and cleanup
    U_list = U_best.tolist()
    V_list = V_best.tolist()
    W_list = W_best.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
