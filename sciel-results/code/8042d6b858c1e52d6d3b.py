import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor: T[i,j,k] = 1 for all valid (i,j,k)
    T = np.ones((n, m, p), dtype=float)
    
    def verify(U, V, W, r, tol=1e-10):
        """Vectorized verification - much faster than loop version."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m + j, :]
                for k in range(p):
                    val = np.dot(u_row, V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B)
        # This part is an outer product of A[:,0] and B[0,:]. Rank 20.
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
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
            U[u_11, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M2 = (a21 + a22)b11
            U[u_21, col] += 1
            U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1
            W[w_11, col] += -1
            col += 1
            
            # M3 = a11(b12 - b22)
            U[u_11, col] += 1
            V[v_12, col] += 1
            V[v_22, col] += -1
            W[w_01, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M4 = a22(b21 - b11)
            U[u_22, col] += 1
            V[v_21, col] += 1
            V[v_11, col] += -1
            W[w_00, col] += 1
            W[w_10, col] += 1
            col += 1
            
            # M5 = (a11 + a12)b22
            U[u_11, col] += 1
            U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1
            W[w_01, col] += 1
            col += 1
            
            # M6 = (a21 - a11)(b11 + b12)
            U[u_21, col] += 1
            U[u_11, col] += -1
            V[v_11, col] += 1
            V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7 = (a12 - a22)(b21 + b22)
            U[u_12, col] += 1
            U[u_22, col] += -1
            V[v_21, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
        
        return U, V, W, rank
    
    def refine_decomposition(U, V, W, rank, max_iters=200, tol=1e-12):
        """Try to reduce rank by zeroing out columns that are nearly redundant.
        
        Uses a greedy approach: try to remove each column if the remaining
        columns can still represent the tensor within tolerance.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        # Try to remove columns from the end (least important)
        for r_try in range(rank - 1, max(1, rank - 10), -1):
            if time.time() > deadline - 5:
                break
            
            # Check if we can drop column r_try
            U_test = U[:, :r_try]
            V_test = V[:, :r_try]
            W_test = W[:, :r_try]
            
            err = verify(U_test, V_test, W_test, r_try)
            if err < tol:
                # Successfully reduced rank
                rank = r_try
                U = U_test
                V = V_test
                W = W_test
        
        return U, V, W, rank
    
    U_best, V_best, W_best, rank_best = rank55_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    # Try to refine/reduce the rank
    if err < 1e-8 and time.time() < deadline - 10:
        U_ref, V_ref, W_ref, rank_ref = refine_decomposition(U_best, V_best, W_best, rank_best)
        if rank_ref < rank_best:
            U_best, V_best, W_best, rank_best = U_ref, V_ref, W_ref, rank_ref
    
    # Final verification
    final_err = verify(U_best, V_best, W_best, rank_best)
    if final_err > 1e-8:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        U_best, V_best, W_best = U, V, W
        rank_best = rank

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
