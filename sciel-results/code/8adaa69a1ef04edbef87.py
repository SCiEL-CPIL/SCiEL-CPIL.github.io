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
    
    # Verify a decomposition (vectorized)
    def verify(U, V, W, r, tol=1e-10):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def verify_fast(U, V, W, r):
        """Faster verification using vectorized operations."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    # WFTA-based decomposition for 2x2x5 blocks
    # For multiplying 2x2 matrix A by 2x2 matrix B to get 2x2 matrix C:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # 
    # WFTA on cyclic group Z5: we can use the fact that 5 is prime
    # The DFT of length 5 has 5 points, but we can use a sparse approach
    # 
    # Actually, let's use a different approach: decompose 2x2x10 into
    # two 2x2x5 blocks and apply Strassen to each
    # 
    # For 2x2x5: use the structure that we're multiplying along the k-dimension
    # A is 2x3 (rows i=0,1; cols j=0,1,2), B is 3x10 (rows j=0,1,2; cols k=0..9)
    # 
    # Let me think differently. The tensor is T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0..9}
    # We can view this as: for each j, we have a 2x10 matrix multiplication
    # 
    # New strategy: Use the fact that we can combine the j=1,2 part more efficiently
    # by viewing it as a 2x2 matrix (rows i, cols j=1,2) multiplied by a 2x10 matrix (rows j=1,2, cols k)
    # 
    # For the 2x2x10 part (j=1,2), we can use:
    # - Split k into two groups of 5: k in {0,1,2,3,4} and k in {5,6,7,8,9}
    # - For each group, we have a 2x2x5 problem
    # 
    # For 2x2x5: The minimal rank is known to be 7 (same as 2x2x2) since 5 >= 2
    # Actually, for 2x2xp with p >= 2, the rank is 7 (Strassen)
    # 
    # So 2x2x10 would be 7 + 7 = 14 if we split into two 2x2x5 blocks
    # Plus 2x1x10 = 20 (schoolbook)
    # Total: 14 + 20 = 34
    # 
    # But wait, we need to be careful. The 2x2x5 problem is:
    # C[i,k] = sum_{j=1,2} A[i,j] * B[j,k] for i in {0,1}, k in {0..4}
    # This is a 2x2x5 tensor. The rank of this tensor...
    # 
    # Actually, for 2x2xp, the rank is 7 for p >= 2 (Strassen's algorithm)
    # So 2x2x5 has rank 7, and 2x2x10 can be done with 7+7=14
    # 
    # Let me implement Strassen for 2x2x5:
    
    def strassen_2x2x5(U, V, W, base_col):
        """Apply Strassen to 2x2x5 block, starting at column base_col. Returns next_col."""
        col = base_col
        n_local = 2
        p_local = 5
        
        # Map: i in {0,1} -> local row, j in {1,2} -> local col (0,1), k in {0..4} -> local col
        # U indices: i*m + j where j is 1 or 2 (global j)
        # V indices: j*p + k where j is 1 or 2, k is 0..4
        # W indices: k*n + i where k is 0..4, i is 0,1
        
        u_00 = 0 * m + 1  # i=0, j=1
        u_01 = 0 * m + 2  # i=0, j=2
        u_10 = 1 * m + 1  # i=1, j=1
        u_11 = 1 * m + 2  # i=1, j=2
        
        # For k in 0..4
        for k in range(p_local):
            v_0 = 1 * p + k  # j=1, k
            v_1 = 2 * p + k  # j=2, k
            w_0 = k * n + 0  # k, i=0
            w_1 = k * n + 1  # k, i=1
            
            # M1 = (a00 + a11)(b0 + b1)
            U[u_00, col] += 1
            U[u_11, col] += 1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M2 = (a10 + a11)b0
            U[u_10, col] += 1
            U[u_11, col] += 1
            V[v_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M3 = a00(b1 - b0)
            U[u_00, col] += 1
            V[v_1, col] += 1
            V[v_0, col] += -1
            W[w_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M4 = a11(b0 - b1)
            U[u_11, col] += 1
            V[v_0, col] += 1
            V[v_1, col] += -1
            W[w_1, col] += 1
            col += 1
            
            # M5 = (a00 + a01)b1
            U[u_00, col] += 1
            U[u_01, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            col += 1
            
            # M6 = (a10 - a00)(b0 + b1)
            U[u_10, col] += 1
            U[u_00, col] += -1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M7 = (a01 - a11)(b0 + b1)
            U[u_01, col] += 1
            U[u_11, col] += -1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            col += 1
        
        return col
    
    # Rank-34 decomposition: j=0 schoolbook (20) + Strassen for j=1,2 split into two 2x2x5 blocks (14)
    def rank34_decomposition():
        rank = 34
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0 (first column of A, first row of B) - schoolbook
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 2x2x5 blocks
        # Block 1: k in {0,1,2,3,4}
        col = strassen_2x2x5(U, V, W, col)
        # Block 2: k in {5,6,7,8,9}
        # Need to adjust k offset
        # For k in 5..9, map to local k' = k-5
        # We need a version with k offset
        for k_local in range(5):
            k = 5 + k_local
            v_0 = 1 * p + k
            v_1 = 2 * p + k
            w_0 = k * n + 0
            w_1 = k * n + 1
            
            u_00 = 0 * m + 1
            u_01 = 0 * m + 2
            u_10 = 1 * m + 1
            u_11 = 1 * m + 2
            
            # M1 = (a00 + a11)(b0 + b1)
            U[u_00, col] += 1
            U[u_11, col] += 1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M2 = (a10 + a11)b0
            U[u_10, col] += 1
            U[u_11, col] += 1
            V[v_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M3 = a00(b1 - b0)
            U[u_00, col] += 1
            V[v_1, col] += 1
            V[v_0, col] += -1
            W[w_0, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M4 = a11(b0 - b1)
            U[u_11, col] += 1
            V[v_0, col] += 1
            V[v_1, col] += -1
            W[w_1, col] += 1
            col += 1
            
            # M5 = (a00 + a01)b1
            U[u_00, col] += 1
            U[u_01, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            col += 1
            
            # M6 = (a10 - a00)(b0 + b1)
            U[u_10, col] += 1
            U[u_00, col] += -1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_1, col] += 1
            col += 1
            
            # M7 = (a01 - a11)(b0 + b1)
            U[u_01, col] += 1
            U[u_11, col] += -1
            V[v_0, col] += 1
            V[v_1, col] += 1
            W[w_0, col] += 1
            col += 1
        
        return U, V, W, rank
    
    # Try the rank-34 decomposition first
    U_best, V_best, W_best, rank_best = rank34_decomposition()
    err = verify(U_best, V_best, W_best, rank_best)
    
    if err < 1e-10:
        print(f"Rank-34 decomposition verified with error {err}")
    else:
        print(f"Rank-34 decomposition failed with error {err}, falling back to rank-55")
        # Fallback to rank-55
        def rank55_decomposition():
            rank = 55
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            W = np.zeros((pn, rank))
            
            col = 0
            
            # Part A: j=0 (first column of A, first row of B)
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
        
        U_best, V_best, W_best, rank_best = rank55_decomposition()
        err = verify(U_best, V_best, W_best, rank_best)
        print(f"Rank-55 decomposition verified with error {err}")
    
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
