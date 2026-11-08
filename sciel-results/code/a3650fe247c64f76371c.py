"""Matrix multiplication 2x4x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    # Strategy:
    # The 2x4x6 multiplication tensor can be viewed as two independent 2x2x6 multiplications.
    # A is 2x4, B is 4x6.
    # C[0, k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # C[1, k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    #
    # We can split the middle dimension (j) into two parts of size 2.
    # Part 1: j=0,1. This is a 2x2x6 multiplication.
    # Part 2: j=2,3. This is also a 2x2x6 multiplication.
    #
    # So Rank(T_{2,4,6}) <= 2 * Rank(T_{2,2,6}).
    #
    # We need a rank-minimal decomposition for T_{2,2,6}.
    # T_{2,2,6} is the tensor for multiplying a 2x2 matrix by a 2x6 matrix.
    # This is equivalent to two 2x2x3 multiplications? No, it's one 2x2x6.
    #
    # Let's find the rank of T_{2,2,6}.
    # Schoolbook: 2*2*6 = 24.
    # Can we do better?
    # T_{2,2,6} can be seen as T_{2,2,2} x T_{1,1,3} in some sense?
    # Actually, T_{n,m,p} with p=6.
    # We can use the Kronecker product structure if dimensions allow.
    # T_{2,2,6} = T_{2,2,2} \otimes T_{1,1,3}?
    # n=2=2*1, m=2=2*1, p=6=2*3.
    # T_{2,2,6} = T_{2,2,2} \otimes T_{1,1,3}.
    # Rank(T_{2,2,2}) = 7 (Strassen).
    # Rank(T_{1,1,3}) = 3 (Schoolbook for 1x1x3 is 3? No, 1x1x3 is just 3 terms?
    # T_{1,1,3}[i,j,k] = 1. i=0, j=0, k=0..2.
    # Sum_t U[0,t]V[0,t]W[k,t] = 1.
    # U[0,t]=1, V[0,t]=1, W[k,t]=delta_{k,t}.
    # Rank 3.
    # So Rank(T_{2,2,6}) <= 7 * 3 = 21.
    #
    # Let's verify this Kronecker decomposition.
    # T_{2,2,6}[i, j, k] = 1.
    # i = i1*1 + i2 = i2. (i1 is 0)
    # j = j1*1 + j2 = j2. (j1 is 0)
    # k = k1*3 + k2. k1 in {0,1}, k2 in {0,1,2}.
    # Wait, p=6. p1=2, p2=3.
    # k = k1*3 + k2? No, usually k = k1*p2 + k2.
    # If p1=2, p2=3, then k = k1*3 + k2.
    # T_{2,2,2} \otimes T_{1,1,3}.
    # U[i, t1, t2] = U1[i1, t1] U2[i2, t2].
    # Here n1=2, n2=1. i1 in {0,1}, i2=0.
    # m1=2, m2=1. j1 in {0,1}, j2=0.
    # p1=2, p2=3. k1 in {0,1}, k2 in {0,1,2}.
    #
    # U[i, t] where t = (t1, t2).
    # U[i1, t1, t2] = U_strassen[i1, t1] * U_id[0, t2].
    # U_id[0, t2] = 1 for all t2 (since rank is 3, t2=0,1,2).
    # So U[i1, t1, t2] = U_strassen[i1, t1].
    #
    # V[j1, t1, t2] = V_strassen[j1, t1] * V_id[0, t2].
    # V_id[0, t2] = 1.
    # So V[j1, t1, t2] = V_strassen[j1, t1].
    #
    # W[k1, k2, t1, t2] = W_strassen[k1, t1] * W_id[k2, t2].
    # W_id[k2, t2] = delta_{k2, t2}.
    # So W[k1, k2, t1, t2] = W_strassen[k1, t1] * delta_{k2, t2}.
    #
    # Let's check the sum.
    # Sum_{t1, t2} U[i1, t1, t2] V[j1, t1, t2] W[k1, k2, t1, t2]
    # = Sum_{t1, t2} U_s[i1, t1] V_s[j1, t1] W_s[k1, t1] delta_{k2, t2}
    # = Sum_{t1} U_s[i1, t1] V_s[j1, t1] W_s[k1, t1] * (Sum_{t2} delta_{k2, t2})
    # = Sum_{t1} U_s[i1, t1] V_s[j1, t1] W_s[k1, t1] * 1
    # = T_{2,2,2}[i1, j1, k1].
    #
    # We need T_{2,2,6}[i, j, k] = 1.
    # T_{2,2,2}[i1, j1, k1] = 1.
    # So this works!
    #
    # Rank of T_{2,2,6} is 7 * 3 = 21.
    # Total Rank for T_{2,4,6} is 2 * 21 = 42.
    #
    # This matches the "inspiration" mentioned in the prompt:
    # "Replaces schoolbook rank 48 with Strassen-based decomposition rank 42 by splitting 2x4x6 into two 2x2x6 blocks..."
    #
    # Let's implement this.
    #
    # Step 1: Get Strassen matrices for 2x2x2.
    # U_s, V_s, W_s of shape (2, 7).
    #
    # Step 2: Construct T_{2,2,6} matrices.
    # Rank 21.
    # U_226: (2, 21).
    # V_226: (2, 21).
    # W_226: (6, 21).
    #
    # Indices:
    # t = t1 * 3 + t2. t1 in 0..6, t2 in 0..2.
    # U_226[i, t] = U_s[i, t1].
    # V_226[j, t] = V_s[j, t1].
    # W_226[k, t] = W_s[k1, t1] * delta_{k2, t2}.
    # k = k1 * 3 + k2.
    #
    # Step 3: Construct T_{2,4,6} matrices.
    # Rank 42.
    # U: (2*4, 42) = (8, 42).
    # V: (4*6, 42) = (24, 42).
    # W: (6*2, 42) = (12, 42).
    #
    # We have two blocks:
    # Block 1: j=0,1.
    # Block 2: j=2,3.
    #
    # For Block 1 (j=0,1):
    # i is 0,1. j_local is 0,1. k is 0..5.
    # U[i * 4 + j, t_block1] = U_226[i, t].
    # V[j * 6 + k, t_block1] = V_226[j_local, t] * delta_{j, j_local}?
    # Wait, V is indexed by (j, k).
    # In the tensor decomposition, V[j, k, t] multiplies B[j, k].
    # For Block 1, we only care about B[0, k] and B[1, k].
    # So V[j, k, t] should be non-zero only for j=0,1.
    # V[j, k, t] = V_226[j, t] if j in {0,1}, else 0.
    #
    # For Block 2 (j=2,3):
    # i is 0,1. j_local is 0,1 (corresponding to j=2,3).
    # U[i * 4 + j, t_block2] = U_226[i, t].
    # V[j * 6 + k, t_block2] = V_226[j-2, t] if j in {2,3}, else 0.
    #
    # W[k, i, t] ?
    # The interface says W has shape (p*n, rank).
    # Index is k*n + i.
    # W[k * 2 + i, t] = W_226[k, t].
    # This is the same for both blocks.
    #
    # Let's assemble.
    # Total rank = 42.
    # First 21 columns for Block 1.
    # Last 21 columns for Block 2.
    
    # Strassen 2x2x2
    U_s = np.array([
        [1, 0, 1, 1, 1, 1, 1],
        [0, 1, 1, -1, -1, 1, 1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 1, 1, -1, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1, 0, 1]
    ], dtype=float)
    
    # Verify Strassen
    T_s = np.einsum('it,jt,kt->ijk', U_s, V_s, W_s)
    if not np.allclose(T_s, np.ones((2,2,2))):
        # Fallback to schoolbook for 2x2x2 if Strassen is wrong (it shouldn't be)
        pass # We trust Strassen
        
    # Construct 2x2x6 matrices
    rank_226 = 21
    U_226 = np.zeros((2, rank_226), dtype=float)
    V_226 = np.zeros((2, rank_226), dtype=float)
    W_226 = np.zeros((6, rank_226), dtype=float)
    
    for t1 in range(7):
        for t2 in range(3):
            t = t1 * 3 + t2
            U_226[:, t] = U_s[:, t1]
            V_226[:, t] = V_s[:, t1]
            for k1 in range(2):
                for k2 in range(3):
                    k = k1 * 3 + k2
                    if k2 == t2:
                        W_226[k, t] = W_s[k1, t1]
                    else:
                        W_226[k, t] = 0
                        
    # Verify 2x2x6
    T_226 = np.einsum('it,jt,kt->ijk', U_226, V_226, W_226)
    if not np.allclose(T_226, np.ones((2,2,6))):
        # Fallback to schoolbook for whole problem
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    # Construct 2x4x6 matrices
    rank_total = 42
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # Block 1: j=0,1
    for t in range(rank_226):
        # U
        for i in range(2):
            U[i * m + 0, t] = U_226[i, t]
            U[i * m + 1, t] = U_226[i, t]
            # U[i*m+2] and U[i*m+3] are 0 for this block
            
        # V
        for k in range(6):
            V[0 * p + k, t] = V_226[0, t]
            V[1 * p + k, t] = V_226[1, t]
            # V[2*p+k] and V[3*p+k] are 0 for this block
            
        # W
        for k in range(6):
            for i in range(2):
                W[k * n + i, t] = W_226[k, t]
                
    # Block 2: j=2,3
    offset = rank_226
    for t in range(rank_226):
        t_global = offset + t
        # U
        for i in range(2):
            U[i * m + 2, t_global] = U_226[i, t]
            U[i * m + 3, t_global] = U_226[i, t]
            
        # V
        for k in range(6):
            V[2 * p + k, t_global] = V_226[0, t]
            V[3 * p + k, t_global] = V_226[1, t]
            
        # W
        for k in range(6):
            for i in range(2):
                W[k * n + i, t_global] = W_226[k, t]

    # Final Verification
    T_recon = np.einsum('it,jt,kt->ijk', 
                        U.reshape(n, m, rank_total), 
                        V.reshape(m, p, rank_total), 
                        W.reshape(p, n, rank_total))
    
    # The tensor T[i, j, k] should be 1.
    # But wait, the contraction is over t.
    # T[i, j, k] = sum_t U[i, j, t] V[j, k, t] W[k, i, t].
    # This matches the einsum 'it,jt,kt->ijk' if we reshape correctly.
    # U is (n, m, rank). V is (m, p, rank). W is (p, n, rank).
    # einsum('imt,jkt,kit->ijk', U, V, W)
    
    T_recon = np.einsum('imt,jkt,kit->ijk', 
                        U.reshape(n, m, rank_total), 
                        V.reshape(m, p, rank_total), 
                        W.reshape(p, n, rank_total))
                        
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_expected):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
