"""Improved program for matrix multiplication 3x4x4 using structured block decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # The multiplication tensor: C[i,k] = sum_j A[i,j] * B[j,k]
    # U: (n*m, rank) = (12, rank), indexed by (i*m+j) = (i*4+j)
    # V: (m*p, rank) = (16, rank), indexed by (j*p+k) = (j*4+k)
    # W: (p*n, rank) = (12, rank), indexed by (k*n+i) = (k*3+i)
    #
    # Condition: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*3+i, r] = 1 for all i,j,k
    
    # Strategy: Use the identity that matrix multiplication can be expressed via
    # a smaller set of multiply-add operations using the Strassen trick on blocks.
    #
    # I'll use a known decomposition. For M_{3,4,4}, we can achieve rank 42 
    # or lower. Let me construct it explicitly.
    #
    # Approach: Partition j = {0,1,2,3} into j_left = {0,1} and j_right = {2,3}
    # Then C[i,k] = sum_{b=0}^{1} A[i,b]*B[b,k] + sum_{b=0}^{1} A[i,b+2]*B[b+2,k]
    #
    # This gives us two (3x2)x(2x4) subproblems. Each 3x2 * 2x4 can be done
    # in rank 3*2*4=24 schoolbook, or we can use Strassen-like tricks.
    #
    # For a 3x2 * 2x4 multiplication:
    # Let A' be 3x2, B' be 2x4, C' = A'*B' be 3x4
    # C'[i,k] = A'[i,0]*B'[0,k] + A'[i,1]*B'[1,k]
    #
    # This is like multiplying a vector (length 2) from the right of A' 
    # by a vector (length 2) from the left of B', summed over the 2 inner dim.
    # 
    # For fixed i and k: C'[i,k] = A'[i,0]*B'[0,k] + A'[i,1]*B'[1,k]
    # We can use 2 terms (one for each j) - that's the schoolbook.
    # Can we do better? Not really for 3x2x2 with 4 outputs per output row...
    #
    # Actually for the subproblem of 3x2 * 2x4:
    # The tensor is 3x2x4. Rank of M_{3,2,4}? 
    # By symmetry duality, rank(M_{3,2,4}) <= min(3*2*4, ...) 
    # 
    # Let me just use schoolbook for the subproblems and see: total rank = 2*3*2*4 = 48. Same.
    #
    # OK let me try a different decomposition strategy.
    #
    # I'll use the fact that we can "share" computations between output elements.
    #
    # Here's an idea: group the 4 output columns k into pairs, and use Strassen on 2x2x2.
    #
    # Let k be partitioned as k = {0,1} and k={2,3}.
    # For fixed k-pair (k0,k1 in {0,1} or {2,3}), C[:, k0] and C[:, k1] depend on all A and B.
    #
    # This is getting complex. Let me just do a numerical search with better initialization.
    
    def verify(U, V, W, tol=1e-9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank = U_arr.shape[1]
        
        T_rec = np.zeros((12, 16, 12))
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
        T_tgt = np.zeros((12, 16, 12))
        for i in range(3):
            for j in range(4):
                for k in range(4):
                    T_tgt[i*4+j, j*4+k, k*3+i] = 1.0
        
        diff = np.max(np.abs(T_rec - T_tgt))
        return diff < tol, diff
    
    # Schoolbook baseline
    rank_sb = 48
    U_sb = [[0.0]*rank_sb for _ in range(12)]
    V_sb = [[0.0]*rank_sb for _ in range(16)]
    W_sb = [[0.0]*rank_sb for _ in range(12)]
    c = 0
    for i in range(3):
        for j in range(4):
            for k in range(4):
                U_sb[i*4+j][c] = 1.0
                V_sb[j*4+k][c] = 1.0
                W_sb[k*3+i][c] = 1.0
                c += 1
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = rank_sb
    
    # Build target tensor
    T = np.zeros((12, 16, 12))
    for i in range(3):
        for j in range(4):
            for k in range(4):
                T[i*4+j, j*4+k, k*3+i] = 1.0
    
    # Try ranks below 48
    for rank in [39, 40, 41, 42, 43, 44, 45, 46, 47]:
        if rank >= best_rank:
            break
        if time.time() > deadline - 5:
            break
        
        for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
            if time.time() > deadline - 3:
                break
            
            np.random.seed(seed)
            U = np.random.randn(12, rank) * 0.01
            V = np.random.randn(16, rank) * 0.01
            W = np.random.randn(12, rank) * 0.01
            
            # ALS iterations
            for it in range(400):
                if time.time() > deadline - 3:
                    break
                    
                # Update U
                VW = np.einsum('br,cr->bc', V, W).reshape(192, rank)
                coeff = VW.T @ VW  # (rank, rank)
                rhs = (T.reshape(12, 192) @ VW).T  # (rank, 12)
                try:
                    U = np.linalg.solve(coeff, rhs).T
                except:
                    U = np.linalg.pinv(coeff) @ rhs.T.T if False else np.zeros((12, rank))
                
                # Update V
                UW = np.einsum('ar,cr->ac', U, W).reshape(144, rank)
                coeff = UW.T @ UW
                rhs = (T.reshape(16, 144) @ UW).T
                try:
                    V = np.linalg.solve(coeff, rhs).T
                except:
                    pass
                
                # Update W
                UV = np.einsum('ar,br->ab', U, V).reshape(192, rank)
                coeff = UV.T @ UV
                rhs = (T.reshape(12, 192) @ UV).T
                try:
                    W = np.linalg.solve(coeff, rhs).T
                except:
                    pass
            
            ok, diff = verify(U, V, W, tol=1e-8)
            if ok and rank < best_rank:
                best_rank = rank
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
