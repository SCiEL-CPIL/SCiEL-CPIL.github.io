"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Schoolbook decomposition as fallback (rank 54)
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    # Try to find a rank-50 decomposition using Strassen-like structure
    # Split the 3x9 matrix B into (2+1) rows: B = [B_top; B_bot] where B_top is 2x9, B_bot is 1x9
    # A is 2x3: A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # C = A * B = [A_left * B_top + A_right * B_bot]
    # 
    # For the 2x2 block (A_left * B_top), we can use Strassen's algorithm (rank 7 instead of 4)
    # For the 2x1 * 1x9 part (A_right * B_bot), we need 2*9 = 18 terms (schoolbook)
    # Total: 7*9 + 18 = 63 + 18 = 81... that's worse.
    #
    # Wait, let me reconsider. The tensor is for 2x3 * 3x9.
    # Let's think of it as: for each of the 2 output rows, we compute a 1x9 vector.
    # C[i, :] = sum_j A[i,j] * B[j, :]
    #
    # For i=0: C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:]
    # For i=1: C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:]
    #
    # Schoolbook: 2*3*9 = 54 terms.
    #
    # Let's try a different approach: group the 9 columns into 3 groups of 3.
    # C[:, 0:3] = A * B[:, 0:3], C[:, 3:6] = A * B[:, 3:6], C[:, 6:9] = A * B[:, 6:9]
    # Each 2x3 * 3x3 multiplication has rank at most 9 (schoolbook), but could be 7 (Strassen).
    # If we use Strassen for each 3x3 block: 3 blocks * 7 = 21 terms per output row? No...
    #
    # Actually, Strassen gives rank 7 for 2x2 matrix multiplication.
    # For 2x3 * 3x3, we can pad to 4x4 and use Strassen, but that's complex.
    #
    # Let me try a direct approach: assume rank 50 and try to find a valid decomposition.
    # We'll use a structured initialization based on the schoolbook and then perturb.
    
    # Try to find rank-50 decomposition
    target_rank = 50
    
    # Initialize with a structured approach:
    # Use 54 terms from schoolbook, but try to merge 4 of them
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    # First 50 terms: use schoolbook for first 50 (i,j,k) triples
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if col >= target_rank:
                    break
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        if col >= target_rank:
            break
    
    # We used 50 terms for the first 50 (i,j,k) triples
    # The remaining 4 triples need to be handled by the last 4 terms (indices 50-53... wait, we only have 50)
    # Actually, we have target_rank=50, so we used all 50 terms for the first 50 triples.
    # The last 4 triples (i,j,k) with index >= 50 need to be represented using the existing 50 terms.
    
    # This is getting complicated. Let me try a different approach:
    # Use random initialization and optimize
    
    rng = np.random.RandomState(42)
    
    # Try multiple random initializations
    for attempt in range(10):
        if time.time() > deadline - 30:
            break
        
        U_rand = rng.randn(n * m, target_rank) * 0.1
        V_rand = rng.randn(m * p, target_rank) * 0.1
        W_rand = rng.randn(p * n, target_rank) * 0.1
        
        # Optimize using gradient descent
        for iter in range(500):
            if time.time() > deadline - 30:
                break
            
            # Compute residual
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                        residual += (val - 1.0) ** 2
            
            if residual < 1e-10:
                break
            
            # Compute gradients (simplified - just do a small random step)
            step_size = 0.01 * (1 + residual) ** 0.5
            U_rand += step_size * rng.randn(n * m, target_rank)
            V_rand += step_size * rng.randn(m * p, target_rank)
            W_rand += step_size * rng.randn(p * n, target_rank)
        
        # Check if we found a valid solution
        max_res = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U_rand[i*m+j, r] * V_rand[j*p+k, r] * W_rand[k*n+i, r]
                    max_res = max(max_res, abs(val - 1.0))
        
        if max_res < 1e-6:
            # Round to half-integers
            U_rounded = np.round(U_rand * 2) / 2
            V_rounded = np.round(V_rand * 2) / 2
            W_rounded = np.round(W_rand * 2) / 2
            
            # Verify rounded solution
            max_res_r = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                        max_res_r = max(max_res_r, abs(val - 1.0))
            
            if max_res_r < 1e-6:
                best = {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                best_rank = target_rank
                return best
    
    # If we didn't find a rank-50 solution, return schoolbook
    return best
# EVOLVE-BLOCK-END
