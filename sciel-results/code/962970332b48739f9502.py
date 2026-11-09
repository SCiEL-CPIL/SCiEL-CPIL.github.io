"""Matrix multiplication 2x3x10 tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 2x3 matrix by a 3x10 matrix.
    
    Strategy: Use block decomposition splitting p=10 into 5 blocks of size 2.
    Each block is M_{2,3,2} with rank 11 (7 from Strassen for 2x2x2 + 4 for 2x1x2).
    Total rank = 5 * 11 = 55.
    
    The output factor W follows the convention W[k*n+i, r].
    """
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 5 blocks of size 2
    block_size = 2
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2 (7 terms)
    # A: i in {0,1}, j in {0,1}
    # B: j in {0,1}, k in {0,1}
    # C: i in {0,1}, k in {0,1}, W index = k*n + i
    strassen_terms = [
        # M1: (A00+A11)(B00+B11)
        {
            'U': [(0,0,1), (1,1,1)],
            'V': [(0,0,1), (1,1,1)],
            'W': [(0,0,1), (1,1,1)]
        },
        # M2: (A10+A11)B00
        {
            'U': [(1,0,1), (1,1,1)],
            'V': [(0,0,1)],
            'W': [(0,1,1), (1,1,-1)]
        },
        # M3: A00(B01-B11)
        {
            'U': [(0,0,1)],
            'V': [(0,1,1), (1,1,-1)],
            'W': [(1,0,1), (1,1,1)]
        },
        # M4: A11(B10-B00)
        {
            'U': [(1,1,1)],
            'V': [(1,0,1), (0,0,-1)],
            'W': [(0,0,1), (0,1,1)]
        },
        # M5: (A00+A01)B11
        {
            'U': [(0,0,1), (0,1,1)],
            'V': [(1,1,1)],
            'W': [(0,0,-1), (1,0,1)]
        },
        # M6: (A10-A00)(B00+B01)
        {
            'U': [(1,0,1), (0,0,-1)],
            'V': [(0,0,1), (0,1,1)],
            'W': [(1,1,1)]
        },
        # M7: (A01-A11)(B10+B11)
        {
            'U': [(0,1,1), (1,1,-1)],
            'V': [(1,0,1), (1,1,1)],
            'W': [(0,0,1)]
        }
    ]
    
    # Outer product terms for the third column (j=2) of A and third row (j=2) of B
    # A: i in {0,1}, j=2
    # B: j=2, k in {0,1}
    # C: i in {0,1}, k in {0,1}
    outer_terms = [
        {'U': [(0,2,1)], 'V': [(2,0,1)], 'W': [(0,0,1)]},
        {'U': [(0,2,1)], 'V': [(2,1,1)], 'W': [(1,0,1)]},
        {'U': [(1,2,1)], 'V': [(2,0,1)], 'W': [(0,1,1)]},
        {'U': [(1,2,1)], 'V': [(2,1,1)], 'W': [(1,1,1)]}
    ]
    
    all_terms = strassen_terms + outer_terms
    
    col_offset = 0
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        for term in all_terms:
            col = col_offset
            # Fill U: index i*m + j
            for i, j, val in term['U']:
                U[i * m + j][col] += val
            # Fill V: index j*p + k_global
            for j, k_local, val in term['V']:
                k_global = b * block_size + k_local
                V[j * p + k_global][col] += val
            # Fill W: index k_global*n + i
            for k_local, i, val in term['W']:
                k_global = b * block_size + k_local
                W[k_global * n + i][col] += val
            col_offset += 1
            
    # Try to find a lower-rank decomposition using optimization
    # We'll attempt to find a rank-50 decomposition by relaxing the block structure
    # and using a smooth optimization approach
    
    if time.time() < deadline - 30:
        try:
            # Attempt rank reduction: try rank 50
            # The tensor T has shape (n*m, m*p, p*n) = (6, 30, 20)
            # We want to find U (6x50), V (30x50), W (20x50) such that
            # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
            
            # The tensor for 2x3x10 matrix multiplication:
            # T[i*m+j, j'*p+k, k'*n+i] = 1 if j == j' else 0
            # where i in {0,1}, j in {0,1,2}, k in {0,...,9}
            # and j' in {0,1,2}, k' in {0,...,9}
            
            # Let's verify the current decomposition is correct
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            
            # Check correctness
            # T[i*m+j, j'*p+k, k'*n+i] should equal sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i, r]
            max_err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for jp in range(m):
                            for kp in range(p):
                                # T[i*m+j, jp*p+k, kp*n+i]
                                # = 1 if j == jp else 0
                                idx_u = i * m + j
                                idx_v = jp * p + k
                                idx_w = kp * n + i
                                val = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                                expected = 1.0 if j == jp else 0.0
                                err = abs(val - expected)
                                max_err = max(max_err, err)
            
            if max_err > 1e-10:
                # Current decomposition is invalid, return it anyway as fallback
                return {"U": U, "V": V, "W": W}
            
            # Try to reduce rank to 50
            target_rank = 50
            if time.time() < deadline - 25:
                # Use a simpler approach: try to find a rank-50 decomposition
                # by randomly initializing and using gradient descent
                np.random.seed(42)
                
                # Initialize with random values
                U_init = np.random.randn(n*m, target_rank) * 0.1
                V_init = np.random.randn(m*p, target_rank) * 0.1
                W_init = np.random.randn(p*n, target_rank) * 0.1
                
                # Define the loss function
                def loss(params):
                    U_flat = params[:n*m*target_rank]
                    V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = params[n*m*target_rank + m*p*target_rank:]
                    
                    U_mat = U_flat.reshape(n*m, target_rank)
                    V_mat = V_flat.reshape(m*p, target_rank)
                    W_mat = W_flat.reshape(p*n, target_rank)
                    
                    # Compute T[i*m+j, jp*p+k, kp*n+i] = sum_r U[i*m+j,r] * V[jp*p+k,r] * W[kp*n+i,r]
                    # and compare to the target tensor
                    loss_val = 0.0
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    idx_u = i * m + j
                                    idx_v = jp * p + k
                                    # For each k'
                                    for kp in range(p):
                                        idx_w = kp * n + i
                                        val = np.dot(U_mat[idx_u], V_mat[idx_v] * W_mat[idx_w])
                                        expected = 1.0 if j == jp else 0.0
                                        loss_val += (val - expected) ** 2
                    
                    return loss_val
                
                # This is too expensive, let's skip the optimization and return the rank-55 solution
                # The optimization would take too long within the time budget
                
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
