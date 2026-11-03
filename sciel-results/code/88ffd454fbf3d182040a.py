import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    S = np.zeros((n*m, m*p, p*n))
    rank = U.shape[1]
    for c in range(rank):
        S += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
    
    return np.allclose(S, T, atol=1e-9)

def construct():
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Strategy: Use block decomposition with Strassen for 2x2x2 sub-blocks
    # 
    # The 2x3x8 problem: A is 2x3, B is 3x8
    # Split B's columns into groups: cols 0-1, cols 2-3, cols 4-5, cols 6-7
    # But we need to handle the 3-row structure of A
    #
    # Alternative: Split A's columns into (0,1) and (1,2)
    # Let A = [A_left | A_right] where A_left is 2x2 (cols 0,1), A_right is 2x2 (cols 1,2)
    # B = [B_left | B_mid | B_right] where B_left is 3x4 (cols 0,1), B_mid is 3x4 (cols 2,3), etc.
    #
    # Better approach: Use the known 2x3x8 decomposition
    # 
    # Let me use a systematic approach:
    # Split the 8 columns of B into 4 groups of 2: B1 (cols 0,1), B2 (cols 2,3), B3 (cols 4,5), B4 (cols 6,7)
    # A has 3 columns: a0, a1, a2
    # 
    # For each group of 2 columns in B, we compute:
    # C[:, 2g:2g+2] = A * B[:, 2g:2g+2]
    # = a0 * B0[:,2g:2g+2] + a1 * B1[:,2g:2g+2] + a2 * B2[:,2g:2g+2]
    #
    # This is a sum of 3 terms, each being a scalar times a 2x2 matrix.
    # Total schoolbook: 4 groups * 3 terms * 4 mults = 48
    #
    # To reduce: Use Strassen-like combinations for the 3-term sum
    # For each group, we need to compute:
    # X = a0*B0 + a1*B1 + a2*B2 (2x2 matrices)
    #
    # We can use the fact that for 2x2 matrices, we can combine the scalar multiplications
    # using precomputed combinations.
    #
    # Actually, let me use a different approach based on known results.
    # The tensor rank of 2x3x8 is at most 44 (using Strassen for sub-blocks).
    #
    # Let me construct it explicitly using the following strategy:
    # 
    # View the problem as: C = A * B where A is 2x3, B is 3x8
    # 
    # Define intermediate products using Strassen-style combinations:
    # 
    # For the 2x3 part, we can write:
    # Let A = [[a00, a01, a02], [a10, a11, a12]]
    # 
    # We'll use 7 Strassen-like products for a "2x2x2" core, then extend.
    #
    # Simpler: Use the decomposition from the literature.
    # 2x3x8 can be done in rank 44 by:
    # - Splitting into two 2x2x8 subproblems (sharing column 1)
    # - Each 2x2x8 uses rank 28 (Strassen on 2x2x2 blocks)
    # - But with sharing, total is less
    #
    # Let me just try gradient descent to find a low-rank decomposition,
    # starting from a structured initialization.
    
    # First, try the structured 44-rank approach
    # 
    # For 2x2x8: Split 8 cols into 4 pairs, each pair uses Strassen (7 mults)
    # So 2x2x8 has rank 4*7 = 28
    # 
    # For 2x3x8: 
    # C = A * B = [A0 A1 A2] * B
    # = A0*B_col0 + A1*B_col1 + A2*B_col2 (where Bj is 2x8 submatrix from row j of B)
    #
    # This is 3 scalar-vector products, but we can combine.
    #
    # Let me use a direct construction:
    # 
    # The key is that we can precompute combinations of rows of A.
    # 
    # Let me use the following 44-rank construction:
    # 
    # For each of the 4 column-pairs of B (cols 0-1, 2-3, 4-5, 6-7):
    #   We need to compute C[:, 2g:2g+2] = A * B[:, 2g:2g+2]
    #   where B[:, 2g:2g+2] is 3x2
    #   
    #   Let Bg = B[:, 2g:2g+2] = [[b00, b01], [b10, b11], [b20, b21]]
    #   
    #   C[:, 2g:2g+2] = [[a00*b00 + a01*b10 + a02*b20, a00*b01 + a01*b11 + a02*b21],
    #                     [a10*b00 + a11*b10 + a12*b20, a10*b01 + a11*b11 + a12*b21]]
    #
    #   This is 8 multiplications per group (schoolbook), 32 total for 4 groups.
    #   Plus we need to handle the A entries, but those are just coefficients.
    #
    #   To use Strassen: For each 2x2x2 sub-problem (one column pair, one row of A),
    #   we can use 7 mults instead of 8. But we have 3 rows of A, so 3*7 = 21 per group?
    #   No, that's not right either.
    #
    # Let me just do gradient descent with a good initialization.
    
    best_rank = n * m * p  # 48
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook as fallback
    U_sb = np.zeros((n*m, best_rank))
    V_sb = np.zeros((m*p, best_rank))
    W_sb = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    # Try to find lower rank using gradient descent
    # Start from rank 44 and work down
    
    for target_rank in range(44, 30, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(trial * 7919 + target_rank * 131)
            scale = 0.3
            U = np.random.randn(n*m, target_rank) * scale
            V = np.random.randn(m*p, target_rank) * scale
            W = np.random.randn(p*n, target_rank) * scale
            
            lr = 0.05
            prev_loss = np.inf
            for step in range(500):
                if time.time() > deadline - 5:
                    break
                
                # Compute residual
                S = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    S += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                R = T - S
                loss = np.sum(R**2)
                
                if loss < 1e-10:
                    break
                if loss > prev_loss * 1.01 and step > 50:
                    lr *= 0.5
                prev_loss = loss
                
                # Gradients
                dU = np.einsum('abd,bc,dc->ac', R, V, W)
                dV = np.einsum('abd,ac,dc->bc', R, U, W)
                dW = np.einsum('abd,ac,bc->dc', R, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check convergence
            S = np.zeros((n*m, m*p, p*n))
            for c in range(target_rank):
                S += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            R = T - S
            if np.linalg.norm(R) < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                S_r = np.zeros((n*m, m*p, p*n))
                for c in range(target_rank):
                    S_r += np.einsum('a,b,c->abc', U_r[:,c], V_r[:,c], W_r[:,c])
                if np.allclose(S_r, T, atol=1e-9):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    found = True
                    break
        
        if found:
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
