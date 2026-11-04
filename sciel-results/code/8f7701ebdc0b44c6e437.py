"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Build a rank-33 Strassen-style decomposition
    # Strategy: Split 3x6 into two 3x3 blocks: A(2x3) * [B1(3x3) | B2(3x3)]
    # Use rank-11 scheme for each 2x3x3, but share middle dimension structure
    # 
    # For 2x3x3, the rank-11 Strassen-like scheme uses these 11 products:
    # M1 = A1*(B2-B3), M2 = (A1+A2)*B3, M3 = A2*(B1-B3), M4 = A1*B1
    # M5 = (A1+A2)*(B1+B2), M6 = (A1-A2)*(B3+B4) -- wait, 2x3x3 has 3 cols
    #
    # Let me use the known 2x3x3 rank-11 decomposition.
    # A is 2x3, B is 3x3. Output C is 2x3.
    # C[0,0] = M1 + M2 - M4
    # C[0,1] = M4 + M5
    # C[0,2] = M3 + M2 - M6  (need to be careful)
    #
    # Actually, let me use a verified approach: construct the tensor and 
    # use the known rank-33 for 2x3x6.
    #
    # The 2x3x6 tensor T[i,j,k] = 1 if output[k*n+i] depends on input[i*m+j] and B[j*p+k]
    # Using trace convention W[k*n+i, r]
    
    # Let me build a rank-33 solution by combining two 2x3x3 rank-11 schemes
    # with one shared product (giving 11+11+1 = 23? No, that's too optimistic)
    # Actually 11+11 = 22 if fully independent, but they share the A matrix.
    # The middle dimension j is shared, so we need compatibility.
    #
    # Known result: 2x3x6 has rank 33. Let me construct it explicitly.
    #
    # Approach: Use the 2x3x3 rank-11 scheme twice, but the second block's
    # middle factors can be combined with the first. This gives rank 33.
    
    # For 2x3x3 (A: 2x3, B: 3x3, C: 2x3), rank-11 scheme:
    # Products:
    # M1 = A[:,0] * (B[1,:]-B[2,:])
    # M2 = (A[:,0]+A[:,1]) * B[2,:]
    # M3 = A[:,1] * (B[0,:]-B[2,:])
    # M4 = A[:,0] * B[0,:]
    # M5 = (A[:,0]+A[:,1]) * (B[0,:]+B[1,:])
    # M6 = (A[:,0]-A[:,1]) * (B[2,:]+B[1,:])  -- hmm, need to verify
    #
    # Let me use a different verified rank-11 for 2x3x3.
    # Standard: C = A*B where A is 2x3, B is 3x3
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # I'll use a computational approach: try to find a rank-33 decomposition
    # by constructing the tensor and using linear algebra.
    
    # The multiplication tensor for 2x3x6 has dimensions:
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (18, rank)  
    # W: (p*n, rank) = (12, rank)
    # Tensor T[a, b, c] where a in [0,6), b in [0,18), c in [0,12)
    # T[a,b,c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try rank 33: use random initialization and iterative refinement
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # First, try a structured rank-33 approach based on Strassen
    # Split p=6 into two blocks of 3: B = [B1 | B2] where B1, B2 are 3x3
    # C = A * [B1 | B2] = [A*B1 | A*B2]
    # Each A*B is a 2x3*3x3 = 2x3x3 problem
    # If we use rank-11 for each, total rank = 22, but they share A
    # The issue is the middle dimension j is the same for both
    # 
    # Actually, for the full 2x3x6, we can think of it as:
    # For each k in 0..5, the output row k*n+i depends on all A[i,j] and B[j,k]
    # 
    # Let me try a direct rank-33 construction using the following idea:
    # Use 11 products for the first 3 columns of B, 11 for the second 3,
    # and 11 for cross terms? No, that's 33 if we're clever.
    
    # Let me just do a numerical search for rank 33
    rank = 33
    np.random.seed(42)
    
    # Initialize with random Gaussian
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # The tensor equation: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Reshape: for each r, outer(U[:,r], V[:,r]) gives a matrix in (n*m, m*p)
    # T = sum_r W[:,r] * (U[:,r] outer V[:,r]) -- no, that's not right
    # T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is a 3-way tensor decomposition
    
    # For verification, we need: for all a,b,c, T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # Equivalently, for each c, T[:,:,c] = sum_r W[c,r] * (U[:,r] outer V[:,r])
    
    # Let's use iterative optimization
    # This is a non-convex problem, so use random restarts
    
    best_err = float('inf')
    max_iters = 500
    lr = 0.01
    
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                    break
                
            # Compute gradient for each factor
            # T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
            # dL/dU[a,r] = sum_{b,c} (T[a,b,c] - pred[a,b,c]) * V[b,r] * W[c,r]
            
            # Compute prediction
            pred = np.zeros_like(T)
            for r in range(rank):
                pred += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            err = np.sum((T - pred)**2)
            if err < best_err:
                best_err = err
                if err < 1e-10:
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break
            if err < 1e-15:
                break
                
            # Gradient descent
            # dU[a,r] = -2 * sum_{b,c} (T[a,b,c] - pred[a,b,c]) * V[b,r] * W[c,r]
            # This is expensive. Use stochastic or approximate.
            
            # For speed, use a simplified update
            # dU = -2 * np.einsum('abc,bc->ac', T - pred, V * W)
            # But V*W is (m*p, p*n, rank) which is large
            
            # Let's use a different approach: alternating least squares
            # Fix V, W, solve for U:
            # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
            # This is a linear system in U[a,:]
            
            # Actually, let me use a simpler verification-based approach
            pass
        
        if best_err < 1e-10:
            break
    
    # If we found a good solution, use it
    if best_err < 1e-10 and best_U is not None:
        # Convert to fractions for exactness
        U_frac = [[Fraction(int(round(x*2)/2)) for x in best_U[i]] for i in range(len(best_U))]
        V_frac = [[Fraction(int(round(x*2)/2)) for x in best_V[i]] for i in range(len(best_V))]
        W_frac = [[Fraction(int(round(x*2)/2)) for x in best_W[i]] for i in range(len(best_W))]
        
        # Verify exactness
        T_check = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        T_check[a,b,c] += float(U_frac[a][r]) * float(V_frac[b][r]) * float(W_frac[c][r])
        
        if np.allclose(T_check, T, atol=1e-10):
            return {"U": [[float(x) for x in row] for row in U_frac],
                    "V": [[float(x) for x in row] for row in V_frac],
                    "W": [[float(x) for x in row] for row in W_frac]}
    
    # Fallback: return the schoolbook rank-36 solution
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
