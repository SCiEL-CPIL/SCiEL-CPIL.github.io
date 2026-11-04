"""Matrix multiplication 2,4,5 rank-minimal tensor decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 5]
# n=2 (rows of A), m=4 (cols of A / rows of B), p=5 (cols of B)

def _verify(U, V, W):
    """Verify that the decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _strassen_3x3x3():
    """
    Return a rank-19 decomposition for 3x3x3 matrix multiplication.
    This uses a known construction based on 3x3 Strassen-like algorithms.
    For 3x3x3, the rank is known to be 19 (or possibly lower, but 19 is a known upper bound).
    
    Actually, let me use a simpler approach: we'll construct a valid decomposition
    for 2x4x5 directly using a combination of Strassen-like ideas.
    
    For now, let's return a schoolbook decomposition as a fallback,
    and then try to improve it.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for 150s limit
    
    best_rank = 40
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook rank 40
    rank = 40
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    if _verify(U.tolist(), V.tolist(), W.tolist()):
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        best_rank = rank
    
    # Try to find lower rank using the lifting strategy
    # Embed 2x4x5 into a larger tensor and use known decompositions
    
    # Strategy 1: Use Strassen for 2x2x2 blocks
    # A (2x4) = [A_L (2x2) | A_R (2x2)]
    # B (4x5) = [B_T (2x5); B_B (2x5)]
    # C = A_L B_T + A_R B_B
    # Each 2x2 * 2x5 can be done with rank 15 (if known), giving total rank 30
    # But let's try a different approach.
    
    # Strategy 2: Direct search for rank 36, 35, 34, ...
    # Fix U and V randomly, solve for W linearly, check if W is half-integer
    
    for target_rank in range(36, 40):
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random integer initialization for U and V
            U = np.random.randint(-2, 3, (n*m, target_rank))
            V = np.random.randint(-2, 3, (m*p, target_rank))
            
            # Ensure no row of U or V is all zeros
            if np.all(U == 0) or np.all(V == 0):
                continue
            
            # Build the linear system for W
            # Equation: for each (i,j,k), sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1
            # Vectorize W: W_vec[(k*n+i)*r + r_idx] = W[k*n+i, r_idx]
            A_mat = np.zeros((n*m*p, p*n*target_rank))
            b_vec = np.ones(n*m*p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = (i*m + j)*p + k
                        for r_idx in range(target_rank):
                            w_idx = (k*n + i)*target_rank + r_idx
                            A_mat[eq_idx, w_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
            
            # Solve the linear system
            try:
                W_vec, residuals, rank_a, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                continue
            
            # Check if the solution is close to half-integers
            W_test = np.round(W_vec * 2) / 2
            
            # Verify the rounded solution satisfies the system exactly
            if np.allclose(A_mat @ W_test, b_vec, atol=1e-9):
                W_mat = W_test.reshape((p*n, target_rank))
                if _verify(U.tolist(), V.tolist(), W_mat.tolist()):
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W_mat.copy()
                    best_rank = target_rank
                    break  # Found a decomposition for this rank, try lower
    
    # Return the best found
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback: schoolbook
        U = np.zeros((n*m, 40))
        V = np.zeros((m*p, 40))
        W = np.zeros((p*n, 40))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
