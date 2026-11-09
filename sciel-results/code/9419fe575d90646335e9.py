"""Matrix multiplication 2x3x8 rank-minimal exact decomposition."""
DIMENSIONS = [2, 3, 8]

import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3x8 matrix multiplication.
    
    Uses numerical optimization to search for lower-rank exact decompositions
    below the schoolbook rank of 48.
    
    Interface: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    W follows the convention W[k*n+i, c].
    """
    deadline = time.time() + 135  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if there exists (i, j, k) such that:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # Using convention: U row = i*m+j (A[i,j]), V row = j*p+k (B[j,k]), W row = k*n+i (C[i,k])
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from low to high, starting below schoolbook
    for rank in range(20, schoolbook_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations for this rank
        found_exact = False
        max_attempts = min(5, max(1, int((deadline - time.time()) / 15))) if rank < schoolbook_rank else 1
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W randomly
            np.random.seed(42 + rank * 100 + attempt)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Use alternating least squares to minimize ||T - sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]||^2
            # This is equivalent to minimizing the Frobenius norm of the residual tensor
            for inner_iter in range(50):
                if time.time() > deadline:
                    break
                
                # Fix V and W, solve for U: minimize ||T - UWV||_F^2
                # For each column r: T_contracted[:, r] = U @ (V^T * W^T) ... 
                # Actually, let's use a simpler approach: iterate over one factor at a time
                
                # Update U: for each (a, b, c), T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # Fix V, W: U[a,:] should satisfy sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
                # This is a linear system: for each a, U[a,:] * (V ⊙ W) = T[a,:,:]
                # where (V ⊙ W)[b,c] = sum_r V[b,r]*W[c,r]
                VW = np.einsum('br,cr->bc', V, W)
                # Solve: U[a,:] @ VW = T[a,:,:] => U[a,:] = T[a,:,:] @ pinv(VW)
                try:
                    U_new = np.einsum('abc,bc->ab', T, np.linalg.pinv(VW))
                except:
                    U_new = U
                U = 0.5 * U + 0.5 * U_new  # damping for stability
                
                # Update V: similar
                UW = np.einsum('ar,cr->ac', U, W)
                try:
                    V_new = np.einsum('abc,ac->bc', T, np.linalg.pinv(UW))
                except:
                    V_new = V
                V = 0.5 * V + 0.5 * V_new
                
                # Update W
                UV = np.einsum('ar,br->ab', U, V)
                try:
                    W_new = np.einsum('abc,ab->cb', T, np.linalg.pinv(UV))
                except:
                    W_new = W
                W = 0.5 * W + 0.5 * W_new
            
            # Check if the decomposition is close to exact
            # Reconstruct: R[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            R = np.einsum('ar,br,cr->abc', U, V, W)
            residual = np.max(np.abs(R - T))
            
            if residual < 1e-8:
                # Found a good numerical solution, try to rationalize
                # Scale and round to check for integer/half-integer entries
                # Try to find a scaling that makes entries close to integers or half-integers
                
                def try_rationalize(mat):
                    """Try to convert a matrix to have integer or half-integer entries."""
                    # Try different scalings
                    for scale in [1, 2, 0.5]:
                        scaled = mat * scale
                        rounded = np.round(scaled * 2) / 2  # round to nearest 0.5
                        if np.max(np.abs(scaled - rounded)) < 1e-6:
                            return rounded.tolist()
                    return None
                
                U_rat = try_rationalize(U)
                V_rat = try_rationalize(V)
                W_rat = try_rationalize(W)
                
                if U_rat is not None and V_rat is not None and W_rat is not None:
                    # Verify exactness with rational entries
                    U_np = np.array(U_rat, dtype=np.float64)
                    V_np = np.array(V_rat, dtype=np.float64)
                    W_np = np.array(W_rat, dtype=np.float64)
                    R_exact = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
                    if np.max(np.abs(R_exact - T)) < 1e-10:
                        found_exact = True
                        if rank < best_rank:
                            best_rank = rank
                            best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                        break
        
        if found_exact and rank < schoolbook_rank:
            break
        
        # If we found an exact decomposition at this rank, we can stop
        # since we're going from low to high rank
    
    # If no better decomposition was found, return schoolbook
    if best_result is None:
        rank = schoolbook_rank
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
        
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
