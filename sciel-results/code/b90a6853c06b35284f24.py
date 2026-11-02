"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook as fallback
    def schoolbook():
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
    
    best_result = None
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # The tensor T has entries T[i,j,k] = 1 if i==k else 0 (for the multiplication)
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
    
    # Strategy: try various structured sparsity patterns
    # For rank r, we need r columns in each factor
    
    # Known: for 3x3x3, rank 7 is possible (Strassen-like)
    # For 3x3x7, let's try ranks from 21 upward
    
    def try_rank(r, pattern_type='structured'):
        """Try to find a rank-r decomposition with given pattern type."""
        if time.time() > deadline:
            return None
        
        # Create the tensor in flattened form
        # T[i,j,k] = 1 if i == k else 0
        # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        # This is a bilinear problem in the factors
        # For fixed U, V, it's linear in W, and vice versa
        
        # Approach: use a greedy/iterative method
        # Or: use the fact that we can fix one factor and solve for others
        
        # Let's try a structured approach:
        # For 3x3x7, try rank = 3*7 = 21 with a Kronecker structure
        
        if pattern_type == 'kron':
            # Try U[i*m+j, r] where r = i*7 + k (for i in 0..2, k in 0..6)
            # V[j*p+k, r] = V[j*7+k, i*7+k] 
            # W[k*n+i, r] = W[k*3+i, i*7+k]
            # This gives rank 21
            # Check: sum_r U[i*3+j, i*7+k'] * V[j*7+k, i*7+k'] * W[k*3+i, i*7+k']
            # = sum_{k'} [U[i*3+j, i*7+k'] * V[j*7+k, i*7+k'] * W[k*3+i, i*7+k']]
            # We need this to equal 1 if i==k else 0
            
            # Let's set:
            # U[i*3+j, i*7+k'] = 1 for all j, k' (within the block for fixed i)
            # V[j*7+k, i*7+k'] = delta_{j, ?} ... this is getting complex
            
            # Simpler: just try to solve numerically
            pass
        
        # Numerical approach: iterative refinement
        # Initialize factors randomly, then alternate least squares
        
        np.random.seed(42)
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Define A, B, C as views/aliases for easier indexing in the update steps
        # A[i,j] corresponds to U[i*m+j, :]
        # B[j,k] corresponds to V[j*p+k, :]
        # C[k,i] corresponds to W[k*n+i, :]
        # We use the same underlying arrays so updates propagate.
        A = U
        B = V
        C = W
        
        # The constraint is: for all (i,j,k):
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        # This is not linear in all three simultaneously.
        # Alternate: fix U and V, solve for W (linear)
        # Fix U and W, solve for V (linear)
        # Fix V and W, solve for U (linear)
        
        # T[i,j,k] = 1 if i == k else 0
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1.0 if i == k else 0.0
        
        # For fixed U and V, the equation for W is:
        # sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = T[i,j,k]
        # For each (k, i), we have a system in W[k*n+i, :]
        # But U and V depend on the index...
        
        # Let me re-index:
        # For fixed (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
        
        # Group by (k, i): for fixed k and i, sum over j:
        # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_j T[i,j,k]
        
        # This is still coupled. Let's use a different approach.
        
        # Direct: treat it as a 3-way tensor decomposition
        # Reshape: for each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        # Sum over r should equal the tensor T reshaped appropriately
        
        # T reshaped: T[i,j,k] with i in [0,n), j in [0,m), k in [0,p)
        # But W is indexed by (k*n+i), so the third mode of the "tensor" for W is (k,i)
        
        # Let me define: A[i,j,r] = U[i*m+j, r]
        # B[j,k,r] = V[j*p+k, r]  
        # C[k,i,r] = W[k*n+i, r]
        # Constraint: sum_r A[i,j,r] * B[j,k,r] * C[k,i,r] = T[i,j,k]
        
        # This is a rank-r decomposition of a 3-tensor with modes (i,j), (j,k), (k,i)
        # but with the constraint that the third mode is (k,i) not (i,k)
        
        # Let's just do alternating minimization numerically
        
        for iteration in range(200):
            if time.time() > deadline:
                return None
            
            # Update W: for each (k,i), solve linear system
            # sum_r W[k*n+i, r] * (sum_j A[i,j,r] * B[j,k,r]) = sum_j T[i,j,k]
            # Let M[k,i,r] = sum_j A[i,j,r] * B[j,k,r]
            # Then: sum_r W[k*n+i, r] * M[k,i,r] = sum_j T[i,j,k] = S[k,i]
            # where S[k,i] = sum_j T[i,j,k] = m if i==k else 0 (since T[i,j,k]=1 iff i==k, and j ranges over m values)
            
            S = np.zeros((p, n))
            for k in range(p):
                for i in range(n):
                    S[k, i] = m if i == k else 0
            
            # M[k,i,r] = sum_j A[i*m+j, r] * B[j*p+k, r]
            M = np.zeros((p, n, r))
            for k in range(p):
                for i in range(n):
                    for j in range(m):
                        M[k, i, :] += A[i*m+j, :] * B[j*p+k, :]
            
            # Solve: for each (k,i), W[k*n+i, :] satisfies sum_r W[k*n+i,r] * M[k,i,r] = S[k,i]
            # This is underdetermined if r > 1. Use least squares.
            for k in range(p):
                for i in range(n):
                    # M[k,i,:] is a row vector, we want W[k*n+i,:] such that dot = S[k,i]
                    # Minimum norm solution: W[k*n+i,:] = S[k,i] * M[k,i,:] / (M[k,i,:] . M[k,i,:])
                    mvec = M[k, i, :]
                    mnorm = np.dot(mvec, mvec)
                    if mnorm > 1e-12:
                        W[k*n+i, :] = S[k, i] * mvec / mnorm
                    else:
                        W[k*n+i, :] = 0
            
            # Update A (U): for each (i,j), solve
            # sum_r A[i*m+j, r] * (sum_k B[j*p+k, r] * C[k*n+i, r]) = T[i,j,k] for all k
            # Let N[i,j,k,r] = B[j*p+k, r] * C[k*n+i, r]
            # Constraint: for each k, sum_r A[i*m+j, r] * N[i,j,k,r] = T[i,j,k]
            # This is a system: A[i*m+j,:] * N[i,j,:].T = T[i,j,:]
            
            N = np.zeros((n, m, p, r))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r_idx in range(r):
                            N[i, j, k, r_idx] = B[j*p+k, r_idx] * C[k*n+i, r_idx]
            
            for i in range(n):
                for j in range(m):
                    # N[i,j,:,r] shape (p, r), T[i,j,:] shape (p,)
                    # Solve: A[i*m+j,:] @ N[i,j,:].T = T[i,j,:]
                    # N[i,j,:].T shape (r, p)
                    Nt = N[i, j, :, :].T  # (r, p)
                    # Least squares: A[i*m+j,:] @ Nt = T[i,j,:]
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(Nt, T[i, j, :], rcond=None)
                        A[i*m+j, :] = sol
                    except:
                        pass
            
            # Update B (V): for each (j,k), solve
            # sum_r B[j*p+k, r] * (sum_i A[i*m+j, r] * C[k*n+i, r]) = T[i,j,k] for all i
            # Let P[j,k,i,r] = A[i*m+j, r] * C[k*n+i, r]
            # Constraint: for each i, sum_r B[j*p+k, r] * P[j,k,i,r] = T[i,j,k]
            
            P = np.zeros((m, p, n, r))
            for j in range(m):
                for k in range(p):
                    for i in range(n):
                        for r_idx in range(r):
                            P[j, k, i, r_idx] = A[i*m+j, r_idx] * C[k*n+i, r_idx]
            
            for j in range(m):
                for k in range(p):
                    Pt = P[j, k, :, :].T  # (r, n)
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(Pt, T[:, j, k], rcond=None)
                        B[j*p+k, :] = sol
                    except:
                        pass
            
            # Check convergence
            # Compute the tensor error
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(A[i*m+j, :], B[j*p+k, :] * C[k*n+i, :])
                        error += (val - T[i, j, k]) ** 2
            
            if error < 1e-10:
                # Success! Convert to lists
                U_list = [[float(x) for x in row] for row in A]
                V_list = [[float(x) for x in row] for row in B]
                W_list = [[float(x) for x in row] for row in W]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try ranks from low to high
    for r in range(21, best_rank):
        if time.time() > deadline:
            break
        result = try_rank(r)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    if best_result is not None:
        return best_result
    else:
        return schoolbook()
# EVOLVE-BLOCK-END
