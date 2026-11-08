import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, rank, seed=42):
        """Verify decomposition with random tests."""
        if time.time() > deadline - 5:
            return False
        rng = np.random.default_rng(seed)
        for trial in range(3):
            A_test = rng.integers(-10, 10, size=(n, m))
            B_test = rng.integers(-10, 10, size=(m, p))
            C_expected = np.matmul(A_test, B_test)
            
            C_calc = np.zeros((n, p))
            for r in range(rank):
                ua = sum(U[i * m + j][r] * A_test[i, j] for i in range(n) for j in range(m))
                vb = sum(V[j * p + k][r] * B_test[j, k] for j in range(m) for k in range(p))
                prod = ua * vb
                for k in range(p):
                    for i in range(n):
                        C_calc[i, k] += prod * W[k * n + i][r]
            
            if not np.allclose(C_expected, C_calc, atol=1e-8):
                return False
        return True
    
    # Try to find better decompositions
    # Strategy: Use block structure - split B's 6 columns into groups
    
    # For 3x3 x 3x6, one approach: treat as computing 3 separate 3x3 x 3x2 multiplications
    # Each 3x3 x 3x2 can be done in some rank, total would be 3 * rank(3x3x3x2)
    
    # Let's try to build a decomposition using bilinear forms on A and blocks of B
    
    # Key idea: The tensor is a sum over k of (A x B[:,k]) terms
    # We can share A-side forms across different k values
    
    # Let me try rank 45, 44, ... down to find the minimum
    
    # First, let's try a structured approach for lower rank
    
    # Approach: Use the fact that for each pair of rows in A, we can create 
    # shared bilinear forms
    
    # Let me try to use a rank-45 solution by combining some terms
    
    # Actually, let me try a systematic search with small integer coefficients
    
    # For small ranks, try to find U, V, W with entries in {-1, 0, 1}
    
    def try_rank(rank, max_coeff=2):
        """Try to find a decomposition of given rank."""
        if time.time() > deadline - 10:
            return None
        
        # The tensor T has entries T[(i,j),(j,k),(i,k)] = 1 for all valid i,j,k
        # We need: T[(i,j),(j,k),(i,k)] = sum_r U[(i,j),r] * V[(j,k),r] * W[(i,k),r]
        
        # This is a system of equations. Let's set up the constraint.
        # Number of constraints: n * m * p = 54 (one for each (i,j,k) triple)
        # Unknowns: rank * (n*m + m*p + p*n) = rank * (9 + 18 + 18) = rank * 45
        
        # For rank >= 54/something, this is underdetermined.
        # Let's try to find a solution using linear algebra.
        
        # Build the constraint matrix
        # Each constraint (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is nonlinear (products of unknowns), so we need a different approach.
        # Instead, fix two of the matrices and solve for the third linearly.
        
        # Strategy: Fix U and V randomly, then solve for W.
        # But W must satisfy: for all (i,k), W[k*n+i, :] is determined by:
        # sum_j,r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = sum_j A[i,j]*B[j,k]
        
        # Actually, the constraint is: for all i,j,k:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is linear in W if U and V are fixed!
        # Let me set up this linear system.
        
        # Unknowns: W has p*n*rank = 18*rank values
        # Constraints: n*m*p = 54 equations
        
        # For rank >= 3, 18*rank >= 54, so we might have solutions.
        
        # Let's try: fix U and V to have simple structure, then solve for W.
        
        # Simple U: U[i*m+j, r] = 1 if r corresponds to (i,j) pair
        # Simple V: V[j*p+k, r] = 1 if r corresponds to (j,k) pair
        # But that's just schoolbook.
        
        # Let me try a different approach: use random U and V, solve for W.
        
        rng = np.random.default_rng(123)
        
        for attempt in range(20):
            if time.time() > deadline - 10:
                return None
            
            # Generate random U and V with small integer entries
            U = rng.integers(-max_coeff, max_coeff + 1, size=(n * m, rank)).astype(float)
            V = rng.integers(-max_coeff, max_coeff + 1, size=(m * p, rank)).astype(float)
            
            # Build linear system for W
            # Unknowns: W[k*n+i, r] for all k in [0,p), i in [0,n), r in [0,rank)
            # Total: p * n * rank unknowns
            
            # Constraint for each (i, j, k):
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            num_constraints = n * m * p
            num_unknowns = p * n * rank
            
            # Build constraint matrix A_sys where A_sys[constraint, unknown]
            # and b[constraint] = 1
            
            A_sys = np.zeros((num_constraints, num_unknowns))
            b = np.ones(num_constraints)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            # W index: k*n + i, r
                            w_idx = (k * n + i) * rank + r
                            A_sys[idx, w_idx] = coeff
                        idx += 1
            
            # Solve the underdetermined system
            # We want a solution W such that A_sys @ W_vec = b
            # Use least squares with regularization, or find minimum norm solution
            
            try:
                # Check if system is consistent
                # Use SVD to find a solution
                U_svd, s, Vt = np.linalg.svd(A_sys, full_matrices=False)
                
                # Pseudoinverse solution
                # W_vec = Vt.T @ diag(1/s) @ U.T @ b
                # But only for components where s > tol
                
                tol = 1e-10
                inv_s = np.zeros_like(s)
                nonzero = s > tol
                inv_s[nonzero] = 1 / s[nonzero]
                
                W_vec = Vt.T @ (inv_s * (U_svd.T @ b))
                
                # Check if solution satisfies constraints
                residual = np.abs(A_sys @ W_vec - b).max()
                if residual > 1e-8:
                    continue
                
                # Reshape W
                W = W_vec.reshape((p * n, rank))
                
                # Check if W entries are close to integers or half-integers
                # Scale to check
                W_scaled = W * 2
                W_rounded = np.round(W_scaled)
                if np.abs(W - W_rounded / 2).max() > 1e-6:
                    continue
                
                # Convert to half-integers
                W_half = W_rounded / 2.0
                
                # Verify
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W_half.tolist()
                
                if verify(U_list, V_list, W_list, rank):
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
            except Exception:
                continue
        
        return None
    
    # Try ranks from 36 upward
    for rank in range(36, 55):
        if time.time() > deadline - 15:
            break
        result = try_rank(rank)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return get_schoolbook()
