import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Try to find a rank-minimal decomposition
    # Start with rank 34 and work down
    
    # The tensor T[i,j,k,r] where:
    # i in 0..n-1 (row of A, 0..1)
    # j in 0..m-1 (col of A = row of B, 0..1) 
    # k in 0..p-1 (col of B, 0..9)
    # r is the rank index
    # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    
    # Actually for matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor has T[i,j,k] = 1 (it's a 3-tensor with entries 0 or 1)
    
    # We need to find U, V, W such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all (i,j,k) where the product is needed
    # and 0 otherwise
    
    # This is a rank minimization problem for a 3-way tensor
    
    # Let's try rank 34 first
    for target_rank in [34, 33, 32, 31, 30]:
        if time.time() > deadline - 60:
            break
            
        # Use random initialization and iterative refinement
        np.random.seed(42)
        
        # Initialize U, V, W randomly
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # The tensor we want to decompose
        # T[i,j,k] = 1 for all valid (i,j,k)
        # Since all entries of A*B contribute to C, T is all ones for valid indices
        
        # Actually, the matrix multiplication tensor has:
        # T[i,j,k] = 1 if we need A[i,j]*B[j,k] in the computation
        # For standard matrix multiplication, T[i,j,k] = 1 for all i,j,k
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        # This is a constrained optimization problem
        
        # Let's use a simpler approach: try to construct explicitly
        
        # For rank 34, we can try to save one multiplication by sharing a Strassen term
        # between two adjacent blocks
        
        # Block 0: columns 0,1
        # Block 1: columns 2,3
        # etc.
        
        # In Strassen, M1 = (A11+A22)(B11+B22)
        # If we can share M1 between blocks, we save one rank
        
        # Let's try: use the same M1 term for blocks 0 and 1
        # M1_shared = (A11+A22) * (B11+B22+B13+B23) -- but this doesn't work directly
        
        # Actually, let me try a different approach: use a rank-6 algorithm if possible
        # But 2x2x2 has rank 7, so we can't do better per block
        
        # Let me try to verify rank 35 is optimal by checking if rank 34 works
        # using a linear algebra approach
        
        # The tensor T has entries T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,...,9}
        # T[i,j,k] = 1 for all i,j,k
        
        # We want to express T as sum of r rank-1 tensors
        # T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # This is equivalent to finding matrices U, V, W of ranks r such that
        # the 3-tensor T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
        
        # Let's try to solve this using alternating least squares
        
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        for iteration in range(1000):
            if time.time() > deadline - 30:
                break
                
            # Update U: for each (i,j), find U[i*m+j, :] that minimizes
            # ||T[i,j,:] - sum_r U[i*m+j,r] * (V[:,r] ⊙ W[:,r])[k]||
            # where ⊙ is element-wise product
            
            for i in range(n):
                for j in range(m):
                    # Compute the "projected" tensor for this (i,j)
                    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all k
                    # Let X[k,r] = V[j*p+k,r] * W[k*n+i,r]
                    # Then we need: U[i*m+j,:] @ X = 1 (vector of ones)
                    
                    X = np.zeros((p, target_rank))
                    for k in range(p):
                        X[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve: X^T @ U[i*m+j,:] = 1
                    # Least squares: U[i*m+j,:] = (X^T X)^{-1} X^T @ 1
                    try:
                        XTX = X.T @ X
                        Xt1 = X.T @ np.ones(p)
                        U[i*m+j, :] = np.linalg.solve(XTX + 1e-10*np.eye(target_rank), Xt1)
                    except:
                        pass
            
            # Update V: for each (j,k), find V[j*p+k, :] that minimizes
            for j in range(m):
                for k in range(p):
                    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i
                    # Let Y[i,r] = U[i*m+j,r] * W[k*n+i,r]
                    # Then we need: V[j*p+k,:] @ Y = 1
                    
                    Y = np.zeros((n, target_rank))
                    for i in range(n):
                        Y[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    try:
                        YTY = Y.T @ Y
                        Yt1 = Y.T @ np.ones(n)
                        V[j*p+k, :] = np.linalg.solve(YTY + 1e-10*np.eye(target_rank), Yt1)
                    except:
                        pass
            
            # Update W: for each (k,i), find W[k*n+i, :] that minimizes
            for k in range(p):
                for i in range(n):
                    # We need: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all j
                    # Let Z[j,r] = U[i*m+j,r] * V[j*p+k,r]
                    # Then we need: W[k*n+i,:] @ Z = 1
                    
                    Z = np.zeros((m, target_rank))
                    for j in range(m):
                        Z[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    try:
                        ZTZ = Z.T @ Z
                        Zt1 = Z.T @ np.ones(m)
                        W[k*n+i, :] = np.linalg.solve(ZTZ + 1e-10*np.eye(target_rank), Zt1)
                    except:
                        pass
            
            # Check convergence
            if iteration % 50 == 0:
                # Compute residual
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(target_rank):
                                val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            residual += (val - 1.0)**2
                
                if residual < 1e-10:
                    # Found a valid decomposition!
                    # Convert to list format
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Round to nearest integer/half-integer if possible
                    U_rounded = [[round(x*2)/2 for x in row] for row in U_list]
                    V_rounded = [[round(x*2)/2 for x in row] for row in V_list]
                    W_rounded = [[round(x*2)/2 for x in row] for row in W_list]
                    
                    # Verify the rounded version
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0.0
                                for r in range(target_rank):
                                    val += U_rounded[i*m+j][r] * V_rounded[j*p+k][r] * W_rounded[k*n+i][r]
                                if abs(val - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_rounded, "V": V_rounded, "W": W_rounded}
    
    # Fall back to rank 35 Strassen
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    col = 0
    for k in range(p // 2):
        for a_vec, b_vec, c_vec in strassen_terms:
            for idx in range(n * m):
                U[idx][col] = a_vec[idx]
            
            V[0 * p + 2 * k][col] = b_vec[0]
            V[0 * p + 2 * k + 1][col] = b_vec[1]
            V[1 * p + 2 * k][col] = b_vec[2]
            V[1 * p + 2 * k + 1][col] = b_vec[3]
            
            W[2 * k * n + 0][col] = c_vec[0]
            W[2 * k * n + 1][col] = c_vec[1]
            W[(2 * k + 1) * n + 0][col] = c_vec[2]
            W[(2 * k + 1) * n + 1][col] = c_vec[3]
            
            col += 1
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return {"U": U, "V": V, "W": W}
