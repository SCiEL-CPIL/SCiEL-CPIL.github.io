import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_sb = [[0] * rank_school for _ in range(n * m)]
    V_sb = [[0] * rank_school for _ in range(m * p)]
    W_sb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_school
    
    # Build the target tensor entries
    # For each (i,j,k), we need sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # The tensor is the identity-like structure for matrix multiplication
    
    # Try ranks from lower to higher, but be smart about it
    # Lower bound: at least max(n*m, m*p, p*n) / something... 
    # Actually, let's just try a range of ranks
    
    # For (2,3,10), let's try ranks from 20 up to 60
    # Use a least-squares / nonlinear approach
    
    np.random.seed(42)
    
    for r in range(15, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        U = np.random.randn(n * m, r)
        V = np.random.randn(m * p, r)
        W = np.random.randn(p * n, r)
        
        # The constraint: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is a system of 60 equations in 56*r variables
        # Use iterative refinement: fix two matrices, solve for the third
        
        # Strategy: alternate between fixing U,V and solving for W, etc.
        # For fixed U,V: for each (k,n_i) pair (i.e., each row of W), 
        # we have constraints from all (i,j) where W[k*n+i, c] appears
        # Actually, let's use a different approach: gradient descent on the residual
        
        # Define the residual
        def compute_residual(U, V, W):
            res = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i,j,k] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
            return res - 1.0
        
        # Try multiple initializations
        found = False
        for init in range(5):
            if time.time() > deadline:
                break
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Simple gradient descent
            lr = 0.01
            for step in range(200):
                if time.time() > deadline:
                    break
                res = compute_residual(U, V, W)
                if np.max(np.abs(res)) < 1e-6:
                    found = True
                    break
                
                # Gradient with respect to U[i*m+j, c]
                # d/dU[i*m+j,c] sum_jk (U[i*m+j,:] * (V[j*p+k,:] * W[k*n+i,:]))^2
                # This gets complex. Let's use a simpler alternating least squares
                
            # Try alternating: fix U,V, solve for W
            # For each (k,i) pair, W[k*n+i, :] appears in constraints for all j
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
            # This is: W[k*n+i, :] * (U[i*m+j,:] * V[j*p+k,:] for all j) = 1
            # So for fixed k,i: sum_c W[k*n+i,c] * (sum over j of U[i*m+j,c]*V[j*p+k,c]) = 1
            # Wait, that's not right. Let me re-derive.
            
            # For fixed (k,i), the constraints involving W[k*n+i,:] are:
            # For each j: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Let A_j[c] = U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c W[k*n+i,c] * A_j[c] = 1 for all j
            # This is a system: W[k*n+i,:] * A_j = 1 for j=0..m-1
            # Where A is an m x r matrix with A[j,c] = U[i*m+j,c]*V[j*p+k,c]
            
            # Similarly for other variables
            
            # Let me just do a simple coordinate descent
            for coord_iter in range(100):
                if time.time() > deadline:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # U[i*m+j,:] appears in constraints for all k
                        # For each k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                        # Let B_k[c] = V[j*p+k,c]*W[k*n+i,c]
                        # Then sum_c U[i*m+j,c]*B_k[c] = 1 for all k
                        # Solve least squares: U[i*m+j,:] @ B.T = ones(p)
                        B = np.zeros((p, r))
                        for k in range(p):
                            B[k,:] = V[j*p+k,:] * W[k*n+i,:]
                        # Solve: U_row @ B.T = ones(p)
                        # This is an underdetermined system if r > p, overdetermined if r < p
                        try:
                            U_new = np.linalg.lstsq(B.T, np.ones(p), rcond=None)[0]
                            U[i*m+j,:] = U_new
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # V[j*p+k,:] appears in constraints for all i
                        # For each i: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                        # Let C_i[c] = U[i*m+j,c]*W[k*n+i,c]
                        # Then sum_c V[j*p+k,c]*C_i[c] = 1 for all i
                        C = np.zeros((n, r))
                        for i in range(n):
                            C[i,:] = U[i*m+j,:] * W[k*n+i,:]
                        try:
                            V_new = np.linalg.lstsq(C.T, np.ones(n), rcond=None)[0]
                            V[j*p+k,:] = V_new
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # W[k*n+i,:] appears in constraints for all j
                        # For each j: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                        # Let D_j[c] = U[i*m+j,c]*V[j*p+k,c]
                        # Then sum_c W[k*n+i,c]*D_j[c] = 1 for all j
                        D = np.zeros((m, r))
                        for j in range(m):
                            D[j,:] = U[i*m+j,:] * V[j*p+k,:]
                        try:
                            W_new = np.linalg.lstsq(D.T, np.ones(m), rcond=None)[0]
                            W[k*n+i,:] = W_new
                        except:
                            pass
                
                # Check convergence
                res = compute_residual(U, V, W)
                if np.max(np.abs(res)) < 1e-5:
                    found = True
                    break
        
        if found:
            best_rank = r
            # Convert to list format
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            # Try to go lower
            continue
        
        # If not found, try next rank
        # But we should also keep the best found so far
    
    return best_result
