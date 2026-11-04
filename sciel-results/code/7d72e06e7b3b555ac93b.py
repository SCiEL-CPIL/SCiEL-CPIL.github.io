"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # We need to find rank-R decomposition where:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Try to find a low-rank decomposition using numerical optimization
    # Start with a guess and iterate
    
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # The constraint: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Let's try various ranks and see if we can find a solution
    # Use a simple iterative approach
    
    best_result = None
    best_rank = nm * mp  # schoolbook rank = 63
    
    # First, create the schoolbook solution as fallback
    U_school = np.zeros((nm, best_rank))
    V_school = np.zeros((mp, best_rank))
    W_school = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank solutions
    # We'll use a heuristic: randomly initialize and use gradient-free optimization
    # on the residual
    
    deadline = time.time() + 120
    
    for R in range(20, best_rank + 1):  # Try ranks from 20 upward
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        found = False
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(42 + R * 100 + trial)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Iteratively optimize to minimize residual
            # Residual: for each (i,j,k), |sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1|
            
            # Use a simple coordinate descent / alternating minimization
            for it in range(200):
                if time.time() > deadline - 30:
                    break
                
                # Update W: for each (k,n+i) row, we need sum_r V[j*p+k,r]*U[i*m+j,r]*W[k*n+i,r] = 1
                # This is a linear system in W for fixed U,V
                # For fixed U,V, the constraint for each (i,j,k) is:
                # sum_r (U[i*m+j,r]*V[j*p+k,r]) * W[k*n+i,r] = 1
                # 
                # Let's collect all constraints. For each (i,j,k), we have a linear equation in W.
                # The coefficient of W[k*n+i, r] in equation (i,j,k) is U[i*m+j,r]*V[j*p+k,r]
                
                # Build the linear system for W
                # Number of equations: n*m*p = 63
                # Unknowns: pn*R = 21*R
                # This is underdetermined if 21*R > 63, i.e., R > 3
                
                # Let's use least squares to find W that minimizes residual
                # A_w is (63, 21*R), b_w is (63,)
                A_w = np.zeros((n*m*p, pn*R))
                b_w = np.ones(n*m*p)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                A_w[eq, (k*n+i)*R + r] = U[i*m+j, r] * V[j*p+k, r]
                            eq += 1
                
                # Solve A_w @ W_flat = b_w in least squares sense
                W_flat, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=None)
                W = W_flat.reshape(pn, R)
                
                # Check residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                            residual += (val - 1)**2
                
                if residual < 1e-6:
                    found = True
                    break
                
                # Update U similarly
                A_u = np.zeros((n*m*p, nm*R))
                b_u = np.ones(n*m*p)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                A_u[eq, (i*m+j)*R + r] = V[j*p+k, r] * W[k*n+i, r]
                            eq += 1
                U_flat, _, _, _ = np.linalg.lstsq(A_u, b_u, rcond=None)
                U = U_flat.reshape(nm, R)
                
                # Update V
                A_v = np.zeros((n*m*p, mp*R))
                b_v = np.ones(n*m*p)
                eq = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            for r in range(R):
                                A_v[eq, (j*p+k)*R + r] = U[i*m+j, r] * W[k*n+i, r]
                            eq += 1
                V_flat, _, _, _ = np.linalg.lstsq(A_v, b_v, rcond=None)
                V = V_flat.reshape(mp, R)
                
                # Check convergence
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                            residual += (val - 1)**2
                
                if residual < 1e-6:
                    found = True
                    break
            
            if found:
                best_rank = R
                best_result = (U, V, W)
                break
        
        if found:
            break
    
    if best_result is None:
        # Return schoolbook
        U, V, W = U_school, V_school, W_school
    else:
        U, V, W = best_result
    
    # Convert to lists with integer/half-integer entries if possible
    # Round to nearest 0.5
    U_list = [[round(U[i,j] * 2) / 2 for j in range(R)] for i in range(nm)]
    V_list = [[round(V[i,j] * 2) / 2 for j in range(R)] for i in range(mp)]
    W_list = [[round(W[i,j] * 2) / 2 for j in range(R)] for i in range(pn)]
    
    # Verify the rounded solution still works
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r] for r in range(R))
                if abs(val - 1) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        U_list = [[float(U_school[i,j]) for j in range(nm*mp)] for i in range(nm)]
        V_list = [[float(V_school[i,j]) for j in range(nm*mp)] for i in range(mp)]
        W_list = [[float(W_school[i,j]) for j in range(nm*mp)] for i in range(pn)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
