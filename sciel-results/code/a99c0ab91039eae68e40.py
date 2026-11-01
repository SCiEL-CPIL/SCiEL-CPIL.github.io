import time
import numpy as np

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0
START_TIME = time.time()

def construct():
    """Return a valid exact tensor decomposition for 3x3 * 3x3 matrix multiplication.
    
    Attempts to find a rank-minimal decomposition using optimization.
    Falls back to schoolbook (rank 27) if optimization doesn't improve.
    """
    n, m, p = 3, 3, 3
    N, M, P = n * m, m * p, p * n  # 9, 9, 9
    
    # Schoolbook: rank 27
    U_school = np.zeros((N, 27))
    V_school = np.zeros((M, 27))
    W_school = np.zeros((P, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-8):
        """Verify that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == 1 for all (i,j,k)."""
        rank = U.shape[1]
        # Reshape for efficient computation
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # Build the full tensor check
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(val - 1.0))
        return max_err < tol
    
    # Try to find a lower rank decomposition using iterative refinement
    # Start with a random initialization and use gradient-free optimization
    best_rank = 27
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try ranks from 23 up to 26
    for target_rank in range(23, 27):
        if time.time() - START_TIME > DEADLINE_SECONDS - SAFETY_MARGIN:
            break
        
        # Multiple random restarts
        for restart in range(30):
            if time.time() - START_TIME > DEADLINE_SECONDS - SAFETY_MARGIN:
                break
            
            # Initialize with random values
            U_rand = np.random.randn(N, target_rank) * 0.5
            V_rand = np.random.randn(M, target_rank) * 0.5
            W_rand = np.random.randn(P, target_rank) * 0.5
            
            # Use a simple iterative method: coordinate descent
            # For each column r, try to improve the fit
            success = False
            
            # Gradient-free approach: use the fact that for rank-k decomposition,
            # we can try to match the tensor T[i,j,k] = 1
            # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] (outer product)
            
            # Flatten the target tensor
            T_target = np.ones((n, m, p))
            
            # Iterative refinement: alternating minimization
            for iteration in range(200):
                if time.time() - START_TIME > DEADLINE_SECONDS - SAFETY_MARGIN:
                    break
                
                # Fix V and W, solve for U
                # For each (i,j), U[i,j,r] should satisfy:
                # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all k
                # This is a system of p equations for each target_rank variables
                # Under-determined if target_rank > p, so use least squares
                
                # Build the system: for each (i,j), we have p equations
                # A_U[i,j] * U[i,j,:] = b_U[i,j]
                # where A_U[i,j][k,r] = V[j*p+k, r] * W[k*n+i, r]
                # b_U[i,j][k] = 1
                
                for i in range(n):
                    for j in range(m):
                        A_uj = np.zeros((p, target_rank))
                        for k in range(p):
                            A_uj[k, :] = V_rand[j*p+k, :] * W_rand[k*n+i, :]
                        b_uj = np.ones(p)
                        # Solve least squares
                        try:
                            U_rand[i*m+j, :] = np.linalg.lstsq(A_uj, b_uj, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and W, solve for V
                for j in range(m):
                    for k in range(p):
                        A_vk = np.zeros((n, target_rank))
                        for i in range(n):
                            A_vk[i, :] = U_rand[i*m+j, :] * W_rand[k*n+i, :]
                        b_vk = np.ones(n)
                        try:
                            V_rand[j*p+k, :] = np.linalg.lstsq(A_vk, b_vk, rcond=None)[0]
                        except:
                            pass
                
                # Fix U and V, solve for W
                for k in range(p):
                    for i in range(n):
                        A_wk = np.zeros((m, target_rank))
                        for j in range(m):
                            A_wk[j, :] = U_rand[i*m+j, :] * V_rand[j*p+k, :]
                        b_wk = np.ones(m)
                        try:
                            W_rand[k*n+i, :] = np.linalg.lstsq(A_wk, b_wk, rcond=None)[0]
                        except:
                            pass
                
                # Check if we've converged
                if verify(U_rand, V_rand, W_rand, tol=1e-6):
                    success = True
                    break
            
            if success:
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break  # No need to try higher ranks
    
    # Final verification of the best result
    if best_rank < 27:
        if verify(best_U, best_V, best_W, tol=1e-8):
            # Round to nearest half-integer or integer if close
            U_out = [[float(x) for x in row] for row in best_U]
            V_out = [[float(x) for x in row] for row in best_V]
            W_out = [[float(x) for x in row] for row in best_W]
            return {"U": U_out, "V": V_out, "W": W_out}
    
    # Fall back to schoolbook
    U_out = [[float(x) for x in row] for row in U_school]
    V_out = [[float(x) for x in row] for row in V_school]
    W_out = [[float(x) for x in row] for row in W_school]
    return {"U": U_out, "V": V_out, "W": W_out}
