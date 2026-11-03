import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0 # 150s limit, keep 10s margin
    
    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school

    # Try to find a lower rank decomposition
    # We search for ranks from 25 down to 15
    # Using a simple random restart with alternating projections
    
    def check_error(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Check: sum_r W[k,i,r] U[i,j,r] V[j,k,r] == delta_{j,j'}
        # We check for a few random i, k pairs to speed up, or all if R is small
        # Actually, we must check all i, k to be sure.
        # n=2, p=5, so 10 pairs. m=4, so 16 entries per pair. Total 160 checks.
        error = 0.0
        for i in range(n):
            for k in range(p):
                # Compute the matrix M_{j,j'} = sum_r W[k,i,r] U[i,j,r] V[j,k,r]
                # U[i, :, r] is a vector of size m
                # V[:, k, r] is a vector of size m
                # Let's extract the relevant slices
                # U_slice: (m, R)
                # V_slice: (m, R)
                # W_vec: (R,)
                U_slice = U[i * m : (i + 1) * m, :]
                V_slice = V[k * p : (k + 1) * p, :] # Wait, V index is j*p+k. So for fixed k, j varies 0..m-1.
                # V rows are indexed by j*p+k.
                # So V_slice should be V[0*p+k, :], V[1*p+k, :], ... V[(m-1)*p+k, :]
                V_indices = [j * p + k for j in range(m)]
                V_slice = V[V_indices, :]
                W_vec = W[k * n + i, :]
                
                # M = sum_r W_vec[r] * (U_slice[:, r] * V_slice[:, r].T)
                # This is a sum of outer products weighted by W_vec
                # M = (U_slice * W_vec) @ V_slice.T
                # U_slice * W_vec broadcasts W_vec over rows?
                # U_slice is (m, R). W_vec is (R,).
                # We want (U_slice[:, r] * W_vec[r]) for each r.
                # This is equivalent to U_slice @ np.diag(W_vec) @ V_slice.T
                # Or (U_slice * W_vec) @ V_slice.T
                M = (U_slice * W_vec) @ V_slice.T
                
                # Target is Identity matrix of size m
                target = np.eye(m)
                error += np.sum((M - target) ** 2)
        return error

    # Search for lower rank
    for R in range(25, 14, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # Scale W to be small initially to avoid divergence?
            # Or scale U, V to be large?
            # Let's try to normalize.
            
            # Alternating Least Squares
            # Fix W, solve for U, V?
            # The system is bilinear.
            # We can fix W and V, solve for U.
            # For each i, k:
            # sum_r W[k,i,r] U[i,j,r] V[j,k,r] = delta_{j,j'}
            # This is not linear in U if V is fixed?
            # U[i,j,r] appears linearly.
            # Let's group by i.
            # For fixed i, we have p equations (one for each k).
            # Each equation is a matrix equation:
            # sum_r W[k,i,r] (U[i,:,r] * V[:,k,r].T) = I_m
            # This is linear in U[i,:,r] if V and W are fixed.
            # Let's set up the linear system for U.
            # Variables: U[i, :, r] for all r. Total m*R variables per i.
            # Equations: For each k (p of them), we have m*m = 16 scalar equations.
            # Total 16*p = 80 equations per i.
            # Unknowns: m*R = 4*R.
            # If R=20, 80 unknowns. 80 equations. Square system.
            
            # Let's implement this ALS step.
            
            # 1. Fix V, W. Solve for U.
            # 2. Fix U, W. Solve for V.
            # 3. Fix U, V. Solve for W.
            
            # Solve for U:
            # For each i:
            #   Build A_matrix (size p*m*m x m*R)
            #   Build b_vector (size p*m*m)
            #   A_matrix[k*m*m + j*m + j', idx] = ...
            #   idx corresponds to (j, r)
            #   Term: W[k,i,r] * V[j,k,r] * U[i,j,r]
            #   So A[k*m*m + j*m + j', j*R + r] = W[k,i,r] * V[j,k,r]
            
            # Solve for V:
            # For each k:
            #   Variables: V[:, k, r] for all r. Total m*R variables.
            #   Equations: For each i (n of them), m*m = 16 scalar equations.
            #   Total 16*n = 32 equations per k.
            #   Unknowns: m*R = 4*R.
            #   If R=20, 80 unknowns. 32 equations. Underdetermined.
            #   We can use least squares.
            
            # Solve for W:
            # For each i, k:
            #   Equation: sum_r W[k,i,r] * (U[i,:,r] * V[:,k,r].T) = I_m
            #   This is m*m = 16 scalar equations for W[k,i,:] (R variables).
            #   Underdetermined if R > 16.
            
            # Let's iterate:
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                    
                # Update U
                for i in range(n):
                    A_mat = np.zeros((p * m * m, m * R))
                    b_vec = np.zeros(p * m * m)
                    for k in range(p):
                        for j in range(m):
                            for j_prime in range(m):
                                row_idx = k * m * m + j * m + j_prime
                                b_vec[row_idx] = 1.0 if j == j_prime else 0.0
                                for r in range(R):
                                    col_idx = j * R + r
                                    A_mat[row_idx, col_idx] = W[k * n + i, r] * V[j * p + k, r]
                    # Solve A_mat @ U_slice = b_vec
                    # U_slice is (m*R,)
                    try:
                        U_slice = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        U[i * m : (i + 1) * m, :] = U_slice.reshape(m, R)
                    except:
                        pass
                        
                # Update V
                for k in range(p):
                    A_mat = np.zeros((n * m * m, m * R))
                    b_vec = np.zeros(n * m * m)
                    for i in range(n):
                        for j in range(m):
                            for j_prime in range(m):
                                row_idx = i * m * m + j * m + j_prime
                                b_vec[row_idx] = 1.0 if j == j_prime else 0.0
                                for r in range(R):
                                    col_idx = j * R + r
                                    A_mat[row_idx, col_idx] = W[k * n + i, r] * U[i * m + j, r]
                    try:
                        V_slice = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                        # V_slice corresponds to V[j*p+k, r] for j=0..m-1
                        for j in range(m):
                            V[j * p + k, :] = V_slice[j * R : (j + 1) * R]
                    except:
                        pass
                        
                # Update W
                for i in range(n):
                    for k in range(p):
                        A_mat = np.zeros((m * m, R))
                        b_vec = np.zeros(m * m)
                        for j in range(m):
                            for j_prime in range(m):
                                row_idx = j * m + j_prime
                                b_vec[row_idx] = 1.0 if j == j_prime else 0.0
                                for r in range(R):
                                    A_mat[row_idx, r] = U[i * m + j, r] * V[j_prime * p + k, r]
                        try:
                            W_vec = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                            W[k * n + i, :] = W_vec
                        except:
                            pass
                            
                # Check error
                err = check_error(U, V, W)
                if err < 1e-6:
                    best_rank = R
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
                if it % 10 == 0 and err < 1e-3:
                    # Converged well enough?
                    pass

            if found:
                break

    # Convert to lists as required
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
