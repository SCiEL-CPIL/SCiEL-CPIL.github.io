import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    dim = n * m * p
    
    # 1. Construct the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Else 0
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
                
    # Flatten T for vector operations
    T_flat = T.flatten()
    
    # 2. Schoolbook solution (Rank 60)
    U_sb = np.zeros((n * m, dim))
    V_sb = np.zeros((m * p, dim))
    W_sb = np.zeros((p * n, dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                c = i * m * p + j * p + k
                U_sb[i * m + j, c] = 1
                V_sb[j * p + k, c] = 1
                W_sb[k * n + i, c] = 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = dim
    
    # 3. Search for lower rank
    # We will try to find a decomposition of rank R < 60
    # Using a non-linear optimization approach
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Try ranks from 25 to 59
    # Lower bound is roughly 20
    for R in range(25, 60):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W with small random integers
            # Scale by 2 to allow half-integers later
            U = np.random.randint(-2, 3, size=(n * m, R)).astype(float)
            V = np.random.randint(-2, 3, size=(m * p, R)).astype(float)
            W = np.random.randint(-2, 3, size=(p * n, R)).astype(float)
            
            # Define the error function
            # E = || sum_r U[:, r] V[:, r] W[:, r] - T ||^2
            # This is a tensor reconstruction problem.
            
            # We can use a simple gradient descent or a library.
            # Given the small size, we can try to solve it using a non-linear solver.
            # However, scipy.optimize.minimize might be slow.
            
            # Alternative: Use the fact that for fixed V, W, the problem is linear in U?
            # No, it's trilinear.
            
            # Let's try a few iterations of a simple update rule or just check if random init is close.
            # Actually, random init is unlikely to be close.
            
            # We can use a "rank-1 update" search?
            # Or we can use the fact that we can solve for one matrix given the others.
            # If we fix V and W, we can solve for U?
            # T_flat = sum_r (U[:, r] * (V[:, r] * W[:, r]).flatten())
            # This is a linear system in U.
            # M = column_stack( (V[:, r] * W[:, r]).flatten() for r in range(R) )
            # U_flat = lstsq(M, T_flat)
            
            # Let's try this iterative approach:
            # 1. Random init V, W
            # 2. Solve for U
            # 3. Solve for V
            # 4. Solve for W
            # 5. Repeat
            
            V_curr = np.random.randint(-2, 3, size=(m * p, R)).astype(float)
            W_curr = np.random.randint(-2, 3, size=(p * n, R)).astype(float)
            
            for iter in range(20):
                if time.time() > deadline:
                    break
                    
                # Solve for U
                # T_flat = sum_r U[:, r] * (V_curr[:, r] * W_curr[:, r]).flatten()
                # Let M_U be (dim x R)
                M_U = np.zeros((dim, R))
                for r in range(R):
                    M_U[:, r] = (V_curr[:, r] * W_curr[:, r]).flatten()
                
                try:
                    U_sol, residuals, rank, s = np.linalg.lstsq(M_U, T_flat, rcond=None)
                    U_curr = U_sol.reshape(n * m, R)
                except:
                    continue
                    
                # Solve for V
                # T_flat = sum_r V[:, r] * (U_curr[:, r] * W_curr[:, r]).flatten()
                M_V = np.zeros((dim, R))
                for r in range(R):
                    M_V[:, r] = (U_curr[:, r] * W_curr[:, r]).flatten()
                
                try:
                    V_sol, residuals, rank, s = np.linalg.lstsq(M_V, T_flat, rcond=None)
                    V_curr = V_sol.reshape(m * p, R)
                except:
                    continue
                    
                # Solve for W
                # T_flat = sum_r W[:, r] * (U_curr[:, r] * V_curr[:, r]).flatten()
                M_W = np.zeros((dim, R))
                for r in range(R):
                    M_W[:, r] = (U_curr[:, r] * V_curr[:, r]).flatten()
                
                try:
                    W_sol, residuals, rank, s = np.linalg.lstsq(M_W, T_flat, rcond=None)
                    W_curr = W_sol.reshape(p * n, R)
                except:
                    continue
                    
            # Check the error
            # Compute T_recon
            T_recon = np.zeros((n * m, m * p, p * n), dtype=float)
            for r in range(R):
                T_recon += np.einsum('i,j,k->ijk', U_curr[:, r], V_curr[:, r], W_curr[:, r])
                
            error = np.linalg.norm(T_recon.flatten() - T_flat)
            
            if error < 1e-6:
                # Found a solution!
                # Check if entries are close to integers or half-integers
                # Scale by 2
                U_int = np.round(U_curr * 2) / 2.0
                V_int = np.round(V_curr * 2) / 2.0
                W_int = np.round(W_curr * 2) / 2.0
                
                # Verify
                T_recon_int = np.zeros((n * m, m * p, p * n), dtype=float)
                for r in range(R):
                    T_recon_int += np.einsum('i,j,k->ijk', U_int[:, r], V_int[:, r], W_int[:, r])
                
                error_int = np.linalg.norm(T_recon_int.flatten() - T_flat)
                
                if error_int < 1e-6:
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    best_rank = R
                    break # Found a solution for this R, try lower R? No, R is increasing.
                    # We want the LOWEST rank.
                    # So if we find a solution for R, we should break the outer loop?
                    # No, we are iterating R from 25 to 59.
                    # If we find a solution for R=25, that's great.
                    # But we might find a solution for R=24 if we started lower.
                    # We started at 25.
                    # If we find a solution, we can return it.
                    # But maybe there is a solution for R=24?
                    # We can break and return.
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
                    
    # If no better solution found, return schoolbook
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
