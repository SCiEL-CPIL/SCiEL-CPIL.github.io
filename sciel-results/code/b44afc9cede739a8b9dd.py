import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    def check_decomposition(U, V, W):
        """
        Check if sum_l U[i*m+j, l] V[j*p+k, l] W[k*n+i, l] == 1 for all i,j,k
        U: (n*m, r), V: (m*p, r), W: (p*n, r)
        """
        r = U.shape[1]
        U_3d = U.reshape(n, m, r)
        V_3d = V.reshape(m, p, r)
        W_3d = W.reshape(p, n, r)
        
        # T[i, j, k] = sum_l U[i, j, l] V[j, k, l] W[k, i, l]
        T = np.einsum('ijl,jkl,kil->ijk', U_3d, V_3d, W_3d)
        
        return np.allclose(T, 1.0, atol=1e-9)

    def refine(U, V, W, max_iters=50):
        """
        ALS refinement step.
        """
        r = U.shape[1]
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n):
                # W_i_mat[k, :] = W[k*n + i, :]
                W_i_mat = np.array([W[k*n + i, :] for k in range(p)]) # (p, r)
                for j in range(m):
                    V_j = V[j*p:(j+1)*p, :] # (p, r)
                    A = V_j * W_i_mat # (p, r)
                    b = np.ones(p)
                    try:
                        # Use lstsq for underdetermined system
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        U_new[i*m + j, :] = x
                    except:
                        U_new[i*m + j, :] = U[i*m + j, :]
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                    b = np.ones(n)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        V_new[j*p + k, :] = x
                    except:
                        V_new[j*p + k, :] = V[j*p + k, :]
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                    b = np.ones(m)
                    try:
                        x = np.linalg.lstsq(A, b, rcond=None)[0]
                        W_new[k*n + i, :] = x
                    except:
                        W_new[k*n + i, :] = W[k*n + i, :]
            W = W_new
            
            # Check convergence and snapping
            if check_decomposition(U, V, W):
                # Try to snap to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if check_decomposition(U_rounded, V_rounded, W_rounded):
                    return U_rounded, V_rounded, W_rounded
                else:
                    # Try integers
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    if check_decomposition(U_int, V_int, W_int):
                        return U_int, V_int, W_int
                    else:
                        # If rounding fails, return None
                        return None
        return None

    # Search for low rank
    # Start from 95 down to 80
    # Use multiple random restarts
    
    # We will iterate through ranks and trials
    # To maximize chances, we interleave ranks or just go down?
    # Going down is better because lower rank is better score.
    
    for r in range(95, 79, -1):
        if time.time() > deadline - 10:
            break
            
        # Number of trials depends on remaining time
        # Estimate time per trial: ~100ms?
        # If 100s left, we can do 1000 trials.
        # But let's cap it to avoid too long single rank search if it's hopeless.
        # However, if we find a solution, we return immediately.
        
        found = False
        trial_count = 0
        while time.time() < deadline - 10:
            if found:
                break
            # Limit trials per rank to avoid getting stuck if no solution exists
            if trial_count > 200: 
                break
            
            np.random.seed(r * 1000 + trial_count)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            result = refine(U, V, W, max_iters=30)
            if result is not None:
                U_res, V_res, W_res = result
                return {
                    "U": U_res.tolist(),
                    "V": V_res.tolist(),
                    "W": W_res.tolist()
                }
            trial_count += 1
            
    # Fallback to schoolbook
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
