"""Matrix multiplication 4,4,7 — rank-minimal tensor decomposition."""
DIMENSIONS = [4, 4, 7]

import time
import numpy as np

def _build_decomp():
    """Build a rank-112 decomposition (schoolbook) for 4x4x7 matmul."""
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112

    U = [[0.0] * rank for _ in range(n * m)]   # 16 x 112
    V = [[0.0] * rank for _ in range(m * p)]   # 28 x 112
    W = [[0.0] * rank for _ in range(p * n)]   # 28 x 112

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W

def construct():
    """Return factor matrices for 4x4x7 matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    nm = n * m   # 16
    mp = m * p   # 28
    pn = p * n   # 28
    
    deadline = time.time() + 135  # safety margin before 150s kill
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as baseline
    rank_school = n * m * p  # 112
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    # Try to find lower rank decompositions using peeling algorithm
    # The peeling algorithm iteratively identifies and subtracts rank-1 components
    
    def compute_error(U, V, W):
        """Compute the max error between T and the decomposition."""
        # T_approx[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # Use einsum for efficiency
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T - T_approx))
    
    def try_peeling(target_rank, n_restarts=3, max_iter=50):
        """Try to find a rank-target_rank decomposition using peeling."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 1.0
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # Alternating minimization with peeling-like approach
            # For each mode, fix the others and solve least squares
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each a, solve T[a,:,:] = sum_r U[a,r] * (V[:,r] outer W[:,r])
                # T[a].reshape(-1) = U[a,:] @ M_U where M_U[r, bc] = V[b,r]*W[c,r]
                M_U = np.zeros((target_rank, mp * pn))
                for r in range(target_rank):
                    M_U[r, :] = np.outer(V[:, r], W[:, r]).reshape(-1)
                
                T_flat = T.reshape(nm, -1)
                # Solve U @ M_U = T_flat => U = T_flat @ pinv(M_U)
                try:
                    U_new = T_flat @ np.linalg.pinv(M_U)
                except:
                    break
                
                # Update V: for each b, T[:,b,:].reshape(-1) = V[b,:] @ M_V
                M_V = np.zeros((target_rank, nm * pn))
                for r in range(target_rank):
                    M_V[r, :] = np.outer(U_new[:, r], W[:, r]).reshape(-1)
                
                T_flat_v = np.zeros((mp, nm * pn))
                for b in range(mp):
                    T_flat_v[b, :] = T[:, b, :].reshape(-1)
                
                try:
                    V_new = T_flat_v @ np.linalg.pinv(M_V)
                except:
                    break
                
                # Update W: for each c, T[:,:,c].reshape(-1) = W[c,:] @ M_W
                M_W = np.zeros((target_rank, nm * mp))
                for r in range(target_rank):
                    M_W[r, :] = np.outer(U_new[:, r], V_new[:, r]).reshape(-1)
                
                T_flat_w = np.zeros((pn, nm * mp))
                for c in range(pn):
                    T_flat_w[c, :] = T[:, :, c].reshape(-1)
                
                try:
                    W_new = T_flat_w @ np.linalg.pinv(M_W)
                except:
                    break
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                err = compute_error(U, V, W)
                if err < 1e-10:
                    if target_rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = target_rank
                    return True
                
            # Check if we found a good solution
            err = compute_error(U, V, W)
            if err < 1e-8 and target_rank < best_rank:
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                best_rank = target_rank
                return True
        
        return False
    
    # Try decreasing ranks from schoolbook down
    # Known lower bounds: for 4x4x4 matmul, rank is between 22 and 25
    # For 4x4x7, we expect rank to be less than 112
    # Try some candidate ranks
    candidate_ranks = [80, 70, 60, 50, 40, 30, 25, 22]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 15:
            break
        if try_peeling(target_rank, n_restarts=2, max_iter=30):
            # Found a valid decomposition at this rank
            break
    
    # Convert numpy arrays to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
