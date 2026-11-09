"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    schoolbook_rank = n * m * p
    best_U = np.zeros((nm, schoolbook_rank))
    best_V = np.zeros((mp, schoolbook_rank))
    best_W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1.0
                best_V[j*p+k, col] = 1.0
                best_W[k*n+i, col] = 1.0
                col += 1
    best_rank = schoolbook_rank
    deadline = time.time() + 130
    
    def check_exact(U, V, W, r):
        T_rec = np.zeros((nm, mp, pn))
        for s in range(r):
            T_rec += np.einsum('i,j,k->ijk', U[:, s], V[:, s], W[:, s])
        return np.max(np.abs(T_rec - T))
    
    def try_rank(R, max_iter=50, tol=1e-12):
        """Try to find rank-R decomposition using Newton-Raphson on polynomial system."""
        for seed in range(20):
            if time.time() > deadline - 10:
                return None
            np.random.seed(seed * 1000003 + R * 7919)
            
            # Initialize with random matrices
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Scale to reasonable magnitude
            U /= np.sqrt(np.sum(U**2))
            V /= np.sqrt(np.sum(V**2))
            W /= np.sqrt(np.sum(W**2))
            
            for iteration in range(max_iter):
                if time.time() > deadline - 10:
                    return None
                
                # Compute residual: F(a,b,c) = sum_s U[a,s]*V[b,s]*W[c,s] - T[a,b,c]
                # Jacobian: dF/da = sum_s V[b,s]*W[c,s] * dU[a,s]/da
                #          dF/db = sum_s U[a,s]*W[c,s] * dV[b,s]/db
                #          dF/dc = sum_s U[a,s]*V[b,s] * dW[c,s]/dc
                
                # Use Gauss-Newton approximation (ignore second-order terms in Jacobian)
                # This is more stable than full Newton for this problem
                
                # Update U: fix V, W, solve for U
                # For each (b,c), we have: sum_s U[a,s] * (V[b,s]*W[c,s]) = T[a,b,c]
                # Let M_U[a, s] = sum over (b,c) of V[b,s]*W[c,s] * ... 
                # Actually, let's do alternating least squares first, then polish
                
                # ALS step 1: update U
                M = np.einsum('bs,cs->sbc', V, W).reshape(R, mp * pn)
                Y = T.reshape(nm, mp * pn)
                MMt = M @ M.T
                try:
                    U_new = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    U_new = Y @ M.T @ np.linalg.pinv(MMt)
                
                # ALS step 2: update V
                M = np.einsum('as,cs->sac', U_new, W).reshape(R, nm * pn)
                Y = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                MMt = M @ M.T
                try:
                    V_new = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    V_new = Y @ M.T @ np.linalg.pinv(MMt)
                
                # ALS step 3: update W
                M = np.einsum('as,bs->sab', U_new, V_new).reshape(R, nm * mp)
                Y = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                MMt = M @ M.T
                try:
                    W_new = Y @ M.T @ np.linalg.inv(MMt)
                except:
                    W_new = Y @ M.T @ np.linalg.pinv(MMt)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                err = check_exact(U, V, W, R)
                if err < tol:
                    # Try to refine to rational entries
                    for scale in [2, 4, 8, 16, 32]:
                        U_r = np.round(U * scale) / scale
                        V_r = np.round(V * scale) / scale
                        W_r = np.round(W * scale) / scale
                        err_r = check_exact(U_r, V_r, W_r, R)
                        if err_r < 1e-9:
                            return (U_r, V_r, W_r)
                    # If no rational found, return the floating point result
                    return (U, V, W)
            
            # After max iterations, check if we got close
            err = check_exact(U, V, W, R)
            if err < 1e-6:
                for scale in [2, 4, 8, 16, 32, 64]:
                    U_r = np.round(U * scale) / scale
                    V_r = np.round(V * scale) / scale
                    W_r = np.round(W * scale) / scale
                    err_r = check_exact(U_r, V_r, W_r, R)
                    if err_r < 1e-9:
                        return (U_r, V_r, W_r)
        
        return None
    
    # Try ranks from schoolbook down
    for target_rank in range(schoolbook_rank - 1, 29, -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, max_iter=30)
        if result is not None:
            U, V, W = result
            err = check_exact(U, V, W, target_rank)
            if err < 1e-9:
                best_U = U
                best_V = V
                best_W = W
                best_rank = target_rank
                break
    
    U = best_U[:, :best_rank]
    V = best_V[:, :best_rank]
    W = best_W[:, :best_rank]
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
