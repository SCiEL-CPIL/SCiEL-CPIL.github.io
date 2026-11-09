import numpy as np
import time
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110.0  # Safety margin

    dim_u = n * m
    dim_v = m * p
    dim_w = p * n

    # Build the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx, col] = 1.0
                    V[v_idx, col] = 1.0
                    W[w_idx, col] = 1.0
                    col += 1
        return U, V, W, rank

    def check_half_integer(U, V, W, T_target):
        """
        Check if rounding to nearest 0.5 gives exact decomposition.
        """
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        
        # Reconstruct and check
        try:
            T_rec = np.einsum('ur,vr,wr->uvw', U_r, V_r, W_r)
            if np.allclose(T_rec, T_target, atol=1e-9):
                return U_r, V_r, W_r
        except Exception:
            pass
        return None

    def als_optimize(rank, n_restarts=10, max_iters=200, T_target=None, dim_u=None, dim_v=None, dim_w=None):
        """
        Alternating Least Squares optimization for tensor decomposition.
        """
        if T_target is None:
            T_target = T
        if dim_u is None:
            dim_u = n * m
        if dim_v is None:
            dim_v = m * p
        if dim_w is None:
            dim_w = p * n
            
        best_err = float('inf')
        best_solution = None
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = np.random.randn(dim_u, rank)
            V = np.random.randn(dim_v, rank)
            W = np.random.randn(dim_w, rank)
            
            current_err = float('inf')
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                    
                # Solve for U: T_{u,v,w} = sum_r U_{u,r} V_{r,v} W_{r,w}
                T_u = T_target.reshape(dim_u, dim_v * dim_w)
                A_u = np.einsum('vr,wr->vwr', V, W).reshape(dim_v * dim_w, rank)
                
                try:
                    U_new = np.linalg.lstsq(A_u, T_u, rcond=None)[0].T
                except:
                    break
                
                # Solve for V
                T_v = T_target.reshape(dim_u * dim_w, dim_v).T
                A_v = np.einsum('ur,wr->uwr', U_new, W).reshape(dim_u * dim_w, rank)
                try:
                    V_new = np.linalg.lstsq(A_v, T_v, rcond=None)[0].T
                except:
                    break

                # Solve for W
                T_w = T_target.reshape(dim_u * dim_v, dim_w).T
                A_w = np.einsum('ur,vr->uvr', U_new, V_new).reshape(dim_u * dim_v, rank)
                try:
                    W_new = np.linalg.lstsq(A_w, T_w, rcond=None)[0].T
                except:
                    break

                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
                err = np.linalg.norm(T_target - T_rec)
                
                if err < 1e-9:
                    break
                if err < current_err:
                    current_err = err
                    if current_err < best_err:
                        best_err = current_err
                        best_solution = (U.copy(), V.copy(), W.copy())
                        # Check if it's a valid half-integer solution
                        sol = check_half_integer(U, V, W, T_target)
                        if sol is not None:
                            return sol
            # End ALS loop
        # End restarts
        
        # Final check on best solution found
        if best_solution is not None:
            sol = check_half_integer(best_solution[0], best_solution[1], best_solution[2], T_target)
            if sol is not None:
                return sol
        return None

    # Border rank lifting technique:
    # 1. Build 2x3x11 tensor
    # 2. Find low-rank decomposition
    # 3. Project down to 2x3x10
    
    p_large = 11
    dim_u_large = n * m
    dim_v_large = m * p_large
    dim_w_large = p_large * n
    
    T_large = np.zeros((dim_u_large, dim_v_large, dim_w_large))
    for i in range(n):
        for j in range(m):
            for k in range(p_large):
                u_idx = i * m + j
                v_idx = j * p_large + k
                w_idx = k * n + i
                T_large[u_idx, v_idx, w_idx] = 1.0

    random.seed(42)
    np.random.seed(42)
    
    # Try border rank lifting with rank 18
    found_large = als_optimize(18, n_restarts=20, max_iters=100, 
                               T_target=T_large, 
                               dim_u=dim_u_large, 
                               dim_v=dim_v_large, 
                               dim_w=dim_w_large)
    
    if found_large is not None:
        U_large, V_large, W_large = found_large
        
        # Project down to 2x3x10
        # We need to eliminate terms corresponding to the 11th column (k=10)
        # The 11th column in V corresponds to indices where v_idx has k=10
        # v_idx = j * 11 + k, so for k=10: v_idx = j*11 + 10 for j in [0,1,2]
        
        # Check if the decomposition has a structure where the 11th column can be eliminated
        # This is complex; for now, let's try a simpler approach:
        # Just try to find a direct decomposition of 2x3x10 with lower rank
        
        pass
    
    # Fallback: Try direct search for 2x3x10 with improved ALS
    # Start from 59, go down
    
    for rank in range(59, 54, -1):
        if time.time() > deadline:
            break
        found = als_optimize(rank, n_restarts=20, max_iters=50)
        if found is not None:
            U, V, W = found
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to Schoolbook
    U, V, W, rank = get_schoolbook()
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
