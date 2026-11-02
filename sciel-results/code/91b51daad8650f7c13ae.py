import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, r, tol=1e-9):
        # Efficient verification using einsum
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # return np.max(np.abs(T_recon - T)) < tol
        # For speed, check only non-zeros? T is sparse.
        # But nm*mp*pn is small (6*30*20 = 3600).
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        
        # Part A: j=0
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
        # Part B: j=1,2 with Strassen on 5 blocks
        for b in range(5):
            k0 = 2 * b
            k1 = 2 * b + 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            u_11 = 0 * m + 1
            u_12 = 0 * m + 2
            u_21 = 1 * m + 1
            u_22 = 1 * m + 2
            
            v_11 = 1 * p + k0
            v_12 = 1 * p + k1
            v_21 = 2 * p + k0
            v_22 = 2 * p + k1
            
            # M1
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M2
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            # M3
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            # M4
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            # M5
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            # M6
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            # M7
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()
    
    # Try to find lower rank
    # We know rank >= 30.
    # We try ranks from 54 down to 40.
    # If found, verify half-integer constraint.
    
    def try_rank(r):
        # Initialize
        # Use random initialization
        for seed in range(10):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(seed * 100 + r)
            U_c = np.random.randn(nm, r)
            V_c = np.random.randn(mp, r)
            W_c = np.random.randn(pn, r)
            
            # ALS
            for it in range(20):
                if time.time() > deadline - 5:
                    return None
                
                # Solve U
                # T_ij_k = sum_r U_ir V_jr W_kr
                # Fix V, W. T_i = sum_r U_ir (V W^T)_r
                # Flatten V, W into matrix M of shape (mp*pn, r)?
                # M[j,k, r] = V[j,r] W[k,r]
                # T[i, j, k] = sum_r U[i,r] M[j,k,r]
                # Reshape T to (nm, mp*pn)
                # Reshape M to (mp*pn, r)
                # T_flat = U @ M.T
                # U = T_flat @ pinv(M.T)
                
                M_VW = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                T_flat = T.reshape(nm, mp * pn)
                try:
                    U_c = T_flat @ np.linalg.pinv(M_VW.T)
                except:
                    return None
                
                # Solve V
                M_UW = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                T_flat_V = T.reshape(mp, nm * pn) # Note: T is (nm, mp, pn). T[j, i, k]
                # T[j, i, k] = sum_r V[j,r] U[i,r] W[k,r]
                # Reshape T to (mp, nm*pn)
                T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_c = T_flat_V @ np.linalg.pinv(M_UW.T)
                except:
                    return None
                
                # Solve W
                M_UV = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_c = T_flat_W @ np.linalg.pinv(M_UV.T)
                except:
                    return None
            
            # Check error
            T_recon = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_recon - T))
            
            if error < 1e-8:
                # Try rounding to half-integers
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2
                
                T_recon_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_recon_r - T)) < 1e-9:
                    return U_r, V_r, W_r, r
                else:
                    # If not half-integer, but exact float, is it allowed?
                    # "integer/half-integer entries"
                    # So we must discard if not half-integer.
                    pass
        
        return None

    # Search for better rank
    for r in range(rank_best - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        
        res = try_rank(r)
        if res is not None:
            U_best, V_best, W_best, rank_best = res
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
