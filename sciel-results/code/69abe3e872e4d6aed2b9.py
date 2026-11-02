import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    start_time = time.time()
    deadline = 140.0
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    def verify(U, V, W, r, tol=1e-9):
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T_recon - T)) < tol

    def rank55_decomposition():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for i in range(n):
            for k in range(p):
                U[i*m + 0, col] = 1
                V[0*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
        
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
            
            U[u_11, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_22, col] += 1
            V[v_11, col] += 1
            W[w_10, col] += 1; W[w_11, col] += -1
            col += 1
            
            U[u_11, col] += 1
            V[v_12, col] += 1; V[v_22, col] += -1
            W[w_01, col] += 1; W[w_11, col] += 1
            col += 1
            
            U[u_22, col] += 1
            V[v_21, col] += 1; V[v_11, col] += -1
            W[w_00, col] += 1; W[w_10, col] += 1
            col += 1
            
            U[u_11, col] += 1; U[u_12, col] += 1
            V[v_22, col] += 1
            W[w_00, col] += -1; W[w_01, col] += 1
            col += 1
            
            U[u_21, col] += 1; U[u_11, col] += -1
            V[v_11, col] += 1; V[v_12, col] += 1
            W[w_11, col] += 1
            col += 1
            
            U[u_12, col] += 1; U[u_22, col] += -1
            V[v_21, col] += 1; V[v_22, col] += 1
            W[w_00, col] += 1
            col += 1
            
        return U, V, W, rank

    U_best, V_best, W_best, rank_best = rank55_decomposition()
    
    def try_rank(r):
        for seed in range(20):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(seed * 100 + r)
            U_c = np.random.randn(nm, r)
            V_c = np.random.randn(mp, r)
            W_c = np.random.randn(pn, r)
            
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                M_VW = np.einsum('jr,kr->jkr', V_c, W_c).reshape(mp * pn, r)
                T_flat = T.reshape(nm, mp * pn)
                try:
                    U_c = T_flat @ np.linalg.pinv(M_VW.T)
                except:
                    return None
                
                M_UW = np.einsum('ir,kr->ikr', U_c, W_c).reshape(nm * pn, r)
                T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
                try:
                    V_c = T_flat_V @ np.linalg.pinv(M_UW.T)
                except:
                    return None
                
                M_UV = np.einsum('ir,jr->ijr', U_c, V_c).reshape(nm * mp, r)
                T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
                try:
                    W_c = T_flat_W @ np.linalg.pinv(M_UV.T)
                except:
                    return None
            
            T_recon = np.einsum('ir,jr,kr->ijk', U_c, V_c, W_c)
            error = np.max(np.abs(T_recon - T))
            
            if error < 1e-8:
                U_r = np.round(U_c * 2) / 2
                V_r = np.round(V_c * 2) / 2
                W_r = np.round(W_c * 2) / 2
                T_recon_r = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
                if np.max(np.abs(T_recon_r - T)) < 1e-9:
                    return U_r, V_r, W_r, r
        
        return None

    for r in range(rank_best - 1, 29, -1):
        if time.time() > deadline - 10:
            break
        res = try_rank(r)
        if res is not None:
            U_best, V_best, W_best, rank_best = res
            break

    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
