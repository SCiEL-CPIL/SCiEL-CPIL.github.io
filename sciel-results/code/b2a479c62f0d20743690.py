"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    deadline = time.time() + 110
    
    # Build target tensor T as (nm, mp, pn)
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    def verify(U, V, W):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.linalg.norm(T_rec - T)

    def solve_U(U, V, W, T):
        U_new = np.zeros_like(U)
        r = U.shape[1]
        Phi_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, r)
        Phi_VW_T = Phi_VW.T
        for a in range(nm):
            target = T[a].flatten()
            x, _, _, _ = np.linalg.lstsq(Phi_VW_T, target, rcond=None)
            U_new[a, :] = x
        return U_new

    def solve_V(U, V, W, T):
        V_new = np.zeros_like(V)
        r = V.shape[1]
        Phi_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, r)
        Phi_UW_T = Phi_UW.T
        for b in range(mp):
            target = T[:, b, :].flatten()
            x, _, _, _ = np.linalg.lstsq(Phi_UW_T, target, rcond=None)
            V_new[b, :] = x
        return V_new

    def solve_W(U, V, W, T):
        W_new = np.zeros_like(W)
        r = W.shape[1]
        Phi_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, r)
        Phi_UV_T = Phi_UV.T
        for c in range(pn):
            target = T[:, :, c].flatten()
            x, _, _, _ = np.linalg.lstsq(Phi_UV_T, target, rcond=None)
            W_new[c, :] = x
        return W_new

    best_result = None
    best_rank = n * m * p  # schoolbook = 96
    
    # Generate schoolbook as safe fallback
    U_fb = np.zeros((nm, best_rank))
    V_fb = np.zeros((mp, best_rank))
    W_fb = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m + j, col] = 1
                V_fb[j*p + k, col] = 1
                W_fb[k*n + i, col] = 1
                col += 1
                
    def to_half_int(M):
        return np.round(M * 2) / 2

    # Try ranks from 35 upward
    for r in range(35, min(96, best_rank + 1)):
        if time.time() > deadline - 10:
            break
            
        found = False
        n_tries = 5
        for trial in range(n_tries):
            if time.time() > deadline - 5:
                break
            
            # Initialization
            U = np.random.randn(nm, r) * 0.5
            V = np.random.randn(mp, r) * 0.5
            W = np.random.randn(pn, r) * 0.5
            
            # ALS
            best_err = verify(U, V, W)
            for iteration in range(30):
                if time.time() > deadline - 5:
                    break
                U = solve_U(U, V, W, T)
                V = solve_V(U, V, W, T)
                W = solve_W(U, V, W, T)
                err = verify(U, V, W)
                if err < best_err:
                    best_err = err
                if err < 1e-8:
                    break
            
            if best_err < 1e-5:
                # Try to round to half-integers
                U_r = to_half_int(U)
                V_r = to_half_int(V)
                W_r = to_half_int(W)
                err_r = verify(U_r, V_r, W_r)
                if err_r < 1e-4:
                    best_rank = r
                    best_result = (U_r, V_r, W_r)
                    found = True
                    break
                else:
                    # Try integer rounding
                    U_r = np.round(U)
                    V_r = np.round(V)
                    W_r = np.round(W)
                    err_r = verify(U_r, V_r, W_r)
                    if err_r < 1e-4:
                        best_rank = r
                        best_result = (U_r, V_r, W_r)
                        found = True
                        break
        
        if found:
            break

    if best_result is not None:
        U, V, W = best_result
        U_list = [[float(U[i, j]) for j in range(len(U[0]))] for i in range(len(U))]
        V_list = [[float(V[i, j]) for j in range(len(V[0]))] for i in range(len(V))]
        W_list = [[float(W[i, j]) for j in range(len(W[0]))] for i in range(len(W))]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(U_fb[i, j]) for j in range(len(U_fb[0]))] for i in range(len(U_fb))]
        V_list = [[float(V_fb[i, j]) for j in range(len(V_fb[0]))] for i in range(len(V_fb))]
        W_list = [[float(W_fb[i, j]) for j in range(len(W_fb[0]))] for i in range(len(W_fb))]
        return {"U": U_list, "V": V_list, "W": W_list}
        
# EVOLVE-BLOCK-END
