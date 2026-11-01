import time
import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 5
    deadline = time.time() + 140
    
    # Return rank-18 solution as fallback
    def _rank18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_strassen_block(k_offset):
            nonlocal col
            U[0][col] = 1; U[3][col] = 1
            V[0*5 + k_offset][col] = 1
            V[1*5 + (k_offset+1)][col] = 1
            W[k_offset*2 + 0][col] = 1
            W[(k_offset+1)*2 + 1][col] = 1
            col += 1
            
            U[2][col] = 1; U[3][col] = 1
            V[0*5 + k_offset][col] = 1
            W[k_offset*2 + 1][col] = 1
            W[(k_offset+1)*2 + 1][col] = -1
            col += 1
            
            U[0][col] = 1
            V[0*5 + (k_offset+1)][col] = 1
            V[1*5 + (k_offset+1)][col] = -1
            W[(k_offset+1)*2 + 0][col] = 1
            W[(k_offset+1)*2 + 1][col] = 1
            col += 1
            
            U[3][col] = 1
            V[1*5 + k_offset][col] = 1
            V[0*5 + k_offset][col] = -1
            W[k_offset*2 + 0][col] = 1
            W[k_offset*2 + 1][col] = 1
            col += 1
            
            U[0][col] = 1; U[1][col] = 1
            V[1*5 + (k_offset+1)][col] = 1
            W[k_offset*2 + 0][col] = -1
            W[(k_offset+1)*2 + 0][col] = 1
            col += 1
            
            U[2][col] = 1; U[0][col] = -1
            V[0*5 + k_offset][col] = 1
            V[0*5 + (k_offset+1)][col] = 1
            W[(k_offset+1)*2 + 1][col] = 1
            col += 1
            
            U[1][col] = 1; U[3][col] = -1
            V[1*5 + k_offset][col] = 1
            V[1*5 + (k_offset+1)][col] = 1
            W[k_offset*2 + 0][col] = 1
            col += 1
        
        add_strassen_block(0)
        add_strassen_block(2)
        
        U[0][col] = 1; V[0*5+4][col] = 1; W[4*2+0][col] = 1; col += 1
        U[1][col] = 1; V[1*5+4][col] = 1; W[4*2+0][col] = 1; col += 1
        U[2][col] = 1; V[0*5+4][col] = 1; W[4*2+1][col] = 1; col += 1
        U[3][col] = 1; V[1*5+4][col] = 1; W[4*2+1][col] = 1; col += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Build target tensor
    def _build_target():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    def _check(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        T = _build_target()
        recon = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
        return np.allclose(recon, T, atol=1e-10)
    
    def _als_solve(rank, max_iter=200, tol=1e-12):
        T = _build_target()
        nA, nB, nC = n*m, m*p, p*n
        
        np.random.seed(42)
        U = np.random.randn(nA, rank)
        V = np.random.randn(nB, rank)
        W = np.random.randn(nC, rank)
        
        for it in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve U
            U_flat = U.flatten()
            VTW = (V.T @ W)  # rank x rank
            # Reshape T to (nA, nB*nC)
            T_mat = T.reshape(nA, nB * nC)
            # Build system: U @ (V.T ⊗ W.T) = T.reshape
            # For each a: sum_r U[a,r] * sum_bc V[b,r]*W[c,r] = T[a,b,c]
            # This is: U @ M = T_mat where M[r, (b,c)] = V[b,r]*W[c,r]
            M = np.zeros((rank, nB * nC))
            for r in range(rank):
                M[r, :] = (V[:, r] * W[:, r]).flatten()
            # Solve U @ M = T_mat
            U_new, _, _, _ = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
            U_new = U_new.T
            
            # Solve V
            M = np.zeros((rank, nA * nC))
            for r in range(rank):
                M[r, :] = (U[:, r] * W[:, r]).flatten()
            T_mat = T.reshape(nB, nA * nC)
            V_new, _, _, _ = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
            V_new = V_new.T
            
            # Solve W
            M = np.zeros((rank, nA * nB))
            for r in range(rank):
                M[r, :] = (U[:, r] * V[:, r]).flatten()
            T_mat = T.reshape(nC, nA * nB)
            W_new, _, _, _ = np.linalg.lstsq(M.T, T_mat.T, rcond=None)
            W_new = W_new.T
            
            err = np.linalg.norm(U_new - U) + np.linalg.norm(V_new - V) + np.linalg.norm(W_new - W)
            U, V, W = U_new, V_new, W_new
            if err < tol:
                break
        
        if not _check(U.tolist(), V.tolist(), W.tolist()):
            return None
        
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try rank 16
    for rank in [16, 17]:
        if time.time() > deadline - 10:
            break
        result = _als_solve(rank)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    return _rank18()
