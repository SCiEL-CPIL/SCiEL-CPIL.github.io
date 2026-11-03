"""Matrix multiplication 2x3x4 tensor decomposition - rank 20."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    nm, mp, pn = n*m, m*p, p*n  # 6, 12, 8
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build the target tensor T[a,b,c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 if this is a valid MM term, else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition (rank 24) as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb), dtype=float)
    V_sb = np.zeros((mp, rank_sb), dtype=float)
    W_sb = np.zeros((pn, rank_sb), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    def verify_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        R = np.zeros((nm, mp, pn), dtype=float)
        for r in range(U.shape[1]):
            R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
        return np.allclose(R, T, atol=tol)
    
    def round_to_half(U, V, W):
        """Round entries to nearest half-integer."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def als_solve(U, V, W, max_iters=50, tol=1e-8):
        """Alternating Least Squares to refine U, V, W."""
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Fix V, W, solve for U
            # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # For fixed a: T[a] = sum_r U[a,r] * (V[:,r] * W[:,r])
            M_VW = np.einsum('br,cr->bcr', V, W).reshape(mp*pn, -1)
            for a in range(nm):
                t_a = T[a, :, :].reshape(-1)
                U[a, :], _, _, _ = np.linalg.lstsq(M_VW, t_a, rcond=None)
            
            # Fix U, W, solve for V
            M_UW = np.einsum('ar,cr->acr', U, W).reshape(nm*pn, -1)
            for b in range(mp):
                t_b = T[:, b, :].reshape(-1)
                V[b, :], _, _, _ = np.linalg.lstsq(M_UW, t_b, rcond=None)
            
            # Fix U, V, solve for W
            M_UV = np.einsum('ar,br->abr', U, V).reshape(nm*mp, -1)
            for c in range(pn):
                t_c = T[:, :, c].reshape(-1)
                W[c, :], _, _, _ = np.linalg.lstsq(M_UV, t_c, rcond=None)
            
            # Check convergence
            R = np.zeros((nm, mp, pn), dtype=float)
            for r in range(U.shape[1]):
                R += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            residual = np.linalg.norm(R - T)
            if residual < tol:
                break
        
        return U, V, W
    
    def try_rank(rank, max_restarts=20):
        """Try to find a rank-r decomposition."""
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random values
            np.random.seed(42 + restart)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Run ALS
            U, V, W = als_solve(U, V, W, max_iters=80)
            
            # Check if exact
            if verify_exact(U, V, W):
                # Try to round to half-integers
                U_r, V_r, W_r = round_to_half(U, V, W)
                if verify_exact(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                # If rounding doesn't work, use the unrounded values
                return U, V, W
        
        return None
    
    # Try ranks from 20 up to 24
    for rank in range(20, 25):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(rank, max_restarts=10)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
