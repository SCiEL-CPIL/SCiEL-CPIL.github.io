"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    A, B, C = n*m, m*p, p*n  # 8, 32, 16
    
    # Build tensor T[a,b,c]
    T = np.zeros((A, B, C), dtype=float)
    for c in range(C):
        i = c % n
        k = c // n
        for a in range(A):
            j = a - i * m
            if 0 <= j < m:
                b = j * p + k
                if 0 <= b < B:
                    T[a, b, c] = 1.0
    
    rank_school = n * m * p  # 64
    
    # Try to find a decomposition with rank < schoolbook
    for rank in range(32, rank_school):
        np.random.seed(42)
        U = np.random.randn(A, rank)
        V = np.random.randn(B, rank)
        W = np.random.randn(C, rank)
        
        err = float('inf')
        for it in range(50):
            # Update U
            P = (V[:, None, :] * W[None, :, :]).reshape(B * C, rank)
            T_flat = T.reshape(A, B * C)
            U_T, _, _, _ = np.linalg.lstsq(P, T_flat.T, rcond=None)
            U = U_T.T
            
            # Update V
            P = (U[:, None, :] * W[None, :, :]).reshape(A * C, rank)
            T_flat = T.transpose(1, 0, 2).reshape(B, A * C)
            V_T, _, _, _ = np.linalg.lstsq(P, T_flat.T, rcond=None)
            V = V_T.T
            
            # Update W
            P = (U[:, None, :] * V[None, :, :]).reshape(A * B, rank)
            T_flat = T.transpose(2, 0, 1).reshape(C, A * B)
            W_T, _, _, _ = np.linalg.lstsq(P, T_flat.T, rcond=None)
            W = W_T.T
            
            # Check
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(recon - T)
            if err < 1e-8:
                break
        
        if err < 1e-8:
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            recon_round = np.einsum('ar,br,cr->abc', U_round, V_round, W_round)
            if np.linalg.norm(recon_round - T) < 1e-8:
                return {
                    "U": U_round.tolist(),
                    "V": V_round.tolist(),
                    "W": W_round.tolist()
                }
    
    # Fallback: schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(A)]
    V = [[0] * rank for _ in range(B)]
    W = [[0] * rank for _ in range(C)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
