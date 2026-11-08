"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build the target tensor
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        return T
    
    T = build_tensor()
    
    # Try to find a lower rank decomposition
    # Start with rank 41 and go down
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try ranks from 41 down to 30, but only if time permits
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        np.random.seed(42 + target_rank)
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Iterative refinement: minimize ||T - sum U_c ⊗ V_c ⊗ W_c||
        # Using alternating least squares
        converged = False
        for it in range(50):
            if time.time() > deadline - 5.0:
                break
            
            # Update U
            for c in range(target_rank):
                # T = sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
                # Fix c, minimize over U[:,c]
                # Residual R = T - sum_{c'≠c} U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']
                R = T.copy()
                for c2 in range(target_rank):
                    if c2 == c:
                        continue
                    R -= np.einsum('a,b,c->abc', U[:,c2], V[:,c2], W[:,c2])
                # R[a,b,c] = U[a,c] * V[b,c] * W[c,c] for fixed c
                # For fixed c, R[a,b,c'] = U[a,c] * (V[:,c] ⊗ W[:,c])[b,c']
                # So U[:,c] = argmin ||R - U[:,c] ⊗ (V[:,c]⊗W[:,c])||
                # This is a least squares problem
                VW = np.outer(V[:,c], W[:,c])  # (m*p, p*n)
                # R reshaped: (n*m, m*p*p*n)
                R_flat = R.transpose(0, 1, 2).reshape(n*m, -1)
                # Solve R_flat ≈ U[:,c] @ VW
                U[:,c], _, _, _ = np.linalg.lstsq(VW.T, R_flat.T, rcond=None)
            
            # Update V
            for c in range(target_rank):
                R = T.copy()
                for c2 in range(target_rank):
                    if c2 == c:
                        continue
                    R -= np.einsum('a,b,c->abc', U[:,c2], V[:,c2], W[:,c2])
                UW = np.outer(U[:,c], W[:,c])  # (n*m, p*n)
                R_flat = R.transpose(1, 0, 2).reshape(m*p, -1)
                V[:,c], _, _, _ = np.linalg.lstsq(UW.T, R_flat.T, rcond=None)
            
            # Update W
            for c in range(target_rank):
                R = T.copy()
                for c2 in range(target_rank):
                    if c2 == c:
                        continue
                    R -= np.einsum('a,b,c->abc', U[:,c2], V[:,c2], W[:,c2])
                UV = np.outer(U[:,c], V[:,c])  # (n*m, m*p)
                R_flat = R.transpose(2, 0, 1).reshape(p*n, -1)
                W[:,c], _, _, _ = np.linalg.lstsq(UV.T, R_flat.T, rcond=None)
            
            # Check convergence
            err = np.linalg.norm(T - np.einsum('ac,bc,dc->abd', U, V, W))
            if err < 1e-6:
                converged = True
                break
        
        if converged:
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            # Verify exactness
            T_recon = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
            if np.allclose(T_recon, T, atol=1e-9):
                best_rank = target_rank
                best_result = {
                    "U": U_snap.tolist(),
                    "V": V_snap.tolist(),
                    "W": W_snap.tolist()
                }
                break
    
    return best_result
# EVOLVE-BLOCK-END
