"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    """Builds the 3x3 matrix multiplication tensor."""
    T = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u = i * 3 + j
                v = j * 3 + k
                w = k * 3 + i
                T[u, v, w] = 1
    return T

def _validate_decomposition(U, V, W):
    """Validates if the decomposition is exact and has nice entries."""
    # Check entries
    for mat in [U, V, W]:
        for val in mat:
            # Check if close to integer or half-integer
            if abs(val - round(val)) > 1e-5:
                if abs(val - (round(val * 2) / 2.0)) > 1e-5:
                    return False
    
    # Check exactness
    T = _build_target_tensor()
    T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
    if not np.allclose(T, T_rec, atol=1e-5):
        return False
    return True

def _als_decompose(T, rank, max_iter=100, tol=1e-8):
    """Alternating Least Squares for CP decomposition."""
    U = np.random.randn(9, rank)
    V = np.random.randn(9, rank)
    W = np.random.randn(9, rank)
    
    prev_error = np.inf
    for _ in range(max_iter):
        # Update U
        # T_uvw = sum_r U_ur V_vr W_wr
        # For fixed V, W:
        # T_u = sum_r U_ur (V_r . W_r) ? No.
        # T is a 3D tensor.
        # We want to minimize || T - sum U_r otimes V_r otimes W_r ||^2
        # Fix V, W.
        # T_uvw = sum_r U_ur (V_vr W_wr)
        # Let M_vw = sum_r V_vr W_wr. This is not right.
        # The term is U_ur * V_vr * W_wr.
        # For a fixed r, the contribution to T_uvw is U_ur * (V_r otimes W_r)_{vw}.
        # So T = sum_r U_r (V_r otimes W_r).
        # Let B_r = V_r otimes W_r (flattened to 81x1 vector? No, 9x9 matrix).
        # T_u = sum_r U_ur * (B_r)_{vw} ... this is getting messy with indices.
        
        # Standard ALS for 3-mode tensor:
        # Mode-1 unfolding T_(1) has shape (9, 81).
        # T_(1) = U * (V * W)
        # Where * is Khatri-Rao product.
        
        # Compute Khatri-Rao of V and W: K_VW shape (81, rank)
        # K_VW[j*9+k, r] = V[j, r] * W[k, r]
        K_VW = np.zeros((81, rank))
        for j in range(9):
            for k in range(9):
                K_VW[j*9+k, :] = V[j, :] * W[k, :]
        
        # T_(1) is T reshaped to (9, 81)
        T_1 = T.reshape(9, 81).T # (81, 9) ? No.
        # T[u, v, w]. Mode 1 is u.
        # T_(1) [u, (v,w)]
        T_1 = T.reshape(9, 81) # (9, 81)
        
        # T_1 = U @ K_VW.T
        # U = T_1 @ K_VW @ inv(K_VW.T @ K_VW)
        try:
            U = np.linalg.lstsq(K_VW.T, T_1, rcond=None)[0].T
        except:
            pass

        # Update V
        K_WU = np.zeros((81, rank))
        for k in range(9):
            for u in range(9):
                K_WU[k*9+u, :] = W[k, :] * U[u, :]
        
        # T_(2) [v, (w,u)]
        T_2 = np.transpose(T, (1, 2, 0)).reshape(9, 81)
        
        try:
            V = np.linalg.lstsq(K_WU.T, T_2, rcond=None)[0].T
        except:
            pass

        # Update W
        K_UV = np.zeros((81, rank))
        for u in range(9):
            for v in range(9):
                K_UV[u*9+v, :] = U[u, :] * V[v, :]
        
        # T_(3) [w, (u,v)]
        T_3 = np.transpose(T, (2, 0, 1)).reshape(9, 81)
        
        try:
            W = np.linalg.lstsq(K_UV.T, T_3, rcond=None)[0].T
        except:
            pass

        # Calculate error
        T_rec = np.einsum('ur,vr,wr->uvw', U, V, W)
        error = np.linalg.norm(T - T_rec)
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
        
    return U, V, W, error

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 26 solution
    def _get_rank26():
        rank = 26
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # 1. A2 B2 (Strassen)
        U[0][col] = 1; U[4][col] = 1
        V[0][col] = 1; V[4][col] = 1
        W[0][col] = 1; W[4][col] = 1
        col += 1
        
        U[3][col] = 1; U[4][col] = 1
        V[0][col] = 1
        W[1][col] = 1; W[4][col] = -1
        col += 1
        
        U[0][col] = 1
        V[1][col] = 1; V[4][col] = -1
        W[3][col] = 1; W[4][col] = 1
        col += 1
        
        U[4][col] = 1
        V[3][col] = 1; V[0][col] = -1
        W[0][col] = 1; W[1][col] = 1
        col += 1
        
        U[0][col] = 1; U[1][col] = 1
        V[4][col] = 1
        W[0][col] = -1; W[3][col] = 1
        col += 1
        
        U[3][col] = 1; U[0][col] = -1
        V[0][col] = 1; V[1][col] = 1
        W[4][col] = 1
        col += 1
        
        U[1][col] = 1; U[4][col] = -1
        V[3][col] = 1; V[4][col] = 1
        W[0][col] = 1
        col += 1
        
        # 2. a b^T
        U[2][col] = 1; V[6][col] = 1; W[0][col] = 1; col += 1
        U[2][col] = 1; V[7][col] = 1; W[3][col] = 1; col += 1
        U[5][col] = 1; V[6][col] = 1; W[1][col] = 1; col += 1
        U[5][col] = 1; V[7][col] = 1; W[4][col] = 1; col += 1
        
        # 3. A2 b
        U[0][col] = 1; V[2][col] = 1; W[6][col] = 1; col += 1
        U[1][col] = 1; V[5][col] = 1; W[6][col] = 1; col += 1
        U[3][col] = 1; V[2][col] = 1; W[7][col] = 1; col += 1
        U[4][col] = 1; V[5][col] = 1; W[7][col] = 1; col += 1
        
        # 4. a b3
        U[2][col] = 1; V[8][col] = 1; W[6][col] = 1; col += 1
        U[5][col] = 1; V[8][col] = 1; W[7][col] = 1; col += 1
        
        # 5. a^T B2
        U[6][col] = 1; V[0][col] = 1; W[2][col] = 1; col += 1
        U[7][col] = 1; V[3][col] = 1; W[2][col] = 1; col += 1
        U[6][col] = 1; V[1][col] = 1; W[5][col] = 1; col += 1
        U[7][col] = 1; V[4][col] = 1; W[5][col] = 1; col += 1
        
        # 6. a3 b^T
        U[8][col] = 1; V[6][col] = 1; W[2][col] = 1; col += 1
        U[8][col] = 1; V[7][col] = 1; W[5][col] = 1; col += 1
        
        # 7. a^T b
        U[6][col] = 1; V[2][col] = 1; W[8][col] = 1; col += 1
        U[7][col] = 1; V[5][col] = 1; W[8][col] = 1; col += 1
        
        # 8. a3 b3
        U[8][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try to find a lower rank solution using ALS
    start_time = time.time()
    time_limit = 100  # seconds
    
    T = _build_target_tensor()
    
    best_solution = _get_rank26()
    
    # Try rank 25
    for rank in [25]:
        if time.time() - start_time > time_limit:
            break
            
        # Multiple random restarts
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
                
            U, V, W, error = _als_decompose(T, rank, max_iter=50)
            
            if error < 1e-4:
                # Snap to grid
                U_snap = np.round(U * 2) / 2.0
                V_snap = np.round(V * 2) / 2.0
                W_snap = np.round(W * 2) / 2.0
                
                if _validate_decomposition(U_snap, V_snap, W_snap):
                    return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    
    return best_solution
# EVOLVE-BLOCK-END
