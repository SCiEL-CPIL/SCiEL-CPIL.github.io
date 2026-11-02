import numpy as np
import time

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, r), V (m*p, r), W (p*n, r)
    # Note: The prompt says W[k*n+i, r].
    # C index is k*n + i.
    # A index is i*m + j.
    # B index is j*p + k.
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i*m + j
                    b_idx = j*p + k
                    c_idx = k*n + i
                    T[a_idx, b_idx, c_idx] = 1.0
        return T

    T = build_tensor()
    
    # Try rank 15
    target_rank = 15
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    best_solution = None
    
    # ALS for Tensor Decomposition
    max_restarts = 100
    max_iters = 30
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Normalize
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        for it in range(max_iters):
            M_VW = np.einsum('br,cr->bcr', V, W)
            U = np.einsum('abc,bcr->ar', T, M_VW)
            
            M_UW = np.einsum('ar,cr->acr', U, W)
            V = np.einsum('abc,acr->br', T, M_UW)
            
            M_UV = np.einsum('ar,br->abr', U, V)
            W = np.einsum('abc,abr->cr', T, M_UV)
            
            # Optional: normalize to prevent drift
            # U /= np.linalg.norm(U)
            # V /= np.linalg.norm(V)
            # W /= np.linalg.norm(W)

        # Check solution
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.linalg.norm(T - T_approx)
        
        if err < 1e-8:
            # Try to snap to half-integers
            def snap_to_half(x):
                return np.round(x * 2) / 2
            
            U_snap = snap_to_half(U)
            V_snap = snap_to_half(V)
            W_snap = snap_to_half(W)
            
            T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
            if np.allclose(T_snap, T, atol=1e-10):
                best_solution = (U_snap, V_snap, W_snap)
                break
        
        # Also check if the un-snapped solution is already close to half-integers
        if np.allclose(U, snap_to_half(U), atol=1e-6) and np.allclose(V, snap_to_half(V), atol=1e-6) and np.allclose(W, snap_to_half(W), atol=1e-6):
             U_snap = snap_to_half(U)
             V_snap = snap_to_half(V)
             W_snap = snap_to_half(W)
             T_snap = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap)
             if np.allclose(T_snap, T, atol=1e-10):
                 best_solution = (U_snap, V_snap, W_snap)
                 break

    if best_solution:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
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
