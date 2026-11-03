import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = 2, 3, 6
    
    # Build target tensor T for 2x3x6
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Strategy: Decompose 2x3x6 into two 2x3x3 blocks.
    # Rank of 2x3x3 is 15. Total rank 30.
    
    n2, m2, p2 = 2, 3, 3
    T233 = np.zeros((n2*m2, m2*p2, p2*n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T233[i*m2+j, j*p2+k, k*n2+i] = 1.0
        
    R233 = 15
    rows_u = n2 * m2  # 6
    rows_v = m2 * p2  # 9
    rows_w = p2 * n2  # 6
    
    def generate_Z(V, W, R):
        Z = np.zeros((R, rows_v * rows_w))
        for r in range(R):
            outer = np.outer(V[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        return Z

    def solve_U(V, W, T_target, R):
        Z = generate_Z(V, W, R)
        T_flat = T_target.flatten().reshape(rows_u, -1)
        try:
            U_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return U_sol.T
        except:
            return None

    def solve_V(U, W, T_target, R):
        Z = np.zeros((R, rows_u * rows_w))
        for r in range(R):
            outer = np.outer(U[:, r], W[:, r])
            Z[r, :] = outer.flatten()
        T_flat = T_target.transpose(1, 0, 2).flatten().reshape(rows_v, -1)
        try:
            V_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return V_sol.T
        except:
            return None

    def solve_W(U, V, T_target, R):
        Z = np.zeros((R, rows_u * rows_v))
        for r in range(R):
            outer = np.outer(U[:, r], V[:, r])
            Z[r, :] = outer.flatten()
        T_flat = T_target.transpose(2, 0, 1).flatten().reshape(rows_w, -1)
        try:
            W_sol, residuals, rank, sv = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
            return W_sol.T
        except:
            return None

    def verify_233(U, V, W):
        T_check = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(T_check, T233, atol=1e-9)

    best_233 = None
    
    # Improved Search:
    # 1. SVD Initialization
    # 2. More Restarts
    # 3. Rounding Polish
    
    # Unfold T233 for SVD initialization
    # T233 shape (6, 9, 6)
    U_mat = T233.reshape(rows_u, -1)
    V_mat = T233.transpose(1, 0, 2).reshape(rows_v, -1)
    W_mat = T233.transpose(2, 0, 1).reshape(rows_w, -1)
    
    try:
        U_svd, _, _ = np.linalg.svd(U_mat, full_matrices=False)
        V_svd, _, _ = np.linalg.svd(V_mat, full_matrices=False)
        W_svd, _, _ = np.linalg.svd(W_mat, full_matrices=False)
    except:
        U_svd = np.random.randn(rows_u, R233)
        V_svd = np.random.randn(rows_v, R233)
        W_svd = np.random.randn(rows_w, R233)

    for restart in range(100):
        if time.time() > deadline - 10:
            break
        
        # Mix SVD and Random initialization
        if restart % 2 == 0:
            U = U_svd[:, :R233].copy()
            V = V_svd[:, :R233].copy()
            W = W_svd[:, :R233].copy()
            # Add small noise to escape symmetry
            U += np.random.randn(rows_u, R233) * 0.1
            V += np.random.randn(rows_v, R233) * 0.1
            W += np.random.randn(rows_w, R233) * 0.1
        else:
            U = np.random.randn(rows_u, R233) * 0.5
            V = np.random.randn(rows_v, R233) * 0.5
            W = np.random.randn(rows_w, R233) * 0.5
        
        converged = False
        for it in range(50):
            if time.time() > deadline - 10:
                break
                
            U_new = solve_U(V, W, T233, R233)
            if U_new is None: break
            V_new = solve_V(U_new, W, T233, R233)
            if V_new is None: break
            W_new = solve_W(U_new, V_new, T233, R233)
            if W_new is None: break
            
            diff = np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            
            if diff < 1e-9:
                converged = True
                break
        
        if not converged:
            continue
            
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        if verify_233(U_rounded, V_rounded, W_rounded):
            best_233 = (U_rounded, V_rounded, W_rounded)
            break

    if best_233:
        U233, V233, W233 = best_233
        R236 = 2 * R233
        U236 = np.zeros((n*m, R236))
        V236 = np.zeros((m*p, R236))
        W236 = np.zeros((p*n, R236))
        
        # Block 1
        U236[:, :R233] = U233
        V236[:m2*p2, :R233] = V233
        W236[:p2*n2, :R233] = W233
        
        # Block 2
        U236[:, R233:] = U233
        V236[m2*p2:, R233:] = V233
        W236[p2*n2:, R233:] = W233
        
        T_final = np.einsum('ir,jr,kr->ijk', U236, V236, W236)
        if np.allclose(T_final, T, atol=1e-9):
            return {
                "U": U236.tolist(),
                "V": V236.tolist(),
                "W": W236.tolist()
            }

    # Fallback to schoolbook rank 36
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
