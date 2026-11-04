import numpy as np
import time
import sys

def construct():
    n, m, p = 2, 3, 7
    N_U = n * m  # 6
    N_V = m * p  # 21
    N_W = p * n  # 14

    start_time = time.time()
    deadline = 140.0

    T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0
    T_flat_target = T_target.flatten()

    def als_step(U, V, W, max_iters=200, tol=1e-12):
        """Alternating least squares to minimize ||T_calc - T_target||."""
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            # Solve for U: fix V, W. T_target[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # Reshape: for each r, U[:,r] * (V*W)[:,r]
            # T_target reshaped: T[a,b,c] -> for fixed a, we need U[a,:] such that
            # sum_r U[a,r] * M_r[b,c] = T[a,b,c] where M_r = V[:,r] * W[:,r]^T
            # This is: U[a,:] @ (V*W)[b,c,r] = T[a,b,c]
            # Equivalently: for each a, U[a,:] solves a linear system
            # Let's flatten: T[a, (b,c)] = sum_r U[a,r] * (V[:,r] * W[:,r])[b,c]
            # So U[a, :] is the least-squares solution of:
            # (V*W)_flat[b,c,r] @ U[a,:] = T[a, b, c] flattened
            # Build A_U: shape (N_V*N_W, rank) where A_U[(b,c), r] = V[b,r]*W[c,r]
            A_U = np.einsum('br,cr->bcr', V, W).reshape(N_V*N_W, rank)
            # Solve U: for each a, U[a,:] = argmin ||A_U @ U[a,:] - T[a,:,:].flat||
            U_new, _, _, _ = np.linalg.lstsq(A_U, T_target.reshape(N_U, N_V*N_W).T, rcond=None)
            U = U_new.T
            
            # Solve for V: fix U, W
            A_V = np.einsum('ar,cr->acr', U, W).reshape(N_U*N_W, rank)
            V_new, _, _, _ = np.linalg.lstsq(A_V, T_target.reshape(N_U, N_W, N_V).transpose(0,2,1).reshape(N_U*N_W, N_V).T, rcond=None)
            V = V_new.T
            
            # Solve for W: fix U, V
            A_W = np.einsum('ar,br->abr', U, V).reshape(N_U*N_V, rank)
            W_new, _, _, _ = np.linalg.lstsq(A_W, T_target.reshape(N_U, N_V, N_W).transpose(2,0,1).reshape(N_W, N_U*N_V).T, rcond=None)
            W = W_new.T
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T_target)
            if abs(prev_err - err) < tol * max(1.0, prev_err):
                break
            prev_err = err
        return U, V, W

    def try_rank_numerical(rank, n_restarts=5, tol=1e-9):
        """Try to find a rank decomposition numerically using ALS with multiple restarts."""
        if time.time() > deadline - 10:
            return None
        
        best_U, best_V, best_W = None, None, None
        best_err = np.inf
        
        for restart in range(n_restarts):
            if time.time() > deadline - 15:
                break
            np.random.seed(42 + restart * 1000)
            U = np.random.randn(N_U, rank) * 0.1
            V = np.random.randn(N_V, rank) * 0.1
            W = np.random.randn(N_W, rank) * 0.1
            
            U, V, W = als_step(U, V, W, max_iters=300)
            
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(T_calc - T_target)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        if best_err < tol:
            # Final polish with tighter tolerance
            best_U, best_V, best_W = als_step(best_U, best_V, best_W, max_iters=500, tol=1e-14)
            T_calc = np.einsum('ar,br,cr->abc', best_U, best_V, best_W)
            if np.allclose(T_calc, T_target, atol=1e-8):
                return best_U, best_V, best_W
        
        return None

    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(N_U)]
        V = [[0] * rank for _ in range(N_V)]
        W = [[0] * rank for _ in range(N_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def get_strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        return U, V, W

    def get_2x1x2_rank4():
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
        return U, V, W

    def get_2x3x1_rank6():
        U = np.zeros((6, 6))
        V = np.zeros((3, 6))
        W = np.zeros((2, 6))
        col = 0
        for i in range(2):
            for j in range(3):
                U[i*3+j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W

    def get_2x3x2_rank11():
        U = np.zeros((6, 11))
        V = np.zeros((9, 11))
        W = np.zeros((4, 11))
        U_s, V_s, W_s = get_strassen_2x2x2()
        U_212, V_212, W_212 = get_2x1x2_rank4()
        U[0, 0:7] = U_s[0, :]
        U[1, 0:7] = U_s[1, :]
        U[3, 0:7] = U_s[2, :]
        U[4, 0:7] = U_s[3, :]
        V[0, 0:7] = V_s[0, :]
        V[1, 0:7] = V_s[1, :]
        V[3, 0:7] = V_s[2, :]
        V[4, 0:7] = V_s[3, :]
        W[0, 0:7] = W_s[0, :]
        W[1, 0:7] = W_s[1, :]
        W[2, 0:7] = W_s[2, :]
        W[3, 0:7] = W_s[3, :]
        U[2, 7:11] = U_212[0, :]
        U[5, 7:11] = U_212[1, :]
        V[6, 7:11] = V_212[0, :]
        V[7, 7:11] = V_212[1, :]
        W[0, 7:11] = W_212[0, :]
        W[1, 7:11] = W_212[1, :]
        W[2, 7:11] = W_212[2, :]
        W[3, 7:11] = W_212[3, :]
        return U, V, W

    def build_2x3x7():
        U11_a, V11_a, W11_a = get_2x3x2_rank11()
        U11_b, V11_b, W11_b = get_2x3x2_rank11()
        U11_c, V11_c, W11_c = get_2x3x2_rank11()
        U6, V6, W6 = get_2x3x1_rank6()
        R_total = 39
        U_full = np.zeros((N_U, R_total))
        V_full = np.zeros((N_V, R_total))
        W_full = np.zeros((N_W, R_total))
        
        U_full[:, 0:11] = U11_a
        for j in range(3):
            for k in range(2):
                V_full[j*7+k, 0:11] = V11_a[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[k*2+i, 0:11] = W11_a[k*2+i, :]
                
        U_full[:, 11:22] = U11_b
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+2), 11:22] = V11_b[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+2)*2+i, 11:22] = W11_b[k*2+i, :]
                
        U_full[:, 22:33] = U11_c
        for j in range(3):
            for k in range(2):
                V_full[j*7+(k+4), 22:33] = V11_c[j*2+k, :]
        for k in range(2):
            for i in range(2):
                W_full[(k+4)*2+i, 22:33] = W11_c[k*2+i, :]
                
        U_full[:, 33:39] = U6
        for j in range(3):
            V_full[j*7+6, 33:39] = V6[j, :]
        for i in range(2):
            W_full[6*2+i, 33:39] = W6[i, :]
            
        return U_full, V_full, W_full

    def try_better_decomposition():
        """Try to find a decomposition with rank < 39."""
        if time.time() > deadline - 20:
            return None
        
        # Try ranks from 38 down to 30
        for rank in range(38, 29, -1):
            if time.time() > deadline - 15:
                break
            result = try_rank_numerical(rank, n_restarts=3)
            if result is not None:
                return result
        return None

    def verify(U, V, W):
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        T_target = np.zeros((N_U, N_V, N_W), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_target[i*m+j, j*p+k, k*n+i] = 1.0
        return np.allclose(T_calc, T_target, atol=1e-9)

    U_full, V_full, W_full = build_2x3x7()

    if not verify(U_full, V_full, W_full):
        return get_schoolbook()

    best_result = (U_full, V_full, W_full)
    best_rank = 39

    # Try to find a better decomposition
    better = try_better_decomposition()
    if better is not None:
        U_b, V_b, W_b = better
        if verify(U_b, V_b, W_b):
            best_result = (U_b, V_b, W_b)
            best_rank = U_b.shape[1]

    return {
        "U": best_result[0].tolist(),
        "V": best_result[1].tolist(),
        "W": best_result[2].tolist()
    }
