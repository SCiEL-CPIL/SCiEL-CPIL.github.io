import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin from 150s limit
    
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Build the target tensor T[i, j, k]
    # T[i, j, k] = 1 for all valid i, j, k
    T = np.ones((n, m, p))
    
    def compute_tensor(U, V, W, R):
        """Compute the approximated tensor from factors."""
        U3 = U.reshape(n, m, R)
        V3 = V.reshape(m, p, R)
        W3 = W.reshape(p, n, R)
        # T_approx[i, j, k] = sum_c U3[i, j, c] * V3[j, k, c] * W3[k, i, c]
        return np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    def try_rank_als(R, num_restarts=50):
        """
        Try to find a decomposition of rank R using Alternating Least Squares.
        """
        
        def solve_U(U, V, W):
            U_new = np.zeros((n, m, R))
            for i in range(n):
                for j in range(m):
                    V_j = V.reshape(m, p, R)[j]  # (p, R)
                    W_i = W.reshape(p, n, R)[:, i]  # (p, R)
                    A = V_j * W_i  # (p, R)
                    b = T[i, j, :]  # (p,)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i, j, :] = x
            return U_new.reshape(n * m, R)
        
        def solve_V(U, V, W):
            V_new = np.zeros((m, p, R))
            for j in range(m):
                for k in range(p):
                    U_j = U.reshape(n, m, R)[:, j]  # (n, R)
                    W_k = W.reshape(p, n, R)[k]  # (n, R)
                    B = U_j * W_k  # (n, R)
                    b = T[:, j, k]  # (n,)
                    x, _, _, _ = np.linalg.lstsq(B, b, rcond=None)
                    V_new[j, k, :] = x
            return V_new.reshape(m * p, R)
        
        def solve_W(U, V, W):
            W_new = np.zeros((p, n, R))
            for k in range(p):
                for i in range(n):
                    U_i = U.reshape(n, m, R)[i]  # (m, R)
                    V_k = V.reshape(m, p, R)[:, k]  # (m, R)
                    C = U_i * V_k  # (m, R)
                    b = T[i, :, k]  # (m,)
                    x, _, _, _ = np.linalg.lstsq(C, b, rcond=None)
                    W_new[k, i, :] = x
            return W_new.reshape(p * n, R)
        
        def check_and_round(U, V, W):
            """Check if the decomposition is exact and round to half-integers."""
            T_approx = compute_tensor(U, V, W, R)
            if np.allclose(T_approx, T, atol=1e-6):
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2.0
                V_rounded = np.round(V * 2) / 2.0
                W_rounded = np.round(W * 2) / 2.0
                
                T_check = compute_tensor(U_rounded, V_rounded, W_rounded, R)
                if np.allclose(T_check, T, atol=1e-9):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist(),
                        "rank": R
                    }
            return None
        
        # Try multiple restarts with different initializations
        for r in range(num_restarts):
            if time.time() - start_time > max_time - 5:
                return None
            
            # Different initialization strategies
            if r % 3 == 0:
                # Random {-1, 0, 1}
                np.random.seed(r)
                U = np.random.choice([-1, 0, 1], size=(n * m, R))
                V = np.random.choice([-1, 0, 1], size=(m * p, R))
                W = np.random.choice([-1, 0, 1], size=(p * n, R))
            elif r % 3 == 1:
                # Random {-1, 1}
                np.random.seed(r + 1000)
                U = np.random.choice([-1, 1], size=(n * m, R))
                V = np.random.choice([-1, 1], size=(m * p, R))
                W = np.random.choice([-1, 1], size=(p * n, R))
            else:
                # Random normal
                np.random.seed(r + 2000)
                U = np.random.normal(size=(n * m, R))
                V = np.random.normal(size=(m * p, R))
                W = np.random.normal(size=(p * n, R))
            
            # Run ALS
            for it in range(30):
                U = solve_U(U, V, W)
                V = solve_V(U, V, W)
                W = solve_W(U, V, W)
                
                # Check convergence / exactness
                result = check_and_round(U, V, W)
                if result:
                    return result
            
        return None
    
    # Try lower ranks
    for R in range(18, 24):
        if time.time() - start_time > max_time - 10:
            break
        # Use more restarts for lower ranks
        num_restarts = 100 if R <= 19 else 50
        res = try_rank_als(R, num_restarts)
        if res:
            return res
    
    # Fallback: Schoolbook decomposition
    rank = target_rank
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
