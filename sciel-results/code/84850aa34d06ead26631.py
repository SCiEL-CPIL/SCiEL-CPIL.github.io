import numpy as np
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def schoolbook():
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

    # Build Tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    best_result = schoolbook()
    
    # Lower bound
    lb = max(n * m, m * p, p * n)
    
    # Search for rank
    for rank in range(lb, 50):
        if time.time() > deadline:
            break
            
        # Multiple restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline:
                break
                
            np.random.seed(rank * 100 + restart)
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                T1 = T.reshape(n * m, -1)
                V_W = np.kron(V, W) # Note: np.kron is not Khatri-Rao
                # Khatri-Rao product of V and W
                # V is mp x rank, W is pn x rank
                # Result is (mp*pn) x rank
                V_W_kr = np.zeros((m * p * p * n, rank))
                for r in range(rank):
                    V_W_kr[:, r] = V[:, r] * W[:, r]
                    
                try:
                    A = V_W_kr.T @ V_W_kr
                    b = V_W_kr.T @ T1.T
                    U = np.linalg.solve(A, b).T
                except np.linalg.LinAlgError:
                    pass
                
                # Update V
                T2 = T.reshape(m * p, -1)
                U_W_kr = np.zeros((n * m * p * n, rank))
                for r in range(rank):
                    U_W_kr[:, r] = U[:, r] * W[:, r]
                try:
                    A = U_W_kr.T @ U_W_kr
                    b = U_W_kr.T @ T2.T
                    V = np.linalg.solve(A, b).T
                except np.linalg.LinAlgError:
                    pass
                    
                # Update W
                T3 = T.reshape(p * n, -1)
                U_V_kr = np.zeros((n * m * m * p, rank))
                for r in range(rank):
                    U_V_kr[:, r] = U[:, r] * V[:, r]
                try:
                    A = U_V_kr.T @ U_V_kr
                    b = U_V_kr.T @ T3.T
                    W = np.linalg.solve(A, b).T
                except np.linalg.LinAlgError:
                    pass
                    
                # Check residual
                T_approx = np.zeros((n * m, m * p, p * n))
                for r in range(rank):
                    T_approx += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                if np.linalg.norm(T - T_approx) < 1e-10:
                    # Try rounding
                    U_int = np.round(U)
                    V_int = np.round(V)
                    W_int = np.round(W)
                    
                    T_check = np.zeros((n * m, m * p, p * n))
                    for r in range(rank):
                        T_check += np.einsum('i,j,k->ijk', U_int[:, r], V_int[:, r], W_int[:, r])
                        
                    if np.allclose(T_check, T, atol=1e-9):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    found = True
                    break
            
            if found:
                break

    return best_result
# EVOLVE-BLOCK-END
