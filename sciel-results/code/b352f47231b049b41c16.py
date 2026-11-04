import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # Schoolbook rank
    rank_school = n * m * p  # 54
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Try to find a lower rank decomposition for 3x3x3, then concatenate
    # to get 3x3x6. If we find rank R3 for 3x3x3, we get rank 2*R3 for 3x3x6.
    
    def check_3x3_decomp(U, V, W, R):
        """Check if U, V, W of shape (9, R) exactly decompose 3x3x3 multiplication."""
        U_np = np.array(U, dtype=np.float64)
        V_np = np.array(V, dtype=np.float64)
        W_np = np.array(W, dtype=np.float64)
        
        # Build target tensor T of shape (9, 9, 9)
        T_target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T_target[i*3+j, j*3+k, k*3+i] = 1.0
        
        # Compute T_comp = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # Efficiently: for each c, add outer product
        T_comp = np.zeros((9, 9, 9))
        for c in range(R):
            T_comp += np.einsum('a,b,c->abc', U_np[:,c], V_np[:,c], W_np[:,c])
        
        if not np.allclose(T_comp, T_target, atol=1e-9):
            return False
        return True

    def try_find_3x3_decomp(R, max_restarts, time_budget):
        """Try to find a rank-R decomposition of 3x3x3 matrix multiplication."""
        best_err = np.inf
        best_UVW = None
        
        for restart in range(max_restarts):
            if time.time() > start_time + time_budget:
                break
            
            # Random initialization
            U = np.random.randn(9, R) * 0.5
            V = np.random.randn(9, R) * 0.5
            W = np.random.randn(9, R) * 0.5
            
            # ALS iterations
            for it in range(200):
                if time.time() > start_time + time_budget:
                    break
                
                # Update U: U[:,c] = T[:,:,c] / (V[:,c] * W[:,c]) in some sense
                # Actually, for tensor T = sum_c u_c v_c w_c, the optimal u_c given v_c, w_c is:
                # u_c = T_(2,3) / (v_c * w_c) where T_(2,3) is the matricization
                # But this is for exact decomposition. For least squares, we use:
                # U[:,c] = (T_unfolded @ diag(V[:,c] * W[:,c])) / sum((V[:,c] * W[:,c])^2)
                
                # Simpler: use gradient descent with small steps
                # Compute T_comp
                T_comp = np.zeros((9, 9, 9))
                for c in range(R):
                    T_comp += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                
                T_target = np.zeros((9, 9, 9))
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            T_target[i*3+j, j*3+k, k*3+i] = 1.0
                
                err = np.linalg.norm(T_comp - T_target)
                if err < best_err:
                    best_err = err
                    best_UVW = (U.copy(), V.copy(), W.copy())
                
                if err < 1e-9:
                    break
                
                # Gradient descent step
                lr = 0.01 / (1 + it * 0.01)
                for c in range(R):
                    dU = 2 * np.einsum('bcd->a', (T_comp - T_target), V[:,c], W[:,c])
                    dV = 2 * np.einsum('acd->b', (T_comp - T_target), U[:,c], W[:,c])
                    dW = 2 * np.einsum('abc->d', (T_comp - T_target), U[:,c], V[:,c])
                    U[:,c] -= lr * dU
                    V[:,c] -= lr * dV
                    W[:,c] -= lr * dW
            
            if time.time() > start_time + time_budget:
                break
        
        if best_UVW is not None and best_err < 1e-8:
            U, V, W = best_UVW
            if check_3x3_decomp(U, V, W, R):
                return U, V, W
        return None

    # Try ranks for 3x3x3: start from 23 (known rank) up to 27
    for R3 in [23, 24, 25, 26, 27]:
        if time.time() > deadline - 30:
            break
        
        # Time budget for this rank attempt
        time_budget = min(40.0, deadline - time.time() - 10.0)
        if time_budget < 5.0:
            break
        
        max_restarts = 10
        result = try_find_3x3_decomp(R3, max_restarts, time_budget)
        
        if result is not None:
            U3, V3, W3 = result
            R = 2 * R3
            if R < rank_school:
                # Build 3x3x6 decomposition by concatenating two 3x3x3 decompositions
                U6 = np.zeros((n*m, R))
                V6 = np.zeros((m*p, R))
                W6 = np.zeros((p*n, R))
                
                # First half: k=0,1,2 (first 3x3x3 block)
                # U6[:, 0:R3] = U3
                U6[:, 0:R3] = U3
                # V6: for j in 0..2, k in 0..2 -> j*p+k where p=6
                # For k=0,1,2: V6[j*6+k, c] = V3[j*3+k, c]
                for j in range(3):
                    for k in range(3):
                        V6[j*6+k, 0:R3] = V3[j*3+k, 0:R3]
                # W6: for k in 0..2, i in 0..2 -> k*n+i
                # For k=0,1,2: W6[k*3+i, c] = W3[k*3+i, c]
                for k in range(3):
                    for i in range(3):
                        W6[k*3+i, 0:R3] = W3[k*3+i, 0:R3]
                
                # Second half: k=3,4,5 (second 3x3x3 block)
                # U6[:, R3:2*R3] = U3
                U6[:, R3:2*R3] = U3
                # V6: for j in 0..2, k in 3..5 -> j*p+k
                # V6[j*6+k, R3+c] = V3[j*3+(k-3), c]
                for j in range(3):
                    for k in range(3):
                        V6[j*6+(k+3), R3:2*R3] = V3[j*3+k, 0:R3]
                # W6: for k in 3..5, i in 0..2 -> k*n+i
                # W6[k*3+i, R3+c] = W3[(k-3)*3+i, c]
                for k in range(3):
                    for i in range(3):
                        W6[(k+3)*3+i, R3:2*R3] = W3[k*3+i, 0:R3]
                
                # Verify
                U6_np = np.array(U6, dtype=np.float64)
                V6_np = np.array(V6, dtype=np.float64)
                W6_np = np.array(W6, dtype=np.float64)
                
                T_target = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_target[i*m+j, j*p+k, k*n+i] = 1.0
                
                T_comp = np.zeros((n*m, m*p, p*n))
                for c in range(R):
                    T_comp += np.einsum('a,b,c->abc', U6_np[:,c], V6_np[:,c], W6_np[:,c])
                
                if np.allclose(T_comp, T_target, atol=1e-9):
                    # Convert to list of lists with half-integer rounding
                    U_list = [[round(U6[i][c] * 2) / 2 for c in range(R)] for i in range(n*m)]
                    V_list = [[round(V6[i][c] * 2) / 2 for c in range(R)] for i in range(m*p)]
                    W_list = [[round(W6[i][c] * 2) / 2 for c in range(R)] for i in range(p*n)]
                    
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
