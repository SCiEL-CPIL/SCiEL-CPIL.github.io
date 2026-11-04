import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Build target tensor unfoldings
    # T1: (n*m, m*p * p*n)
    # T2: (m*p, n*m * p*n)
    # T3: (p*n, n*m * m*p)
    
    size1 = n * m
    size2 = m * p
    size3 = p * n
    
    T1 = np.zeros((size1, size2 * size3))
    T2 = np.zeros((size2, size1 * size3))
    T3 = np.zeros((size3, size1 * size2))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx1 = i * m + j
                idx2 = j * p + k
                idx3 = k * n + i
                
                # T1: row idx1, col idx2 * size3 + idx3
                T1[idx1, idx2 * size3 + idx3] = 1.0
                
                # T2: row idx2, col idx1 * size3 + idx3
                T2[idx2, idx1 * size3 + idx3] = 1.0
                
                # T3: row idx3, col idx1 * size2 + idx2
                T3[idx3, idx1 * size2 + idx2] = 1.0

    def khatri_rao(A, B):
        # A: (I, R), B: (J, R) -> (I*J, R)
        I, R = A.shape
        J, _ = B.shape
        return (A[:, :, None] * B[None, :, :]).reshape(I * J, R)

    def try_rank(R, time_limit):
        start_time = time.time()
        for _ in range(10): # 10 restarts
            if time.time() > time_limit:
                break
            V = np.random.randn(size2, R)
            W = np.random.randn(size3, R)
            for _ in range(50):
                # Update U
                VW = khatri_rao(V, W)
                U = T1 @ np.linalg.pinv(VW)
                # Update V
                UW = khatri_rao(U, W)
                V = T2 @ np.linalg.pinv(UW)
                # Update W
                UV = khatri_rao(U, V)
                W = T3 @ np.linalg.pinv(UV)
            
            # Check error
            VW = khatri_rao(V, W)
            U = T1 @ np.linalg.pinv(VW)
            UW = khatri_rao(U, W)
            V = T2 @ np.linalg.pinv(UW)
            UV = khatri_rao(U, V)
            W = T3 @ np.linalg.pinv(UV)
            
            # Reconstruct
            T1_rec = U @ khatri_rao(V, W).T
            err = np.linalg.norm(T1 - T1_rec)
            
            if err < 1e-5:
                # Snap to integers
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                
                # Verify
                T1_check = U_int @ khatri_rao(V_int, W_int).T
                if np.allclose(T1, T1_check):
                    return U_int.tolist(), V_int.tolist(), W_int.tolist()
        return None

    deadline = time.time() + 110
    for R in range(rank_sb - 1, 50, -1):
        if time.time() > deadline:
            break
        res = try_rank(R, deadline - 5)
        if res:
            best_solution = {"U": res[0], "V": res[1], "W": res[2]}
            best_rank = R
            break

    return best_solution
