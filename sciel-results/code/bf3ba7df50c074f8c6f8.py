import time
import numpy as np

DIMENSIONS = [3, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def verify_decomp(U, V, W, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def trivial_decomp():
        U = [[0] * 27 for _ in range(9)]
        V = [[0] * 27 for _ in range(9)]
        W = [[0] * 27 for _ in range(9)]
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i * 3 + j][col] = 1
                    V[j * 3 + k][col] = 1
                    W[k * 3 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    target_rank = 23
    best_result = None

    # Vectorized ALS
    for restart in range(100):
        if time.time() > deadline - 10:
            break

        np.random.seed(1000 + restart)
        # Initialize with small random values
        U = np.random.randn(9, target_rank) * 0.5
        V = np.random.randn(9, target_rank) * 0.5
        W = np.random.randn(9, target_rank) * 0.5

        # Reshape for easier indexing
        # U[i, j, r] where index is i*m + j
        # V[j, k, r] where index is j*p + k
        # W[k, i, r] where index is k*n + i
        U_3d = U.reshape(3, 3, target_rank)
        V_3d = V.reshape(3, 3, target_rank)
        W_3d = W.reshape(3, 3, target_rank)

        for iteration in range(500):
            if time.time() > deadline - 5:
                break

            # Update U
            for i in range(3):
                for j in range(3):
                    # Construct M_ij: shape (3, R)
                    # Row k is V[j, k, :] * W[k, i, :]
                    M_ij = V_3d[j, :, :] * W_3d[:, i, :] # Broadcasting: (3, R) * (3, R) -> (3, R)
                    # Solve M_ij.T @ u = ones(3)
                    # M_ij.T @ M_ij @ u = M_ij.T @ ones(3)
                    try:
                        A = M_ij.T @ M_ij
                        b = M_ij.T @ np.ones(3)
                        # Add small regularization for stability
                        A += 1e-8 * np.eye(target_rank)
                        U_3d[i, j, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Update V
            for j in range(3):
                for k in range(3):
                    # Construct M_jk: shape (3, R)
                    # Row i is U[i, j, :] * W[k, i, :]
                    M_jk = U_3d[:, j, :] * W_3d[k, :, :] # Broadcasting: (3, R) * (3, R) -> (3, R)
                    # Note: W index is k*n+i, so W_3d[k, i, :]
                    # U index is i*m+j, so U_3d[i, j, :]
                    # The term is U[i,j,r] * W[k,i,r]. Sum over r? No, product is element-wise.
                    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i.
                    # Fix j, k. V[j,k,r] is unknown.
                    # Sum_r V[j,k,r] (U[i,j,r] W[k,i,r]) = 1 for all i.
                    # Let M_jk[i, r] = U[i,j,r] * W[k,i,r].
                    # Then M_jk.T @ v = ones(3).
                    try:
                        A = M_jk.T @ M_jk
                        b = M_jk.T @ np.ones(3)
                        A += 1e-8 * np.eye(target_rank)
                        V_3d[j, k, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Update W
            for k in range(3):
                for i in range(3):
                    # Construct M_ki: shape (3, R)
                    # Row j is U[i, j, :] * V[j, k, :]
                    M_ki = U_3d[i, :, :] * V_3d[:, k, :] # Broadcasting: (3, R) * (3, R) -> (3, R)
                    # We want sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all j.
                    # Fix k, i. W[k,i,r] is unknown.
                    # Sum_r W[k,i,r] (U[i,j,r] V[j,k,r]) = 1 for all j.
                    # Let M_ki[j, r] = U[i,j,r] * V[j,k,r].
                    # Then M_ki.T @ w = ones(3).
                    try:
                        A = M_ki.T @ M_ki
                        b = M_ki.T @ np.ones(3)
                        A += 1e-8 * np.eye(target_rank)
                        W_3d[k, i, :] = np.linalg.solve(A, b)
                    except np.linalg.LinAlgError:
                        pass

            # Check error
            if iteration % 50 == 0:
                max_err = 0.0
                # Flatten back to 2D for calculation or use 3D
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            s = np.sum(U_3d[i, j, :] * V_3d[j, k, :] * W_3d[k, i, :])
                            err = abs(s - 1.0)
                            if err > max_err:
                                max_err = err
                if max_err < 1e-8:
                    break

        # Final verification
        U_flat = U_3d.reshape(9, target_rank)
        V_flat = V_3d.reshape(9, target_rank)
        W_flat = W_3d.reshape(9, target_rank)
        
        if verify_decomp(U_flat.tolist(), V_flat.tolist(), W_flat.tolist()):
            best_result = {"U": U_flat.tolist(), "V": V_flat.tolist(), "W": W_flat.tolist()}
            break

    if best_result is not None:
        return best_result

    return trivial_decomp()
