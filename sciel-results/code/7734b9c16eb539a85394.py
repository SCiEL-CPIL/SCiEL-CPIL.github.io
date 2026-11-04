import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4

    # Target tensor T[x, y, z] where x in 0..7, y in 0..15, z in 0..7
    # T[x, y, z] = 1 if x=4i+k, y=4k+j, z=2j+i
    T = np.zeros((8, 16, 8), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                x = i * m + k
                y = k * p + j
                z = j * n + i
                T[x, y, z] = 1

    # Known Rank 28 Solution (4 Strassen blocks)
    rank_known = 28
    U_known = [[0] * rank_known for _ in range(n * m)]
    V_known = [[0] * rank_known for _ in range(m * p)]
    W_known = [[0] * rank_known for _ in range(p * n)]
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows):
        U = U_known
        V = V_known
        W = W_known
        
        # M1
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # M2
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # M3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # M4
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # M5
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # M6
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # M7
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    X1 = [0, 1, 4, 5]
    Y1 = [0, 1, 4, 5]
    Z1 = [0, 2, 1, 3]
    add_block(0, X1, Y1, Z1)

    X2 = [2, 3, 6, 7]
    Y2 = [8, 9, 12, 13]
    Z2 = [0, 2, 1, 3]
    add_block(7, X2, Y2, Z2)

    X3 = [0, 1, 4, 5]
    Y3 = [2, 3, 6, 7]
    Z3 = [4, 6, 5, 7]
    add_block(14, X3, Y3, Z3)

    X4 = [2, 3, 6, 7]
    Y4 = [10, 11, 14, 15]
    Z4 = [4, 6, 5, 7]
    add_block(21, X4, Y4, Z4)

    best_rank = rank_known
    best_U, best_V, best_W = U_known, V_known, W_known

    start_time = time.time()
    deadline = start_time + 110

    # Attempt to find Rank 27
    target_rank = 27

    def check_rank_solution(U, V, W, R):
        """
        Checks if U, V, W (lists of lists of floats) form a valid decomposition.
        Returns True if valid.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Compute T_est
        T_est = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
        
        # Check closeness
        if np.allclose(T_est, T, atol=1e-9):
            return True
        return False

    def compute_error(U_np, V_np, W_np):
        """
        Computes Frobenius norm of error.
        """
        T_est = np.einsum('xr,yr,zr->xyz', U_np, V_np, W_np)
        return np.linalg.norm(T_est - T)

    def compute_gradients(U_np, V_np, W_np, T_est):
        """
        Computes gradients of 0.5 * ||T_est - T||^2 w.r.t U, V, W.
        E = T_est - T
        dL/dU[x,r] = sum_{y,z} E[x,y,z] V[y,r] W[z,r]
        """
        E = T_est - T
        dU = np.einsum('xyz,yr,zr->xr', E, V_np, W_np)
        dV = np.einsum('xyz,xr,zr->yr', E, U_np, W_np)
        dW = np.einsum('xyz,xr,yr->zr', E, U_np, V_np)
        return dU, dV, dW

    found = False
    
    # Search for Rank 27
    # Strategy: Random initialization + Gradient Descent + Rounding
    num_restarts = 50
    steps_per_restart = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initialize
        U_try = np.random.randn(8, target_rank) * 0.5
        V_try = np.random.randn(16, target_rank) * 0.5
        W_try = np.random.randn(8, target_rank) * 0.5
        
        # Alternatively, start from Rank 28, remove a column, and adjust
        if restart % 2 == 0:
             # Random start
             pass
        else:
             # Structured start: Take U_known, V_known, W_known (Rank 28)
             # Remove column 0
             # And add small noise to all
             U_sub = np.array(U_known)[:, 1:].copy()
             V_sub = np.array(V_known)[:, 1:].copy()
             W_sub = np.array(W_known)[:, 1:].copy()
             # Pad to target_rank? No, target is 27. Rank 28 - 1 = 27.
             U_try = U_sub + np.random.randn(8, target_rank) * 0.1
             V_try = V_sub + np.random.randn(16, target_rank) * 0.1
             W_try = W_sub + np.random.randn(8, target_rank) * 0.1

        # Gradient Descent
        lr = 0.01
        best_err = np.inf
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
            
            T_est = np.einsum('xr,yr,zr->xyz', U_try, V_try, W_try)
            err = np.linalg.norm(T_est - T)
            
            if err < best_err:
                best_err = err
                # Check if we found an exact solution (within tolerance)
                if err < 1e-5:
                    # Try to round to half-integers
                    U_rounded = np.round(U_try * 2) / 2
                    V_rounded = np.round(V_try * 2) / 2
                    W_rounded = np.round(W_try * 2) / 2
                    
                    if check_rank_solution(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), target_rank):
                        best_rank = target_rank
                        best_U = U_rounded.tolist()
                        best_V = V_rounded.tolist()
                        best_W = W_rounded.tolist()
                        found = True
                        break
            
            if found:
                break
            
            # Compute gradients
            dU, dV, dW = compute_gradients(U_try, V_try, W_try, T_est)
            
            # Update
            U_try -= lr * dU
            V_try -= lr * dV
            W_try -= lr * dW
            
            # Optional: Decay learning rate
            # lr *= 0.99

        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}

BET: {mechanism: "Identical to parent program, ensuring stability and avoiding invalid mutations.", q: 0.95, delta: [0, 0]}
