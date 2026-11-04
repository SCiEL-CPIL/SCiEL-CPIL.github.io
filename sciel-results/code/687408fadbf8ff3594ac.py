import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Trivial decomposition
    trivial_rank = n * m * p
    U_trivial = np.zeros((n * m, trivial_rank), dtype=np.float64)
    V_trivial = np.zeros((m * p, trivial_rank), dtype=np.float64)
    W_trivial = np.zeros((p * n, trivial_rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, col] = 1.0
                V_trivial[j * p + k, col] = 1.0
                W_trivial[k * n + i, col] = 1.0
                col += 1
                
    # Store non-zero indices of the target tensor
    # T[I, J, K] = 1 if I=(i,j), J=(j,k), K=(i,k)
    indices_I = []
    indices_J = []
    indices_K = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices_I.append(i * m + j)
                indices_J.append(j * p + k)
                indices_K.append(k * n + i)
                
    indices_I = np.array(indices_I)
    indices_J = np.array(indices_J)
    indices_K = np.array(indices_K)
    target_vals = np.ones(len(indices_I))
    
    def compute_error(U, V, W):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need sum_r U[I_s, r] V[J_s, r] W[K_s, r]
        # This is a batched contraction.
        # U_I = U[indices_I, :] shape (24, R)
        # V_J = V[indices_J, :] shape (24, R)
        # W_K = W[indices_K, :] shape (24, R)
        # Product = U_I * V_J * W_K element-wise? No.
        # We need sum over r.
        # So: (U_I * V_J * W_K).sum(axis=1)
        U_I = U[indices_I]
        V_J = V[indices_J]
        W_K = W[indices_K]
        vals = (U_I * V_J * W_K).sum(axis=1)
        return np.sum((vals - target_vals) ** 2)

    def compute_gradients(U, V, W):
        U_I = U[indices_I]
        V_J = V[indices_J]
        W_K = W[indices_K]
        vals = (U_I * V_J * W_K).sum(axis=1)
        diff = vals - target_vals
        
        # Gradient wrt U[I, r] is sum over s where I_s == I of diff[s] * V[J_s, r] * W[K_s, r]
        # This is sparse.
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Efficiently, we can iterate over the 24 terms
        # But 24 is small, so we can just do a loop or use bincount
        # Let's use a loop for clarity and speed since 24 is tiny
        for s in range(len(indices_I)):
            I = indices_I[s]
            J = indices_J[s]
            K = indices_K[s]
            d = diff[s]
            
            # dU[I, r] += d * V[J, r] * W[K, r]
            dU[I, :] += d * V[J, :] * W[K, :]
            # dV[J, r] += d * U[I, r] * W[K, r]
            dV[J, :] += d * U[I, :] * W[K, :]
            # dW[K, r] += d * U[I, r] * V[J, :]
            dW[K, :] += d * U[I, :] * V[J, :]
            
        return dU, dV, dW

    def verify_decomposition(U, V, W, tol=1e-9):
        U_I = U[indices_I]
        V_J = V[indices_J]
        W_K = W[indices_K]
        vals = (U_I * V_J * W_K).sum(axis=1)
        return np.all(np.abs(vals - target_vals) < tol)

    def check_half_integer(U, V, W, tol=1e-9):
        for mat in [U, V, W]:
            if not np.all(np.abs(mat * 2 - np.round(mat * 2)) < tol):
                return False
        return True

    def try_rank(R, num_trials=50):
        for trial in range(num_trials):
            if time.time() > deadline:
                return None
            
            np.random.seed(42 + R + trial)
            # Initialize with small random values
            U = np.random.randn(n * m, R) * 0.1
            V = np.random.randn(m * p, R) * 0.1
            W = np.random.randn(p * n, R) * 0.1
            
            # Gradient Descent
            lr = 0.01
            for step in range(1000):
                if time.time() > deadline:
                    return None
                err = compute_error(U, V, W)
                if err < 1e-12:
                    break
                dU, dV, dW = compute_gradients(U, V, W)
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate
                if step % 100 == 0:
                    lr *= 0.9
                    
            # Final check
            err = compute_error(U, V, W)
            if err < 1e-9:
                # Snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if verify_decomposition(U_snap, V_snap, W_snap) and check_half_integer(U_snap, V_snap, W_snap):
                    return (U_snap, V_snap, W_snap)
        return None

    # Try ranks from 23 down to 19
    for R in range(23, 18, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(R)
        if result is not None:
            U, V, W = result
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to trivial
    return {
        "U": U_trivial.tolist(),
        "V": V_trivial.tolist(),
        "W": W_trivial.tolist()
    }
