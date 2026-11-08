import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

    def validate(U, V, W, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)

    def snap_to_half(arr):
        return np.round(arr * 2) / 2

    def build_rank46():
        # ... same as before ...
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            U[0 * m + (j_off + 1)][c] += 1
            U[1 * m + (j_off + 1)][c] -= 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1

        add_strassen(0, 0, 0)
        add_strassen(0, 2, 7)
        add_strassen(2, 0, 14)
        add_strassen(2, 2, 21)
        
        col = 28
        for i in range(2):
            for k in range(4):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        for i in range(2):
            for j in range(4):
                U[i * m + j][col] = 1
                V[j * p + 4][col] = 1
                W[4 * n + i][col] = 1
                col += 1
        for i in range(2):
            U[i * m + 4][col] = 1
            V[4 * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
        return U, V, W

    U46, V46, W46 = build_rank46()
    
    # Simulated Annealing for Rank 45
    np.random.seed(42)
    
    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
    for c_remove in range(46):
        mask = [i for i in range(46) if i != c_remove]
        U = U46[:, mask].copy()
        V = V46[:, mask].copy()
        W = W46[:, mask].copy()
        
        # Add small noise
        U += np.random.randn(*U.shape) * 0.05
        V += np.random.randn(*V.shape) * 0.05
        W += np.random.randn(*W.shape) * 0.05
        
        loss = compute_loss(U, V, W)
        
        # Gradient-based refinement with random restarts
        T_sa = 1.0
        best_loss = loss
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for i in range(500):
            # Try gradient step
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            # Compute gradients
            grad_U = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            grad_V = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            grad_W = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adaptive learning rate
            lr = 0.01 / (1 + i * 0.01)
            
            U_new = U - lr * grad_U
            V_new = V - lr * grad_V
            W_new = W - lr * grad_W
            
            loss_new = compute_loss(U_new, V_new, W_new)
            
            # Accept if better or with probability
            if loss_new < loss or np.random.rand() < np.exp(-(loss_new - loss) / T_sa):
                U, V, W = U_new, V_new, W_new
                loss = loss_new
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            else:
                # Random perturbation to escape local minima
                if i % 10 == 0:
                    idx_U = np.random.randint(U.size)
                    idx_V = np.random.randint(V.size)
                    idx_W = np.random.randint(W.size)
                    
                    U_flat = U.flatten()
                    V_flat = V.flatten()
                    W_flat = W.flatten()
                    
                    U_flat[idx_U] += np.random.randn() * 0.05
                    V_flat[idx_V] += np.random.randn() * 0.05
                    W_flat[idx_W] += np.random.randn() * 0.05
                    
                    U = U_flat.reshape(U.shape)
                    V = V_flat.reshape(V.shape)
                    W = W_flat.reshape(W.shape)
                    loss = compute_loss(U, V, W)
            
            T_sa *= 0.98
            
            # Snap check periodically
            if i % 50 == 0:
                U_s = snap_to_half(U)
                V_s = snap_to_half(V)
                W_s = snap_to_half(W)
                if validate(U_s, V_s, W_s):
                    return {
                        "U": [[float(x) for x in row] for row in U_s],
                        "V": [[float(x) for x in row] for row in V_s],
                        "W": [[float(x) for x in row] for row in W_s]
                    }
        
        # Final snap on best solution found
        U_s = snap_to_half(best_U)
        V_s = snap_to_half(best_V)
        W_s = snap_to_half(best_W)
        if validate(U_s, V_s, W_s):
            return {
                "U": [[float(x) for x in row] for row in U_s],
                "V": [[float(x) for x in row] for row in V_s],
                "W": [[float(x) for x in row] for row in W_s]
            }

    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
