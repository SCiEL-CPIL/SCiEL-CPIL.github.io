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
    for c_remove in range(46):
        mask = [i for i in range(46) if i != c_remove]
        U = U46[:, mask].copy()
        V = V46[:, mask].copy()
        W = W46[:, mask].copy()
        
        # Add noise
        U += np.random.randn(*U.shape) * 0.1
        V += np.random.randn(*V.shape) * 0.1
        W += np.random.randn(*W.shape) * 0.1
        
        # Anneal
        T_sa = 1.0
        for i in range(200):
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            
            if loss < 1e-10:
                break
            
            # Random perturbation
            idx_U = np.random.randint(U.size)
            idx_V = np.random.randint(V.size)
            idx_W = np.random.randint(W.size)
            
            U_flat = U.flatten()
            V_flat = V.flatten()
            W_flat = W.flatten()
            
            U_flat[idx_U] += np.random.randn() * 0.1
            V_flat[idx_V] += np.random.randn() * 0.1
            W_flat[idx_W] += np.random.randn() * 0.1
            
            U = U_flat.reshape(U.shape)
            V = V_flat.reshape(V.shape)
            W = W_flat.reshape(W.shape)
            
            prod_new = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff_new = prod_new - T
            loss_new = np.sum(diff_new * diff_new)
            
            if loss_new < loss or np.random.rand() < np.exp(-(loss_new - loss) / T_sa):
                loss = loss_new
            else:
                U = U_flat.reshape(U.shape) # Revert
                V = V_flat.reshape(V.shape)
                W = W_flat.reshape(W.shape)
                
            T_sa *= 0.95
            
            # Snap check
            if i % 20 == 0:
                U_s = snap_to_half(U)
                V_s = snap_to_half(V)
                W_s = snap_to_half(W)
                if validate(U_s, V_s, W_s):
                    return {
                        "U": [[float(x) for x in row] for row in U_s],
                        "V": [[float(x) for x in row] for row in V_s],
                        "W": [[float(x) for x in row] for row in W_s]
                    }
        
        # Final snap
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
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
