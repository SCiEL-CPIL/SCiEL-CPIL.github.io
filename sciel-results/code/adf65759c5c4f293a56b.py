import time
import numpy as np
import itertools
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

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

    def snap_to_integer(arr):
        return np.round(arr)

    def snap_to_quarter(arr):
        return np.round(arr * 4) / 4

    def try_snap_validate(U, V, W):
        for snap_func in [snap_to_half, snap_to_integer, snap_to_quarter]:
            U_s = snap_func(U)
            V_s = snap_func(V)
            W_s = snap_func(W)
            if validate(U_s, V_s, W_s):
                return True, U_s, V_s, W_s
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))

        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            # P5
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7
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

    def optimize_decomposition(U, V, W, max_steps=500, lr=0.05):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            loss = np.sum(diff * diff)
            if loss < 1e-10:
                break
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            U -= lr * gradU
            V -= lr * gradV
            W -= lr * gradW
            lr *= 0.99
            if lr < 0.001:
                lr = 0.001
        return U, V, W, loss

    def try_find_rank_45(U46, V46, W46):
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        np.random.seed(42)
        
        for col_to_remove in range(46):
            if time.time() > deadline - 10:
                break
            
            mask = [i for i in range(46) if i != col_to_remove]
            U_init = U46[:, mask].copy()
            V_init = V46[:, mask].copy()
            W_init = W46[:, mask].copy()
            
            for attempt in range(5):
                if time.time() > deadline - 10:
                    break
                
                U_try = U_init + np.random.randn(*U_init.shape) * 0.01
                V_try = V_init + np.random.randn(*V_init.shape) * 0.01
                W_try = W_init + np.random.randn(*W_init.shape) * 0.01
                
                U_opt, V_opt, W_opt, loss = optimize_decomposition(U_try, V_try, W_try, max_steps=200, lr=0.05)
                
                if loss < best_loss:
                    best_loss = loss
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                
                if loss < 1e-8:
                    valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
                    if valid:
                        return U_s, V_s, W_s
        
        if best_U is not None and best_loss < 1e-6:
            valid, U_s, V_s, W_s = try_snap_validate(best_U, best_V, best_W)
            if valid:
                return U_s, V_s, W_s
        
        return None, None, None

    U46, V46, W46 = build_rank46()
    if not validate(U46, V46, W46):
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }

    best_U, best_V, best_W = U46, V46, W46
    best_rank = 46

    if time.time() < deadline - 20:
        U45, V45, W45 = try_find_rank_45(U46, V46, W46)
        if U45 is not None:
            best_U, best_V, best_W = U45, V45, W45
            best_rank = 45

    return {
        "U": [[float(x) for x in row] for row in best_U],
        "V": [[float(x) for x in row] for row in best_V],
        "W": [[float(x) for x in row] for row in best_W]
    }
# EVOLVE-BLOCK-END
