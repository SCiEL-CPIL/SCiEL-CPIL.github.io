import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10

    # Build target tensor
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

    def try_snap_validate(U, V, W):
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if validate(U_s, V_s, W_s):
            return True, U_s, V_s, W_s
        
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        if validate(U_i, V_i, W_i):
            return True, U_i, V_i, W_i
            
        return False, None, None, None

    def build_rank46():
        rank = 46
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        
        def add_strassen(j_off, k_off, col_start):
            c = col_start
            # P1 = (A00 + A11)(B00 + B11)
            U[0 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P2 = (A10 + A11) B00
            U[1 * m + j_off][c] += 1
            U[1 * m + (j_off + 1)][c] += 1
            V[j_off * p + k_off][c] += 1
            W[k_off * n + 1][c] += 1
            W[(k_off + 1) * n + 1][c] -= 1
            c += 1
            # P3 = A00 (B01 - B11)
            U[0 * m + j_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P4 = A11 (B10 - B00)
            U[1 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + k_off][c] += 1
            V[j_off * p + k_off][c] -= 1
            W[k_off * n + 0][c] += 1
            W[k_off * n + 1][c] += 1
            c += 1
            # P5 = (A00 + A01) B11
            U[0 * m + j_off][c] += 1
            U[0 * m + (j_off + 1)][c] += 1
            V[(j_off + 1) * p + (k_off + 1)][c] += 1
            W[k_off * n + 0][c] -= 1
            W[(k_off + 1) * n + 0][c] += 1
            c += 1
            # P6 = (A10 - A00)(B00 + B01)
            U[1 * m + j_off][c] += 1
            U[0 * m + j_off][c] -= 1
            V[j_off * p + k_off][c] += 1
            V[j_off * p + (k_off + 1)][c] += 1
            W[(k_off + 1) * n + 1][c] += 1
            c += 1
            # P7 = (A01 - A11)(B10 + B11)
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
    if not validate(U46, V46, W46):
        return {
            "U": [[float(x) for x in row] for row in np.eye(U_dim)],
            "V": [[float(x) for x in row] for row in np.eye(V_dim)],
            "W": [[float(x) for x in row] for row in np.eye(W_dim)]
        }

    def objective(params):
        R = len(params) // (U_dim + V_dim + W_dim)
        U = params[:U_dim*R].reshape(U_dim, R)
        V = params[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
        W = params[U_dim*R+V_dim*R:].reshape(W_dim, R)
        
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        tensor_error = np.sum(diff * diff)
        
        # Penalty for non-half-integers
        int_penalty = np.sum((2*U - np.round(2*U))**2) + np.sum((2*V - np.round(2*V))**2) + np.sum((2*W - np.round(2*W))**2)
        
        return tensor_error + 0.1 * int_penalty

    def gradients(params):
        R = len(params) // (U_dim + V_dim + W_dim)
        U = params[:U_dim*R].reshape(U_dim, R)
        V = params[U_dim*R:U_dim*R+V_dim*R].reshape(V_dim, R)
        W = params[U_dim*R+V_dim*R:].reshape(W_dim, R)
        
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        
        gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
        gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
        gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
        
        # Add integer penalty gradients
        gradU += 0.1 * 4 * (U - np.round(2*U)/2)
        gradV += 0.1 * 4 * (V - np.round(2*V)/2)
        gradW += 0.1 * 4 * (W - np.round(2*W)/2)
        
        return np.concatenate([gradU.flatten(), gradV.flatten(), gradW.flatten()])

    best_result = None
    
    # Try to reduce rank to 45
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(attempt)
        
        # Strategy 1: Zero out a random column from rank 46
        col_to_zero = np.random.randint(46)
        mask = [i for i in range(46) if i != col_to_zero]
        U_init = U46[:, mask].copy()
        V_init = V46[:, mask].copy()
        W_init = W46[:, mask].copy()
        
        # Add small noise to escape local minima
        U_init += np.random.randn(*U_init.shape) * 0.1
        V_init += np.random.randn(*V_init.shape) * 0.1
        W_init += np.random.randn(*W_init.shape) * 0.1
        
        params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        res = minimize(objective, params, jac=gradients, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < 1e-5:
            U_opt = res.x[:U_dim*45].reshape(U_dim, 45)
            V_opt = res.x[U_dim*45:U_dim*45+V_dim*45].reshape(V_dim, 45)
            W_opt = res.x[U_dim*45+V_dim*45:].reshape(W_dim, 45)
            
            valid, U_s, V_s, W_s = try_snap_validate(U_opt, V_opt, W_opt)
            if valid:
                best_result = (U_s, V_s, W_s)
                break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        if validate(U_r, V_r, W_r):
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }

    # Fallback to rank 46
    return {
        "U": [[float(x) for x in row] for row in U46],
        "V": [[float(x) for x in row] for row in V46],
        "W": [[float(x) for x in row] for row in W46]
    }
