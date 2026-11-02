import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 140

    # Get the rank-5 decomposition for 1x2x3
    U_123, V_123, W_123 = _get_rank5_123()
    
    # Get the rank-7 decomposition for 2x2x2 (Strassen)
    U_222, V_222, W_222 = _get_strassen_222()
    
    # Combine via Kronecker product
    # Dimensions:
    # U_222: (2, 7), V_222: (2, 7), W_222: (2, 7)
    # U_123: (1, 5), V_123: (2, 5), W_123: (3, 5)
    # Result U: (2*1, 7*5) = (2, 35)
    # Result V: (2*2, 7*5) = (4, 35)
    # Result W: (2*3, 7*5) = (6, 35)
    
    U = np.zeros((n, 35))
    V = np.zeros((m, 35))
    W = np.zeros((p, 35))
    
    for i in range(7):
        for j in range(5):
            idx = i * 5 + j
            # U
            U[0, idx] += U_222[0, i] * U_123[0, j]
            U[1, idx] += U_222[1, i] * U_123[0, j]
            # V
            V[0, idx] += V_222[0, i] * V_123[0, j]
            V[1, idx] += V_222[0, i] * V_123[1, j]
            V[2, idx] += V_222[1, i] * V_123[0, j]
            V[3, idx] += V_222[1, i] * V_123[1, j]
            # W
            W[0, idx] += W_222[0, i] * W_123[0, j]
            W[1, idx] += W_222[0, i] * W_123[1, j]
            W[2, idx] += W_222[0, i] * W_123[2, j]
            W[3, idx] += W_222[1, i] * W_123[0, j]
            W[4, idx] += W_222[1, i] * W_123[1, j]
            W[5, idx] += W_222[1, i] * W_123[2, j]
            
    # Verify
    if _verify(U, V, W, n, m, p):
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to rank 42
        return _strassen_rank42(n, m, p)

def _get_rank5_123():
    # Solve for rank 5 decomposition of 1x2x3
    # Variables: U (1x5), V (2x5), W (3x5)
    # Flatten to 1*5 + 2*5 + 3*5 = 30 variables
    
    def objective(params):
        U = params[0:5].reshape(1, 5)
        V = params[5:15].reshape(2, 5)
        W = params[15:30].reshape(3, 5)
        
        # Target tensor T[0, j, k] = 1
        err = 0.0
        for j in range(2):
            for k in range(3):
                val = 0.0
                for t in range(5):
                    val += U[0, t] * V[j, t] * W[k, t]
                err += (val - 1.0) ** 2
        return err

    # Initial guess
    np.random.seed(42)
    params0 = np.random.randn(30) * 0.1
    
    res = minimize(objective, params0, method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    if res.fun < 1e-8:
        U = res.x[0:5].reshape(1, 5)
        V = res.x[5:15].reshape(2, 5)
        W = res.x[15:30].reshape(3, 5)
        return U, V, W
    else:
        # Return dummy if failed, will fallback
        return np.zeros((1,5)), np.zeros((2,5)), np.zeros((3,5))

def _get_strassen_222():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 1],
        [0, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, -1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [1, -1, 1, 0, 1, 1, 0]
    ])
    return U, V, W

def _verify(U, V, W, n, m, p):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(U.shape[1]):
                    val += U[i, t] * V[j, t] * W[k, t]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _strassen_rank42(n, m, p):
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(i_start, j_start, k_start, col_offset):
        a_idx = (i_start * m + j_start, i_start * m + j_start + 1,
                 (i_start + 1) * m + j_start, (i_start + 1) * m + j_start + 1)
        e_idx = (j_start * p + k_start, j_start * p + k_start + 1,
                 (j_start + 1) * p + k_start, (j_start + 1) * p + k_start + 1)
        c11_idx = k_start * n + i_start
        c12_idx = (k_start + 1) * n + i_start
        c21_idx = k_start * n + (i_start + 1)
        c22_idx = (k_start + 1) * n + (i_start + 1)
        
        col = col_offset
        U[a_idx[0]][col] = 1; U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1; V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1; W[c22_idx][col] = 1
        col += 1
        U[a_idx[2]][col] = 1; U[a_idx[3]][col] = 1
        V[e_idx[0]][col] = 1
        W[c21_idx][col] = 1; W[c22_idx][col] = -1
        col += 1
        U[a_idx[0]][col] = 1
        V[e_idx[1]][col] = 1; V[e_idx[3]][col] = -1
        W[c12_idx][col] = 1; W[c22_idx][col] = 1
        col += 1
        U[a_idx[3]][col] = 1
        V[e_idx[2]][col] = 1; V[e_idx[0]][col] = -1
        W[c11_idx][col] = 1; W[c21_idx][col] = 1
        col += 1
        U[a_idx[0]][col] = 1; U[a_idx[1]][col] = 1
        V[e_idx[3]][col] = 1
        W[c11_idx][col] = -1; W[c12_idx][col] = 1
        col += 1
        U[a_idx[2]][col] = 1; U[a_idx[0]][col] = -1
        V[e_idx[0]][col] = 1; V[e_idx[1]][col] = 1
        W[c22_idx][col] = 1
        col += 1
        U[a_idx[1]][col] = 1; U[a_idx[3]][col] = -1
        V[e_idx[2]][col] = 1; V[e_idx[3]][col] = 1
        W[c11_idx][col] = 1
        col += 1
        return col
    
    col = 0
    col = add_strassen_block(0, 0, 0, col)
    col = add_strassen_block(0, 0, 2, col)
    col = add_strassen_block(0, 0, 4, col)
    col = add_strassen_block(0, 2, 0, col)
    col = add_strassen_block(0, 2, 2, col)
    col = add_strassen_block(0, 2, 4, col)
    
    return {"U": [[float(x) for x in row] for row in U], 
            "V": [[float(x) for x in row] for row in V], 
            "W": [[float(x) for x in row] for row in W]}
