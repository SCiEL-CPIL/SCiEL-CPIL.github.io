import random
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate with exact fraction arithmetic."""
    r = len(W[0]) if W else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    val += Fraction(U[idx_u][c]) * Fraction(V[idx_v][c]) * Fraction(W[idx_w][c])
                if val != Fraction(1, 1):
                    return False
    return True

def validate_float(U, V, W, n, m, p, tol=1e-9):
    """Fast float validation."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def build_strassen_blocks(n, m, p):
    """
    Build a decomposition using Strassen's algorithm for 2x2x2 blocks
    where possible, and schoolbook for the rest.
    """
    terms = []
    
    def add_strassen(i_start, j_start, k_start):
        u = {(i_start, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start+1, j_start+1): 1}
        v = {(j_start, k_start): 1}
        w = {(k_start, i_start+1): 1, (k_start+1, i_start+1): -1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1}
        v = {(j_start, k_start+1): 1, (j_start+1, k_start+1): -1}
        w = {(k_start+1, i_start): 1, (k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start+1): 1}
        v = {(j_start+1, k_start): 1, (j_start, k_start): -1}
        w = {(k_start, i_start): 1, (k_start, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start): 1, (i_start, j_start+1): 1}
        v = {(j_start+1, k_start+1): 1}
        w = {(k_start, i_start): -1, (k_start+1, i_start): 1}
        terms.append((u, v, w))
        
        u = {(i_start+1, j_start): 1, (i_start, j_start): -1}
        v = {(j_start, k_start): 1, (j_start, k_start+1): 1}
        w = {(k_start+1, i_start+1): 1}
        terms.append((u, v, w))
        
        u = {(i_start, j_start+1): 1, (i_start+1, j_start+1): -1}
        v = {(j_start+1, k_start): 1, (j_start+1, k_start+1): 1}
        w = {(k_start, i_start): 1}
        terms.append((u, v, w))

    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    u = {(i, j): 1}
                    v = {(j, k): 1}
                    w = {(k, i): 1}
                    terms.append((u, v, w))

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    
    add_schoolbook(0, 2, 4, 5, 0, 2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    add_schoolbook(0, 2, 4, 5, 4, 5)

    rank = len(terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j][c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k][c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i][c] = val
            
    return U, V, W

def tensor_error(U, V, W, n, m, p):
    """Calculate L2 error of the tensor decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = W.shape[1]
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                error += (val - 1.0) ** 2
    return error

def try_optimize_rank(n, m, p, rank, deadline, max_iter=2000, lr=0.01):
    """
    Try to find a decomposition of given rank using gradient descent with proper gradients.
    Returns (U, V, W) as lists if found, else None.
    """
    np.random.seed(123 + rank)
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3
    
    best_error = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Compute error and gradients
        error = 0.0
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                    err = val - 1.0
                    error += err * err
                    
                    # Gradients: d/dU[idx_u,c] = 2*err * V[idx_v,c]*W[idx_w,c]
                    dU[idx_u, :] += 2 * err * (V[idx_v, :] * W[idx_w, :])
                    dV[idx_v, :] += 2 * err * (U[idx_u, :] * W[idx_w, :])
                    dW[idx_w, :] += 2 * err * (U[idx_u, :] * V[idx_v, :])
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < 1e-12:
            break
        
        # Adaptive learning rate
        step_size = lr / (1.0 + it * 0.001)
        
        U -= step_size * dU
        V -= step_size * dV
        W -= step_size * dW
        
        # Occasionally restart from best if stuck
        if it > 0 and it % 500 == 0 and error > 1e-4:
            U = best_U + np.random.randn(n * m, rank) * 0.02
            V = best_V + np.random.randn(m * p, rank) * 0.02
            W = best_W + np.random.randn(p * n, rank) * 0.02
    
    # Final check on best found
    if best_error < 1e-8:
        # Try to snap to rational values
        U_snap = np.round(best_U * 100) / 100
        V_snap = np.round(best_V * 100) / 100
        W_snap = np.round(best_W * 100) / 100
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if validate_exact(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        
        # Try with more precision
        U_snap2 = np.round(best_U * 1000) / 1000
        V_snap2 = np.round(best_V * 1000) / 1000
        W_snap2 = np.round(best_W * 1000) / 1000
        
        U_list2 = U_snap2.tolist()
        V_list2 = V_snap2.tolist()
        W_list2 = W_snap2.tolist()
        
        if validate_exact(U_list2, V_list2, W_list2, n, m, p):
            return U_list2, V_list2, W_list2
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s buffer
    
    # Build Strassen-based decomposition as fallback
    U_strassen, V_strassen, W_strassen = build_strassen_blocks(n, m, p)
    best_result = (U_strassen, V_strassen, W_strassen)
    best_rank = len(W_strassen[0])
    
    # Verify the Strassen decomposition is valid
    if not validate_float(U_strassen, V_strassen, W_strassen, n, m, p):
        U_strassen, V_strassen, W_strassen = schoolbook_decomposition(n, m, p)
        best_result = (U_strassen, V_strassen, W_strassen)
        best_rank = len(W_strassen[0])
    
    # Try to find lower ranks using gradient optimization
    candidate_ranks = [30, 28, 26, 24, 22, 20]
    
    for rank in candidate_ranks:
        if rank >= best_rank:
            continue
        if time.time() > deadline - 30:
            break
            
        # Try multiple random starts
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
                
            np.random.seed(attempt * 1000 + rank)
            result = try_optimize_rank(n, m, p, rank, deadline, max_iter=1500, lr=0.01)
            if result:
                U, V, W = result
                best_result = (U, V, W)
                best_rank = rank
                found = True
                break
        
        if found:
            continue
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
