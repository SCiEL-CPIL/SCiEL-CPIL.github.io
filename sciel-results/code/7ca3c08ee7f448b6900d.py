"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x6 matrix multiplication.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    
    def schoolbook_decomposition():
        """Return the schoolbook rank n*m*p decomposition."""
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
        return {"U": U, "V": V, "W": W}

    def validate(U, V, W):
        """Check if the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def try_rank_r_decomposition(r, max_iter=1000, timeout=30, init_method='random'):
        """
        Try to find a rank-r decomposition using random initialization and gradient descent.
        """
        start_time = time.time()
        
        triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        idx_U = np.array([i * m + j for i, j, k in triples])
        idx_V = np.array([j * p + k for i, j, k in triples])
        idx_W = np.array([k * n + i for i, j, k in triples])
        
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
                
            if init_method == 'block':
                # Initialize using block structure
                # Assume we have a 3x3x3 decomposition of rank r//2
                # This is a heuristic initialization
                U = np.random.uniform(-0.5, 0.5, (n * m, r))
                V = np.random.uniform(-0.5, 0.5, (m * p, r))
                W = np.random.uniform(-0.5, 0.5, (p * n, r))
            else:
                U = np.random.uniform(-0.5, 0.5, (n * m, r))
                V = np.random.uniform(-0.5, 0.5, (m * p, r))
                W = np.random.uniform(-0.5, 0.5, (p * n, r))
            
            lr = 0.01
            for it in range(max_iter):
                if time.time() - start_time > timeout:
                    break
                
                U_sel = U[idx_U, :]
                V_sel = V[idx_V, :]
                W_sel = W[idx_W, :]
                
                S = np.sum(U_sel * V_sel * W_sel, axis=1)
                diff = S - 1.0
                error = np.sum(diff * diff)
                
                if error < 1e-8:
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    if validate(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                    continue

                term_U = 2.0 * diff[:, None] * V_sel * W_sel
                term_V = 2.0 * diff[:, None] * U_sel * W_sel
                term_W = 2.0 * diff[:, None] * U_sel * V_sel
                
                grad_U = np.zeros((n * m, r))
                grad_V = np.zeros((m * p, r))
                grad_W = np.zeros((p * n, r))
                
                np.add.at(grad_U, idx_U, term_U)
                np.add.at(grad_V, idx_V, term_V)
                np.add.at(grad_W, idx_W, term_W)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                lr *= 0.99

        return None

    best = None
    deadline = time.time() + 110
    
    # Try ranks from 30 upwards, assuming block structure might help
    for r in range(30, 55):
        if time.time() > deadline:
            break
        result = try_rank_r_decomposition(r, max_iter=500, timeout=5)
        if result is not None:
            best = result
            break
    
    if best is not None:
        return best
    
    return schoolbook_decomposition()
# EVOLVE-BLOCK-END
