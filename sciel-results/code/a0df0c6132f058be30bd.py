"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    start_time = time.time()
    deadline = start_time + 140
    
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W):
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    def solve_linear(A, b):
        """Solve Ax=b, return x or None if inconsistent."""
        try:
            return np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            return None
    
    def alternating_min(r, max_iters=100):
        nonlocal best_result, best_rank
        if r >= best_rank:
            return False
        
        for attempt in range(5):
            if time.time() > deadline:
                return False
            
            # Random init
            U = np.random.choice([-1, 0, 1], size=(n*m, r)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m*p, r)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p*n, r)).astype(float)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Fix U, W; solve for V
                # For each (i,j,k): sum_c V[j*p+k,c] * U[i*m+j,c]*W[k*n+i,c] = 1
                # Group by (j,k): for each (j,k), we have n equations (one per i) in r unknowns
                V_new = np.zeros((m*p, r))
                for j in range(m):
                    for k in range(p):
                        # n equations, r unknowns
                        A = np.zeros((n, r))
                        b = np.ones(n)
                        for i in range(n):
                            for c in range(r):
                                A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        if np.linalg.matrix_rank(A) > 0:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V_new[j*p+k] = sol
                
                # Fix V, W; solve for U
                U_new = np.zeros((n*m, r))
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, r))
                        b = np.ones(p)
                        for k in range(p):
                            for c in range(r):
                                A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        if np.linalg.matrix_rank(A) > 0:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U_new[i*m+j] = sol
                
                # Fix U, V; solve for W
                W_new = np.zeros((p*n, r))
                for k in range(p):
                    for i in range(n):
                        A = np.zeros((m, r))
                        b = np.ones(m)
                        for j in range(m):
                            for c in range(r):
                                A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        if np.linalg.matrix_rank(A) > 0:
                            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W_new[k*n+i] = sol
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                if verify(U.tolist(), V.tolist(), W.tolist()):
                    # Round to nearest integer or half-integer
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if verify(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                        best_result = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                        best_rank = r
                        return True
                    else:
                        # Use the unrounded version if it verifies
                        if verify(U.tolist(), V.tolist(), W.tolist()):
                            best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                            best_rank = r
                            return True
                        break
        return False
    
    # Try ranks from 99 up
    for r in range(99, best_rank):
        if time.time() > deadline:
            break
        if alternating_min(r):
            break
    
    return best_result
# EVOLVE-BLOCK-END
