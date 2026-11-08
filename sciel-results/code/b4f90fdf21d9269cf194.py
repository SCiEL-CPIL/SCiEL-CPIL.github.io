import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 5]

def verify_decomposition(n, m, p, U, V, W, tol=1e-9):
    """
    Verify that the decomposition is exact.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    We need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_vec = np.array(U[i * m + j])
                v_vec = np.array(V[j * p + k])
                w_vec = np.array(W[k * n + i])
                prod = np.sum(u_vec * v_vec * w_vec)
                if abs(prod - 1.0) > tol:
                    return False
    return True

def construct_rank_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: A2 B2 (j=2, k=0..4) -> 10 terms
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: A1 B13 (j=0,1, k=4) -> 4 terms
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: A1 B11 (j=0,1, k=0,1) using Strassen -> 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: A1 B12 (j=0,1, k=2,3) using Strassen -> 7 terms
    # M1
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M2
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    
    # M3
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M4
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    
    # M5
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    
    # M6
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M7
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def search_lower_rank(n, m, p, target_rank, time_budget=10, max_iter=200):
    """
    Try to find a decomposition of the given rank using gradient descent.
    """
    start_time = time.time()
    
    # Objective: minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    def objective_and_grads(U, V, W):
        obj = 0.0
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(target_rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = val - 1.0
                    obj += err * err
                    # Gradients
                    for r in range(target_rank):
                        u = U[i * m + j, r]
                        v = V[j * p + k, r]
                        w = W[k * n + i, r]
                        grad_U[i * m + j, r] += 2 * err * v * w
                        grad_V[j * p + k, r] += 2 * err * u * w
                        grad_W[k * n + i, r] += 2 * err * u * v
        
        return obj, grad_U, grad_V, grad_W
    
    # Multiple random restarts
    best_obj = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Random initialization
        U = np.random.randn(n * m, target_rank) * 0.3
        V = np.random.randn(m * p, target_rank) * 0.3
        W = np.random.randn(p * n, target_rank) * 0.3
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            obj, gU, gV, gW = objective_and_grads(U, V, W)
            
            if obj < best_obj:
                best_obj = obj
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if obj < 1e-10:
                break
            
            # Adaptive learning rate
            U -= lr * gU
            V -= lr * gV
            W -= lr * gW
            
            # Decay learning rate
            lr *= 0.999
    
    if best_obj < 1e-8 and best_U is not None:
        # Try to round to nice values
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded version
        if verify_decomposition(n, m, p, 
                                [[float(x) for x in row] for row in U_rounded],
                                [[float(x) for x in row] for row in V_rounded],
                                [[float(x) for x in row] for row in W_rounded]):
            return {
                "U": [[float(x) for x in row] for row in U_rounded],
                "V": [[float(x) for x in row] for row in V_rounded],
                "W": [[float(x) for x in row] for row in W_rounded]
            }
        
        # Return unrounded if it's very close
        if best_obj < 1e-6:
            return {
                "U": [[float(x) for x in row] for row in best_U],
                "V": [[float(x) for x in row] for row in best_V],
                "W": [[float(x) for x in row] for row in best_W]
            }
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with known rank 28 solution
    best_solution = construct_rank_28()
    best_rank = 28
    
    # Try to find rank 27
    if time.time() < deadline - 20:
        sol = search_lower_rank(n, m, p, 27, time_budget=15, max_iter=300)
        if sol is not None:
            best_solution = sol
            best_rank = 27
    
    # Try rank 26 if time permits
    if best_rank > 26 and time.time() < deadline - 15:
        sol = search_lower_rank(n, m, p, 26, time_budget=10, max_iter=200)
        if sol is not None:
            best_solution = sol
            best_rank = 26
    
    # Try rank 25 if time permits
    if best_rank > 25 and time.time() < deadline - 10:
        sol = search_lower_rank(n, m, p, 25, time_budget=8, max_iter=150)
        if sol is not None:
            best_solution = sol
            best_rank = 25
    
    return {"U": best_solution["U"], "V": best_solution["V"], "W": best_solution["W"]}
