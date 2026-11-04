"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    n, m, p = DIMENSIONS

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def try_rank(rank, max_iters=200):
        """Try to find a half-integer decomposition of given rank."""
        if time.time() > deadline - 5:
            return None
        
        # Use the structure: for each i, we need the bilinear form to work.
        # Initialize with a structured guess based on Kronecker products.
        # U has shape (n*m, rank) = (6, rank)
        # V has shape (m*p, rank) = (27, rank)
        # W has shape (p*n, rank) = (18, rank)
        
        # Try multiple random restarts
        for restart in range(30):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization with small integers
            U = np.random.choice([-1, 0, 1], size=(n*m, rank)).astype(float)
            V = np.random.choice([-1, 0, 1], size=(m*p, rank)).astype(float)
            W = np.random.choice([-1, 0, 1], size=(p*n, rank)).astype(float)
            
            # Alternating minimization: fix two, solve for third
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    return None
                
                # Solve for U: for each (i,j), we need sum_c U[i*m+j,c] * (sum_k V[j*p+k,c]*W[k*n+i,c]) = 1
                # But this is coupled. Instead, do proper ALS.
                
                # For each (i,j): minimize sum_k (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                # This is linear in U[i*m+j,:]. The gradient w.r.t. U[i*m+j,c] is:
                # -2 * sum_k V[j*p+k,c]*W[k*n+i,c] * (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'] - 1)
                # For the linear system: let B[c] = sum_k V[j*p+k,c]*W[k*n+i,c]
                # Let S = sum_k (sum_c' U[i*m+j,c']*V[j*p+k,c']*W[k*n+i,c'])
                # We want: sum_c B[c] * U[i*m+j,c] = S + 1 ... this is nonlinear.
                
                # Simpler: use gradient descent with a few steps
                lr = 0.01
                for i in range(n):
                    for j in range(m):
                        grad = np.zeros(rank)
                        for k in range(p):
                            v = V[j * p + k, :]
                            w = W[k * n + i, :]
                            prod = v * w
                            val = np.dot(U[i*m+j, :] * prod, np.ones(rank))  # sum_c U*V*W
                            # Actually val = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                            val = np.dot(U[i*m+j, :] * v, w)
                            grad += (val - 1.0) * prod
                        U[i*m+j, :] -= lr * grad
                
                for j in range(m):
                    for k in range(p):
                        grad = np.zeros(rank)
                        for i in range(n):
                            u = U[i * m + j, :]
                            w = W[k * n + i, :]
                            prod = u * w
                            val = np.dot(V[j*p+k, :] * u, w)
                            grad += (val - 1.0) * prod
                        V[j*p+k, :] -= lr * grad
                
                for k in range(p):
                    for i in range(n):
                        grad = np.zeros(rank)
                        for j in range(m):
                            u = U[i * m + j, :]
                            v = V[j * p + k, :]
                            prod = u * v
                            val = np.dot(W[k*n+i, :] * u, v)
                            grad += (val - 1.0) * prod
                        W[k*n+i, :] -= lr * grad
                
                # Check if we're close to a solution
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u = U[i * m + j, :]
                            v = V[j * p + k, :]
                            w = W[k * n + i, :]
                            val = np.dot(u * v, w)
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-6:
                    # Snap to half-integers and verify
                    U_snap = [[int(round(x * 2)) / 2.0 for x in row] for row in U.tolist()]
                    V_snap = [[int(round(x * 2)) / 2.0 for x in row] for row in V.tolist()]
                    W_snap = [[int(round(x * 2)) / 2.0 for x in row] for row in W.tolist()]
                    if verify(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # Try small ranks first
    for rank in range(6, 55):
        if time.time() > deadline - 10:
            break
        res = try_rank(rank)
        if res is not None:
            return res
    
    return schoolbook()
# EVOLVE-BLOCK-END
