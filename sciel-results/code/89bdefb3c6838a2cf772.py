import time
import random
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    deadline = start_time + 110.0

    # Target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

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

    def verify_exact(U, V, W, n, m, p):
        rank = len(U[0])
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 1:
                            return False
                    else:
                        s = Fraction(0)
                        for r in range(rank):
                            s += Fraction(U[a][r]) * Fraction(V[b][r]) * Fraction(W[c][r])
                        if s != 0:
                            return False
        return True

    def tensor_error(U, V, W):
        err = np.einsum('ar,br,cr->abc', U, V, W) - T
        return np.linalg.norm(err)

    def als_solve(U, V, W, R, max_iter=20, tol=1e-12):
        for _ in range(max_iter):
            # Solve U
            new_U = np.zeros_like(U)
            # M is the same for all a
            M_U = (V[:, None, :] * W[None, :, :]).reshape(m * p * p * n, R)
            for a in range(n * m):
                t = T[a].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_U, t, rcond=None)
                new_U[a] = sol
            U = new_U
            
            # Solve V
            new_V = np.zeros_like(V)
            M_V = (U[:, None, :] * W[None, :, :]).reshape(n * m * p * n, R)
            for b in range(m * p):
                t = T[:, b, :].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_V, t, rcond=None)
                new_V[b] = sol
            V = new_V
            
            # Solve W
            new_W = np.zeros_like(W)
            M_W = (U[:, None, :] * V[None, :, :]).reshape(n * m * m * p, R)
            for c in range(p * n):
                t = T[:, :, c].flatten()
                sol, residuals, rank_s, sv = np.linalg.lstsq(M_W, t, rcond=None)
                new_W[c] = sol
            W = new_W
            
            err = tensor_error(U, V, W)
            if err < tol:
                break
                
        return U, V, W, err

    def try_quantize(U, V, W, target_rank):
        # Try rounding to nearest 0.5
        U_q = np.round(U * 2) / 2
        V_q = np.round(V * 2) / 2
        W_q = np.round(W * 2) / 2
        
        U_list = U_q.tolist()
        V_list = V_q.tolist()
        W_list = W_q.tolist()
        
        if verify_exact(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try rounding to nearest integer
        U_i = np.round(U)
        V_i = np.round(V)
        W_i = np.round(W)
        
        U_list_i = U_i.tolist()
        V_list_i = V_i.tolist()
        W_list_i = W_i.tolist()
        
        if verify_exact(U_list_i, V_list_i, W_list_i, n, m, p):
            return {"U": U_list_i, "V": V_list_i, "W": W_list_i}
            
        return None

    best = schoolbook()
    best_rank = n * m * p

    # Search for lower rank
    # Try ranks 41 down to 30
    for target_rank in range(41, 29, -1):
        if time.time() > deadline - 15.0:
            break
        
        found = False
        # Try several random initializations
        for attempt in range(30):
            if time.time() > deadline - 5.0:
                break
                
            # Initialize U, V, W with small random values
            U = np.random.randn(n * m, target_rank) * 0.5
            V = np.random.randn(m * p, target_rank) * 0.5
            W = np.random.randn(p * n, target_rank) * 0.5
            
            U, V, W, err = als_solve(U, V, W, target_rank, max_iter=50)
            
            if err < 1e-8:
                result = try_quantize(U, V, W, target_rank)
                if result is not None:
                    best = result
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break

    return best
