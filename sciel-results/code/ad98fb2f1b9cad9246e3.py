import numpy as np
import time

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    
    # Fallback: Rank 22
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return {"U": U, "V": V, "W": W}

    if time.time() - _START_TIME > TIME_LIMIT:
        return build_rank_22()

    best_result = build_rank_22()
    best_rank = 22

    # Search for lower ranks
    for rank in [21, 20]:
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break
            
        found = find_exact_decomposition(rank)
        if found:
            U, V, W = found
            if verify(U, V, W, rank):
                best_result = {"U": U, "V": V, "W": W}
                best_rank = rank
                # If we found 20, we can stop. If 21, try 20.
                if rank == 20:
                    return best_result

    return best_result

def verify(U, V, W, rank):
    n, m, p = DIMENSIONS
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def find_exact_decomposition(rank):
    n, m, p = DIMENSIONS
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    np.random.seed(42)
    
    def objective(params):
        U = params[:num_u*rank].reshape(num_u, rank)
        V = params[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
        W = params[num_u*rank+num_v*rank:].reshape(num_w, rank)
        
        res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term = U[i*m+j] * V[j*p+k] * W[k*n+i]
                    s = np.sum(term)
                    res += (s - 1.0)**2
        return res

    def gradient(params):
        U = params[:num_u*rank].reshape(num_u, rank)
        V = params[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
        W = params[num_u*rank+num_v*rank:].reshape(num_w, rank)
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                    r = s - 1.0
                    
                    gU[i*m+j] += 2 * r * V[j*p+k] * W[k*n+i]
                    gV[j*p+k] += 2 * r * U[i*m+j] * W[k*n+i]
                    gW[k*n+i] += 2 * r * U[i*m+j] * V[j*p+k]
        
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])

    for restart in range(10):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break
            
        x0 = np.random.randn(num_u*rank + num_v*rank + num_w*rank) * 0.5
        # Add some structure? Maybe initialize with rank 22 projected?
        # For now, random.
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
        except Exception:
            continue
        
        if res.fun < 1e-8:
            x_opt = res.x
            U = x_opt[:num_u*rank].reshape(num_u, rank)
            V = x_opt[num_u*rank:num_u*rank+num_v*rank].reshape(num_v, rank)
            W = x_opt[num_u*rank+num_v*rank:].reshape(num_w, rank)
            
            # Snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            
            U_list = [[float(x) for x in row] for row in U_snap]
            V_list = [[float(x) for x in row] for row in V_snap]
            W_list = [[float(x) for x in row] for row in W_snap]
            
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
            
            # Try integers
            U_snap_int = np.round(U)
            V_snap_int = np.round(V)
            W_snap_int = np.round(W)
            
            U_list_int = [[float(x) for x in row] for row in U_snap_int]
            V_list_int = [[float(x) for x in row] for row in V_snap_int]
            W_list_int = [[float(x) for x in row] for row in W_snap_int]
            
            if verify(U_list_int, V_list_int, W_list_int, rank):
                return U_list_int, V_list_int, W_list_int

    return None

from scipy.optimize import minimize
