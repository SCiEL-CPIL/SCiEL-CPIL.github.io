import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_rank = 21
    
    # Generate valid triples
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                triples.append((u, v, w))
                
    num_triples = len(triples)
    
    # Function to flatten U, V, W into a single vector
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    size_u = n * m * target_rank
    size_v = m * p * target_rank
    size_w = p * n * target_rank
    total_size = size_u + size_v + size_w
    
    def unpack(x):
        U = x[:size_u].reshape(n*m, target_rank)
        V = x[size_u:size_u+size_v].reshape(m*p, target_rank)
        W = x[size_u+size_v:].reshape(p*n, target_rank)
        return U, V, W

    def residuals(x):
        U, V, W = unpack(x)
        res = np.zeros(num_triples)
        for idx, (u, v, w) in enumerate(triples):
            # Sum over rank c
            # U[u, c] * V[v, c] * W[w, c]
            res[idx] = np.sum(U[u, :] * V[v, :] * W[w, :]) - 1.0
        return res

    def jacobian(x):
        U, V, W = unpack(x)
        # Jacobian shape: (num_triples, total_size)
        J = np.zeros((num_triples, total_size))
        for idx, (u, v, w) in enumerate(triples):
            # d(res[idx]) / d(U[u, c]) = V[v, c] * W[w, c]
            J[idx, u*target_rank : (u+1)*target_rank] = V[v, :] * W[w, :]
            
            # d(res[idx]) / d(V[v, c]) = U[u, c] * W[w, c]
            J[idx, size_u + v*target_rank : size_u + (v+1)*target_rank] = U[u, :] * W[w, :]
            
            # d(res[idx]) / d(W[w, c]) = U[u, c] * V[v, c]
            J[idx, size_u + size_v + w*target_rank : size_u + size_v + (w+1)*target_rank] = U[u, :] * V[v, :]
        return J

    best_solution = None
    start_time = time.time()
    max_time = 110 # Leave 10s margin
    
    # Try a few random initializations
    # To help convergence, we can initialize with the schoolbook solution for rank 24, 
    # but we need rank 21. 
    # Random initialization with small integers might work better for finding integer solutions.
    
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
            
        # Random initialization with values in {-1, -0.5, 0, 0.5, 1}
        x0 = np.random.choice([-1, -0.5, 0, 0.5, 1], size=total_size)
        
        try:
            sol = least_squares(residuals, x0, jac=jacobian, method='lm')
            res_norm = np.linalg.norm(sol.fun)
            if res_norm < 1e-6:
                # Check if close to half-integers
                U, V, W = unpack(sol.x)
                # Round to nearest 0.5
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for u, v, w in triples:
                    val = np.sum(U_round[u, :] * V_round[v, :] * W_round[w, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        valid = False
                        break
                
                if valid:
                    # Return the found solution
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
        except Exception:
            continue
            
    # If no rank 21 solution found, fall back to schoolbook rank 24
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
