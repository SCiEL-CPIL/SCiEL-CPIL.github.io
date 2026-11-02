"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 3]

import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=3
    # U shape: (n*m, R) = (4, R)
    # V shape: (m*p, R) = (6, R)
    # W shape: (p*n, R) = (6, R)
    
    # Construct the correct target tensor for M_{n,m,p}
    # T[u, v, w] where:
    # u = i*m + j
    # v = j*p + l
    # w = i*p + l  (Note: Output C is n x p. Flattened index i*p + l)
    # Wait, the prompt says W shape is (p*n, R).
    # And "official trace convention W[k*n+i,r]".
    # In the previous code, W was indexed by k*n+l.
    # Let's stick to the standard flattening for C (n x p): index = i*p + l.
    # But the prompt says W shape is p*n.
    # If W is p*n, the index is k*n + l?
    # If the evaluator expects W to be indexed by (col_of_C, row_of_C) or similar?
    # "W[k*n+i,r]" suggests index = k*n + i.
    # If k is col (p) and i is row (n), then index = k*n + i.
    # This is column-major flattening of C (n x p) -> (p, n).
    # Let's assume the evaluator uses this convention.
    # So W index w = l*n + i, where l is col of B/C, i is row of A/C.
    # Let's verify with U and V.
    # U shape n*m. Index i*m + j. (Row-major A).
    # V shape m*p. Index j*p + k? No, j*p + l. (Row-major B).
    # If W is column-major C, then w = l*n + i.
    
    # Let's build the target tensor T[u, v, w] based on:
    # u = i*m + j
    # v = j*p + l
    # w = l*n + i
    # T[u,v,w] = 1 if j_A == j_B and i_A == i_C and l_B == l_C
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                u = i * m + j
                v = j * p + l
                w = l * n + i
                T_target[u, v, w] = 1.0

    def try_rank(R, seed=42):
        np.random.seed(seed)
        
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Use a simple gradient descent with momentum
        # Loss = sum (T - U V W)^2
        # To speed up, we can compute the error tensor E = T - sum_r U_r V_r W_r
        # E[u,v,w] = T[u,v,w] - sum_r U[u,r] V[v,r] W[w,r]
        
        # Gradients:
        # dU[u,r] = sum_{v,w} E[u,v,w] V[v,r] W[w,r]
        # dV[v,r] = sum_{u,w} E[u,v,w] U[u,r] W[w,r]
        # dW[w,r] = sum_{u,v} E[u,v,w] U[u,r] V[v,r]
        
        lr = 0.05
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        for iteration in range(10000):
            # Compute current tensor
            T_current = np.einsum('ur,vr,wr->uvw', U, V, W)
            E = T_target - T_current
            loss = np.sum(E**2)
            
            if loss < 1e-10:
                break
            
            # Compute gradients
            # dU[u,r] = sum_{v,w} E[u,v,w] V[v,r] W[w,r]
            # This is (E * W^T) @ V ? No.
            # E is (u,v,w). V is (v,r). W is (w,r).
            # dU[u,r] = sum_v sum_w E[u,v,w] V[v,r] W[w,r]
            # Let M1[v,w,r] = V[v,r] W[w,r]
            # dU[u,r] = sum_{v,w} E[u,v,w] M1[v,w,r]
            
            # Efficiently:
            # Temp1[v,w] = E[u,v,w] for fixed u?
            # Better use einsum.
            # dU = np.einsum('uvw,vr,wr->ur', E, V, W)
            # dV = np.einsum('uvw,ur,wr->vr', E, U, W)
            # dW = np.einsum('uvw,ur,vr->wr', E, U, V)
            
            dU = np.einsum('uvw,vr,wr->ur', E, V, W)
            dV = np.einsum('uvw,ur,wr->vr', E, U, W)
            dW = np.einsum('uvw,ur,vr->wr', E, U, V)
            
            # Update with momentum
            vU = momentum * vU - lr * dU
            vV = momentum * vV - lr * dV
            vW = momentum * vW - lr * dW
            
            U += vU
            V += vV
            W += vW
            
            # Optional: normalize factors to prevent drift
            # Not strictly necessary if lr is small
        
        T_final = np.einsum('ur,vr,wr->uvw', U, V, W)
        max_err = np.max(np.abs(T_final - T_target))
        
        if max_err < 1e-6:
            return U, V, W, max_err
        return None, None, None, max_err

    # Try ranks from 10 down to 8? No, 10 is the known min.
    # Try 10 first with multiple seeds.
    for R in [10, 11, 12]:
        found = False
        for seed in range(20):
            U, V, W, err = try_rank(R, seed)
            if U is not None:
                # Try to round to integers or half-integers
                # Scale factors
                best_solution = None
                for scale in [1, 2, 4, 8, 16, 32]:
                    sU = U * scale
                    sV = V * scale
                    sW = W * scale
                    
                    # Try integer rounding
                    U_int = np.round(sU)
                    V_int = np.round(sV)
                    W_int = np.round(sW)
                    
                    # Check error of rounded solution
                    T_round = np.einsum('ur,vr,wr->uvw', U_int, V_int, W_int)
                    err_round = np.max(np.abs(T_round - T_target))
                    
                    if err_round < 1e-9:
                        # Check if values are close to integers
                        if np.max(np.abs(U_int - sU)) < 1e-4 and \
                           np.max(np.abs(V_int - sV)) < 1e-4 and \
                           np.max(np.abs(W_int - sW)) < 1e-4:
                            best_solution = (U_int, V_int, W_int)
                            break
                    
                    # Try half-integer rounding (scale by 2, round, divide by 2)
                    # Actually, just check if 2*val is close to int
                    U_half = np.round(sU * 2) / 2.0
                    V_half = np.round(sV * 2) / 2.0
                    W_half = np.round(sW * 2) / 2.0
                    
                    T_half = np.einsum('ur,vr,wr->uvw', U_half, V_half, W_half)
                    err_half = np.max(np.abs(T_half - T_target))
                    
                    if err_half < 1e-9:
                        if np.max(np.abs(U_half - sU)) < 1e-4 and \
                           np.max(np.abs(V_half - sV)) < 1e-4 and \
                           np.max(np.abs(W_half - sW)) < 1e-4:
                            if best_solution is None or R < 10: # Prefer lower rank if we were trying higher?
                                # We are iterating R from 10 up. So first found is best.
                                best_solution = (U_half, V_half, W_half)
                                break
                            
                if best_solution is not None:
                    U_final, V_final, W_final = best_solution
                    # Convert to lists
                    U_list = [[float(x) for x in row] for row in U_final]
                    V_list = [[float(x) for x in row] for row in V_final]
                    W_list = [[float(x) for x in row] for row in W_final]
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no exact solution found for this R, continue to next R
        # If R == 12, we can just use schoolbook as fallback if optimization failed
        if R == 12:
             # Schoolbook fallback
             rank = n * m * p
             U = [[0] * rank for _ in range(n * m)]
             V = [[0] * rank for _ in range(m * p)]
             W = [[0] * rank for _ in range(p * n)]
             column = 0
             for i in range(n):
                 for j in range(m):
                     for l in range(p):
                         u = i * m + j
                         v = j * p + l
                         w = l * n + i
                         U[u][column] = 1
                         V[v][column] = 1
                         W[w][column] = 1
                         column += 1
             return {"U": U, "V": V, "W": W}

    # Fallback if loop finishes without return (should not happen with R=12 logic)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                u = i * m + j
                v = j * p + l
                w = l * n + i
                U[u][column] = 1
                V[v][column] = 1
                W[w][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
