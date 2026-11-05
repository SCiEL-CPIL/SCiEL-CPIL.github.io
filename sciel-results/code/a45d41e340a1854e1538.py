"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 30
    pn = p * n  # 12
    
    # The tensor: T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    # For all i in [0,n), j in [0,m), k in [0,p)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the target tensor as a list of (a, b, c) triples that should equal 1
    target_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_triples.append((a, b, c))
    
    # Total number of constraints: n*m*p = 60
    # We need: sum_c U[a,c] * V[b,c] * W[c,c] = 1 for each (a,b,c) in target_triples
    
    best_result = None
    best_rank = n * m * p  # Start with schoolbook
    
    # Try ranks from a low estimate up to schoolbook
    # For 2x5x6, let's try ranks from 35 to 60
    for target_rank in range(35, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(200):
            if time.time() > deadline:
                break
            
            np.random.seed(trial * 7919 + 13)
            random.seed(trial * 7919 + 13)
            
            # Initialize with small random values
            U = np.random.uniform(-1, 1, (nm, target_rank))
            V = np.random.uniform(-1, 1, (mp, target_rank))
            W = np.random.uniform(-1, 1, (pn, target_rank))
            
            # Alternating minimization
            for it in range(500):
                if time.time() > deadline:
                    break
                
                # Update U: for each row a, solve for U[a, :]
                # For fixed b, c: U[a,c] * V[b,c] * W[c,c] should contribute
                # This is a bilinear problem, so we use least squares
                # For each a, we need: sum_c U[a,c] * (V[b,c] * W[c,c]) = 1 for each (a,b,c)
                # Group constraints by a
                # For a given a, the constraints are: for each (b,c) where (a,b,c) is in target_triples
                # sum_c' U[a,c'] * V[b,c'] * W[c',c] = 1  -- wait, this is getting complex
                
                # Let me use a different approach: gradient descent on the residual
                # Residual for each (a,b,c): r = sum_c U[a,c]*V[b,c]*W[c,c] - 1
                # Gradient w.r.t U[a0, c0]: V[b,c0] * W[c0,c0] * (r - 1) ... 
                
                # Actually, let me just do simple gradient descent
                lr = 0.01
                U_grad = np.zeros_like(U)
                V_grad = np.zeros_like(V)
                W_grad = np.zeros_like(W)
                
                for a, b, c in target_triples:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    err = val - 1.0
                    # d(err)/d(U[a,c0]) = V[b,c0] * W[c,c0]
                    U_grad[a, :] += err * (V[b, :] * W[c, :])
                    V_grad[b, :] += err * (U[a, :] * W[c, :])
                    W_grad[c, :] += err * (U[a, :] * V[b, :])
                
                U -= lr * U_grad
                V -= lr * V_grad
                W -= lr * W_grad
                
                # Check convergence
                max_err = 0
                for a, b, c in target_triples:
                    val = np.dot(U[a, :], V[b, :] * W[c, :])
                    max_err = max(max_err, abs(val - 1.0))
                if max_err < 1e-8:
                    found = True
                    break
            
            if found:
                # Round to nearest half-integer
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify
                valid = True
                for a, b, c in target_triples:
                    val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                
                if valid:
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    break
        
        if found and best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
