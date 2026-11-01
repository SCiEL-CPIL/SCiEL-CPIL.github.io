import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build the tensor T: T[a, b, c] = 1 if c corresponds to the output of a*b
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Try ranks from 40 down
    best_result = None
    
    for rank in range(40, 19, -1):  # Try 40, 39, ..., 20
        if time.time() > deadline - 20:
            break
        
        # Try to find U, V, W with this rank
        # Use a simple iterative approach
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Initialize
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # Iterate
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor estimate
                # T_est[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
                # This is expensive, so let's be smart
                
                # For each c, compute the outer product structure
                # Actually, let's just compute the error directly for the non-zero entries
                
                # Compute error on the 40 non-zero entries
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a, :], V[b, :] * W[c, :])
                            error += (val - 1.0) ** 2
                
                # Also check some zero entries
                for _ in range(50):
                    a = np.random.randint(nm)
                    b = np.random.randint(mp)
                    c = np.random.randint(pn)
                    if T[a, b, c] == 0:
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += val ** 2
                
                if error < 1e-10:
                    found = True
                    break
                
                # Gradient step (simplified)
                # This is a rough heuristic
                lr = 0.01 / (1 + it * 0.1)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            # Gradient of (dot(U[a], V[b]*W[c]) - 1)^2 w.r.t. U[a]
                            grad_U = 2 * (np.dot(U[a, :], V[b, :] * W[c, :]) - 1.0) * (V[b, :] * W[c, :])
                            U[a, :] -= lr * grad_U
                            
                            # Gradient w.r.t. V[b]
                            grad_V = 2 * (np.dot(U[a, :], V[b, :] * W[c, :]) - 1.0) * (U[a, :] * W[c, :])
                            V[b, :] -= lr * grad_V
                            
                            # Gradient w.r.t. W[c]
                            grad_W = 2 * (np.dot(U[a, :], V[b, :] * W[c, :]) - 1.0) * (U[a, :] * V[b, :])
                            W[c, :] -= lr * grad_W
                
                if time.time() > deadline - 5:
                    break
            
            if found:
                break
        
        if found:
            # Convert to lists with integer/half-integer entries
            U_list = [[round(U[i, j] * 2) / 2 for j in range(rank)] for i in range(nm)]
            V_list = [[round(V[i, j] * 2) / 2 for j in range(rank)] for i in range(mp)]
            W_list = [[round(W[i, j] * 2) / 2 for j in range(rank)] for i in range(pn)]
            
            # Verify
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = sum(U_list[a][d] * V_list[b][d] * W_list[c][d] for d in range(rank))
                        if abs(val - 1.0) > 1e-9:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                continue
            
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is None:
        # Fallback to schoolbook
        rank = 40
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
