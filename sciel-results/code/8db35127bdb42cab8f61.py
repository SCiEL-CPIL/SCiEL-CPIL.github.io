import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145.0
    
    # Try ranks from lower bound up
    min_rank = max(n, m, p)
    
    for R in range(min_rank, n*m*p + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this rank
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize V and W randomly
            V = np.random.rand(m*p, R)
            W = np.random.rand(p*n, R)
            
            # Iterate alternating minimization
            converged = False
            for _ in range(100):
                # Solve for U
                U = np.zeros((n*m, R))
                for i in range(n):
                    for j in range(m):
                        u_idx = i*m + j
                        # A_u is p x R
                        A_u = np.zeros((p, R))
                        for k in range(p):
                            v_idx = j*p + k
                            w_idx = k*n + i
                            for c in range(R):
                                A_u[k, c] = V[v_idx, c] * W[w_idx, c]
                        # Solve A_u U_u = ones(p)
                        # Use least squares if not square or singular
                        try:
                            U_u = np.linalg.lstsq(A_u, np.ones(p), rcond=None)[0]
                            U[u_idx, :] = U_u
                        except:
                            pass
                
                # Solve for V
                V_new = np.zeros((m*p, R))
                for j in range(m):
                    for k in range(p):
                        v_idx = j*p + k
                        A_v = np.zeros((n, R)) # Equations for i
                        for i in range(n):
                            u_idx = i*m + j
                            w_idx = k*n + i
                            for c in range(R):
                                A_v[i, c] = U[u_idx, c] * W[w_idx, c]
                        try:
                            V_v = np.linalg.lstsq(A_v, np.ones(n), rcond=None)[0]
                            V_new[v_idx, :] = V_v
                        except:
                            pass
                V = V_new
                
                # Solve for W
                W_new = np.zeros((p*n, R))
                for k in range(p):
                    for i in range(n):
                        w_idx = k*n + i
                        A_w = np.zeros((m, R)) # Equations for j
                        for j in range(m):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            for c in range(R):
                                A_w[j, c] = U[u_idx, c] * V[v_idx, c]
                        try:
                            W_w = np.linalg.lstsq(A_w, np.ones(m), rcond=None)[0]
                            W_new[w_idx, :] = W_w
                        except:
                            pass
                W = W_new
                
                # Check convergence / validity
                # Calculate residual
                residual = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i*m + j
                            v_idx = j*p + k
                            w_idx = k*n + i
                            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                            residual += (val - 1)**2
                            
                if residual < 1e-8:
                    # Round to half integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    # Verify exactness
                    if verify(U_rounded, V_rounded, W_rounded, n, m, p):
                        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
                        
    # Fallback
    return schoolbook(n, m, p)

def verify(U, V, W, n, m, p):
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                if abs(val - 1) > 1e-9:
                    return False
    # Check zero entries (sample)
    # For random indices that don't match
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        jp = np.random.randint(m) # j'
        k = np.random.randint(p)
        kp = np.random.randint(p) # k'
        ip = np.random.randint(n) # i'
        
        if j == jp and i == ip and k == kp:
            continue
            
        u_idx = i*m + j
        v_idx = jp*p + k
        w_idx = kp*n + ip
        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        if abs(val) > 1e-9:
            return False
    return True

def schoolbook(n, m, p):
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
