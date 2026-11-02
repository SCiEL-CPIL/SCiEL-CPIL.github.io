import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, rank, n, m, p, tol=1e-8):
    """Verify that the decomposition is exact."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            U_row = U_np[i*m+j, :]
            for jp in range(m):
                for k in range(p):
                    V_row = V_np[jp*p+k, :]
                    for ip in range(n):
                        for kp in range(p):
                            W_row = W_np[kp*n+ip, :]
                            val = np.dot(U_row * V_row, W_row)
                            target = 1.0 if (i == ip and k == kp and j == jp) else 0.0
                            if abs(val - target) > tol:
                                return False
    return True

def _round_to_half_int(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * np.array(x)
    rounded = np.round(doubled)
    if np.all(np.abs(doubled - rounded) < tol):
        return (rounded / 2.0).tolist()
    return np.array(x).tolist()

def _construct_trivial(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return U.tolist(), V.tolist(), W.tolist(), rank

def _als_decompose(n, m, p, rank, max_iter=100, tol=1e-10, seed=0, time_limit=None, start_time=None):
    """ALS decomposition of the matrix multiplication tensor."""
    if start_time is None:
        start_time = time.time()
    
    np.random.seed(seed)
    U = np.random.randn(n*m, rank) * 0.1
    V = np.random.randn(m*p, rank) * 0.1
    W = np.random.randn(p*n, rank) * 0.1
    
    # Precompute index arrays for targets
    # For U update: target[i,j, jp,k,ip,kp] = 1 if jp==j, ip==i, k==kp else 0
    # We'll build b vectors on the fly
    
    num_eqns = m * p * n * p  # total equations
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            return None, it
        
        # Precompute V*W products for U update
        # P_U[jp,k,ip,kp] = V[jp*p+k,:] * W[kp*n+ip,:]
        P_U = np.zeros((m, p, n, p, rank))
        for jp in range(m):
            for k in range(p):
                V_row = V[jp*p+k, :]
                for ip in range(n):
                    for kp in range(p):
                        W_row = W[kp*n+ip, :]
                        P_U[jp, k, ip, kp, :] = V_row * W_row
        P_U_flat = P_U.reshape(-1, rank)  # (m*p*n*p, rank)
        
        # Update U
        for i in range(n):
            for j in range(m):
                # b[idx] = 1 if jp==j, ip==i, k==kp else 0
                b = np.zeros(num_eqns)
                idx = 0
                for jp in range(m):
                    for k in range(p):
                        for ip in range(n):
                            for kp in range(p):
                                if jp == j and ip == i and k == kp:
                                    b[idx] = 1.0
                                idx += 1
                
                try:
                    U[i*m+j, :], _, _, _ = np.linalg.lstsq(P_U_flat, b, rcond=None)
                except:
                    pass
        
        # Precompute U*W products for V update
        P_V = np.zeros((n, m, n, p, rank))
        for i in range(n):
            for j in range(m):
                U_row = U[i*m+j, :]
                for ip in range(n):
                    for kp in range(p):
                        W_row = W[kp*n+ip, :]
                        P_V[i, j, ip, kp, :] = U_row * W_row
        P_V_flat = P_V.reshape(-1, rank)
        
        # Update V
        for jp in range(m):
            for k in range(p):
                b = np.zeros(num_eqns)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for ip in range(n):
                            for kp in range(p):
                                if i == ip and k == kp and j == jp:
                                    b[idx] = 1.0
                                idx += 1
                
                try:
                    V[jp*p+k, :], _, _, _ = np.linalg.lstsq(P_V_flat, b, rcond=None)
                except:
                    pass
        
        # Precompute U*V products for W update
        P_W = np.zeros((n, m, m, p, rank))
        for i in range(n):
            for j in range(m):
                U_row = U[i*m+j, :]
                for jp in range(m):
                    for k in range(p):
                        V_row = V[jp*p+k, :]
                        P_W[i, j, jp, k, :] = U_row * V_row
        P_W_flat = P_W.reshape(-1, rank)
        
        # Update W
        for kp in range(p):
            for ip in range(n):
                b = np.zeros(num_eqns)
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                if i == ip and k == kp and j == jp:
                                    b[idx] = 1.0
                                idx += 1
                
                try:
                    W[kp*n+ip, :], _, _, _ = np.linalg.lstsq(P_W_flat, b, rcond=None)
                except:
                    pass
        
        # Check convergence
        if it % 10 == 0:
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                        err += (val - 1.0)**2
            if err < tol:
                break
    
    return (U, V, W), it

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    # Start with trivial solution
    U_triv, V_triv, W_triv, rank_trivial = _construct_trivial(n, m, p)
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}
    best_rank = rank_trivial
    
    # Try to find lower rank decompositions
    for rank in range(rank_trivial - 1, max(1, rank_trivial - 40), -1):
        if time.time() - start_time > deadline:
            break
        
        for seed in range(5):
            if time.time() - start_time > deadline:
                break
            
            remaining = deadline - (time.time() - start_time)
            if remaining < 5:
                break
                
            result, iters = _als_decompose(n, m, p, rank, max_iter=80, seed=seed, 
                                           time_limit=remaining,
                                           start_time=start_time)
            if result is None:
                continue
            
            U, V, W = result
            if _verify(U, V, W, rank, n, m, p):
                U_rounded = _round_to_half_int(U)
                V_rounded = _round_to_half_int(V)
                W_rounded = _round_to_half_int(W)
                
                if _verify(U_rounded, V_rounded, W_rounded, rank, n, m, p, tol=1e-6):
                    best_result = {"U": U_rounded, "V": V_rounded, "W": W_rounded}
                    best_rank = rank
                    break
                else:
                    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = rank
                    break
        
        if best_rank < rank_trivial:
            break
    
    return best_result
