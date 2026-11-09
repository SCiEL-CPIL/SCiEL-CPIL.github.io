import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False

    # T[(i,j), (jp,k), (kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U_arr[i*m+j, c] * V_arr[jp*p+k, c] * W_arr[kp*n+ip, c]
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > 1e-6:
                                return False
    return True

def _check_half_integer(A):
    """Check if all entries are integer or half-integer."""
    A2 = 2 * np.array(A, dtype=float)
    # Check if A2 is close to integer
    return np.allclose(A2, np.round(A2), atol=1e-6)

def _round_to_half_integer(A):
    """Round entries to nearest half-integer."""
    A_arr = np.array(A, dtype=float)
    A2 = 2 * A_arr
    A2_rounded = np.round(A2)
    return (A2_rounded / 2.0).tolist()

def _try_rank(r, n, m, p, max_iters=200, n_restarts=5, deadline=None):
    """
    Try to find a rank-r decomposition.
    """
    if deadline is not None and time.time() > deadline - 10:
        return None
        
    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 10:
            break
            
        np.random.seed(restart * 1000 + r)
        
        # Initialize
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Alternating Least Squares / Gradient Descent
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 10:
                break
                
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            loss = 0.0
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    # s = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
                                    s = np.dot(U[i*m+j, :], np.dot(V[jp*p+k, :], W[kp*n+ip, :]))
                                    err = s - rhs
                                    loss += err * err
                                    # Gradients
                                    dU[i*m+j, :] += 2 * err * np.dot(V[jp*p+k, :], W[kp*n+ip, :])
                                    dV[jp*p+k, :] += 2 * err * np.dot(U[i*m+j, :], W[kp*n+ip, :])
                                    dW[kp*n+ip, :] += 2 * err * np.dot(U[i*m+j, :], V[jp*p+k, :])
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if loss < 1e-8:
                break
                
            # Adaptive learning rate
            lr = 0.05 / (1 + it * 0.05)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
    if best_loss < 1e-6:
        # Try to round to half-integers
        U_r = _round_to_half_integer(best_U)
        V_r = _round_to_half_integer(best_V)
        W_r = _round_to_half_integer(best_W)
        
        if _verify(U_r, V_r, W_r, n, m, p):
            return U_r, V_r, W_r
            
        # If rounding failed, try to see if the continuous solution is "close" to half-integers
        # and if we can solve a local adjustment? 
        # For now, if rounding fails, we return None to be safe, 
        # unless the continuous solution itself is effectively integer/half-integer.
        if _check_half_integer(best_U) and _check_half_integer(best_V) and _check_half_integer(best_W):
             return best_U.tolist(), best_V.tolist(), best_W.tolist()

    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank
    # Known bounds suggest rank < 48. 
    # Try 40, 36, 32...
    ranks_to_try = [40, 36, 32, 28]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        if r >= best_rank:
            continue
            
        res = _try_rank(r, n, m, p, max_iters=100, n_restarts=5, deadline=deadline)
        if res is not None:
            U_l, V_l, W_l = res
            best_rank = r
            best_result = {"U": U_l, "V": V_l, "W": W_l}
            # If we found a lower rank, we can stop or try even lower
            # But lower might be harder. Let's try a few more if time permits.
            
    return best_result
