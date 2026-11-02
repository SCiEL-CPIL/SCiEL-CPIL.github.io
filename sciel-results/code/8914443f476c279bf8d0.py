import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor.
    Returns True if exact, False otherwise."""
    rank = len(U[0])
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Convert to numpy for faster computation
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check all entries
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for ip in range(n):
                        for kp in range(p):
                            # Expected: 1 if i==ip and j==jp and k==kp, else 0
                            expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                            
                            # Computed: sum_r U[i*m+j, r] * V[jp*p+k, r] * W[kp*n+ip, r]
                            computed = 0.0
                            for r in range(rank):
                                computed += U_np[i*m+j, r] * V_np[jp*p+k, r] * W_np[kp*n+ip, r]
                            
                            if abs(computed - expected) > 1e-9:
                                return False
    return True

def _round_to_half_integer(mat):
    """Round matrix entries to nearest half-integer."""
    return np.round(np.array(mat) * 2) / 2.0

def _try_integer_kernel_search(n, m, p, time_limit=10):
    """
    Try to find a low-rank decomposition using integer kernel search.
    This is a heuristic approach that looks for integer solutions.
    """
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    # Build the multiplication tensor
    T = np.zeros((dim_W, dim_U, dim_V), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0
    
    # Try ranks from 19 to 23
    for r in range(19, 24):
        # Random initialization with integer values
        np.random.seed(r * 100)
        
        for attempt in range(5):
            if time.time() > time_limit:
                break
                
            # Initialize with small integers
            U = np.random.randint(-2, 3, size=(dim_U, r))
            V = np.random.randint(-2, 3, size=(dim_V, r))
            W = np.random.randint(-2, 3, size=(dim_W, r))
            
            # ALS iterations with integer rounding
            for it in range(20):
                # Update U
                for b in range(dim_U):
                    X = np.zeros((dim_W * dim_V, r))
                    y = T[:, b, :].flatten()
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], V[:, rr])
                    try:
                        u_b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        U[b, :] = np.round(u_b * 2) / 2
                    except:
                        pass
                
                # Update V
                for c in range(dim_V):
                    X = np.zeros((dim_W * dim_U, r))
                    y = T[:, :, c].flatten()
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], U[:, rr])
                    try:
                        v_c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        V[c, :] = np.round(v_c * 2) / 2
                    except:
                        pass
                
                # Update W
                for a in range(dim_W):
                    X = np.zeros((dim_U * dim_V, r))
                    y = T[a, :, :].flatten()
                    for rr in range(r):
                        X[:, rr] = np.kron(U[:, rr], V[:, rr])
                    try:
                        w_a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        W[a, :] = np.round(w_a * 2) / 2
                    except:
                        pass
                
                # Check convergence
                T_rec = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
                
                error = np.linalg.norm(T - T_rec)
                if error < 1e-8:
                    # Verify exactness
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_schoolbook = n * m * p  # 24
    
    start_time = time.time()
    time_limit = 115  # Safety margin
    
    # First, try integer kernel search (fast, limited time)
    int_result = _try_integer_kernel_search(n, m, p, time_limit=10)
    if int_result is not None:
        return int_result
    
    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    # T[a, b, c] = 1 if indices match multiplication term
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i  # Index in W (C)
                b = i * m + j  # Index in U (A)
                c = j * p + k  # Index in V (B)
                T[a, b, c] = 1.0
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Try ranks from 19 to 23
    for r in range(19, rank_schoolbook):
        if time.time() - start_time > time_limit:
            break
        
        # Try multiple random initializations
        max_attempts = 10
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize U, V, W randomly
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            
            # ALS Iterations
            max_iters = 30
            converged = False
            
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                    
                # Update U
                for b in range(dim_U):
                    X = np.zeros((dim_W * dim_V, r))
                    y = T[:, b, :].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], V[:, rr])
                    
                    try:
                        u_b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        U[b, :] = u_b
                    except:
                        pass
                        
                # Update V
                for c in range(dim_V):
                    X = np.zeros((dim_W * dim_U, r))
                    y = T[:, :, c].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(W[:, rr], U[:, rr])
                    
                    try:
                        v_c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        V[c, :] = v_c
                    except:
                        pass
                        
                # Update W
                for a in range(dim_W):
                    X = np.zeros((dim_U * dim_V, r))
                    y = T[a, :, :].flatten()
                    
                    for rr in range(r):
                        X[:, rr] = np.kron(U[:, rr], V[:, rr])
                    
                    try:
                        w_a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                        W[a, :] = w_a
                    except:
                        pass
                
                # Check convergence
                T_rec = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
                    
                error = np.linalg.norm(T - T_rec)
                if error < 1e-8:
                    converged = True
                    break
                    
            if converged:
                # Try to round to nearest 0.5
                U_rounded = _round_to_half_integer(U)
                V_rounded = _round_to_half_integer(V)
                W_rounded = _round_to_half_integer(W)
                
                # Verify rounded solution
                T_rec_rounded = np.zeros((dim_W, dim_U, dim_V))
                for rr in range(r):
                    T_rec_rounded += np.einsum('a,b,c->abc', W_rounded[:, rr], U_rounded[:, rr], V_rounded[:, rr])
                
                error_rounded = np.linalg.norm(T - T_rec_rounded)
                
                if error_rounded < 1e-8:
                    # Return rounded version
                    U_list = U_rounded.tolist()
                    V_list = V_rounded.tolist()
                    W_list = W_rounded.tolist()
                    
                    # Final exact verification
                    if _verify_exact(U_list, V_list, W_list, n, m, p):
                        best_result = {"U": U_list, "V": V_list, "W": W_list}
                        best_rank = r
                        found = True
                        break
                else:
                    # If rounding fails, check if float solution is exact enough
                    # Try to find a clean solution by checking if values are close to half-integers
                    U_close = np.all(np.abs(U * 2 - np.round(U * 2)) < 1e-6)
                    V_close = np.all(np.abs(V * 2 - np.round(V * 2)) < 1e-6)
                    W_close = np.all(np.abs(W * 2 - np.round(W * 2)) < 1e-6)
                    
                    if U_close and V_close and W_close:
                        U_list = (np.round(U * 2) / 2).tolist()
                        V_list = (np.round(V * 2) / 2).tolist()
                        W_list = (np.round(W * 2) / 2).tolist()
                        
                        if _verify_exact(U_list, V_list, W_list, n, m, p):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            best_rank = r
                            found = True
                            break
    
    # If no low rank found, return schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(dim_U)]
        V = [[0] * rank for _ in range(dim_V)]
        W = [[0] * rank for _ in range(dim_W)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
