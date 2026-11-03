import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition using heuristic search
    # Start from rank 40 up to 47
    for r in range(40, 48):
        sol = _find_decomposition_heuristic(n, m, p, r)
        if sol is not None:
            return sol
            
    # Fallback to trivial decomposition
    return _trivial_decomposition(n, m, p)

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _find_decomposition_heuristic(n, m, p, r):
    # Tensor T[i, j, k] = 1
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
                
    # Initialize V, W randomly with small integers
    # Use a fixed seed for reproducibility in the attempt, or random?
    # Random is better for exploration.
    np.random.seed(42) # Fixed seed for deterministic behavior in eval? 
    # Actually, let's use random seed each time? No, 120s limit.
    # Let's try a few random initializations.
    
    for attempt in range(10):
        V = np.random.randint(0, 2, size=(m * p, r))
        W = np.random.randint(0, 2, size=(p * n, r))
        
        for iter in range(20):
            # Solve for U
            U = np.zeros((n * m, r))
            for i in range(n):
                for j in range(m):
                    # System: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = T[i, j, k]
                    # For fixed i, j:
                    # Vector b_k = T[i, j, k] for k=0..p-1
                    # Matrix A[k, l] = V[j*p+k, l] * W[k*n+i, l]
                    A = np.zeros((p, r))
                    b = np.zeros(p)
                    for k in range(p):
                        b[k] = T[i, j, k]
                        for l in range(r):
                            A[k, l] = V[j*p + k, l] * W[k*n + i, l]
                    
                    # Solve A x = b in least squares sense
                    # x is U[i*m+j, :]
                    if r > 0:
                        try:
                            U[i*m+j, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            U[i*m+j, :] = 0.0
                            
            # Solve for V
            V_new = np.zeros((m * p, r))
            for j in range(m):
                for k in range(p):
                    # System: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = T[i, j, k]
                    # For fixed j, k:
                    # Vector b_i = T[i, j, k] for i=0..n-1
                    # Matrix A[i, l] = U[i*m+j, l] * W[k*n+i, l]
                    A = np.zeros((n, r))
                    b = np.zeros(n)
                    for i in range(n):
                        b[i] = T[i, j, k]
                        for l in range(r):
                            A[i, l] = U[i*m + j, l] * W[k*n + i, l]
                            
                    if r > 0:
                        try:
                            V_new[j*p+k, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            V_new[j*p+k, :] = 0.0
            V = V_new
            
            # Solve for W
            W_new = np.zeros((p * n, r))
            for k in range(p):
                for i in range(n):
                    # System: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = T[i, j, k]
                    # For fixed k, i:
                    # Vector b_j = T[i, j, k] for j=0..m-1
                    # Matrix A[j, l] = U[i*m+j, l] * V[j*p+k, l]
                    A = np.zeros((m, r))
                    b = np.zeros(m)
                    for j in range(m):
                        b[j] = T[i, j, k]
                        for l in range(r):
                            A[j, l] = U[i*m + j, l] * V[j*p + k, l]
                            
                    if r > 0:
                        try:
                            W_new[k*n+i, :] = np.linalg.lstsq(A, b, rcond=None)[0]
                        except:
                            W_new[k*n+i, :] = 0.0
            W = W_new
            
            # Check if we can round to integers/half-integers and be exact
            # Round U, V, W to nearest integer
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            # Check exactness
            if _check_decomposition(n, m, p, U_int, V_int, W_int):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
            # Also try half-integers?
            U_half = np.round(U * 2) / 2
            V_half = np.round(V * 2) / 2
            W_half = np.round(W * 2) / 2
            if _check_decomposition(n, m, p, U_half, V_half, W_half):
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
    return None

def _check_decomposition(n, m, p, U, V, W):
    # T[i, j, k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    # Must equal 1
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for l in range(r):
                    s += U[i*m + j, l] * V[j*p + k, l] * W[k*n + i, l]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
