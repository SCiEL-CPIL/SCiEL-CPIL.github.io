import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 12]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=2 (cols of A, rows of B), p=12 (cols of B)
    
    def build_tensor():
        """Build the matrix multiplication tensor of shape (n*m, m*p, p*n)."""
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                u = i * m + j
                for k in range(p):
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        return T
    
    T = build_tensor()
    
    def strassen_decomposition():
        """Construct the Strassen-based rank-42 decomposition."""
        num_blocks = p // 2  # 6 blocks
        rank = num_blocks * 7  # 42
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Strassen 7-product decomposition for 2x2x2
        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({0: -1, 2: 1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]
        
        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
                
                for a_idx, val in u_coeffs.items():
                    U[a_idx, col] = val
                
                for local_b, val in v_coeffs.items():
                    if local_b == 0:
                        global_b = 2 * t
                    elif local_b == 1:
                        global_b = 2 * t + 1
                    elif local_b == 2:
                        global_b = m * p // 2 + 2 * t
                    else:
                        global_b = m * p // 2 + 2 * t + 1
                    V[global_b, col] = val
                
                for local_c, val in w_coeffs.items():
                    if local_c == 0:
                        global_w = (2 * t) * n + 0
                    elif local_c == 1:
                        global_w = (2 * t + 1) * n + 0
                    elif local_c == 2:
                        global_w = (2 * t) * n + 1
                    else:
                        global_w = (2 * t + 1) * n + 1
                    W[global_w, col] = val
        
        return U, V, W
    
    def verify(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        T_recon = np.zeros((n*m, m*p, p*n))
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n*m, m*p, p*n)
        T_orig = build_tensor()
        return np.allclose(T_recon, T_orig, atol=1e-10)
    
    # Start with the known rank-42 Strassen decomposition
    best_U, best_V, best_W = strassen_decomposition()
    best_rank = 42
    
    if not verify(best_U, best_V, best_W):
        # Fallback: naive decomposition (rank 48)
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, idx] = 1.0
                    V[v, idx] = 1.0
                    W[w, idx] = 1.0
                    idx += 1
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Try to find a lower rank decomposition using optimization
    # We'll try ranks below 42 and see if we can find a valid decomposition
    
    def try_rank(r):
        """Try to find a rank-r decomposition."""
        if r < best_rank:
            # Initialize with random values
            np.random.seed(42)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Objective: minimize the Frobenius norm of the difference
            def objective(params):
                U_flat = params[:n*m*r].reshape(n*m, r)
                V_flat = params[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_flat = params[n*m*r + m*p*r:].reshape(p*n, r)
                
                T_recon = np.zeros((n*m, m*p, p*n))
                for rr in range(r):
                    T_recon += np.outer(np.outer(U_flat[:, rr], V_flat[:, rr]), W_flat[:, rr]).reshape(n*m, m*p, p*n)
                
                return np.sum((T_recon - T)**2)
            
            # Start from a good initial point (scaled Strassen-like)
            init_params = np.concatenate([best_U[:, :r].flatten(), best_V[:, :r].flatten(), best_W[:, :r].flatten()])
            
            try:
                result = minimize(objective, init_params, method='L-BFGS-B', 
                                 options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12})
                if result.fun < 1e-8:
                    U_final = result.x[:n*m*r].reshape(n*m, r)
                    V_final = result.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                    W_final = result.x[n*m*r + m*p*r:].reshape(p*n, r)
                    if verify(U_final, V_final, W_final):
                        return U_final, V_final, W_final
            except:
                pass
        return None
    
    # Try ranks from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = r
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
