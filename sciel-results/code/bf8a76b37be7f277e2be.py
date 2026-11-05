import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank
    target_rank = 21
    
    # Define the target tensor T[i, j, k] = 1 if we are computing C[i, k] from A[i, j] and B[j, k]?
    # The tensor T is such that <T, A \otimes B \otimes C> = sum_{i,j,k} A[i,j] B[j,k] C[k,i] = Tr(AB C^T)?
    # Actually, the decomposition identity is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is equivalent to:
    # sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} (sum_j' A[i,j'] B[j',k]) A[i,j] B[j,k]
    # This is not a simple dot product.
    
    # The standard tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i, j, k] is not defined directly.
    # The bilinear map is Phi(A, B) = AB.
    # The tensor T is in A^* \otimes B^* \otimes C^*.
    # T(a, b, c) = <AB, c^T>?
    # Let's use the definition from the seed:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Reconstruction: C[i,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want this to equal sum_j A[i,j] B[j,k].
    
    # Let's define the target matrix multiplication tensor in the flattened form.
    # We want to find U, V, W such that for all A, B:
    # sum_{i,k} C[i,k] (something) ...
    # Actually, we can just minimize the error on a random set of matrices A, B.
    
    def compute_C(U, V, W, A, B):
        # A is n x m, B is m x p
        # U is n*m x R, V is m*p x R, W is p*n x R
        C = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    val += U[i*m + j, :] @ V[j*p + k, :] * W[k*n + i, :]
                    # Wait, the dot product is over r.
                    # term_r = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # Sum over r.
                C[i, k] = val
        return C

    # Better vectorized compute
    def compute_C_fast(U, V, W):
        # Returns a tensor or function
        pass

    # We will minimize the squared error on random matrices.
    np.random.seed(42)
    
    # Initial guess: Use the seed (rank 24) and truncate to 21?
    # Or random initialization.
    # Random initialization is safer for escaping local minima if we do multiple restarts.
    
    best_score = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Generate random test matrices
    num_tests = 10
    A_tests = [np.random.randn(n, m) for _ in range(num_tests)]
    B_tests = [np.random.randn(m, p) for _ in range(num_tests)]
    
    def objective(params):
        # params is flat array of U, V, W
        # U: n*m * R, V: m*p * R, W: p*n * R
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        size_W = p * n * target_rank
        
        U = params[:size_U].reshape(n * m, target_rank)
        V = params[size_U:size_U + size_V].reshape(m * p, target_rank)
        W = params[size_U + size_V:].reshape(p * n, target_rank)
        
        err = 0.0
        for idx in range(num_tests):
            A = A_tests[idx]
            B = B_tests[idx]
            C_target = A @ B
            
            # Compute C from U, V, W
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    # Sum over j and r
                    # U[i*m + j, :] is vector of length R
                    # V[j*p + k, :] is vector of length R
                    # W[k*n + i, :] is vector of length R
                    # Term is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is element-wise product sum.
                    # We can do this efficiently?
                    # For each i, k, we need sum_j (U[i*m+j] * V[j*p+k] * W[k*n+i])
                    # Note W[k*n+i] is constant for fixed i, k.
                    w_vec = W[k*n + i, :]
                    for j in range(m):
                        u_vec = U[i*m + j, :]
                        v_vec = V[j*p + k, :]
                        # sum_r u_vec[r] * v_vec[r] * w_vec[r]
                        C_calc[i, k] += np.sum(u_vec * v_vec * w_vec)
            
            err += np.sum((C_calc - C_target) ** 2)
        return err

    # Try a few random restarts
    start_time = time.time()
    max_restarts = 5
    
    for restart in range(max_restarts):
        if time.time() - start_time > 100: break
        
        # Initialize
        # Scale matters. Let's try small random values.
        init_scale = 1.0
        params = np.random.randn(n * m * target_rank + m * p * target_rank + p * n * target_rank) * init_scale
        
        # Use scipy optimize
        try:
            from scipy.optimize import minimize
            res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.success:
                if res.fun < best_score:
                    best_score = res.fun
                    size_U = n * m * target_rank
                    size_V = m * p * target_rank
                    best_U = res.x[:size_U].reshape(n * m, target_rank).copy()
                    best_V = res.x[size_U:size_U + size_V].reshape(m * p, target_rank).copy()
                    best_W = res.x[size_U + size_V:].reshape(p * n, target_rank).copy()
        except Exception:
            pass

    # Check if we found a good solution
    if best_U is not None and best_score < 1e-4:
        # Round to nearest 0.5 to satisfy "integer/half-integer"
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded solution
        err = 0.0
        for idx in range(num_tests):
            A = A_tests[idx]
            B = B_tests[idx]
            C_target = A @ B
            C_calc = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    w_vec = W_rounded[k*n + i, :]
                    for j in range(m):
                        u_vec = U_rounded[i*m + j, :]
                        v_vec = V_rounded[j*p + k, :]
                        C_calc[i, k] += np.sum(u_vec * v_vec * w_vec)
            err += np.sum((C_calc - C_target) ** 2)
        
        if err < 1e-4:
            return {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
    
    # Fallback to seed (rank 24)
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
