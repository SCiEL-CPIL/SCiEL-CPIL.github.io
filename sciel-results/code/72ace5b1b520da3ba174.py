import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Try to find a rank 44 decomposition by reusing a rank 11 decomposition of 2x3x2
    # The 2x3x2 tensor T_sub has entries T_sub[i, j, k] = 1 if (i,j,k) corresponds to a multiplication term
    # i in 0..1, j in 0..2, k in 0..1
    
    # We will attempt to solve for 11 terms for the 2x3x2 case
    # If successful, we replicate it 4 times for the 2x3x8 case (rank 44)
    # If it fails, we fall back to schoolbook rank 48
    
    def get_target_tensor_232():
        T = np.zeros((2, 3, 2))
        for i in range(2):
            for j in range(3):
                for k in range(2):
                    T[i, j, k] = 1
        return T

    def flatten_u(u):
        return u.flatten()

    def unflatten_u(f):
        return f.reshape(2, 3)

    def flatten_v(v):
        return v.flatten()

    def unflatten_v(f):
        return f.reshape(3, 2)

    def flatten_w(w):
        return w.flatten()

    def unflatten_w(f):
        return f.reshape(2, 2)

    def compute_error_232(params):
        # params: 11 * (6 + 6 + 4) = 176 floats
        T = get_target_tensor_232()
        error = 0.0
        for c in range(11):
            u = unflatten_u(params[c*16:c*16+6])
            v = unflatten_v(params[c*16+6:c*16+12])
            w = unflatten_w(params[c*16+12:c*16+16])
            # Term is u[i,j] * v[j,k] * w[k,i]
            for i in range(2):
                for j in range(3):
                    for k in range(2):
                        term_val = u[i, j] * v[j, k] * w[k, i]
                        # T[i,j,k] is 1
                        error += (term_val - T[i, j, k])**2
        return error

    # Initialize params
    np.random.seed(42)
    x0 = np.random.rand(11 * 16)
    
    # Optimize
    res = minimize(compute_error_232, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
    
    if res.success and res.fun < 1e-8:
        # We found a rank 11 decomposition for 2x3x2
        # Construct the rank 44 decomposition for 2x3x8
        rank = 44
        U = [[0.0] * rank for _ in range(n * m)] # 6 x 44
        V = [[0.0] * rank for _ in range(m * p)] # 24 x 44
        W = [[0.0] * rank for _ in range(p * n)] # 16 x 44
        
        # Extract the 11 terms
        terms = []
        for c in range(11):
            u = unflatten_u(res.x[c*16:c*16+6])
            v = unflatten_v(res.x[c*16+6:c*16+12])
            w = unflatten_w(res.x[c*16+12:c*16+16])
            terms.append((u, v, w))
            
        # Map to 2x3x8
        # 8 is 4 blocks of 2
        for b in range(4):
            for c_idx in range(11):
                col = b * 11 + c_idx
                u, v, w = terms[c_idx]
                
                # u is 2x3, maps to U[i, j] where i in 0..1, j in 0..2
                for i in range(2):
                    for j in range(3):
                        U[i * 3 + j][col] = u[i, j]
                        
                # v is 3x2, maps to V[j, k_global] where k_global = 2*b + k_local
                for j in range(3):
                    for k_local in range(2):
                        k_global = 2 * b + k_local
                        V[j * 8 + k_global][col] = v[j, k_local]
                        
                # w is 2x2, maps to W[k_global, i]
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    for i in range(2):
                        W[k_global * 2 + i][col] = w[k_local, i]
                        
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook rank 48
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
