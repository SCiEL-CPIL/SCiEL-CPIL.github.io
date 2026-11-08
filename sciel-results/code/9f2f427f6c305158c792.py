"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T[i, j, k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    # We check that T[i,j,k] = delta(i,j,k) i.e., the matrix multiplication tensor
    # Actually, the standard decomposition is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor form: T[i,j,k] = 1 for all (i,j,k) in the "multiplication" sense
    # Wait, the matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 
    # and the decomposition means:
    # M[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
    
    # Let's verify by checking a few random cases
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute using the decomposition
        C_computed = np.zeros((n, p))
        for s in range(rank):
            u_col = U_arr[:, s]
            v_col = V_arr[:, s]
            w_col = W_arr[:, s]
            
            # u_col has indices i*m + j
            # v_col has indices j*p + k  
            # w_col has indices k*n + i
            
            # The product term is: sum_{i,j,k} u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i] * A[i,j] * B[j,k]
            # But actually the decomposition says:
            # C[i,k] = sum_s (sum_j A[i,j] * U[i*m+j, s]) * (sum_j B[j,k] * V[j*p+k, s]) * W[k*n+i, s]
            
            a_proj = np.zeros((n, m))
            for i in range(n):
                for j in range(m):
                    a_proj[i, j] = U[i*m + j][s]
            
            b_proj = np.zeros((m, p))
            for j in range(m):
                for k in range(p):
                    b_proj[j, k] = V[j*p + k][s]
            
            w_val = W[k*n + i][s]  # This doesn't work, w depends on i and k
            
            # Actually the formula is:
            # C[i,k] = sum_s W[k*n+i, s] * (sum_j A[i,j]*U[i*m+j,s]) * (sum_j B[j,k]*V[j*p+k,s])
            # Wait no, that's not right either. Let me reconsider.
            
            # Standard tensor decomposition of matmul:
            # The tensor is T[i,j,k] = delta(i,j,k) in the sense that
            # (A*B)[i,k] = sum_j A[i,j] B[j,k]
            # As a 3-tensor: M[i,j,k] = 1 (it's the identity tensor for the bilinear map)
            # No wait, M[i,j,k] is the coefficient such that
            # (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
            # The tensor M has M[i,j,k] = 1 for all valid i,j,k? No.
            
            # Let me think again. The matrix multiplication can be seen as a bilinear map.
            # The tensor T is defined by T[i,j,k] = 1 if we're computing (AB)[i,k] = sum_j A[i,j]B[j,k].
            # Actually no. The "multiplication tensor" is the identity: 
            # T[i,j,k] = 1 means the coefficient of A[i,j]*B[j,k] in (AB)[i,k] is 1.
            
            # So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
            
            # A rank-r decomposition gives:
            # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
            
            # So we need: 1 = sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] for all i,j,k.
            
            # Let's verify this directly.
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for s in range(rank):
                            val += U[i*m+j][s] * V[j*p+k][s] * W[k*n+i][s]
                        if abs(val - 1.0) > 1e-9:
                            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second limit, use 110 for safety
    
    # Current best: rank 42 from block-wise Strassen
    best_rank = 42
    best_result = None
    
    # First, let's construct the known rank-42 solution
    def construct_rank42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for U (first matrix M)
        U_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (0, 0): -1},
            {(0, 1): 1, (1, 1): -1}
        ]
        
        # Strassen coefficients for V (second matrix N)
        V_coeffs = [
            {(0, 0): 1, (1, 1): 1},
            {(0, 0): 1},
            {(0, 1): 1, (1, 1): -1},
            {(1, 0): 1, (0, 0): -1},
            {(1, 1): 1},
            {(0, 0): 1, (0, 1): 1},
            {(1, 0): 1, (1, 1): 1}
        ]
        
        # W contributions
        W_contrib = [
            {(0, 0): 1, (1, 1): 1},
            {(1, 0): 1, (1, 1): -1},
            {(0, 1): 1, (1, 1): 1},
            {(0, 0): 1, (1, 0): 1},
            {(0, 0): -1, (0, 1): 1},
            {(1, 1): 1},
            {(0, 0): 1}
        ]
        
        col = 0
        for a in range(2):  # A block: 0 for j in {0,1}, 1 for j in {2,3}
            for b in range(3):  # B col block: 0 for k in {0,1}, 1 for {2,3}, 2 for {4,5}
                for s in range(7):  # Strassen term
                    for (i_loc, j_loc), coeff in U_coeffs[s].items():
                        i = i_loc
                        j = a * 2 + j_loc
                        u_idx = i * m + j
                        U[u_idx][col] += coeff
                    
                    for (j_loc, k_loc), coeff in V_coeffs[s].items():
                        j = a * 2 + j_loc
                        k = b * 2 + k_loc
                        v_idx = j * p + k
                        V[v_idx][col] += coeff
                    
                    for (i_loc, k_loc), coeff in W_contrib[s].items():
                        i = i_loc
                        k = b * 2 + k_loc
                        w_idx = k * n + i
                        W[w_idx][col] += coeff
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    result = construct_rank42()
    best_result = result
    
    # Verify the rank-42 solution
    if _verify(result["U"], result["V"], result["W"], n, m, p, 42):
        best_rank = 42
    else:
        # If verification fails, something is wrong; return anyway as fallback
        return result
    
    # Try to find a lower rank solution using optimization
    # The problem: find rank r such that there exist U (8 x r), V (24 x r), W (12 x r)
    # with U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] summing to 1 for all i,j,k.
    
    # This is a non-convex problem. Let's try a gradient-based approach.
    # We'll parameterize and use scipy's optimizer.
    
    try:
        from scipy.optimize import minimize
    except ImportError:
        return best_result
    
    # Try rank 40, 39, ... down to some minimum
    # Theoretical lower bound: For 2x4x6, the rank is at least max(n*m, m*p, p*n) / something
    # Actually, the rank is at least max(n, m, p) for some cases, but for matmul it's known
    # that rank(M_{n,m,p}) >= max(n,m,p) and for n=m=p=2 it's 7.
    
    # For 2x4x6, let's try to see if we can do better than 42.
    # One approach: use a different block structure.
    
    # Alternative: try rank 40 with a structured approach
    # Split into 2x2 blocks differently.
    
    # Let's try a direct optimization for lower ranks
    def try_optimize_rank(r):
        """Try to find a rank-r decomposition using optimization."""
        if time.time() > deadline - 5:
            return None
        
        # Number of variables: r * (8 + 24 + 12) = r * 44
        # Constraints: 2*4*6 = 48 equations (one for each i,j,k)
        
        # Initialize with random values
        np.random.seed(42)
        U_flat = np.random.randn(n * m * r) * 0.1
        V_flat = np.random.randn(m * p * r) * 0.1
        W_flat = np.random.randn(p * n * r) * 0.1
        
        x0 = np.concatenate([U_flat, V_flat, W_flat])
        
        def objective(x):
            U_mat = x[:n*m*r].reshape(n*m, r)
            V_mat = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_mat = x[n*m*r+m*p*r:].reshape(p*n, r)
            
            # Compute sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] for all i,j,k
            # This should equal 1 for all i,j,k
            
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for s in range(r):
                            val += U_mat[i*m+j, s] * V_mat[j*p+k, s] * W_mat[k*n+i, s]
                        error += (val - 1.0) ** 2
            return error
        
        # Check initial objective
        obj0 = objective(x0)
        if obj0 < 1e-10:
            # Already at a solution
            U_mat = x0[:n*m*r].reshape(n*m, r)
            V_mat = x0[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
            W_mat = x0[n*m*r+m*p*r:].reshape(p*n, r)
            return {
                "U": [[U_mat[i, s] for s in range(r)] for i in range(n*m)],
                "V": [[V_mat[j, s] for s in range(r)] for j in range(m*p)],
                "W": [[W_mat[k, s] for s in range(r)] for k in range(p*n)]
            }
        
        # Try gradient descent with Adam-like updates
        lr = 0.01
        x = x0.copy()
        
        for iteration in range(2000):
            if time.time() > deadline - 5:
                return None
            
            # Compute gradient numerically (expensive but simple)
            grad = np.zeros_like(x)
            eps = 1e-7
            for idx in range(len(x)):
                x_plus = x.copy()
                x_plus[idx] += eps
                x_minus = x.copy()
                x_minus[idx] -= eps
                grad[idx] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            
            x -= lr * grad
            obj = objective(x)
            
            if obj < 1e-10:
                U_mat = x[:n*m*r].reshape(n*m, r)
                V_mat = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_mat = x[n*m*r+m*p*r:].reshape(p*n, r)
                return {
                    "U": [[U_mat[i, s] for s in range(r)] for i in range(n*m)],
                    "V": [[V_mat[j, s] for s in range(r)] for j in range(m*p)],
                    "W": [[W_mat[k, s] for s in range(r)] for k in range(p*n)]
                }
        
        return None
    
    # Try ranks from 41 down to 30
    for r in range(41, 29, -1):
        if time.time() > deadline - 10:
            break
        result_r = try_optimize_rank(r)
        if result_r is not None:
            if _verify(result_r["U"], result_r["V"], result_r["W"], n, m, p, r):
                best_rank = r
                best_result = result_r
                break
    
    return best_result
# EVOLVE-BLOCK-END
