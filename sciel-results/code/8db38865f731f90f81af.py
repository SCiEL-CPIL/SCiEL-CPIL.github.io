import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 3, 7]

def _verify_decomposition(U, V, W, n, m, p, R):
    """Verify that the decomposition exactly represents the matrix multiplication tensor."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    # Check: sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta_jj' * delta_kk' * delta_ii'
    for i in range(n):
        for i_prime in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            expected = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                            actual = 0.0
                            for r in range(R):
                                actual += U[i * m + j, r] * V[j_prime * p + k, r] * W[k_prime * n + i_prime, r]
                            if abs(actual - expected) > 1e-8:
                                return False
    return True

def _als_solve(U, V, W, n, m, p, R, max_iter=200, tol=1e-10):
    """Alternating least squares to find exact tensor decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        # Solve for W: fix U, V
        # For each (i, i_prime, j, j_prime, k, k_prime):
        #   sum_r U[i*m+j, r] * V[j_prime*p+k, r] * W[k_prime*n+i_prime, r] = delta_jj' * delta_kk' * delta_ii'
        # Group by (k_prime, i_prime): for each such pair, we have a linear system in W[k_prime*n+i_prime, :]
        
        # Build linear system for W
        # Variables: W[k_prime*n+i_prime, r] for k_prime in [0,p), i_prime in [0,n), r in [0,R)
        # Total variables: p*n*R
        # Equations: for each (i, j, j_prime, k, k_prime, i_prime): one equation
        # But we can group: for fixed (i, j, j_prime, k, i_prime), the equation over k_prime is:
        #   sum_r (U[i*m+j, r] * V[j_prime*p+k, r]) * W[k_prime*n+i_prime, r] = delta_jj' * delta_kk' * delta_ii'
        
        # Let's build a big linear system
        num_w_vars = p * n * R
        num_eqs = n * m * m * p * p * n  # (i, j, j', k, k', i')
        
        # This is too large to build explicitly. Let's use a smarter approach.
        # For fixed (i, i_prime, j, j_prime, k), the equations over k_prime are:
        #   For each k_prime: sum_r A_r * W[k_prime*n+i_prime, r] = b
        # where A_r = U[i*m+j, r] * V[j_prime*p+k, r] and b = delta_jj' * delta_kk' * delta_ii'
        
        # So for fixed (i, i_prime, j, j_prime, k), we have p equations (one per k_prime) and R unknowns (W[k_prime*n+i_prime, r] for r=0..R-1, but these are different unknowns for different k_prime!)
        
        # Actually, W[k_prime*n+i_prime, r] are all different variables. So for fixed (i, i_prime, j, j_prime, k), we have p*R unknowns and p equations. This is underdetermined if p*R > p, i.e., R > 1.
        
        # Better approach: build the full system.
        # Let's vectorize W as a vector of length p*n*R.
        # For equation (i, j, j_prime, k, k_prime, i_prime):
        #   sum_r U[i*m+j, r] * V[j_prime*p+k, r] * W[k_prime*n+i_prime, r] = delta_jj' * delta_kk' * delta_ii'
        
        # The coefficient of W[kp*ni_p, r] in this equation is U[i*m+j, r] * V[j_prime*p+k, r] if (kp, ni_p) = (k_prime, i_prime), else 0.
        
        # So the system matrix A has entries:
        # A[equation_idx, var_idx] = U[i*m+j, r] * V[j_prime*p+k, r] where equation_idx corresponds to (i,j,j',k,k',i') and var_idx corresponds to (k',i',r)
        
        # This is sparse! Each equation only involves R variables.
        
        # Let's build it as a dense matrix for small sizes.
        # num_eqs = 1764, num_w_vars = 21*14 = 294 for R=21
        # 1764 x 294 dense matrix is manageable.
        
        A_mat = np.zeros((num_eqs, num_w_vars))
        b_vec = np.zeros(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                for r in range(R):
                                    A_mat[eq_idx, k_prime * n * R + i_prime * R + r] = U[i * m + j, r] * V[j_prime * p + k, r]
                                b_vec[eq_idx] = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                                eq_idx += 1
        
        # Solve least squares
        W_flat, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        W = W_flat.reshape(p * n, R)
        
        # Solve for U: fix V, W
        # For each (i, j, j_prime, k, k_prime, i_prime):
        #   sum_r U[i*m+j, r] * V[j_prime*p+k, r] * W[k_prime*n+i_prime, r] = delta_jj' * delta_kk' * delta_ii'
        # Group by (i, j): variables are U[i*m+j, r] for r=0..R-1
        # For fixed (j_prime, k, k_prime, i_prime), the equation over (i, j) is:
        #   sum_r U[i*m+j, r] * (V[j_prime*p+k, r] * W[k_prime*n+i_prime, r]) = delta_jj' * delta_kk' * delta_ii'
        
        num_u_vars = n * m * R
        A_mat_u = np.zeros((num_eqs, num_u_vars))
        for i in range(n):
            for j in range(m):
                for j_prime in range(m):
                    for k in range(p):
                        for k_prime in range(p):
                            for i_prime in range(n):
                                eq_idx2 = ((i * m + j) * m * p * p * n + j_prime * p * p * n + k * p * n + k_prime * n + i_prime)
                                # Wait, let me recompute the equation index
                                eq_idx2 = 0
                                # Rebuild: the equation index should match the order above
                                # Above order: i, j, j_prime, k, k_prime, i_prime
                                # Let me just recompute
                                pass
        
        # This is getting complex. Let me use a cleaner indexing.
        # Equation index: ((i*m + j)*m + j_prime)*p*p*n + k*p*n + k_prime*n + i_prime
        # Hmm, let me just rebuild with consistent indexing.
        
        # Actually, let me restart with a cleaner implementation.
        break  # Placeholder, will rewrite
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Trivial rank 42 decomposition as fallback
    R_trivial = n * m * p
    U_trivial = np.zeros((n * m, R_trivial))
    V_trivial = np.zeros((m * p, R_trivial))
    W_trivial = np.zeros((p * n, R_trivial))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
    
    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = R_trivial
    
    # Try ALS for lower ranks
    for R in range(m * p, R_trivial + 1):  # Start from lower bound 21
        if time.time() > deadline - 30:
            break
        
        # Initialize with random values
        np.random.seed(42 + R)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # ALS iterations
        converged = False
        for it in range(50):
            if time.time() > deadline - 30:
                break
            
            # Solve for W given U, V
            num_eqs = n * m * m * p * p * n
            num_w_vars = p * n * R
            
            A_mat = np.zeros((num_eqs, num_w_vars))
            b_vec = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                for i_prime in range(n):
                                    for r in range(R):
                                        A_mat[eq_idx, k_prime * n * R + i_prime * R + r] = U[i * m + j, r] * V[j_prime * p + k, r]
                                    b_vec[eq_idx] = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                                    eq_idx += 1
            
            W_flat, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W = W_flat.reshape(p * n, R)
            
            # Solve for U given V, W
            num_u_vars = n * m * R
            A_mat_u = np.zeros((num_eqs, num_u_vars))
            b_vec_u = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                for i_prime in range(n):
                                    for r in range(R):
                                        A_mat_u[eq_idx, i * m * R + j * R + r] = V[j_prime * p + k, r] * W[k_prime * n + i_prime, r]
                                    b_vec_u[eq_idx] = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                                    eq_idx += 1
            
            U_flat, _, _, _ = np.linalg.lstsq(A_mat_u, b_vec_u, rcond=None)
            U = U_flat.reshape(n * m, R)
            
            # Solve for V given U, W
            num_v_vars = m * p * R
            A_mat_v = np.zeros((num_eqs, num_v_vars))
            b_vec_v = np.zeros(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for j_prime in range(m):
                        for k in range(p):
                            for k_prime in range(p):
                                for i_prime in range(n):
                                    for r in range(R):
                                        A_mat_v[eq_idx, j_prime * p * R + k * R + r] = U[i * m + j, r] * W[k_prime * n + i_prime, r]
                                    b_vec_v[eq_idx] = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                                    eq_idx += 1
            
            V_flat, _, _, _ = np.linalg.lstsq(A_mat_v, b_vec_v, rcond=None)
            V = V_flat.reshape(m * p, R)
            
            # Check convergence
            if it > 0:
                # Compute residual
                max_err = 0
                for i in range(n):
                    for i_prime in range(n):
                        for j in range(m):
                            for j_prime in range(m):
                                for k in range(p):
                                    for k_prime in range(p):
                                        expected = 1.0 if (j == j_prime and k == k_prime and i == i_prime) else 0.0
                                        actual = np.sum(U[i*m+j] * V[j_prime*p+k] * W[k_prime*n+i_prime])
                                        max_err = max(max_err, abs(actual - expected))
                
                if max_err < 1e-8:
                    converged = True
                    break
        
        if converged:
            # Verify fully
            if _verify_decomposition(U, V, W, n, m, p, R):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = R
                break  # Found minimum rank, stop
    
    return best_result
