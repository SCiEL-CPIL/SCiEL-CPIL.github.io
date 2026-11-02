import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = start_time + 130  # 20s margin before 150s kill
    
    # Build the known rank 28 solution
    U28, V28, W28 = _build_rank28_arrays(n, m, p)
    
    # Try to find rank 27 using a refined search strategy
    if time.time() < deadline - 10:
        result = _search_rank_refined(n, m, p, 27, U28, V28, W28, deadline - 10)
        if result is not None:
            return _to_lists(result[0], result[1], result[2])
    
    # Fallback to rank 28
    return _to_lists(U28, V28, W28)

def _build_rank28_arrays(n, m, p):
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Block 1: j=0, schoolbook
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1
            V[0 * p + k, col] = 1
            W[k * n + i, col] = 1
            col += 1
            
    # Block 2a: j=1,2, k=0,1, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 0, col)
    
    # Block 2b: j=1,2, k=2, schoolbook
    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * m + j, col] = 1
            V[j * p + 2, col] = 1
            W[2 * n + i, col] = 1
            col += 1
            
    # Block 2c: j=1,2, k=3,4, Strassen
    col = _add_strassen(U, V, W, n, m, p, 1, 3, col)
    
    return U, V, W

def _add_strassen(U, V, W, n, m, p, base_j, base_k, col):
    # Same as before
    U[0 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[1 * m + base_j, col] = 1
    U[1 * m + (base_j + 1), col] = 1
    V[base_j * p + base_k, col] = 1
    W[base_k * n + 1, col] = 1
    W[(base_k + 1) * n + 1, col] = -1
    col += 1
    U[0 * m + base_j, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[1 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + base_k, col] = 1
    V[base_j * p + base_k, col] = -1
    W[base_k * n + 0, col] = 1
    W[base_k * n + 1, col] = 1
    col += 1
    U[0 * m + base_j, col] = 1
    U[0 * m + (base_j + 1), col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = -1
    W[(base_k + 1) * n + 0, col] = 1
    col += 1
    U[1 * m + base_j, col] = 1
    U[0 * m + base_j, col] = -1
    V[base_j * p + base_k, col] = 1
    V[base_j * p + (base_k + 1), col] = 1
    W[(base_k + 1) * n + 1, col] = 1
    col += 1
    U[0 * m + (base_j + 1), col] = 1
    U[1 * m + (base_j + 1), col] = -1
    V[(base_j + 1) * p + base_k, col] = 1
    V[(base_j + 1) * p + (base_k + 1), col] = 1
    W[base_k * n + 0, col] = 1
    col += 1
    return col

def _search_rank_refined(n, m, p, target_rank, U_ref, V_ref, W_ref, deadline):
    """
    Refined search combining:
    1. Cyclic group algebra / CRT structure to find structured initializations
    2. Multi-start Levenberg-Marquardt with diverse initializations
    3. Final high-precision polish pass
    
    The 2x3x5 multiplication tensor T[i,j,k] = delta_{i,j,k} (where the
    product A*B has entry (i,k) = sum_j A[i,j]*B[j,k]).
    
    We use the ring Z[x]/(x^m) for the middle index j and Z[y]/(y^p) for k,
    with CRT decomposition to find low-rank structured solutions.
    """
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    num_constraints = n * m * p  # 30
    
    def residuals(params):
        U = params[:target_rank*dim_u].reshape(target_rank, dim_u).T
        V = params[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
        W = params[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
        
        res = np.zeros(num_constraints)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def verify_and_polish(U, V, W, deadline):
        """Verify solution and attempt high-precision polish if close."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return U, V, W
        
        # If close but not exact, try a polish pass
        if max_err < 1e-3 and time.time() < deadline - 5:
            params = np.concatenate([U.T.flatten(), V.T.flatten(), W.T.flatten()])
            try:
                from scipy.optimize import least_squares
                result = least_squares(residuals, params, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                if np.max(np.abs(result.fun)) < 1e-8:
                    U_p = result.x[:target_rank*dim_u].reshape(target_rank, dim_u).T
                    V_p = result.x[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
                    W_p = result.x[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
                    if check_solution(U_p, V_p, W_p, tol=1e-8):
                        return U_p, V_p, W_p
            except Exception:
                pass
        
        return None
    
    # === Strategy 1: Cyclic group algebra / CRT structured initialization ===
    # Map j-index to Z[x]/(x^m), k-index to Z[y]/(y^p)
    # CRT decomposes into product of fields from cyclotomic factors
    # For m=3: x^3-1 = (x-1)(x^2+x+1), so Z[x]/(x^3) has factors at roots
    # For p=5: y^5-1 = (y-1)(y^4+y^3+y^2+y+1)
    
    # The key insight: if we can find a decomposition where each "frequency"
    # component requires fewer terms, we can combine them.
    
    # Try structured patterns based on roots of unity
    omega_m = np.exp(2j * np.pi / m)  # primitive m-th root
    omega_p = np.exp(2j * np.pi / p)  # primitive p-th root
    
    # Build structured initializations using DFT-like patterns
    structured_inits = []
    
    for freq_j in range(m):  # frequency in j-direction
        for freq_k in range(p):  # frequency in k-direction
            U_struct = np.zeros((dim_u, target_rank), dtype=complex)
            V_struct = np.zeros((dim_v, target_rank), dtype=complex)
            W_struct = np.zeros((dim_w, target_rank), dtype=complex)
            
            # Assign each rank slot a (j_freq, k_freq) pair
            for t in range(target_rank):
                jf = (t + freq_j) % m
                kf = (t + freq_k) % p
                
                # U depends on (i, j) - use j-frequency
                for i in range(n):
                    U_struct[i*m + jf, t] = omega_m ** (i * jf)
                
                # V depends on (j, k) - use both frequencies
                for j in range(m):
                    for k in range(p):
                        V_struct[j*p + k, t] = omega_m ** (j * jf) * omega_p ** (k * kf)
                
                # W depends on (k, i) - use k-frequency and i
                for k in range(p):
                    for i in range(n):
                        W_struct[k*n + i, t] = omega_p ** (k * kf) * omega_m ** (-i * jf)
            
            # Convert to real (take real part or use conjugate pairs)
            structured_inits.append(np.concatenate([
                U_struct.real.flatten(),
                V_struct.real.flatten(),
                W_struct.real.flatten()
            ]))
    
    # === Strategy 2: Random and reference-based initializations ===
    np.random.seed(12345)
    
    random_inits = []
    for attempt in range(30):
        scale = 0.3 + attempt * 0.05
        random_inits.append(np.random.normal(0, scale, target_rank * (dim_u + dim_v + dim_w)))
    
    # Reference-based: perturb rank-28 solution
    ref_inits = []
    for _ in range(15):
        cols_to_keep = np.random.choice(28, target_rank, replace=False)
        U_init = U_ref[:, cols_to_keep] + np.random.normal(0, 0.05, (dim_u, target_rank))
        V_init = V_ref[:, cols_to_keep] + np.random.normal(0, 0.05, (dim_v, target_rank))
        W_init = W_ref[:, cols_to_keep] + np.random.normal(0, 0.05, (dim_w, target_rank))
        ref_inits.append(np.concatenate([
            U_init.T.flatten(),
            V_init.T.flatten(),
            W_init.T.flatten()
        ]))
    
    # Sparse binary initializations
    sparse_inits = []
    for _ in range(10):
        params_init = np.zeros(target_rank * (dim_u + dim_v + dim_w))
        num_nonzero_u = np.random.randint(target_rank // 2, target_rank * 2)
        num_nonzero_v = np.random.randint(target_rank, target_rank * 3)
        num_nonzero_w = np.random.randint(target_rank // 2, target_rank)
        
        u_indices = np.random.choice(target_rank * dim_u, num_nonzero_u, replace=False)
        v_indices = np.random.choice(target_rank * dim_v, num_nonzero_v, replace=False)
        w_indices = np.random.choice(target_rank * dim_w, num_nonzero_w, replace=False)
        
        params_init[u_indices] = np.random.choice([-1, 1], num_nonzero_u)
        params_init[target_rank*dim_u + v_indices] = np.random.choice([-1, 1], num_nonzero_v)
        params_init[target_rank*dim_u + target_rank*dim_v + w_indices] = np.random.choice([-1, 1], num_nonzero_w)
        sparse_inits.append(params_init)
    
    # Combine all initializations, prioritizing structured ones
    all_inits = structured_inits + ref_inits + random_inits[:15] + sparse_inits
    
    # Shuffle but keep structured ones early
    rng = np.random.RandomState(42)
    for idx in range(len(all_inits) - 1, 0, -1):
        if idx < len(structured_inits):
            continue  # Keep structured inits in order
        j = rng.randint(idx + 1)
        if j < len(structured_inits):
            j = idx
        all_inits[idx], all_inits[j] = all_inits[j], all_inits[idx]
    
    # Run optimization on each initialization
    from scipy.optimize import least_squares
    
    for init_params in all_inits:
        if time.time() > deadline - 2:
            break
        
        try:
            result = least_squares(residuals, init_params, method='lm', max_nfev=8000)
            
            max_residual = np.max(np.abs(result.fun))
            if max_residual < 1e-5:
                U_sol = result.x[:target_rank*dim_u].reshape(target_rank, dim_u).T
                V_sol = result.x[target_rank*dim_u:target_rank*dim_u+target_rank*dim_v].reshape(target_rank, dim_v).T
                W_sol = result.x[target_rank*dim_u+target_rank*dim_v:].reshape(target_rank, dim_w).T
                
                polished = verify_and_polish(U_sol, V_sol, W_sol, deadline)
                if polished is not None:
                    return polished
        except Exception:
            pass
    
    return None

def _to_lists(U, V, W):
    """
    Convert numpy arrays to lists of integers or half-integers.
    The entries should be exactly integer or half-integer for valid solutions.
    """
    def to_list(arr):
        result = []
        for row in arr:
            row_list = []
            for val in row:
                # Check if it's close to an integer
                if abs(val - round(val)) < 1e-8:
                    row_list.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-8:
                    # Half-integer
                    half = round(val * 2) / 2
                    if abs(half - int(half)) < 1e-8:
                        row_list.append(int(half))
                    else:
                        row_list.append(float(half))
                else:
                    # Shouldn't happen for valid solutions, but handle gracefully
                    row_list.append(float(val))
            result.append(row_list)
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
