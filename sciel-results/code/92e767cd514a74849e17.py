import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Output factor follows the official trace convention W[k*n+i, r].
    
    The tensor equation is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 112
    
    def compute_error_fast(U_np, V_np, W_np):
        """
        Compute the sum of squared errors for all (i,j,k) using vectorized operations.
        """
        rank = U_np.shape[1]
        # Reshape for einsum
        U_mat = U_np.reshape(n, m, rank)  # (n, m, rank)
        V_mat = V_np.reshape(m, p, rank)  # (m, p, rank)
        W_mat = W_np.reshape(p, n, rank)  # (p, n, rank)
        
        # Compute S[i,j,k] = sum_r U[i,m,j,r] * V[j,p,k,r] * W[k,n,i,r]
        # Using einsum: 'ijr,jkr,ikr->ijk' won't work directly, need to be careful
        # S[i,j,k] = sum_r U_mat[i,j,r] * V_mat[j,k,r] * W_mat[k,i,r]
        S = np.einsum('ijr,jkr,ikr->ijk', U_mat, V_mat, W_mat)
        error = np.sum((S - 1.0) ** 2)
        return error

    def verify_fast(U_np, V_np, W_np, tol=1e-9):
        """Verify if the decomposition is exact."""
        rank = U_np.shape[1]
        U_mat = U_np.reshape(n, m, rank)
        V_mat = V_np.reshape(m, p, rank)
        W_mat = W_np.reshape(p, n, rank)
        S = np.einsum('ijr,jkr,ikr->ijk', U_mat, V_mat, W_mat)
        return np.allclose(S, 1.0, atol=tol)

    def quantize(arr):
        """Quantize to nearest half-integer."""
        return np.round(arr * 2) / 2

    def polish_discrete(U_np, V_np, W_np, max_steps=50):
        """
        Local search in the discrete space of half-integers to reduce error.
        Uses gradient-guided variable selection.
        """
        U_np = quantize(U_np).copy()
        V_np = quantize(V_np).copy()
        W_np = quantize(W_np).copy()
        
        current_error = compute_error_fast(U_np, V_np, W_np)
        if current_error < 1e-9:
            return U_np, V_np, W_np, True

        best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
        best_err = current_error
        
        rank = U_np.shape[1]
        
        # Precompute indices
        U_indices = [(i, j) for i in range(n*m) for j in range(rank)]
        V_indices = [(i, j) for i in range(m*p) for j in range(rank)]
        W_indices = [(i, j) for i in range(p*n) for j in range(rank)]
        
        all_vars = [('U', U_indices), ('V', V_indices), ('W', W_indices)]
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try more trials per step for better convergence
            num_trials = 100
            for _ in range(num_trials):
                # Pick a random variable
                var_name, indices = random.choice(all_vars)
                idx, jdx = random.choice(indices)
                
                if var_name == 'U':
                    old_val = U_np[idx, jdx]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        U_np[idx, jdx] = new_val
                        err = compute_error_fast(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        U_np[idx, jdx] = old_val
                elif var_name == 'V':
                    old_val = V_np[idx, jdx]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        V_np[idx, jdx] = new_val
                        err = compute_error_fast(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        V_np[idx, jdx] = old_val
                else:
                    old_val = W_np[idx, jdx]
                    for new_val in [old_val - 0.5, old_val + 0.5]:
                        W_np[idx, jdx] = new_val
                        err = compute_error_fast(U_np, V_np, W_np)
                        if err < best_err - 1e-12:
                            best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
                            best_err = err
                            improved = True
                            break
                        W_np[idx, jdx] = old_val
            
            if not improved:
                break
            if best_err < 1e-9:
                break

        return best_U, best_V, best_W, best_err < 1e-9

    def to_list(arr):
        return [[float(arr[i][j]) for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    # Schoolbook solution as fallback
    U_sch = np.zeros((n * m, schoolbook_rank))
    V_sch = np.zeros((m * p, schoolbook_rank))
    W_sch = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j, col] = 1
                V_sch[j * p + k, col] = 1
                W_sch[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sch, V_sch, W_sch
    best_rank = schoolbook_rank

    random.seed(42)
    np.random.seed(42)
    
    def try_rank(target_rank, num_restarts=5):
        """Try to find a valid decomposition at the given rank."""
        nonlocal best_U, best_V, best_W, best_rank
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Structured initialization based on partial schoolbook
            U_r = np.zeros((n * m, target_rank))
            V_r = np.zeros((m * p, target_rank))
            W_r = np.zeros((p * n, target_rank))
            
            # Initialize with random sparse structure
            for r in range(target_rank):
                # Each tensor factor gets a random subset of non-zero entries
                # U: choose random (i,j) positions
                num_u_nonzero = random.randint(2, 4)
                u_positions = random.sample(range(n*m), num_u_nonzero)
                for pos in u_positions:
                    U_r[pos, r] = random.choice([-1.0, 0.5, 1.0, 1.5])
                
                # V: choose random (j,k) positions
                num_v_nonzero = random.randint(2, 4)
                v_positions = random.sample(range(m*p), num_v_nonzero)
                for pos in v_positions:
                    V_r[pos, r] = random.choice([-1.0, 0.5, 1.0, 1.5])
                
                # W: choose random (k,i) positions
                num_w_nonzero = random.randint(2, 4)
                w_positions = random.sample(range(p*n), num_w_nonzero)
                for pos in w_positions:
                    W_r[pos, r] = random.choice([-1.0, 0.5, 1.0, 1.5])
            
            def flatten_params(U, V, W):
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unflatten_params(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank:]
                return U_flat.reshape(n*m, target_rank), V_flat.reshape(m*p, target_rank), W_flat.reshape(p*n, target_rank)
            
            def objective(params):
                U, V, W = unflatten_params(params)
                return compute_error_fast(U, V, W)
            
            initial_params = flatten_params(U_r, V_r, W_r)
            
            try:
                # Use L-BFGS-B with gradient
                res = minimize(objective, initial_params, method='L-BFGS-B', 
                               options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-6, 'maxfun': 5000})
                
                U_opt, V_opt, W_opt = unflatten_params(res.x)
                
                # Quantize
                U_q = quantize(U_opt)
                V_q = quantize(V_opt)
                W_q = quantize(W_opt)
                
                # Check if exact
                if verify_fast(U_q, V_q, W_q):
                    best_U, best_V, best_W = U_q, V_q, W_q
                    best_rank = target_rank
                    return True
                
                # If not exact, try discrete polish
                U_pol, V_pol, W_pol, success = polish_discrete(U_opt, V_opt, W_opt)
                if success:
                    best_U, best_V, best_W = U_pol, V_pol, W_pol
                    best_rank = target_rank
                    return True
                    
            except Exception as e:
                pass
        
        return False

    # Search ranks from lower to higher, prioritizing finding low-rank solutions
    # Start from a reasonable lower bound
    for target_rank in range(35, 112):
        if time.time() > deadline - 20:
            break
            
        if target_rank >= best_rank:
            continue
            
        # Allocate more restarts for lower ranks (harder but more valuable)
        if target_rank < 50:
            num_restarts = 8
        elif target_rank < 70:
            num_restarts = 5
        else:
            num_restarts = 3
        
        if try_rank(target_rank, num_restarts):
            # Found a solution at this rank, continue searching for lower
            continue
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
# EVOLVE-BLOCK-END
