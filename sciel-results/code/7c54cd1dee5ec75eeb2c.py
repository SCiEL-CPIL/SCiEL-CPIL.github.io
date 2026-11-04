"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # leave margin before 150s wall-clock limit
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    schoolbook_rank = n * m * p  # 60
    
    # Schoolbook decomposition as fallback
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Build the full tensor as a 3D array for verification
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1
        return T
    
    T = build_tensor()
    
    def verify_decomposition(U_mat, V_mat, W_mat):
        """Check if U * V * W (tensor product) equals T exactly"""
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(U_mat.shape[1]):
            recon += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r])
        
        diff = np.abs(recon - T)
        max_diff = np.max(diff) if recon.size > 0 else 0
        return max_diff < 1e-8, max_diff
    
    # Use Kruskal-Cantor rank bound analysis via LP relaxation
    # The tensor support is the set of (i,j,k) tuples where the tensor is non-zero
    # For matrix multiplication, this is {(i,j,k) : i in [0,n), j in [0,m), k in [0,p)}
    
    # The Kruskal-Cantor bound gives a lower bound on rank based on the
    # structure of the tensor. We use an LP relaxation to estimate achievable rank.
    
    # Strategy: Try to find a decomposition with rank R by solving a system
    # where we allow entries to be in {0, 1/2, 1, -1/2, -1}
    
    # For small matrices, we can try structured decompositions
    # Let's try a block approach: partition the multiplication into blocks
    # and use Strassen-like tricks on each block
    
    # First, let's try to find if rank 55 or so is achievable
    # by using a hybrid approach: some schoolbook terms + some Strassen-like terms
    
    # Actually, let's use a more systematic approach:
    # 1. Start with schoolbook (rank 60)
    # 2. Try to eliminate terms by finding linear dependencies
    # 3. Use the fact that for 3x4x5, we might be able to save a few terms
    
    # Let's try random search with better initialization
    random.seed(42)
    
    from scipy.optimize import minimize
    
    def make_flattened_params(R, init_type='random'):
        """Create flattened parameter vector for U, V, W with rank R"""
        if init_type == 'schoolbook':
            # Start from schoolbook and perturb
            U_flat = np.zeros((n*m, R))
            V_flat = np.zeros((m*p, R))
            W_flat = np.zeros((p*n, R))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < R:
                            U_flat[i*m+j, col] = 1
                            V_flat[j*p+k, col] = 1
                            W_flat[k*n+i, col] = 1
                        col += 1
            # Add small random perturbation to break symmetry
            U_flat += np.random.randn(n*m, R) * 0.01
            V_flat += np.random.randn(m*p, R) * 0.01
            W_flat += np.random.randn(p*n, R) * 0.01
        else:
            U_flat = np.random.randn(n*m, R) * 0.1
            V_flat = np.random.randn(m*p, R) * 0.1
            W_flat = np.random.randn(p*n, R) * 0.1
        
        params = np.concatenate([U_flat.flatten(), V_flat.flatten(), W_flat.flatten()])
        return params, U_flat, V_flat, W_flat
    
    def unpack_params(params, R):
        U_size = n*m*R
        V_size = m*p*R
        U_flat = params[:U_size].reshape(n*m, R)
        V_flat = params[U_size:U_size+V_size].reshape(m*p, R)
        W_flat = params[U_size+V_size:].reshape(p*n, R)
        return U_flat, V_flat, W_flat
    
    def objective(params, R):
        if time.time() > deadline - 5:
            return 1e10
        U_mat, V_mat, W_mat = unpack_params(params, R)
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        return error
    
    def objective_grad(params, R):
        """Compute objective and gradient"""
        U_mat, V_mat, W_mat = unpack_params(params, R)
        
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        
        diff = recon - T
        grad_U = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat) * 2
        grad_V = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat) * 2
        grad_W = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat) * 2
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad
    
    # Try to find solutions for ranks below schoolbook
    best_solution = None
    
    # Try ranks from 55 down to 30, with schoolbook initialization for higher ranks
    for R in range(55, 29, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts with different initializations
        num_restarts = 4
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            init_type = 'schoolbook' if R >= 55 else 'random'
            params, U0, V0, W0 = make_flattened_params(R, init_type)
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    lambda p: objective(p, R),
                    params,
                    jac=lambda p: objective_grad(p, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < 1e-6:
                    U_mat, V_mat, W_mat = unpack_params(result.x, R)
                    is_valid, max_diff = verify_decomposition(U_mat, V_mat, W_mat)
                    if is_valid:
                        best_solution = (R, U_mat, V_mat, W_mat)
                        best_rank = R
                        break
            
            except Exception:
                pass
        
        if best_solution is not None and best_solution[0] == R:
            # Found a solution at this rank, try to go lower
            continue
    
    # If we found a better solution, use it; otherwise use schoolbook
    if best_solution is not None:
        R, U_mat, V_mat, W_mat = best_solution
        best_rank = R
    else:
        U_mat, V_mat, W_mat = U_sb, V_sb, W_sb
    
    # Convert to lists as required by the interface
    U_list = U_mat.tolist()
    V_list = V_mat.tolist()
    W_list = W_mat.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
