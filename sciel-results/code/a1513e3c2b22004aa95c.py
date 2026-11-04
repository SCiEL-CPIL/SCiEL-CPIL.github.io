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
    
    # Try to find lower rank via random search with verification
    # Known lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Strassen-like or other decompositions might achieve rank between 20 and 60
    
    # Use iterative randomized approach: start with random matrices, 
    # verify if they decompose the tensor exactly (up to tolerance)
    
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 
    # where the multiplication is: C[i,k] = sum_j A[i,j]*B[j,k]
    # In flattened form: T[i*m+j, j*p+k, k*n+i] = 1 for the "standard" 
    # but actually the tensor is defined as: for each (i,j,k), the 
    # multiplication contributes A[i,j]*B[j,k] to C[i,k]
    # So the tensor is: T[(i,j), (j,k), (i,k)] = 1 in the appropriate indexing
    
    # Let's define the tensor properly:
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition says: for all (a, b, c) where a=(i,j), b=(j',k'), c=(k'',i''),
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if j=j' and k'=k'' and i=i'', else 0
    # More precisely: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (where j must match between U and V, and k must match between V and W, 
    #  and i must match between W and U)
    
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
        # Compute the reconstructed tensor
        # recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is equivalent to: reshape U to (n*m, rank), V to (m*p, rank), W to (p*n, rank)
        # Then for each r, outer product U[:,r] * V[:,r] * W[:,r]
        # Sum over r
        
        # Efficient computation: 
        # Create rank-1 contributions
        # We can do this as: for each r, add U[:,r] * V[:,r] * W[:,r]
        # But this is O(n*m * m*p * p*n * rank) which is fine for small cases
        
        # Alternative: use einsum
        # recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        # But for exactness, let's compute carefully
        
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(U_mat.shape[1]):
            recon += np.outer(np.outer(U_mat[:, r], V_mat[:, r]), W_mat[:, r])
        
        # Check exact match (within very tight tolerance for floating point)
        diff = np.abs(recon - T)
        max_diff = np.max(diff) if recon.size > 0 else 0
        return max_diff < 1e-8, max_diff
    
    # Random search strategy
    # For a given rank R, generate random U, V, W and check
    # But this is extremely unlikely to find exact solutions randomly
    # 
    # Better approach: Use the structure. The matrix multiplication tensor 
    # has known decompositions. For 3x4x5, we can try to find a decomposition
    # by solving a system of equations.
    #
    # Actually, let's try a different approach: use the fact that the tensor
    # can be decomposed using bilinear algorithms. 
    #
    # For now, let's try small rank improvements by searching over structured 
    # decompositions.
    
    # Try rank from lower bound up to schoolbook
    # Use a smarter approach: start with schoolbook and try to merge/eliminate terms
    
    # Alternative: use scipy or numpy to solve for a given rank
    # Set up as a system: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # This is a system of quadratic equations. Hard to solve directly.
    
    # Let's try a practical approach: use known results or try specific ranks
    # The rank of matrix multiplication is an active area of research.
    # For 3x4x5, the rank is not known to be much below 60 in general,
    # but specific structured decompositions might help.
    
    # Try a few random restarts with constrained search
    random.seed(42)
    
    # Try ranks from 20 up to 59 (below schoolbook)
    # Use a heuristic: for each rank, try to find a solution
    # by starting from random initial conditions and using gradient-based
    # optimization on the objective ||recon - T||^2
    
    from scipy.optimize import minimize
    
    def make_flattened_params(R):
        """Create flattened parameter vector for U, V, W with rank R"""
        U_flat = np.random.randn(n*m, R) * 0.1
        V_flat = np.random.randn(m*p, R) * 0.1
        W_flat = np.random.randn(p*n, R) * 0.1
        # Concatenate
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
        # Compute reconstruction error
        # Use einsum for efficiency
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        return error
    
    def objective_grad(params, R):
        """Compute objective and gradient"""
        U_mat, V_mat, W_mat = unpack_params(params, R)
        
        # recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # d(recon)/dU[a,r] = V[b,r]*W[c,r] (summed over b,c for the gradient)
        
        recon = np.einsum('ar,br,cr->abc', U_mat, V_mat, W_mat)
        error = np.sum((recon - T)**2)
        
        # Gradient
        diff = recon - T
        grad_U = np.einsum('abc,br,cr->ar', diff, V_mat, W_mat) * 2
        grad_V = np.einsum('abc,ar,cr->br', diff, U_mat, W_mat) * 2
        grad_W = np.einsum('abc,ar,br->cr', diff, U_mat, V_mat) * 2
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return error, grad
    
    # Try to find solutions for ranks below schoolbook
    best_solution = None
    
    for R in range(30, schoolbook_rank):  # Try ranks from 30 to 59
        if time.time() > deadline - 10:
            break
        
        # Try a few random restarts
        num_restarts = 3
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
                
            params, U0, V0, W0 = make_flattened_params(R)
            
            # Use L-BFGS-B
            try:
                result = minimize(
                    lambda p: objective(p, R),
                    params,
                    jac=lambda p: objective_grad(p, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
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
            break
    
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
