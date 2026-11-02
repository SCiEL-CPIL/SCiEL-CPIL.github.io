"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 52
    
    # Build the tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try to find lower rank decompositions
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Generate schoolbook decomposition
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1.0
                V_sb[j*p+k, col] = 1.0
                W_sb[k*n+i, col] = 1.0
                col += 1
    
    best_decomp = (U_sb, V_sb, W_sb)
    
    # Try ranks below schoolbook
    # The tensor rank lower bound for matrix mult (n,m,p) is at least max(n*m, m*p, p*n)/something
    # For (2,2,13): lower bound is at least ceil(2*13/2) = 13? No, that's not right.
    # Actually, the rank must be at least the max of the dimensions of the slices.
    # T[i, :, :] has rank at most m*p = 26 for each i, but that's not a lower bound on tensor rank.
    # A better lower bound: the tensor rank is at least the max over all "slices" of the matrix rank.
    
    # Let's try a few candidate ranks and use random initialization + gradient descent
    # to see if we can find valid decompositions.
    
    def verify_decomp(U, V, W, tol=1e-6):
        """Check if U, V, W decompose T."""
        # Reconstruct tensor
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.allclose(T_recon, T, atol=tol, rtol=tol)
    
    # Try some structured approaches first
    # Approach: exploit the fact that for each output (i,k), the computation is
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This is a sum of m terms for each of n*p outputs.
    
    # Let's try rank = n*p + m*p - n*m + ... various formulas
    # For (2,2,13): try ranks like 39, 40, 42, 44, 46, 48, 50
    
    candidate_ranks = [50, 48, 46, 44, 42, 40, 39, 38, 36, 34, 32, 30, 26, 24, 22, 20, 18, 16, 14, 13]
    
    np.random.seed(42)
    
    for rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
            
        # Try a few random initializations
        max_attempts = 3 if rank > 30 else 5
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                    break
            
            # Random initialization
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
            
            # Normalize to help optimization
            # Scale so that the reconstruction is close in scale
            T_norm = np.linalg.norm(T)
            recon_norm = np.linalg.norm(np.einsum('ac,bc,dc->abd', U, V, W))
            if recon_norm > 0:
                scale = T_norm / recon_norm
                U = U * scale**(1/3)
                V = V * scale**(1/3)
                W = W * scale**(1/3)
            
            # Use gradient descent to minimize ||recon - T||_F
            # Parameters: flatten U, V, W
            def objective(params):
                U_p = params[:n*m*rank].reshape(n*m, rank)
                V_p = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_p = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                recon = np.einsum('ac,bc,dc->abd', U_p, V_p, W_p)
                return np.sum((recon - T)**2)
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Use L-BFGS-B with limited iterations
            try:
                result = minimize(objective, params0, method='L-BFGS-B', 
                                  options={'maxiter': 50, 'maxfun': 100, 'gtol': 1e-8})
                if result.fun < 1e-4:
                    U_opt = result.x[:n*m*rank].reshape(n*m, rank)
                    V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Verify
                    if verify_decomp(U_opt, V_opt, W_opt, tol=1e-4):
                        best_rank = rank
                        best_decomp = (U_opt, V_opt, W_opt)
                        break
            except Exception:
                pass
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_decomp is not None and best_rank < schoolbook_rank:
        U, V, W = best_decomp
        # Try to clean up to integers or half-integers
        # Round small values to 0, and see if rounding to nearest 0.5 works
        def clean_matrix(M):
            M_clean = M.copy()
            # Round to nearest 0.5
            M_rounded = np.round(M_clean * 2) / 2
            # Check if rounding preserves the decomposition
            if verify_decomp(M_rounded[0] if isinstance(M_rounded, tuple) else M_rounded, 
                            V if M is U else (U if M is V else W),
                            W if M is U else (V if M is V else U), tol=1e-3):
                return M_rounded
            return M_clean
        
        # Actually, let's just check if the values are already close to half-integers
        def is_half_integer(M, tol=1e-3):
            diff = M * 2 - np.round(M * 2)
            return np.all(np.abs(diff) < tol)
        
        if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
    else:
        U, V, W = best_decomp
    
    # Convert to lists of lists with exact rational-like values
    def to_list_of_lists(M):
        rows = []
        for row in M:
            r = []
            for val in row:
                # Round to nearest 0.5 for cleanliness
                v = round(val * 2) / 2
                # If very close to integer, make it integer
                if abs(v - round(v)) < 1e-9:
                    v = int(round(v))
                r.append(v)
            rows.append(r)
        return rows
    
    U_list = to_list_of_lists(U)
    V_list = to_list_of_lists(V)
    W_list = to_list_of_lists(W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
