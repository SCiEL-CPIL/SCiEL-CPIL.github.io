"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    NM, MP, PN = n*m, m*p, p*n
    
    # Build target tensor T[i,j,k] = 1 if valid
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    
    T_flat = T.flatten()
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = np.array(U).reshape(n, m, rank)
        V_r = np.array(V).reshape(m, p, rank)
        W_r = np.array(W).reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    def strassen_block(i0, j0, k0, size, U, V, W, offset):
        """
        Embed a 2x2 Strassen decomposition for sub-block [i0:i0+size, j0:j0+size, k0:k0+size].
        Returns (U_block, V_block, W_block, new_offset) where each is list of (row, col, value) tuples.
        """
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # For our 3x4x6, we'll use size-1 blocks (individual elements) with Strassen-like
        # combinations. Actually, let's think differently.
        
        # A more practical approach: use known small matrix multiplication decompositions.
        # For 3x4x6, we can decompose into overlapping 2x2 blocks.
        
        # Let's use a direct algebraic construction:
        # The tensor T[i,j,k] = delta_{i,j,k} (1 for all valid triples)
        # We need U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all (i,j,k)
        # and 0 otherwise.
        
        # Strategy: Use the fact that for matrix multiplication, we can write:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Tensor form: T[i,j,k] = 1
        # We need rank-r decomposition: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = T[i,j,k]
        
        # For the 3x4x6 case, let's try a structured approach:
        # Group indices and use bilinear relations.
        
        pass
    
    def build_strassen_based_decomposition():
        """
        Build a structured decomposition using Strassen-inspired block structure.
        
        Key insight: For 3x4x6 matrix multiplication, we can partition:
        - A is 3x4, B is 4x6, C is 3x6
        - We can decompose this into overlapping 2x2 Strassen blocks
        
        Approach: Use the 2x2 Strassen algorithm (rank 7) as building blocks,
        then combine them for the full 3x4x6 case.
        
        Partition:
        - A = [A1 A2] where A1 is 3x2, A2 is 3x2
        - B = [B1; B2] where B1 is 2x6, B2 is 2x6
        - C = A1*B1 + A2*B2
        
        For each product term, use Strassen-like decomposition.
        """
        # Start with schoolbook and try to reduce via structured combinations
        
        # Actually, let's use a more systematic approach:
        # The matrix multiplication tensor M_{n,m,p} has rank r(n,m,p).
        # Known: r(2,2,2) = 7 (Strassen)
        # For larger matrices, we can use tensor product constructions.
        
        # Let's try to find a decomposition by solving a linear system
        # with a structured sparsity pattern.
        
        # Method: Assume the decomposition has a specific block structure
        # and solve for the free parameters.
        
        # For 3x4x6, try rank = 35 (empirically, small matrix mult tensors
        # often have rank around 0.6-0.8 * n*m*p)
        
        # Let's use a greedy structured approach:
        # 1. Start with schoolbook (rank 72)
        # 2. Try to merge terms that are linearly dependent
        
        # Alternative: Use the fact that T[i,j,k] = 1 for all valid (i,j,k)
        # This is a "complete" tensor, so we need:
        # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1 for all i,j,k
        
        # This is equivalent to finding a rank-r decomposition of the all-ones
        # 3-tensor with the specific index structure.
        
        # Let's try a parameterized family:
        # U[i,j,c] = a[i,c] * b[j,c]
        # V[j,k,c] = b[j,c] * d[k,c]  
        # W[k,i,c] = d[k,c] * a[i,c]
        # Then U*V*W = a[i,c]^2 * b[j,c]^2 * d[k,c]^2
        # Sum_c a[i,c]^2 * b[j,c]^2 * d[k,c]^2 = 1 for all i,j,k
        
        # This requires: sum_c (a[i,c]*b[j,c]*d[k,c])^2 = 1
        # Let x_c = a[i,c]*b[j,c]*d[k,c], then sum_c x_c^2 = 1
        
        # For this to work for ALL i,j,k independently, we need the vectors
        # to be orthogonal in a specific way.
        
        # Actually, let's just use numerical optimization with a good initialization
        # based on the Strassen structure.
        
        return None
    
    def als_decompose_structured(rank, max_iters=100, tol=1e-14, seed=0):
        """
        ALS with structured initialization based on Strassen-like patterns.
        """
        np.random.seed(seed)
        
        # Initialize with a structured pattern:
        # Use the fact that for small blocks, Strassen gives rank-7 decompositions
        # We'll initialize with a pattern that respects the block structure
        
        U = np.zeros((n, m, rank))
        V = np.zeros((m, p, rank))
        W = np.zeros((p, n, rank))
        
        # Structured initialization:
        # For each "virtual" 2x2 block, assign Strassen-like patterns
        # Then add random perturbation for the remaining ranks
        
        # Simple structured init:
        # Assign each (i,j,k) triple to a specific rank component
        # with some sharing
        
        # Initialize with random but scaled values
        U = np.random.randn(n, m, rank) * 0.5
        V = np.random.randn(m, p, rank) * 0.5
        W = np.random.randn(p, n, rank) * 0.5
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    A_mat = np.einsum('kc,kc->ck', V[j], W[:, i])
                    b = T[i, j, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A_mat.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    A_mat = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                    b = T[:, j, k]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A_mat.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    A_mat = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                    b = T[i, :, k]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A_mat.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-8:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_exact(U_rounded, V_rounded, W_rounded, rank):
                return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
            
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
            
            # Polish with L-BFGS-B
            if final_err < 1.0:
                try:
                    from scipy.optimize import minimize
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_x, V_x, W_x)
                        return np.sum((T_rec - T)**2)
                    
                    res = minimize(objective, x0, method='L-BFGS-B', 
                                   options={'maxiter': 100, 'ftol': 1e-18, 'gtol': 1e-12})
                    
                    if res.success or res.fun < 1e-10:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        # Round and check
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        if check_exact(U_rounded, V_rounded, W_rounded, rank):
                            return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
                        
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        if check_exact(U_int, V_int, W_int, rank):
                            return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
                except Exception:
                    pass
        
        return None
    
    def try_rank(rank, num_restarts=10):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            result = als_decompose_structured(rank, max_iters=80, tol=1e-14, seed=seed)
            if result is not None:
                return result
        return None
    
    # Search for lower rank
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(12, 24, 18) = 24
    # For matrix multiplication, better bounds exist but let's search from 30
    best_result = None
    
    # Try ranks from 30 upward
    for rank in range(30, rank_schoolbook):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=6)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
