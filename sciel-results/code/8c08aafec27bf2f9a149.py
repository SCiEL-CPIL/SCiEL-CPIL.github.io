"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import time

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i is row, j is col, k is output.
    Actually, the tensor T[i, j, k] represents: C[k] += A[i] * B[j] where i indexes A's rows,
    j indexes B's cols, and k = (i-1)*p + j is the output position.
    
    More precisely: for C = AB, C[k*n+i] gets contribution from A[i*m+j] * B[j*p+k].
    So tensor T[a, b, c] where a = i*m+j (input A index), b = j*p+k (input B index), 
    c = k*n+i (output index), T[a,b,c] = 1 if the indices are consistent (same j).
    """
    # T is a 3D tensor of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=object)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    return T

def _tensor_to_matrix(T):
    """Reshape 3D tensor (I, J, K) into matrix (I, J*K) for rank computation."""
    I, J, K = T.shape
    return T.reshape(I, J * K)

def _matrix_rank_rational(M):
    """Compute rank of a matrix with rational entries using exact Gaussian elimination."""
    if M.size == 0:
        return 0
    rows, cols = M.shape
    M = np.array(M, dtype=object)
    rank = 0
    for c in range(cols):
        # Find pivot
        pivot = None
        for r in range(rank, rows):
            if M[r, c] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap
        M[rank], M[pivot] = M[pivot].copy(), M[rank].copy()
        # Eliminate below
        pivot_val = M[rank, c]
        for r in range(rank + 1, rows):
            if M[r, c] != 0:
                factor = M[r, c] / pivot_val
                for cc in range(c, cols):
                    M[r, cc] = M[r, cc] - factor * M[rank, cc]
        rank += 1
    return rank

def _try_find_decomposition(n, m, p, max_rank, deadline):
    """Try to find a rank-r decomposition for matrix multiplication."""
    start_time = time.time()
    
    # Build the tensor
    T = _build_tensor(n, m, p)
    
    # The decomposition is: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # For each r, we need U[:,r], V[:,r], W[:,r] such that the product matches.
    # 
    # This is a bilinear problem. We can fix U and V and solve for W, or use iterative methods.
    #
    # Alternative: Use the fact that for small problems, we can try random initializations
    # and use gradient descent or alternating minimization.
    
    # For now, let's try a structured approach:
    # The schoolbook has rank n*m*p. Can we do better?
    # 
    # Known results: For 2x3x6, the tensor rank might be less than 36.
    # Let's try to compute the rank of the tensor matricizations.
    
    # Matricization 1: (n*m) x (m*p * p*n) -> not standard
    # Standard matricizations:
    # T1: (n*m) x (m*p * p*n) - flatten last two
    # T2: (m*p) x (n*m * p*n) - flatten first and third
    # T3: (p*n) x (n*m * m*p) - flatten first two
    
    # The rank of the tensor is at least the max of the ranks of its matricizations.
    # But the tensor rank can be higher than the matricization ranks.
    
    # Let's compute matricization ranks to get a lower bound.
    T1 = T.reshape(n * m, m * p * p * n)
    T2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)
    
    # Compute ranks using SVD (approximate, but for integer tensors should be exact)
    r1 = np.linalg.matrix_rank(T1.astype(float))
    r2 = np.linalg.matrix_rank(T2.astype(float))
    r3 = np.linalg.matrix_rank(T3.astype(float))
    
    lower_bound = max(r1, r2, r3)
    
    # Try to find a decomposition with rank = lower_bound
    # Use alternating minimization with exact arithmetic
    
    rank = lower_bound
    if rank > max_rank:
        rank = max_rank
    
    # Initialize with random values
    np.random.seed(42)
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Alternating minimization
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Fix U and V, solve for W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For each c: sum_r (sum_{a,b} T[a,b,c] * U[a,r] * V[b,r]) = W[c,r] * (sum over r' ...)
        # Actually, this is not linear in W when U and V are fixed.
        # 
        # Let me reformulate. The constraint is:
        # For all a,b,c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        #
        # This is a system of (n*m)*(m*p)*(p*n) equations in (n*m + m*p + p*n)*rank unknowns.
        #
        # For small problems, we can set up the linear system.
        
        # Actually, let's use a different approach: 
        # Fix two factors and solve for the third using least squares.
        
        # Fix U and V, solve for W:
        # For each c and r: W[c,r] = ?
        # The equation is: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c] for all a,b,c
        #
        # This is not directly separable. Let me think again.
        
        # Actually, the standard approach is to use the fact that:
        # T = sum_r U_r ⊗ V_r ⊗ W_r (outer products)
        #
        # So we're looking for rank-r decomposition of the 3D tensor.
        #
        # For small tensors, we can use:
        # 1. Random initialization
        # 2. Alternating least squares (ALS)
        
        # ALS: Fix U and V, solve for W by minimizing ||T - sum_r U_r ⊗ V_r ⊗ W_r||^2
        # This is linear in W.
        
        # Build the linear system for W
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U,V: this is linear in W[c,r]
        #
        # Let's vectorize: for each (a,b), we have:
        # T[a,b,:] = sum_r (U[a,r] * V[b,r]) * W[:,r]
        #
        # So: T[a,b,:] = W @ (U[a,:] * V[b,:])^T  (element-wise product)
        #
        # Let M_ab = U[a,:] * V[b,:] (element-wise, shape (rank,))
        # Then T[a,b,:] = W @ M_ab^T
        #
        # Stacking over all (a,b): 
        # T_flat = W @ (all M_ab stacked)^T
        #
        # Where T_flat has shape (n*m*m*p, p*n) and the matrix on the right has shape (p*n, n*m*m*p*rank)
        
        # Build the system
        nm = n * m
        mp = m * p
        pn = p * n
        
        # For each (a, b), form the coefficient vector
        # A[c, (a,b,r)] = U[a,r] * V[b,r]
        # Then T[a,b,c] = sum_r A[c, (a,b,r)] * W[c,r]
        # Wait, that's not right either.
        
        # Let me be more careful.
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        #
        # For fixed U and V, this is linear in W.
        # Let's write it as: for each c, T[:, :, c] = sum_r (U * V[:, r]) * W[c, r]
        # where (U * V[:, r])[a, b] = U[a, r] * V[b, r]
        
        # So: T[:, :, c] = sum_r W[c, r] * (U[:, r] ⊗ V[:, r])
        #
        # This means: T[:,:,c] is a linear combination of the rank-1 matrices U[:,r] ⊗ V[:,r]
        # with coefficients W[c,r].
        
        # Let's set up the least squares problem.
        # Flatten T to shape (nm*mp, pn)
        T_mat = T.reshape(nm * mp, pn)
        
        # Build the design matrix: for each (a,b,c), the coefficient of W[c,r] is U[a,r]*V[b,r]
        # So the design matrix has shape (nm*mp*pn, rank) where entry ((a,b), c, r) = U[a,r]*V[b,r] if c is the output index
        # Actually, let's think of it as: for each output index c, we have a system:
        # T[:,:,c] = W[c,:] @ (U ⊙ V)^T  where ⊙ is element-wise
        #
        # Let's build it properly.
        
        # For each c in range(pn):
        #   T[:,:,c] (shape (nm, mp)) = sum_r W[c,r] * (U[:,r] * V[:,r])  (outer product, shape (nm, mp))
        #
        # Flatten: T[:,:,c].flatten() (shape (nm*mp,)) = sum_r W[c,r] * (U[:,r] * V[:,r]).flatten()
        #
        # So for each c: T_c = W[c,:] @ A_c^T where A_c[:,r] = (U[:,r] * V[:,r]).flatten()
        #
        # This is a linear system in W[c,:].
        
        W_new = np.zeros((pn, rank))
        for c in range(pn):
            # Build A_c: shape (nm*mp, rank)
            A_c = np.zeros((nm * mp, rank))
            for r in range(rank):
                A_c[:, r] = (U[:, r] * V[:, r]).flatten()
            
            # Solve: A_c @ W_new[c,:] = T[:,:,c].flatten()
            T_c = T_mat[:, c]
            
            # Least squares
            try:
                sol, residuals, rank_c, sv = np.linalg.lstsq(A_c, T_c, rcond=None)
                W_new[c] = sol
            except:
                W_new[c] = W[c]
        
        W = W_new
        
        # Similarly for U and V
        # Fix V and W, solve for U
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V,W: T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
        # So: T[a,b,c] = U[a,:] @ (V[b,:] * W[c,:])^T
        #
        # For each a: T[a,:,:] (shape (mp, pn)) = sum_r U[a,r] * (V[:,r] * W[:,r]) (outer product, shape (mp, pn))
        # Flatten: T[a,:,:].flatten() = U[a,:] @ (V[:,r] * W[:,r]).flatten()
        
        U_new = np.zeros((nm, rank))
        for a in range(nm):
            A_a = np.zeros((mp * pn, rank))
            for r in range(rank):
                A_a[:, r] = (V[:, r] * W[:, r]).flatten()
            T_a = T.reshape(nm, mp * pn)[a]
            try:
                sol, residuals, rank_a, sv = np.linalg.lstsq(A_a, T_a, rcond=None)
                U_new[a] = sol
            except:
                U_new[a] = U[a]
        U = U_new
        
        # Fix U and W, solve for V
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U,W: T[a,b,c] = sum_r V[b,r] * (U[a,r] * W[c,r])
        # For each b: T[:,b,:] (shape (nm, pn)) = sum_r V[b,r] * (U[:,r] * W[:,r]) (outer product)
        
        V_new = np.zeros((mp, rank))
        for b in range(mp):
            A_b = np.zeros((nm * pn, rank))
            for r in range(rank):
                A_b[:, r] = (U[:, r] * W[:, r]).flatten()
            T_b = T.reshape(nm, mp, pn)[:, b, :].flatten()
            try:
                sol, residuals, rank_b, sv = np.linalg.lstsq(A_b, T_b, rcond=None)
                V_new[b] = sol
            except:
                V_new[b] = V[b]
        V = V_new
        
        # Check convergence
        if iteration % 10 == 0:
            # Compute the reconstruction error
            error = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        recon = 0.0
                        for r in range(rank):
                            recon += U[a, r] * V[b, r] * W[c, r]
                        error += abs(recon - T[a, b, c])
            if error < 1e-8:
                break
    
    # Check if the decomposition is valid
    max_error = 0.0
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                recon = 0.0
                for r in range(rank):
                    recon += U[a, r] * V[b, r] * W[c, r]
                max_error = max(max_error, abs(recon - T[a, b, c]))
    
    if max_error < 1e-8:
        # Convert to half-integers if possible
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        # Verify
        max_error_int = 0.0
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    recon = 0.0
                    for r in range(rank):
                        recon += U_int[a, r] * V_int[b, r] * W_int[c, r]
                    max_error_int = max(max_error_int, abs(recon - T[a, b, c]))
        
        if max_error_int < 1e-8:
            return U_int, V_int, W_int
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # First, get the schoolbook decomposition as a fallback
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = (rank_schoolbook, U_sb, V_sb, W_sb)
    
    # Try to find a lower rank decomposition
    T = _build_tensor(n, m, p)
    
    # Compute lower bound from matricization ranks
    T1 = T.reshape(n * m, m * p * p * n).astype(float)
    T2 = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n).astype(float)
    T3 = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p).astype(float)
    
    r1 = np.linalg.matrix_rank(T1)
    r2 = np.linalg.matrix_rank(T2)
    r3 = np.linalg.matrix_rank(T3)
    lower_bound = max(r1, r2, r3)
    
    # Try ranks from lower_bound to rank_schoolbook-1
    for rank in range(lower_bound, min(rank_schoolbook, 36)):
        if time.time() > deadline - 30:  # Safety margin
            break
        
        result = _try_find_decomposition(n, m, p, rank, deadline - 30)
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            # Verify
            T_check = _build_tensor(n, m, p)
            valid = True
            for a in range(n * m):
                for b in range(m * p):
                    for c in range(p * n):
                        recon = 0.0
                        for r in range(rank):
                            recon += U_list[a][r] * V_list[b][r] * W_list[c][r]
                        if abs(recon - T_check[a, b, c]) > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
