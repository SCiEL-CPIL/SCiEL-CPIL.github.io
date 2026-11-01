import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 8]

def _verify_fast(U, V, W, n=2, m=4, p=8, trials=20, tol=1e-8):
    """Fast verification using vectorized numpy operations."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    if rank == 0:
        return False
    
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n*m, rank) or V_mat.shape != (m*p, rank) or W_mat.shape != (p*n, rank):
        return False

    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        U_reshaped = U_mat.reshape(n, m, rank)
        V_reshaped = V_mat.reshape(m, p, rank)
        W_reshaped = W_mat.reshape(p, n, rank)

        factor_A = np.einsum('ij,ijc->ic', A, U_reshaped)
        factor_B = np.einsum('jk,jkc->kc', B, V_reshaped)
        C_computed = np.einsum('ic,kc,kic->ik', factor_A, factor_B, W_reshaped)

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _get_target_tensor(n, m, p):
    """
    Construct the target tensor T such that T[i,j,k] = 1 if (i,j,k) is part of the multiplication.
    Actually, the tensor for matrix mult is defined by:
    C[i,k] = sum_j A[i,j] B[j,k]
    In tensor decomposition form:
    sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    where T[i,j,k] = 1 for all valid i,j,k.
    Wait, the standard tensor for matrix multiplication M_{n,m,p} has entries:
    M[i, j, k] = 1 if the indices correspond to a valid term in the sum.
    Specifically, the tensor is a 3-tensor of shape (n, m, p) where M[i,j,k]=1 for all i,j,k?
    No. The tensor is defined such that:
    (A @ B)[i, k] = sum_{j} A[i, j] B[j, k]
    The decomposition requires:
    sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1  for all i, j, k.
    So the target tensor T is simply the all-ones tensor of shape (n, m, p).
    """
    return np.ones((n, m, p))

def _build_mps_initialization(n, m, p, r, bond_dim):
    """
    Initialize the MPS (Tensor Ring) parameters.
    The factors U, V, W are constructed from the MPS matrices.
    
    U has shape (n, m, r). We flatten (n, m) to (n*m).
    V has shape (m, p, r). We flatten (m, p) to (m*p).
    W has shape (p, n, r). We flatten (p, n) to (p*n).
    
    In the MPS formulation, we usually have a ring of tensors.
    However, here we have three distinct factor matrices.
    A simpler "Ring" approach for this specific 3-factor structure:
    We can view the rank-r decomposition as a 3-tensor network.
    
    Let's use a different parameterization:
    Instead of full U, V, W, we parameterize them using a set of "core" matrices 
    that share structure, or just use the full matrices but optimize them directly?
    Direct optimization of U, V, W with r=50 is 50*(8+32+16) = 2600 parameters.
    L-BFGS can handle this. The previous strategy failed because it used local perturbations.
    Global optimization from a good start (Strassen) might find a lower rank.
    
    Let's try direct optimization of U, V, W for a target rank r.
    We start with the Strassen rank 56 solution, zero-pad to a larger rank if needed?
    No, we want to find a rank < 56.
    
    Strategy:
    1. Start with rank 56 Strassen solution.
    2. Try to optimize for rank 55, 54, ... down to 30.
    3. For a fixed rank r, we initialize U, V, W with the first r columns of the rank 56 solution.
    4. Run L-BFGS-B to minimize ||T - T_decomp||^2.
    5. If the residual is small enough, we have a valid rank-r decomposition.
    
    To make it faster, we can fix one of the matrices (e.g., W) and optimize U and V, or optimize all.
    Optimizing all is non-convex.
    
    Let's try optimizing U and V, and solving for W analytically?
    For fixed U, V, the optimal W is given by the least squares solution.
    This reduces the parameter count and makes the objective smoother.
    
    Let's define the objective function for fixed W? No, W is linear.
    Let's define the objective for fixed U, V and solve for W.
    
    Parameters: U (n*m, r), V (m*p, r).
    For a given U, V, we can compute the "partial tensor" P[i, k, c] = sum_j U[i, j, c] V[j, k, c].
    Then we need W[k, i, c] such that sum_c P[i, k, c] W[k, i, c] = 1.
    This is a linear system for W.
    Let W_vec be the vector of all W entries.
    The equation is: sum_c P[i, k, c] W[k, i, c] = 1 for all i, k.
    This is a system of n*p equations in n*p*r unknowns?
    Actually, for each (i, k), we have one equation:
    sum_c P[i, k, c] W[k, i, c] = 1.
    The unknowns for this specific (i, k) are W[k, i, 0], ..., W[k, i, r-1].
    So for each (i, k), we have a system:
    P[i, k, :] @ W[k, i, :] = 1.
    This is r unknowns, 1 equation. It's underdetermined if r > 1.
    We want the minimum norm solution: W[k, i, :] = P[i, k, :]^T / ||P[i, k, :]||^2.
    
    So the algorithm is:
    1. Parameterize U and V.
    2. Compute P[i, k, c] = sum_j U[i, j, c] V[j, k, c].
    3. Compute W[k, i, c] = P[i, k, c] / sum_c' P[i, k, c']^2 (if norm > eps).
    4. Compute the residual: R = 1 - sum_c P[i, k, c] W[k, i, c].
       Note: sum_c P W = sum_c P * (P / ||P||^2) = ||P||^2 / ||P||^2 = 1.
       Wait, if we choose W this way, the residual is ALWAYS 0?
       No, the residual is 0 for the equation sum_c P[i,k,c] W[k,i,c] = 1.
       BUT, the decomposition must hold for ALL i, j, k.
       The equation sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 must hold for all i, j, k.
       My linear system above only ensures that the "summed over j" version works?
       No, the tensor decomposition is:
       T[i, j, k] = sum_c U[i, j, c] V[j, k, c] W[k, i, c].
       For matrix multiplication, T[i, j, k] = 1 for all i, j, k.
       
       So we need:
       sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1 for ALL i, j, k.
       
       If we fix U, V, and choose W to satisfy the equation for the "effective" tensor P[i, k, c] = sum_j U[i, j, c] V[j, k, c],
       that is NOT sufficient. The equation must hold for each specific j as well.
       The tensor T is not just the sum over j. The term U[i,j,c] V[j,k,c] W[k,i,c] is specific to j.
       
       So we cannot simply solve for W given U, V. The system is:
       For each (i, j, k):
       sum_c U[i, j, c] V[j, k, c] W[k, i, c] = 1.
       
       This is a system of n*m*p equations.
       Unknowns: W[k, i, c].
       For a fixed (i, k), the unknowns are W[k, i, 0..r-1].
       The equations involving W[k, i, :] are for all j:
       sum_j (sum_c U[i, j, c] V[j, k, c] W[k, i, c]) = n? No.
       Each equation is separate.
       For a fixed (i, k), we have m equations (one for each j) and r unknowns (W[k, i, 0..r-1]).
       Equation j: sum_c (U[i, j, c] V[j, k, c]) W[k, i, c] = 1.
       Let M_j[c] = U[i, j, c] V[j, k, c].
       Then M_j @ W[k, i, :] = 1 for j=0..m-1.
       This is a system of m equations in r unknowns.
       If r >= m, we can solve this in the least squares sense.
       If r < m, it might be inconsistent.
       
       So, for fixed U, V, the optimal W for each (i, k) is the least squares solution to:
       A_ik @ w_ik = 1
       where A_ik is an m x r matrix with A_ik[j, c] = U[i, j, c] V[j, k, c].
       
       This makes the problem tractable!
       We can optimize U and V, and compute W analytically for each (i, k) pair.
       
       Objective Function:
       1. Take parameters U_flat, V_flat.
       2. Reshape to U (n, m, r), V (m, p, r).
       3. For each i in 0..n-1, k in 0..p-1:
          - Construct A_ik (m x r): A_ik[j, c] = U[i, j, c] * V[j, k, c].
          - Solve A_ik @ w = 1 for w (length r) using least squares.
          - Store w as W[k, i, :].
          - Compute residual for this (i, k): ||A_ik @ w - 1||^2.
       4. Sum residuals over all i, k.
       
       This minimizes the error in the "contracted" tensor?
       Wait. The original condition is sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for ALL j.
       My least squares solution minimizes sum_j (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2.
       This is exactly the Frobenius norm of the residual tensor restricted to the (i, k) slices.
       Minimizing this over all i, k minimizes the total Frobenius norm of the residual tensor T_decomp - T_target.
       
       This is a valid strategy!
       Parameters: U (n*m*r), V (m*p*r).
       For n=2, m=4, p=8, r=50:
       U params: 2*4*50 = 400.
       V params: 4*8*50 = 1600.
       Total: 2000 parameters.
       
       This is much better than optimizing W as well (which would add p*n*r = 8*2*50 = 800 params, total 2800).
       And it handles the non-linearity by keeping the W part linear/analytic.
    """
    pass

def _objective(params, n, m, p, r):
    """
    Objective function for L-BFGS-B.
    params: flattened U and V.
    Returns: Frobenius norm squared of the residual tensor.
    """
    U_size = n * m * r
    V_size = m * p * r
    
    U_flat = params[:U_size]
    V_flat = params[U_size:]
    
    U = U_flat.reshape(n, m, r)
    V = V_flat.reshape(m, p, r)
    
    total_residual = 0.0
    
    # Pre-compute V transposed for efficiency?
    # V[j, k, c]
    
    for i in range(n):
        for k in range(p):
            # Construct A_ik (m x r)
            # A_ik[j, c] = U[i, j, c] * V[j, k, c]
            # U[i, :, :] is (m, r)
            # V[:, k, :] is (m, r)
            A_ik = U[i, :, :] * V[:, k, :]
            
            # Solve A_ik @ w = 1
            # A_ik is (m, r). w is (r,).
            # We want to minimize ||A_ik @ w - 1||^2.
            try:
                w, residuals, rank, sv = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                # Compute residual
                diff = A_ik @ w - 1.0
                total_residual += np.dot(diff, diff)
            except:
                # If singular or error, return large value
                return 1e10
    
    return total_residual

def _optimize_rank(n, m, p, r, start_params, start_time, deadline):
    """
    Try to optimize U and V for a given rank r, starting from start_params.
    Returns (success, U, V, W) if success.
    """
    if time.time() > deadline - 10:
        return False, None, None, None
    
    # Initial guess for U, V
    # start_params is flattened U, V from a previous solution or random
    
    # We need to be careful with the initial guess.
    # If we start from Strassen (rank 56) and want rank 50, we take the first 50 columns.
    
    res = minimize(_objective, start_params, args=(n, m, p, r), method='L-BFGS-B', 
                   options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
    
    if time.time() > deadline - 10:
        return False, None, None, None
    
    # Check if the solution is good enough
    # The objective is the sum of squared errors.
    # There are n*p slices. Each slice has m equations.
    # Total equations: n*m*p = 64.
    # We want the error to be very small.
    # Let's say if objective < 1e-6, it's a success.
    
    if res.fun < 1e-6:
        U_size = n * m * r
        V_size = m * p * r
        U_flat = res.x[:U_size]
        V_flat = res.x[U_size:]
        
        U = U_flat.reshape(n, m, r)
        V = V_flat.reshape(m, p, r)
        
        # Reconstruct W
        W = np.zeros((p, n, r))
        for i in range(n):
            for k in range(p):
                A_ik = U[i, :, :] * V[:, k, :]
                try:
                    w, _, _, _ = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                    W[k, i, :] = w
                except:
                    pass
        
        # Convert to required format: U (n*m, r), V (m*p, r), W (p*n, r)
        U_out = U.reshape(n*m, r).tolist()
        V_out = V.reshape(m*p, r).tolist()
        W_out = W.reshape(p*n, r).tolist()
        
        # Final verification
        if _verify_fast(U_out, V_out, W_out, n, m, p, trials=10, tol=1e-7):
            return True, U_out, V_out, W_out
    
    return False, None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Build the standard Strassen decomposition (Rank 56)
    # We will use this as the initial guess for lower ranks.
    
    # First, let's explicitly build the Rank 56 Strassen solution to use as a seed.
    # We can reuse the logic from the current program or build it here.
    # Let's build a simple Rank 64 schoolbook first? No, Strassen is better.
    
    # Let's implement a quick Strassen builder for 2x4x8.
    # We can decompose 2x4 * 4x8 into 8 blocks of 2x2 * 2x2.
    # Each 2x2 * 2x2 has rank 7.
    # Total rank 56.
    
    def _build_strassen_2x2x2():
        U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, -1],
            [0, 1, 0, 1]
        ], dtype=float)
        V = np.array([
            [1, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, -1, 0],
            [0, -1, 0, 1],
            [0, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=float)
        W = np.array([
            [1, 0, 0, 1, -1, 0, 1],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [1, -1, 1, 0, 0, 1, 0]
        ], dtype=float)
        return U, V, W

    U_s, V_s, W_s = _build_strassen_2x2x2()
    
    # Build the full 56-rank decomposition
    rank_56 = 56
    U_56 = np.zeros((n*m, rank_56))
    V_56 = np.zeros((m*p, rank_56))
    W_56 = np.zeros((p*n, rank_56))
    
    # A is 2x4. Split into A_left (2x2) and A_right (2x2).
    # B is 4x8. Split into B_top (2x8) and B_bot (2x8).
    # B_top and B_bot are split into 4 blocks of 2x2 each.
    
    col_offset = 0
    # Left part: A_left (cols 0,1 of A) * B_top (rows 0,1 of B)
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            # U: A_left[i,j] -> index i*2+j
            for r in range(4):
                U_56[r, col] = U_s[f, r]
            # V: B_top[j, k] -> index j*8 + (2*block + r%2)
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = V_s[f, r]
            # W: C[i, k] -> index k*2 + i
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = W_s[r, f]
        col_offset += 7
        
    # Right part: A_right (cols 2,3 of A) * B_bot (rows 2,3 of B)
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            # U: A_right[i,j] -> index i*2 + 2 + j
            for r in range(4):
                U_56[2 + r, col] = U_s[f, r]
            # V: B_bot[j, k] -> index (2+j)*8 + (2*block + r%2)
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V_56[j * 8 + k, col] = V_s[f, r]
            # W: C[i, k] -> index k*2 + i
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W_56[k * 2 + i, col] = W_s[r, f]
        col_offset += 7
    
    # Verify Rank 56
    if not _verify_fast(U_56.tolist(), V_56.tolist(), W_56.tolist(), n, m, p, trials=5, tol=1e-9):
        # Fallback to schoolbook if Strassen fails (shouldn't happen)
        schoolbook_rank = n * m * p
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
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    best_U = U_56.tolist()
    best_V = V_56.tolist()
    best_W = W_56.tolist()
    best_rank = 56
    
    # 2. Try to reduce rank using optimization
    # We will try ranks from 55 down to 30.
    # For each rank r, we initialize U, V with the first r columns of the 56-rank solution.
    
    # Flatten U_56, V_56 for initialization
    U_56_flat = U_56.flatten()
    V_56_flat = V_56.flatten()
    
    for r in range(55, 29, -1):
        if time.time() > deadline - 15:
            break
            
        # Initialize params with first r columns
        # U_56 is (n*m, 56). We want (n*m, r).
        U_init = U_56[:, :r].flatten()
        V_init = V_56[:, :r].flatten()
        params_init = np.concatenate([U_init, V_init])
        
        # Run optimization
        success, U_opt, V_opt, W_opt = _optimize_rank(n, m, p, r, params_init, start_time, deadline)
        
        if success:
            best_U = U_opt
            best_V = V_opt
            best_W = W_opt
            best_rank = r
            # We can break here if we find a very low rank, or continue to try lower.
            # Let's continue to try to go lower, but be careful with time.
            # If we found rank 40, maybe we can find 30.
            # But optimization gets harder as we go lower.
            # Let's just try a few more steps.
    
    return {"U": best_U, "V": best_V, "W": best_W}

# Keep the helper functions from the seed program if needed, but we defined our own.
# The task requires keeping function names and signatures of the seed program.
# The seed program had:
# _verify_fast
# _build_strassen_2x2x2
# _build_block_decomposition
# _try_linear_reduction
# construct

# We should probably include the original _build_block_decomposition and _try_linear_reduction
# as no-ops or simple wrappers to satisfy any potential interface checks, 
# although the prompt says "Keep the function names and signatures of the seed program".
# It usually means `construct` must exist. But to be safe, I will include the other functions
# as they were in the seed, but my `construct` will use the new strategy.

def _build_strassen_2x2x2():
    """Standard Strassen decomposition for 2x2 * 2x2 -> 2x2, rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, 1]
    ], dtype=float)
    V = np.array([
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, -1, 0, 1],
        [0, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U.tolist(), V.tolist(), W.tolist()

def _build_block_decomposition(n, m, p):
    """
    Build a block-Strassen decomposition.
    """
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    U_s, V_s, W_s = _build_strassen_2x2x2()
    U_s = np.array(U_s)
    V_s = np.array(V_s)
    W_s = np.array(W_s)
    
    col_offset = 0
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[r][col] = U_s[f][r]
            for r in range(4):
                j = r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    for block in range(4):
        for f in range(7):
            col = col_offset + f
            for r in range(4):
                U[2 + r][col] = U_s[f][r]
            for r in range(4):
                j = 2 + r // 2
                k = 2 * block + r % 2
                V[j * 8 + k][col] = V_s[f][r]
            for r in range(4):
                i = r // 2
                k = 2 * block + r % 2
                W[k * 2 + i][col] = W_s[r][f]
        col_offset += 7
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_linear_reduction(U, V, W, n, m, p, start_time, deadline):
    """
    Placeholder for the old reduction method.
    We are using the new optimization strategy in construct().
    """
    return U, V, W
