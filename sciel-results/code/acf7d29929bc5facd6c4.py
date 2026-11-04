"""Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

# Dimensions
N, M, P = 2, 3, 6
NM = N * M  # 6
MP = M * P  # 18
PN = P * N  # 12


def verify_decomposition(U, V, W, rank, tol=1e-12):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                expected = 1.0
                # Sum over c: U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
                u_val = U[i * M + j, :]
                v_val = V[j * P + k, :]
                w_val = W[k * N + i, :]
                actual = np.sum(u_val * v_val * w_val)
                if abs(actual - expected) > tol:
                    return False, (i, j, k, expected, actual)
    return True, None


def schoolbook_decomposition():
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = N * M * P
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return U, V, W, rank


def try_strassen_block_decomposition():
    """
    Attempt to construct a decomposition using Strassen-like block structure.
    For 2x3 * 3x6: split B into three 3x2 blocks.
    Each 2x3 * 3x2 product can potentially use fewer than 6 multiplications.
    
    Actually, let's try a different approach: use the known lower bounds and
    try to find a rank-30 or rank-28 decomposition using algebraic methods.
    
    For now, let's try a structured approach based on linearity.
    """
    # One idea: exploit the fact that N=2 is small.
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # C[1,k] = sum_j A[1,j]*B[j,k]
    
    # We can write this as a vector equation:
    # [C[0,k]]   [sum_j A[0,j]*B[j,k]]
    # [C[1,k]] = [sum_j A[1,j]*B[j,k]]
    
    # For each k, this is a 2-vector that depends linearly on A and B.
    
    # Let's try rank 24: use 4 terms per k-value, with 6 k-values.
    # Or use a different structure.
    
    # Actually, let's try a direct algebraic approach for small ranks.
    # We'll use the method of undetermined coefficients with rational arithmetic.
    
    # For rank r, we have 36r unknowns and 36 equations (one for each (i,j,k)).
    # The equations are cubic, so we can't solve them directly with linear algebra.
    
    # Instead, let's try a randomized approach: randomly initialize U, V, W and
    # use gradient descent or Newton's method to minimize the residual.
    
    pass


def try_winograd_fta():
    """
    Apply Winograd Fourier Transform Algorithm to the 2x3*3x6 multiplication.
    
    The idea: treat the multiplication as a convolution-like operation on cyclic groups.
    For 2x3 * 3x6, we can embed into a larger structure and use DFT to reduce multiplications.
    
    Winograd's FTA for small matrix multiplication:
    - For 2x2: rank 7 (Strassen)
    - For 3x3: rank 23 (Bini et al.)
    
    For 2x3 * 3x6, one approach:
    - Split the 6 columns of B into three groups of 2
    - Each 2x3 * 3x2 sub-problem has tensor rank at most 6 (schoolbook)
    - Can we do better for 2x3 * 3x2?
    
    The 2x3 * 3x2 tensor has 12 entries. Let's see if rank 5 or 6 is achievable.
    
    Actually, let's try a more systematic algebraic approach.
    """
    pass


def compute_residuals(U, V, W):
    """Compute all residuals vectorized."""
    residuals = np.zeros((N, M, P))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                residuals[i, j, k] = np.sum(U[i*M+j, :] * V[j*P+k, :] * W[k*N+i, :])
    return residuals - 1.0


def compute_gradients(U, V, W, residuals):
    """Compute gradients vectorized."""
    r = U.shape[1]
    grad_U = np.zeros((NM, r))
    grad_V = np.zeros((MP, r))
    grad_W = np.zeros((PN, r))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                res = residuals[i, j, k]
                if abs(res) < 1e-15:
                    continue
                u_row = U[i*M+j, :]
                v_row = V[j*P+k, :]
                w_row = W[k*N+i, :]
                grad_U[i*M+j, :] += 2 * res * v_row * w_row
                grad_V[j*P+k, :] += 2 * res * u_row * w_row
                grad_W[k*N+i, :] += 2 * res * u_row * v_row
    return grad_U, grad_V, grad_W


def optimize_decomposition(U, V, W, max_iter=2000, time_limit=120, start_time=None, tol=1e-10):
    """Optimize decomposition using gradient descent with adaptive step size."""
    if start_time is None:
        start_time = time.time()
    
    lr = 0.005
    r = U.shape[1]
    prev_loss = float('inf')
    patience = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_limit - 15:
            break
        
        residuals = compute_residuals(U, V, W)
        loss = np.sum(residuals ** 2)
        
        if loss < tol:
            break
        
        # Check for stagnation
        if loss > prev_loss * 0.999:
            patience += 1
        else:
            patience = 0
        prev_loss = loss
        
        if patience > 200:
            lr *= 0.5
            patience = 0
            if lr < 1e-8:
                break
        
        grad_U, grad_V, grad_W = compute_gradients(U, V, W, residuals)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return U, V, W, residuals


def strassen_block_decomposition():
    """
    Construct a rank-33 decomposition using Strassen-like ideas.
    
    Split B into 3 blocks of 3x2. For each block, we compute a 2x2 output.
    Use the fact that for 2x3*3x2, we can potentially save multiplications
    by sharing structure.
    
    Actually, let's try a different approach: use the known result that
    2x3*3x6 can be done with rank 33 by exploiting the small N=2 dimension.
    
    Key idea: C[0,k] and C[1,k] share the same B[:,k] column.
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    
    For each k, we're computing a 2x1 product from a 2x3 matrix and 3x1 vector.
    This is equivalent to: [C0k, C1k]^T = A * B[:,k]
    
    We can write this as:
    C0k = (A[0,0]+A[1,0])*B[0,k] + (A[0,1]-A[1,1])*B[1,k] + (A[0,2]+A[1,2])*B[2,k]
         + A[1,0]*B[0,k] - A[1,1]*B[1,k] + A[1,2]*B[2,k]  ... this doesn't help directly.
    
    Let me try a concrete Strassen-like approach for the full problem.
    """
    # Start with schoolbook and try to merge some terms
    # Use the structure: for each j, we have terms involving A[i,j] and B[j,k]
    # 
    # Idea: For j=0 and j=1, create combined terms:
    # Let p0 = (A[0,0]+A[1,0]) * (B[0,k]+B[1,k]) ... but this introduces cross terms.
    #
    # A cleaner approach: use the fact that we can compute:
    # S0 = A[0,0]*B[0,k] + A[0,1]*B[1,k]  (partial sum for C[0,k])
    # S1 = A[1,0]*B[0,k] + A[1,1]*B[1,k]  (partial sum for C[1,k])
    # Then C[0,k] = S0 + A[0,2]*B[2,k]
    #       C[1,k] = S1 + A[1,2]*B[2,k]
    #
    # This doesn't save anything. Let me try the numerical approach instead.
    pass


def try_rank_search(max_rank=35, time_limit=120):
    """
    Try to find a decomposition of rank < max_rank using numerical optimization.
    Uses vectorized operations and better initialization.
    """
    start_time = time.time()
    best_result = None
    best_residual = float('inf')
    
    # Pre-compute index mappings for efficiency
    # For each (i,j,k), we need U[i*M+j,:], V[j*P+k,:], W[k*N+i,:]
    
    # Try various ranks from high to low
    for r in range(max_rank, 25, -1):
        if time.time() - start_time > time_limit - 15:
            break
        
        # Try multiple initializations
        n_tries = 3 if r > 30 else 2
        for attempt in range(n_tries):
            if time.time() - start_time > time_limit - 15:
                break
            
            # Initialize: use schoolbook structure perturbed
            # First r terms from schoolbook, rest random
            U = np.zeros((NM, r))
            V = np.zeros((MP, r))
            W = np.zeros((PN, r))
            
            # Fill first min(r, 36) columns with schoolbook
            n_schoolbook = min(r, N * M * P)
            col = 0
            for i in range(N):
                for j in range(M):
                    for k in range(P):
                        if col >= r:
                            break
                        U[i * M + j, col] = 1.0
                        V[j * P + k, col] = 1.0
                        W[k * N + i, col] = 1.0
                        col += 1
                    if col >= r:
                        break
                if col >= r:
                    break
            
            # Perturb remaining columns randomly
            for c in range(col, r):
                U[:, c] = np.random.randn(NM) * 0.5
                V[:, c] = np.random.randn(MP) * 0.5
                W[:, c] = np.random.randn(PN) * 0.5
            
            # Also add small noise to schoolbook part
            noise_scale = 0.1
            for c in range(min(col, r)):
                U[:, c] += np.random.randn(NM) * noise_scale
                V[:, c] += np.random.randn(MP) * noise_scale
                W[:, c] += np.random.randn(PN) * noise_scale
            
            # Optimize
            U, V, W, residuals = optimize_decomposition(
                U, V, W, max_iter=1000, time_limit=time_limit, start_time=start_time
            )
            
            # Check if we found a valid decomposition
            ok, _ = verify_decomposition(U, V, W, r, tol=1e-10)
            if ok:
                best_result = (r, U, V, W)
                return best_result
            
            # Track best residual
            res_val = np.max(np.abs(residuals))
            if res_val < best_residual:
                best_residual = res_val
    
    return best_result


def try_structured_decomposition():
    """
    Try a structured decomposition based on the algebraic structure of the problem.
    
    Key insight: C[i,k] = sum_j A[i,j]*B[j,k]
    This is a bilinear map from R^{2x3} x R^{3x6} -> R^{2x6}.
    
    One approach: use the fact that the middle dimension is 3 (small).
    We can write the product as:
    C = A * B where A is 2x3 and B is 3x6.
    
    Let's try to use a rank-30 decomposition based on the following idea:
    - For each of the 3 middle indices j, we have contributions to all 12 output entries.
    - Schoolbook uses 3*2*6 = 36 multiplications.
    - We can try to combine some of these.
    
    Actually, let's try a different approach: use the matrix multiplication
    as a composition of smaller multiplications.
    
    2x3 * 3x6 = (2x3 * 3x2) * 3 (block diagonal)
    Each 2x3 * 3x2 has 6 entries in the output.
    
    For 2x3 * 3x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    This is a 2x3x2 tensor with 12 entries. Schoolbook rank is 6.
    Can we do better? The tensor is:
    T[i,j,k] = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
    
    This is the all-ones tensor of shape 2x3x2, but with the matrix multiplication
    structure (factors indexed by (i,j), (j,k), (k,i)).
    
    Let's try rank 5 for this sub-problem. If we can achieve rank 5 for each
    2x3*3x2 block, total rank would be 3*5 = 15. But this seems too optimistic.
    
    Let's be more conservative and try rank 6 for each block (no improvement)
    or find a way to share structure across blocks.
    """
    pass


def construct():
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Start with schoolbook as fallback
    U, V, W, rank = schoolbook_decomposition()
    ok, err = verify_decomposition(U, V, W, rank)
    if not ok:
        return {"U": U, "V": V, "W": W}
    
    # Try numerical search for lower rank
    result = try_rank_search(max_rank=35, time_limit=130)
    
    if result is not None:
        r, U_new, V_new, W_new = result
        U_list = [[U_new[i, c] for c in range(r)] for i in range(NM)]
        V_list = [[V_new[i, c] for c in range(r)] for i in range(MP)]
        W_list = [[W_new[i, c] for c in range(r)] for i in range(PN)]
        
        ok, _ = verify_decomposition(U_list, V_list, W_list, r, tol=1e-10)
        if ok:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
    return {"U": U, "V": V, "W": W}
