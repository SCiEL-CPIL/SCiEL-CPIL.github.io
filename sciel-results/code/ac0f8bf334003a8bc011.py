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


def try_rank_search(max_rank=35, time_limit=120):
    """
    Try to find a decomposition of rank < max_rank using numerical optimization.
    We minimize the sum of squared residuals:
    sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    """
    start_time = time.time()
    best_result = None
    best_residual = float('inf')
    
    # Try various ranks from high to low
    for r in range(max_rank, 17, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try multiple random initializations
        n_tries = 5 if r > 25 else 3
        for attempt in range(n_tries):
            if time.time() - start_time > time_limit - 10:
                break
            
            np.random.seed(42 + r * 100 + attempt)
            
            # Initialize with random values
            U = np.random.randn(NM, r) * 0.1
            V = np.random.randn(MP, r) * 0.1
            W = np.random.randn(PN, r) * 0.1
            
            # Use iterative optimization
            lr = 0.01
            for iteration in range(500):
                if time.time() - start_time > time_limit - 10:
                    break
                
                # Compute residual and gradients
                residual = np.zeros((N, M, P))
                grad_U = np.zeros((NM, r))
                grad_V = np.zeros((MP, r))
                grad_W = np.zeros((PN, r))
                
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            # Residual for (i,j,k)
                            res = 0.0
                            for c in range(r):
                                res += U[i*M+j, c] * V[j*P+k, c] * W[k*N+i, c]
                            residual[i, j, k] = res - 1.0
                            
                            # Gradients
                            for c in range(r):
                                grad_U[i*M+j, c] += 2 * residual[i, j, k] * V[j*P+k, c] * W[k*N+i, c]
                                grad_V[j*P+k, c] += 2 * residual[i, j, k] * U[i*M+j, c] * W[k*N+i, c]
                                grad_W[k*N+i, c] += 2 * residual[i, j, k] * U[i*M+j, c] * V[j*P+k, c]
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Check convergence
                if np.max(np.abs(residual)) < 1e-10:
                    break
                
                # Adaptive learning rate
                if iteration % 50 == 0:
                    lr *= 0.95
            
            # Check if we found a valid decomposition
            ok, _ = verify_decomposition(U, V, W, r, tol=1e-10)
            if ok:
                best_result = (r, U, V, W)
                return best_result
            
            # Track best residual
            res_val = np.max(np.abs(residual))
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
    
    # Try to find a better decomposition
    # Strategy 1: Try structured algebraic approaches
    # Strategy 2: Try numerical optimization
    
    # First, let's try a few known structured approaches
    
    # Approach: Try rank 30 using a block structure
    # Split the 6 columns of B into 3 groups of 2
    # For each group, use a 2x3*3x2 algorithm
    
    # Let's try to manually construct a rank-30 decomposition
    # by combining Strassen-like ideas.
    
    # Actually, let's try the numerical search approach
    result = try_rank_search(max_rank=35, time_limit=120)
    
    if result is not None:
        r, U_new, V_new, W_new = result
        # Convert to list format
        U_list = [[U_new[i, c] for c in range(r)] for i in range(NM)]
        V_list = [[V_new[i, c] for c in range(r)] for i in range(MP)]
        W_list = [[W_new[i, c] for c in range(r)] for i in range(PN)]
        
        # Verify
        ok, _ = verify_decomposition(U_list, V_list, W_list, r, tol=1e-10)
        if ok:
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no better decomposition found, return schoolbook
    return {"U": U, "V": V, "W": W}
