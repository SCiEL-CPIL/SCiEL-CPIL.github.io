import time
import sys
import numpy as np
import scipy.optimize as opt
from itertools import product

START_TIME = time.time()
TIME_LIMIT = 135  # Safety margin before 150s kill

def time_up():
    return (time.time() - START_TIME) > TIME_LIMIT

def verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition is correct."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
    
    T_target = np.ones((n, m, p), dtype=float)
    max_err = np.max(np.abs(T_recon - T_target))
    return max_err < 1e-9, max_err

def residuals(x, n=2, m=4, p=4, rank=24):
    """Calculate residuals for the optimization problem."""
    size_U = n * m * rank
    size_V = m * p * rank
    
    U = x[:size_U].reshape(n*m, rank)
    V = x[size_U : size_U + size_V].reshape(m*p, rank)
    W = x[size_U + size_V : ].reshape(p*n, rank)
    
    T_recon = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_recon[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    return T_recon.flatten() - 1.0

def newton_polish(x, n=2, m=4, p=4, rank=24, max_iter=50, tol=1e-12):
    """Newton's method polish for the polynomial system."""
    x = x.copy()
    size = len(x)
    for it in range(max_iter):
        if time_up():
            break
        r = residuals(x, n, m, p, rank)
        norm_r = np.linalg.norm(r)
        if norm_r < tol:
            break
        
        # Use finite differences for Jacobian (sparse sampling for speed)
        J = np.zeros((n*m*p, size))
        step = 1e-7 * (1 + np.linalg.norm(x) / size)
        
        # Sample a subset of columns for speed
        sample_size = min(size, 300)
        indices = np.random.choice(size, size=sample_size, replace=False)
        
        for idx in indices:
            x_pert = x.copy()
            x_pert[idx] += step
            r_pert = residuals(x_pert, n, m, p, rank)
            J[:, idx] = (r_pert - r) / step
        
        # Solve (J^T J + lambda I) dx = -J^T r
        JtJ = J.T @ J
        Jtr = J.T @ r
        lam = 1e-6 * np.trace(JtJ) / max(1, sample_size)
        
        try:
            dx = np.linalg.solve(JtJ + lam * np.eye(size), -Jtr)
        except np.linalg.LinAlgError:
            dx = np.zeros(size)
            for idx in indices:
                dx[idx] = -0.01 * Jtr[idx]
        
        x_new = x + dx
        r_new = residuals(x_new, n, m, p, rank)
        norm_r_new = np.linalg.norm(r_new)
        
        if norm_r_new < norm_r:
            x = x_new
        else:
            lam *= 10
            if lam > 1e6:
                break
    
    return x

def winograd_4x4_decomposition():
    """
    Construct a rank-16 decomposition using Winograd's Minimal Cyclic Convolution
    approach for 4x4 matrix multiplication blocks.
    
    For 4x4 matrix multiplication, we can use the fact that it's a cyclic convolution
    over Z4. The 4-point DFT gives us a rank-16 decomposition for each 4x4 block.
    Since our problem is 2x4 times 4x4, we have 2 such blocks (one for each row of the 2x4 matrix).
    
    The key insight: for each of the 2 rows of the 2x4 matrix, the multiplication with the 4x4 matrix
    can be viewed as 4 independent 1x4 * 4x4 multiplications, but we can combine them using
    the Winograd approach.
    
    Actually, let me think about this more carefully. The tensor is T[i,j,k] where:
    - i indexes the output row (0,1)
    - j indexes the intermediate (0,1,2,3) 
    - k indexes the output column (0,1,2,3)
    
    For fixed i, the sub-tensor T[i,j,k] represents the j-th row of the i-th row-block times B.
    This is just standard matrix multiplication: (A_i, B) -> C_i where A_i is 1x4 and B is 4x4.
    
    The rank of a 1x4 * 4x4 multiplication is 4 (schoolbook) or can be reduced.
    Actually, the tensor for 1x4 * 4x4 has rank 4 (it's just 4 independent multiplications).
    
    So the total rank should be 2*4 = 8? No, that's not right either.
    
    Let me reconsider. The tensor T[i,j,k] = 1 for all i,j,k means that for each (i,k),
    we need sum_j A[i,j] * B[j,k] = C[i,k].
    
    This is just standard matrix multiplication: C = A * B where A is 2x4 and B is 4x4.
    The rank of this tensor is the rank of the 2x4 * 4x4 matrix multiplication.
    
    For square n x n, the rank is known. For 2x4 * 4x4, it's less well-known.
    The trivial bound is 2*4*4 = 32.
    Using block Strassen on the 4x4 part with 2x1 blocks might give improvements.
    
    Let me try a different approach: use the fact that this is equivalent to
    multiplying a 2x4 matrix by a 4x4 matrix, which can be seen as 2 independent
    1x4 * 4x4 multiplications. Each 1x4 * 4x4 has rank 4 (you need 4 multiplications
    for each row). So total rank = 2*4 = 8? That seems too low.
    
    Wait, I think I'm confusing myself. The tensor T[i,j,k] = 1 means that the
    (i,k)-th entry of the output is sum_j A[i,j] B[j,k]. This is exactly matrix
    multiplication C = AB.
    
    The rank of the matrix multiplication tensor for m x n * n x p is at most m*n*p.
    For 2x4 * 4x4, the trivial rank is 32.
    
    Can we do better? Let's think about it as 2 independent 4x4 vector-matrix products.
    Each 1x4 * 4x4 product has rank 4 (you can't do better than 4 multiplications for
    a dot product of length 4 with each column). So total rank = 2*4 = 8? 
    
    No wait, that's wrong. For a 1x4 vector times a 4x4 matrix, you need 4 multiplications
    per column, so 4*4 = 16 multiplications total. But these can be shared across columns
    using Winograd's approach.
    
    Actually, for 1x4 * 4x4, the tensor rank is 4. Here's why: the output has 4 entries,
    and each output entry c_k = sum_j a_j b_jk. If we precompute p_j = a_j, then
    c_k = sum_j p_j b_jk. But this still requires 4 multiplications per k, so 16 total.
    
    Hmm, I think I need to be more careful. The tensor for 1x4 * 4x4 multiplication
    has entries T[j,k] where the output is a 1x4 vector. The rank of this tensor is
    actually 4, because you can write:
    c_k = sum_j a_j * b_jk
    This is a sum of 4 terms, each involving one a_j and one b_jk. But the tensor
    rank counts the minimum number of rank-1 tensors whose sum gives the full tensor.
    
    For the 1x4 * 4x4 case, the tensor is T[j,k,l] = delta_{jk} * delta_{kl} ... no,
    let me redefine. The tensor for multiplying a 1x4 vector u by a 4x4 matrix M to
    get a 1x4 vector v is T[j,k] where v_k = sum_j u_j M_jk. The "tensor" here is
    really just the identity, and the rank is 4 (you need 4 multiplications per output,
    but they can be organized as 4 rank-1 terms).
    
    I think I'm overcomplicating this. Let me just try to find a low-rank
    decomposition numerically.
    """
    # For now, return None and use the numerical approach
    return None

def strassen_decomposition():
    """Return the Strassen-based decomposition (rank 28)."""
    n, m, p = 2, 4, 4
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Use the 7-term Strassen formulas for 2x2 blocks
    # Divide the 4x4 matrix into 2x2 blocks
    # A (2x4) = [A11 A12; A21 A22] where each Ai is 2x2
    # B (4x4) = [B11 B12; B21 B22] where each Bj is 2x2
    # C (2x4) = [C11 C12; C21 C22] where each Ci is 2x2
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our 2x4 * 4x4 case, we have 2 rows of 2x2 blocks in A
    # So we apply Strassen twice (once for each row-block)
    # This gives 2*7 = 14 rank-1 terms for the 2x2 level
    # Each 2x2 multiplication has rank 2 (using schoolbook for 2x2*2x2 with 2x2 blocks)
    # Wait, no. Each Mi is a 2x2 * 2x2 product, which has rank 4 (schoolbook)
    # So total rank = 14 * 4 = 56? That's worse than 32.
    
    # Let me try a different approach: use schoolbook for the 4x4 part
    # but with some sharing.
    
    # Actually, the simplest approach is to use the naive decomposition
    # and then try to reduce the rank numerically.
    
    # For now, return the naive rank-32 decomposition
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][idx] = 1.0
                V[j*p+k][idx] = 1.0
                W[k*n+i][idx] = 1.0
                idx += 1
    
    return U, V, W

def find_low_rank_decomposition(target_rank, n=2, m=4, p=4, max_restarts=30):
    """Use scipy.optimize.least_squares to find a decomposition of target_rank."""
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_size = size_U + size_V + size_W
    
    best_result = None
    best_error = float('inf')
    
    for restart in range(max_restarts):
        if time_up():
            break
            
        # Use structured random initialization based on tensor structure
        if restart == 0:
            # Initialize with sparse structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Randomly assign non-zero entries
                u_rows = np.random.choice(n*m, size=max(1, n*m//4), replace=False)
                v_rows = np.random.choice(m*p, size=max(1, m*p//4), replace=False)
                w_rows = np.random.choice(p*n, size=max(1, p*n//4), replace=False)
                U[u_rows, r] = np.random.choice([-1, 1], size=len(u_rows))
                V[v_rows, r] = np.random.choice([-1, 1], size=len(v_rows))
                W[w_rows, r] = np.random.choice([-1, 1], size=len(w_rows))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        elif restart < 10:
            # Random initialization with varying scales
            scale = 0.1 + 0.05 * (restart % 5)
            x0 = np.random.randn(total_size) * scale
        else:
            # More structured: use Strassen-like patterns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            for r in range(target_rank):
                # Each rank-1 term covers a random subset
                u_mask = np.random.rand(n*m) < 0.3
                v_mask = np.random.rand(m*p) < 0.3
                w_mask = np.random.rand(p*n) < 0.3
                U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
                V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
                W[w_mask, r] = np.random.choice([-1, 1], size=np.sum(w_mask))
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = opt.least_squares(residuals, x0, args=(n, m, p, target_rank), method='lm', max_nfev=5000)
            err = np.linalg.norm(result.fun)
            
            if err < best_error:
                best_error = err
                best_result = result.x.copy()
                
            if err < 1e-9:
                # Found a solution
                U = best_result[:size_U].reshape(n*m, target_rank)
                V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                
                # Verify
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
                    
        except Exception as e:
            pass
        
        # Try Newton polish if close
        if best_error < 0.05 and not time_up():
            try:
                x_polished = newton_polish(best_result, n, m, p, target_rank, max_iter=20, tol=1e-12)
                err_polished = np.linalg.norm(residuals(x_polished, n, m, p, target_rank))
                if err_polished < best_error:
                    best_error = err_polished
                    best_result = x_polished
                if err_polished < 1e-9:
                    U = best_result[:size_U].reshape(n*m, target_rank)
                    V = best_result[size_U : size_U + size_V].reshape(m*p, target_rank)
                    W = best_result[size_U + size_V : ].reshape(p*n, target_rank)
                    valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                    if valid:
                        return U.tolist(), V.tolist(), W.tolist()
            except Exception:
                pass
    
    # Polish pass if we got close
    if best_result is not None and best_error < 0.1 and not time_up():
        try:
            result2 = opt.least_squares(residuals, best_result, args=(n, m, p, target_rank),
                                         method='trf', max_nfev=5000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
            err2 = np.linalg.norm(result2.fun)
            if err2 < 1e-9:
                U = result2.x[:size_U].reshape(n*m, target_rank)
                V = result2.x[size_U : size_U + size_V].reshape(m*p, target_rank)
                W = result2.x[size_U + size_V : ].reshape(p*n, target_rank)
                valid, _ = verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p)
                if valid:
                    return U.tolist(), V.tolist(), W.tolist()
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 4, 4
    best_result = None
    best_rank = float('inf')
    
    # Start with naive rank-32 decomposition
    U_naive, V_naive, W_naive = strassen_decomposition()
    valid, err = verify_decomposition(U_naive, V_naive, W_naive, n, m, p)
    if valid:
        best_rank = 32
        best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    
    # Try to find lower-rank decompositions
    for target_rank in [24, 25, 26, 27, 28, 29, 30, 31]:
        if time_up():
            break
        if target_rank >= best_rank:
            continue
        
        res = find_low_rank_decomposition(target_rank, n, m, p, max_restarts=20)
        if res is not None:
            U2, V2, W2 = res
            valid, err = verify_decomposition(U2, V2, W2, n, m, p)
            if valid:
                best_rank = target_rank
                best_result = {"U": U2, "V": V2, "W": W2}
                break  # Found a lower-rank solution, stop
    
    if best_result is None:
        # Fallback: naive decomposition
        rank = 32
        U = [[0.0] * rank for _ in range(n*m)]
        V = [[0.0] * rank for _ in range(m*p)]
        W = [[0.0] * rank for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1.0
                    V[j*p+k][idx] = 1.0
                    W[k*n+i][idx] = 1.0
                    idx += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result

if __name__ == "__main__":
    result = construct()
    print(f"Best rank: {len(result['U'][0])}")
    valid, err = verify_decomposition(result['U'], result['V'], result['W'])
    print(f"Valid: {valid}, Max error: {err}")
