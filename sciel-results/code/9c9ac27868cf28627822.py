"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Reconstruct tensor
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U_arr[:, r], V_arr[:, r], W_arr[:, r])
    
    T_true = _build_tensor(n, m, p)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(arr, tol=1e-9):
    """Check if all entries are integer or half-integer."""
    for val in np.array(arr).flat:
        v = float(val)
        if abs(2 * v - round(2 * v)) > tol:
            return False
    return True

def _to_list(arr):
    """Convert numpy array to list of lists with rounded half-integer values."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = float(val)
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _try_structured_w_search(n, m, p, time_limit=120.0):
    """
    Try to find a low-rank decomposition by using structured W matrices.
    
    Key insight: The tensor T has structure where T[i*m+j, j*p+k, k*n+i] = 1.
    This means the non-zero entries are organized by (i,j,k) triples.
    
    Strategy: Generate W matrices that respect the k-index structure.
    For each candidate W, solve for U and V and check if the solution works.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    support = np.argwhere(T != 0)
    support_set = set(map(tuple, support))
    
    # Compute matricization ranks for lower bounds
    T_mat1 = T.reshape(n * m, m * p * p * n)
    T_mat2 = T.reshape(n * m * m * p, p * n)
    T_mat3 = T.reshape(n * m * p * n, m * p)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(42)
    
    # Try ranks from lower bound up to schoolbook - 1
    for r in range(max(rank_lower, 20), n * m * p):
        if time.time() - start_time > time_limit:
            break
        
        # Generate structured W matrices
        for w_type in range(8):
            if time.time() - start_time > time_limit:
                break
            
            for attempt in range(30):
                if time.time() - start_time > time_limit:
                    break
                
                W_rand = np.zeros((p * n, r))
                
                if w_type == 0:
                    # Random {-1, 0, 1}
                    W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
                elif w_type == 1:
                    # Sparse: 2 non-zeros per row
                    W_rand = np.zeros((p * n, r))
                    for i in range(p * n):
                        cols = np.random.choice(r, size=2, replace=False)
                        W_rand[i, cols] = np.random.choice([-1, 1], size=2)
                elif w_type == 2:
                    # Block by k: each k-block gets a subset of columns
                    W_rand = np.zeros((p * n, r))
                    cols_per_k = r // p
                    for k in range(p):
                        start_col = k * cols_per_k
                        end_col = start_col + cols_per_k
                        for i in range(n):
                            row = k * n + i
                            if start_col < r:
                                W_rand[row, start_col:start_col+1] = 1
                elif w_type == 3:
                    # More zeros
                    W_rand = np.random.choice([-1, 0, 0, 0, 1], size=(p * n, r))
                elif w_type == 4:
                    # Gaussian threshold
                    W_rand = np.random.randn(p * n, r)
                    W_rand[W_rand < -0.5] = -1
                    W_rand[W_rand > 0.5] = 1
                    W_rand[(W_rand >= -0.5) & (W_rand <= 0.5)] = 0
                elif w_type == 5:
                    # Identity-like: each row has one 1
                    W_rand = np.zeros((p * n, r))
                    for i in range(p * n):
                        col = i % r
                        W_rand[i, col] = 1
                elif w_type == 6:
                    # Random with more structure: group by i
                    W_rand = np.zeros((p * n, r))
                    for k in range(p):
                        for i in range(n):
                            row = k * n + i
                            col = (k + i * p) % r
                            W_rand[row, col] = 1
                else:
                    # Random binary
                    W_rand = np.random.choice([0, 1], size=(p * n, r))
                
                if np.linalg.matrix_rank(W_rand) < r:
                    continue
                
                try:
                    WtW = W_rand.T @ W_rand
                    WtW_inv = np.linalg.inv(WtW)
                except np.linalg.LinAlgError:
                    continue
                
                # Compute X_vec
                TW = np.einsum('abc,cr->abr', T, W_rand)
                X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
                
                U_sol = np.zeros((n * m, r))
                V_sol = np.zeros((m * p, r))
                all_rank1 = True
                
                for r_idx in range(r):
                    C_r = X_vec[:, :, r_idx]
                    svals = np.linalg.svd(C_r, compute_uv=False)
                    if len(svals) > 1:
                        if svals[1] > 1e-10 * max(svals[0], 1e-12):
                            all_rank1 = False
                            break
                    if svals[0] < 1e-12:
                        U_sol[:, r_idx] = 0
                        V_sol[:, r_idx] = 0
                    else:
                        U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                        U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                        V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
                
                if not all_rank1:
                    continue
                
                # Verify on support
                T_recon_support = np.zeros(len(support))
                for idx, (a, b, c) in enumerate(support):
                    val = 0.0
                    for r_idx in range(r):
                        val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
                    T_recon_support[idx] = val
                
                if not np.allclose(T_recon_support, 1.0, atol=1e-8):
                    continue
                
                # Check half-integer constraint
                if _check_half_integer(U_sol) and _check_half_integer(V_sol) and _check_half_integer(W_rand):
                    best_rank = r
                    best_U = U_sol.copy()
                    best_V = V_sol.copy()
                    best_W = W_rand.copy()
                    break
                else:
                    # Try rounding to half-integers
                    U_round = np.round(U_sol * 2) / 2
                    V_round = np.round(V_sol * 2) / 2
                    W_round = np.round(W_rand * 2) / 2
                    
                    # Verify rounded version
                    T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
                    for r_idx in range(r):
                        T_recon += np.einsum('i,j,k->ijk', U_round[:, r_idx], V_round[:, r_idx], W_round[:, r_idx])
                    
                    if np.allclose(T_recon, T, atol=1e-8):
                        best_rank = r
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        break
        
        if best_rank == r:
            continue
    
    if best_U is not None:
        return _to_list(best_U), _to_list(best_V), _to_list(best_W)
    
    return None

def _try_polynomial_system_solve(n, m, p, time_limit=120.0):
    """
    Formulate the rank minimization as a system of polynomial equations and solve
    using Newton's method on the polynomial system.
    
    For a given rank r, we have variables U (n*m*r), V (m*p*r), W (p*n*r).
    The constraints are:
    For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    For each (a,b,c) not in support: sum_r U[a, r] * V[b, r] * W[c, r] = 0
    
    This is a system of polynomial equations. We can use Newton's method
    with random initializations to find solutions.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    support = np.argwhere(T != 0)
    
    # Compute lower bound
    T_mat1 = T.reshape(n * m, m * p * p * n)
    T_mat2 = T.reshape(n * m * m * p, p * n)
    T_mat3 = T.reshape(n * m * p * n, m * p)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    np.random.seed(123)
    
    # Try ranks from lower bound
    for r in range(max(rank_lower, 15), min(n * m * p, 40)):
        if time.time() - start_time > time_limit:
            break
        
        for attempt in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Newton's method iterations
            converged = False
            for it in range(20):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute residuals
                residuals = np.zeros((len(support) + 100,))
                
                # For support points, residual = 1 - sum
                for idx, (a, b, c) in enumerate(support):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    residuals[idx] = 1.0 - val
                
                # For some non-support points, residual = sum (should be 0)
                non_support_count = min(100, (n*m)*(m*p)*(p*n) - len(support))
                np.random.seed(attempt * 1000 + it)
                for idx in range(non_support_count):
                    a = np.random.randint(0, n * m)
                    b = np.random.randint(0, m * p)
                    c = np.random.randint(0, p * n)
                    if (a, b, c) in set(map(tuple, support)):
                        continue
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    residuals[len(support) + idx] = val
                
                # Check convergence
                max_residual = np.max(np.abs(residuals))
                if max_residual < 1e-10:
                    converged = True
                    break
                
                # Compute Jacobian (simplified - use finite differences for speed)
                # This is expensive, so we use a simpler approach:
                # Just do gradient descent on the squared residuals
                
                # Gradient with respect to U
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                # For support points
                for idx, (a, b, c) in enumerate(support):
                    res = residuals[idx]
                    dU[a, :] += res * V[b, :] * W[c, :]
                    dV[b, :] += res * U[a, :] * W[c, :]
                    dW[c, :] += res * U[a, :] * V[b, :]
                
                # For non-support points
                for idx in range(non_support_count):
                    a = np.random.randint(0, n * m)
                    b = np.random.randint(0, m * p)
                    c = np.random.randint(0, p * n)
                    if (a, b, c) in set(map(tuple, support)):
                        continue
                    res = residuals[len(support) + idx]
                    dU[a, :] += res * V[b, :] * W[c, :]
                    dV[b, :] += res * U[a, :] * W[c, :]
                    dW[c, :] += res * U[a, :] * V[b, :]
                
                # Update with small step
                step = 0.01
                U -= step * dU
                V -= step * dV
                W -= step * dW
            
            if not converged:
                continue
            
            # Verify the solution
            T_recon = np.zeros((n * m, m * p, p * n), dtype=np.float64)
            for r_idx in range(r):
                T_recon += np.einsum('i,j,k->ijk', U[:, r_idx], V[:, r_idx], W[:, r_idx])
            
            if not np.allclose(T_recon, T, atol=1e-8):
                continue
            
            # Check half-integer constraint
            if _check_half_integer(U) and _check_half_integer(V) and _check_half_integer(W):
                best_rank = r
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                break
            else:
                # Try rounding
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                
                T_recon2 = np.zeros((n * m, m * p, p * n), dtype=np.float64)
                for r_idx in range(r):
                    T_recon2 += np.einsum('i,j,k->ijk', U_round[:, r_idx], V_round[:, r_idx], W_round[:, r_idx])
                
                if np.allclose(T_recon2, T, atol=1e-8):
                    best_rank = r
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    break
        
        if best_rank == r:
            continue
    
    if best_U is not None:
        return _to_list(best_U), _to_list(best_V), _to_list(best_W)
    
    return None

def _try_direct_construction(n, m, p, time_limit=120.0):
    """
    Try to directly construct a decomposition by exploiting the tensor structure.
    
    For the 2x4x4x7 case, we can try to group the products.
    A is 2x4, B is 4x7.
    
    Idea: Split A into two 2x2 blocks and B into two 2x7 blocks.
    A = [A1 A2], B = [B1; B2]
    Result = A1*B1 + A2*B2
    
    Each 2x2 * 2x7 product can be done with rank 7 (Strassen-like).
    So total rank could be 14 if we can make the two parts independent.
    
    But we need to check if this works with the half-integer constraint.
    """
    start_time = time.time()
    
    # For now, just return schoolbook if we can't find better quickly
    # This is a placeholder for a more sophisticated approach
    return None

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_result = None
    best_rank = n * m * p
    
    # Try structured W search
    try:
        remaining = deadline - time.time()
        if remaining > 10.0:
            result = _try_structured_w_search(n, m, p, time_limit=min(remaining, 80.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    rank = len(U[0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
    except:
        pass
    
    # Try polynomial system solve
    try:
        remaining = deadline - time.time()
        if remaining > 10.0:
            result = _try_polynomial_system_solve(n, m, p, time_limit=min(remaining, 50.0))
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    rank = len(U[0])
                    if rank < best_rank:
                        best_rank = rank
                        best_result = (U, V, W)
    except:
        pass
    
    # Return best result or fall back to schoolbook
    if best_result is not None:
        U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
