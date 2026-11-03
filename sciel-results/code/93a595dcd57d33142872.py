"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _schoolbook(n, m, p):
    """Return the standard schoolbook rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k."""
    # Vectorized verification
    U_mat = U  # (n*m, rank)
    V_mat = V  # (m*p, rank)
    W_mat = W  # (p*n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_mat[i * m + j, :], V_mat[j * p + k, :])
                s = np.sum(s * W_mat[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _vectorized_als(U, V, W, n, m, p, max_iters=500, tol=1e-9, deadline=None):
    """
    Alternating least squares for rank-r decomposition.
    T^1 = U * M_VW  where M_VW[(j,k), r] = V[j,r] * W[k,r]
    T^2 = V * M_WU  where M_WU[(k,i), r] = W[k,r] * U[i,r]
    T^3 = W * M_UV  where M_UV[(j,i), r] = U[i,r] * V[j,r]
    
    We want T^1 to be the flattening of the tensor [1 if (i,j,k) is a valid index]
    Actually, the target tensor T has entries T[i,j,k] = 1 for all i,j,k.
    So T^1[i, (j,k)] = 1, T^2[j, (k,i)] = 1, T^3[k, (i,j)] = 1.
    
    Update rules:
    U = T^1 * M_VW^+
    V = T^2 * M_WU^+
    W = T^3 * M_UV^+
    """
    rank = U.shape[1]
    T1 = np.ones((n * m, m * p))  # T^1[i, (j,k)] = 1
    T2 = np.ones((m * p, p * n))  # T^2[j, (k,i)] = 1
    T3 = np.ones((p * n, n * m))  # T^3[k, (i,j)] = 1
    
    prev_err = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline - 2.0:
            break
            
        # Build M_VW: shape (m*p, rank)
        # M_VW[(j,k), r] = V[j*p+k, r] * W[k*n+i, r] ... wait, W depends on i.
        # The mode-1 unfolding is T^1[i, (j,k)] = sum_r U[i,r] V[j,r] W[k,r] ?
        # No, the tensor is sum_r U[i,r] V[j,r] W[k,r].
        # So T^1[i, (j,k)] = sum_r U[i,r] * (V[j,r] * W[k,r]).
        # Here V is indexed by (j, r) but V is stored as (m*p, rank).
        # V[j, r] means V[j, r] in the tensor sense, which is V[j*p + 0? No.
        # The factor matrices are U (n*m, rank), V (m*p, rank), W (p*n, rank).
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # So for fixed i, T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
        # This is NOT a standard CP decomposition where factors are separate.
        # The factors share indices in a complex way.
        # U row index is i*m+j. V row index is j*p+k. W row index is k*n+i.
        
        # Let's re-derive the linear system for U.
        # For fixed i, j:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k.
        # This is a linear system for U[i*m+j, :].
        # Let A_k = V[j*p+k, :] * W[k*n+i, :] (element-wise product, size rank).
        # Then sum_r U[i*m+j, r] * A_k[r] = 1 for k=0..p-1.
        # This is: U[i*m+j, :] @ A^T = 1, where A is (p, rank) with A[k, :] = A_k.
        # So U[i*m+j, :] = 1 @ (A^T)^+ = 1 @ pinv(A^T).
        
        # This means we CAN solve for each row of U independently, but the system
        # depends on V and W. It is not a single matrix multiplication for all rows.
        # However, we can vectorize the construction of A.
        
        # For efficiency, let's just loop over i, j, k but use numpy operations.
        # Since n=3, m=3, p=6, the loops are small (9 * 6 = 54 systems).
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # Solve U[i*m+j, :] @ A^T = 1
                try:
                    # U[i*m+j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                    U[i * m + j, :] = np.linalg.pinv(A.T) @ np.ones(p)
                except:
                    pass
        
        # Update V
        # For fixed j, k:
        # sum_r V[j*p+k, r] * U[i*m+j, r] * W[k*n+i, r] = 1 for all i.
        # Let B_i = U[i*m+j, :] * W[k*n+i, :] (element-wise).
        # sum_r V[j*p+k, r] * B_i[r] = 1 for i=0..n-1.
        # V[j*p+k, :] @ B^T = 1, where B is (n, rank).
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    V[j * p + k, :] = np.linalg.pinv(B.T) @ np.ones(n)
                except:
                    pass
        
        # Update W
        # For fixed k, i:
        # sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] = 1 for all j.
        # Let C_j = U[i*m+j, :] * V[j*p+k, :] (element-wise).
        # sum_r W[k*n+i, r] * C_j[r] = 1 for j=0..m-1.
        # W[k*n+i, :] @ C^T = 1, where C is (m, rank).
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    W[k * n + i, :] = np.linalg.pinv(C.T) @ np.ones(m)
                except:
                    pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (s - 1.0) ** 2
        err = np.sqrt(err)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    school_rank = n * m * p
    
    # Fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, school_rank
    
    # Lower bound is max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But for matrix multiplication, rank is at least max(n, m, p) * something?
    # Strassen's rank for 2x2x2 is 7. For 3x3x3, it's known to be at least 19?
    # Let's try from 19 to school_rank-1.
    
    # To save time, we'll try ranks in increasing order but with limited iterations per rank.
    # If we find a valid one, we return it.
    
    # We will use a smarter initialization: start from schoolbook and remove some columns?
    # Or just random.
    
    # Let's try a range of ranks.
    # Since the search space is large, we might not find the minimum rank.
    # But we should try to beat 54.
    
    # Let's try ranks: 53, 52, ..., 40.
    # With 100 seconds, we can afford to try a few.
    
    ranks_to_try = list(range(school_rank - 1, 30, -1))  # 53 down to 31
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10.0:
            break
        
        found = False
        for seed in range(5):  # 5 restarts per rank
            if time.time() > deadline - 10.0:
                break
            
            rng = np.random.RandomState(seed + rank)
            U = rng.randn(n * m, rank)
            V = rng.randn(m * p, rank)
            W = rng.randn(p * n, rank)
            
            # Run ALS
            U, V, W, err = _vectorized_als(U, V, W, n, m, p, max_iters=200, deadline=deadline)
            
            if err < 1e-5:
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    found = True
                    break
        
        if found:
            # We found a lower rank, continue to see if we can go lower?
            # The task says "Find a rank-minimal exact tensor decomposition".
            # But "score is the rank". So lower is better.
            # If we find rank 50, and then find 48, we should return 48.
            # So we should not break immediately, unless we hit the theoretical lower bound.
            # However, time is limited.
            # Let's continue to the next rank (lower) but with less time budget.
            pass
    
    # If no better rank found, return schoolbook
    # But wait, if we found a better rank, we should keep searching for even lower?
    # Yes.
    
    # However, the loop above goes from high to low.
    # So if we find 53, we try 52, etc.
    # The last found rank is the best (lowest).
    
    # One issue: if we find rank 40, we might have spent too much time on higher ranks.
    # Let's optimize the search order.
    # Maybe try the most promising ranks first?
    # But we don't know which are promising.
    
    # Let's just go in order.
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
