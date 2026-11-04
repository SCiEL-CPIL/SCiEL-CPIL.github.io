"""
Rank-minimal exact tensor decomposition for 2x3 by 3x10 matrix multiplication.

Design (kept from seed family):
  Partition the middle dimension (3) into 2 + 1.
  - Part 1: A[0:2,:] (2x2) * B[0:2,:] (2x10). Split the 10 output columns into
    five 2-wide blocks; apply classic rank-7 Strassen to each 2x2 x 2x2 block.
    Subtotal: 5 * 7 = 35.
  - Part 2: A[2,:] (2x1) * B[2,:] (1x10). Schoolbook: 2*1*10 = 20.
  Total rank: 55.

Refinements:
  1. Multi-trial exact verification.
  2. Graceful fallback to schoolbook (rank 60).
  3. Deadline guard.
  4. Attempt to reduce rank below 55 via ALS-based rank minimization,
     starting from the rank-55 solution and trying to project out terms.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 140.0  # Safety margin before the 150 s kill


def _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234):
    """Verify the decomposition with random integer inputs (exact arithmetic).

    Returns (ok, max_err). Uses vectorized ops for speed.
    """
    rng = np.random.RandomState(seed)
    max_err = 0.0
    for _ in range(num_trials):
        A = rng.randint(-20, 21, size=(n, m)).astype(float)
        B = rng.randint(-20, 21, size=(m, p)).astype(float)
        C_direct = A @ B

        A_flat = A.flatten()
        B_flat = B.flatten()
        u_sums = U.T @ A_flat          # (rank,)
        v_sums = V.T @ B_flat          # (rank,)
        scalars = u_sums * v_sums      # (rank,)
        C_flat = W @ scalars           # (p*n,)
        C_decomp = C_flat.reshape(p, n).T  # (n, p)

        err = float(np.max(np.abs(C_decomp - C_direct)))
        if err > max_err:
            max_err = err
        if err > 1e-9:
            return False, max_err
    return True, max_err


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor as a 3D array T[i,j,k]."""
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0  # A[i,j]*B[j,k] -> C[i,k]
    return T


def _als_reduce_rank(U, V, W, target_rank, max_iters=200, tol=1e-12, deadline=None):
    """Try to reduce the rank of a decomposition to target_rank using ALS.
    
    Starts from the current (U,V,W) and attempts to find a valid rank-target_rank
    decomposition. Returns (success, U_new, V_new, W_new).
    """
    n, m, p = DIMENSIONS
    dim_A = n * m  # 6
    dim_B = m * p  # 30
    dim_C = p * n  # 20
    
    # Build the target tensor (flattened)
    T = _build_tensor(n, m, p)
    T_flat = T.flatten()  # (n*m*p,) = (60,)
    
    # Start from current solution, truncate to target_rank
    current_rank = U.shape[1]
    if current_rank <= target_rank:
        return False, None, None, None
    
    # Use the first target_rank columns as initialization
    U0 = U[:, :target_rank].copy()
    V0 = V[:, :target_rank].copy()
    W0 = W[:, :target_rank].copy()
    
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_err = np.inf
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each column u_i, minimize ||T - sum_j u_j (v_j ⊗ w_j)||
        # This is a linear system: (V⊗W) u_i = residual_i
        # More efficiently: treat as matrix completion
        # T_mat[i, j] = sum_k U[i,k] * V[j,k] * W[k] ... no, that's not right.
        
        # The tensor is T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # Matricize along the C dimension: for each (i,j), T[i,j,:] = sum_r U[i,r]*V[j,r] * W[:,r]
        # This is a sum of rank-1 matrices in the (i,j) vs k space.
        
        # Alternative: iterate over r, solve for W[:,r] given U,V:
        # For fixed r, W[:,r] should satisfy:
        # sum_{r'} U[i,r']*V[j,r']*W[k,r'] = T[i,j,k] for all i,j,k
        # => W[k,r] = (sum_{i,j} T[i,j,k] * U[i,r] * V[j,r]) / (sum_{i,j} U[i,r]^2 * V[j,r]^2)
        
        # Let's do coordinate descent on each factor
        for r in range(target_rank):
            # Solve for W[:, r]
            num = np.zeros(p * n)
            for i in range(n):
                for j in range(m):
                    val = U[i*m+j, r] * V[j*p+0, r]  # This isn't right for general case
                    # Actually V index is (j, k) but we need to sum over k
                    pass
            
            # Better approach: use the matricized form
            # T is (n, m, p). Fix i,j: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is hard to separate. Let me use a different formulation.
            
            # Use the standard ALS for 3-way tensor:
            # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # where U is (n, rank), V is (m, rank), W is (p, rank)
            # But our U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
            # So U_{(i,j),r} V_{(j,k),r} W_{(k,i),r}
            
            # This is a "matrix multiplication" tensor, not a standard 3-way tensor.
            # The standard ALS doesn't directly apply. Let me use a different approach.
            
            # For fixed r, the contribution is U[:,r] ⊗ V[:,r] ⊗ W[:,r] in a specific sense.
            # The residual is T_flat - sum_{r'} (U[:,r'] * V[:,r'] reshaped) * W[:,r']
            
            # Let me just do a simple gradient-free approach: try random perturbations.
            pass
        
        # Simpler: use the fact that for each r, given other factors,
        # we can solve for W[:,r] via least squares.
        # T_flat = sum_r (U[:,r] * V[:,r])_reshaped * W[:,r]
        # where (U[:,r] * V[:,r]) is the element-wise product of U[:,r] (6,) and V[:,r] (30,)
        # reshaped to (6*30=180)? No, that's not right either.
        
        # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For fixed r, define A_r[i,j] = U[i*m+j, r], B_r[j,k] = V[j*p+k, r], C_r[k,i] = W[k*n+i, r]
        # Then T[i,j,k] = sum_r A_r[i,j] * B_r[j,k] * C_r[k,i]
        
        # This is the matrix multiplication tensor! The decomposition is valid if
        # sum_r A_r ⊙ B_r ⊙ C_r = T (element-wise in the appropriate sense)
        
        # For fixed r, to solve for C_r = W[:,r] (shape (p*n,) = (20,)):
        # T[i,j,k] = A_r[i,j] * B_r[j,k] * C_r[k,i] + (contribution from other r's)
        # => C_r[k,i] = (T[i,j,k] - sum_{r'≠r} A_{r'}[i,j]*B_{r'}[j,k]*C_{r'}[k,i]) / (A_r[i,j]*B_r[j,k])
        # This is over-determined (multiple (i,j,k) give constraints on C_r[k,i]).
        # Use least squares: for each (k,i), C_r[k,i] = sum_{j} (T[i,j,k]*A_r[i,j]*B_r[j,k]) / sum_j (A_r[i,j]^2 * B_r[j,k]^2)
        
        # Let me implement this properly.
        # Residual after removing r-th term:
        # R[i,j,k] = T[i,j,k] - sum_{r'≠r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r']
        # Then C_r[k,i] should minimize sum_{i,j,k} (R[i,j,k] - U[i*m+j,r]*V[j*p+k,r]*C_r[k,i])^2
        # For fixed (k,i): C_r[k,i] = sum_j R[i,j,k]*U[i*m+j,r]*V[j*p+k,r] / sum_j (U[i*m+j,r]*V[j*p+k,r])^2
        
        # This is O(n*m*p*rank^2) per iteration which is fine for our sizes.
        
        # Let me redo the ALS loop properly.
        break  # Skip the broken loop, use a cleaner approach below
    
    # Clean ALS implementation
    U_new = U[:, :target_rank].copy().astype(float)
    V_new = V[:, :target_rank].copy().astype(float)
    W_new = W[:, :target_rank].copy().astype(float)
    
    T = _build_tensor(n, m, p)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update W: for each r, solve for W_new[:,r]
        for r in range(target_rank):
            # Compute residual without r-th term
            # R[i,j,k] = T[i,j,k] - sum_{r'≠r} U_new[i*m+j,r']*V_new[j*p+k,r']*W_new[k*n+i,r']
            # For efficiency, compute the full sum first, then subtract r-th term
            # Full = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # Use vectorized: 
            # U_mat: (n, m, rank) from U_new
            # V_mat: (m, p, rank) from V_new  
            # W_mat: (p, n, rank) from W_new
            
            U_mat = U_new.reshape(n, m, target_rank)
            V_mat = V_new.reshape(m, p, target_rank)
            W_mat = W_new.reshape(p, n, target_rank)
            
            # Compute contribution of r-th term: U_mat[:,:,r] * V_mat[:,:,r] * W_mat[:,:,r]
            # But indices don't align directly. Let me think...
            # T[i,j,k] = sum_r U_mat[i,j,r] * V_mat[j,k,r] * W_mat[k,i,r]
            
            # For fixed r, the r-th term is U_mat[i,j,r] * V_mat[j,k,r] * W_mat[k,i,r]
            # Residual R[i,j,k] = T[i,j,k] - sum_{r'≠r} U_mat[i,j,r']*V_mat[j,k,r']*W_mat[k,i,r']
            
            # To solve for W_mat[k,i,r]:
            # Minimize sum_{i,j,k} (R[i,j,k] - U_mat[i,j,r]*V_mat[j,k,r]*W_mat[k,i,r])^2
            # For fixed (k,i): W_mat[k,i,r] = sum_j R[i,j,k]*U_mat[i,j,r]*V_mat[j,k,r] / sum_j (U_mat[i,j,r]*V_mat[j,k,r])^2
            
            # Compute the sum of all terms first
            # S[i,j,k] = sum_r U_mat[i,j,r]*V_mat[j,k,r]*W_mat[k,i,r]
            # This is expensive to compute naively. For our small sizes it's fine.
            
            # Build S by summing over r
            S = np.zeros((n, m, p))
            for r2 in range(target_rank):
                # S[i,j,k] += U_mat[i,j,r2] * V_mat[j,k,r2] * W_mat[k,i,r2]
                # Vectorized: for each i,j,k, sum over r2
                # U_mat[i,j,:] * V_mat[j,k,:] * W_mat[k,i,:]
                # This is (n, m, p, rank) -> sum over last axis
                term = U_mat[:, :, r2]  # (n, m)
                # Need to multiply element-wise with V_mat[j,k,r2] and W_mat[k,i,r2]
                # Result[i,j,k] = U_mat[i,j,r2] * V_mat[j,k,r2] * W_mat[k,i,r2]
                # Shape: (n, m, p)
                # U_mat[i,j,r2] is (n,m), V_mat[j,k,r2] is (m,p), W_mat[k,i,r2] is (p,n)
                # Result[i,j,k] = U_mat[i,j,r2] * V_mat[j,k,r2] * W_mat[k,i,r2]
                # = (U_mat[:, :, r2] @ V_mat[:, :, r2].T) ... no
                # Let me just do it with broadcasting
                # U_term: (n, 1, 1) broadcast to (n, m, p)? No.
                # U_mat[i,j,r2]: shape (n, m)
                # V_mat[j,k,r2]: shape (m, p)  
                # W_mat[k,i,r2]: shape (p, n)
                # Result[i,j,k] = U_mat[i,j,r2] * V_mat[j,k,r2] * W_mat[k,i,r2]
                # This is an einsum: 'ij,jk,ki->ijk'
                term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
                S += term_3d
            
            # Now R[i,j,k] = T[i,j,k] - (S - term_r) = T[i,j,k] - S + term_r
            # where term_r[i,j,k] = U_mat[i,j,r] * V_mat[j,k,r] * W_mat[k,i,r]
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r  # This is the residual when we set W_mat[:,:,r] = 0
            
            # Solve for W_mat[k,i,r]:
            # W_mat[k,i,r] = sum_j R[i,j,k] * U_mat[i,j,r] * V_mat[j,k,r] / sum_j (U_mat[i,j,r] * V_mat[j,k,r])^2
            # Numerator[k,i] = sum_j R[i,j,k] * U_mat[i,j,r] * V_mat[j,k,r]
            # = sum_j R[i,j,k] * U_mat[i,j,r] * V_mat[j,k,r]
            # = einsum('ijk,ij,jk->ki', R, U_mat[:,:,r], V_mat[:,:,r])
            num = np.einsum('ijk,ij,jk->ki', R, U_mat[:, :, r], V_mat[:, :, r])
            # Denominator[k,i] = sum_j (U_mat[i,j,r] * V_mat[j,k,r])^2
            prod = U_mat[:, :, r] * V_mat[:, :, r]  # (n, m, p)? No, U_mat is (n,m), V_mat is (m,p)
            # prod[i,j,k] = U_mat[i,j,r] * V_mat[j,k,r]
            prod_3d = np.einsum('ij,jk->ijk', U_mat[:, :, r], V_mat[:, :, r])
            den = np.einsum('ijk,ijk->ki', prod_3d, prod_3d)  # (p, n)
            
            # Avoid division by zero
            mask = den > 1e-15
            W_mat[:, :, r][mask] = num[mask] / den[mask]
            W_mat[:, :, r][~mask] = 0.0
        
        # Update U: for each r, solve for U_mat[i,j,r]
        for r in range(target_rank):
            U_mat = U_new.reshape(n, m, target_rank)
            V_mat = V_new.reshape(m, p, target_rank)
            W_mat = W_new.reshape(p, n, target_rank)
            
            S = np.zeros((n, m, p))
            for r2 in range(target_rank):
                term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
                S += term_3d
            
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r
            
            # U_mat[i,j,r] = sum_{k} R[i,j,k] * V_mat[j,k,r] * W_mat[k,i,r] / sum_k (V_mat[j,k,r] * W_mat[k,i,r])^2
            num = np.einsum('ijk,jk,ki->ij', R, V_mat[:, :, r], W_mat[:, :, r])
            prod_3d = np.einsum('jk,ki->ijk', V_mat[:, :, r], W_mat[:, :, r])
            den = np.einsum('ijk,ijk->ij', prod_3d, prod_3d)
            
            mask = den > 1e-15
            U_mat[:, :, r][mask] = num[mask] / den[mask]
            U_mat[:, :, r][~mask] = 0.0
        
        # Update V: for each r, solve for V_mat[j,k,r]
        for r in range(target_rank):
            U_mat = U_new.reshape(n, m, target_rank)
            V_mat = V_new.reshape(m, p, target_rank)
            W_mat = W_new.reshape(p, n, target_rank)
            
            S = np.zeros((n, m, p))
            for r2 in range(target_rank):
                term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
                S += term_3d
            
            term_r = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r], V_mat[:, :, r], W_mat[:, :, r])
            R = T - S + term_r
            
            # V_mat[j,k,r] = sum_{i} R[i,j,k] * U_mat[i,j,r] * W_mat[k,i,r] / sum_i (U_mat[i,j,r] * W_mat[k,i,r])^2
            num = np.einsum('ijk,ij,ki->jk', R, U_mat[:, :, r], W_mat[:, :, r])
            prod_3d = np.einsum('ij,ki->ijk', U_mat[:, :, r], W_mat[:, :, r])
            den = np.einsum('ijk,ijk->jk', prod_3d, prod_3d)
            
            mask = den > 1e-15
            V_mat[:, :, r][mask] = num[mask] / den[mask]
            V_mat[:, :, r][~mask] = 0.0
        
        # Check convergence
        U_new = U_mat.reshape(-1, target_rank)
        V_new = V_mat.reshape(-1, target_rank)
        W_new = W_mat.reshape(-1, target_rank)
        
        # Compute error
        S = np.zeros((n, m, p))
        for r2 in range(target_rank):
            term_3d = np.einsum('ij,jk,ki->ijk', U_mat[:, :, r2], V_mat[:, :, r2], W_mat[:, :, r2])
            S += term_3d
        err = np.max(np.abs(S - T))
        if err < best_err:
            best_err = err
            best_U = U_new.copy()
            best_V = V_new.copy()
            best_W = W_new.copy()
        
        if err < tol:
            break
    
    return best_err < 1e-9, best_U, best_V, best_W


def _schoolbook(n, m, p):
    """Guaranteed-correct schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W


def _strassen_decomposition(n, m, p):
    """Build the rank-55 Strassen-on-2x2-blocks + schoolbook decomposition."""
    rank = 35 + 20  # 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    idx = 0

    # Part 1: Strassen for 2x2 x 2x10 (five 2x2x2 blocks)
    for block in range(5):
        k0 = 2 * block
        k1 = 2 * block + 1

        # M1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        #     -> C[0,k0] += M1, C[1,k1] += M1
        U[0 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M2 = (A[1,0] + A[1,1]) * B[0,k0]
        #     -> C[0,k1] += M2, C[1,k1] -= M2
        U[1 * 3 + 0, idx] = 1.0
        U[1 * 3 + 1, idx] = 1.0
        V[0 * 10 + k0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        W[k1 * 2 + 1, idx] = -1.0
        idx += 1

        # M3 = A[0,0] * (B[0,k1] - B[1,k1])
        #     -> C[0,k1] += M3, C[1,k1] += M3
        U[0 * 3 + 0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        V[1 * 10 + k1, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M4 = A[1,1] * (B[1,k0] - B[0,k0])
        #     -> C[0,k0] += M4, C[1,k0] += M4
        U[1 * 3 + 1, idx] = 1.0
        V[1 * 10 + k0, idx] = 1.0
        V[0 * 10 + k0, idx] = -1.0
        W[k0 * 2 + 0, idx] = 1.0
        W[k0 * 2 + 1, idx] = 1.0
        idx += 1

        # M5 = (A[0,0] + A[0,1]) * B[1,k1]
        #     -> C[0,k0] -= M5, C[0,k1] += M5
        U[0 * 3 + 0, idx] = 1.0
        U[0 * 3 + 1, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = -1.0
        W[k1 * 2 + 0, idx] = 1.0
        idx += 1

        # M6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[0,k1])
        #     -> C[1,k1] += M6
        U[1 * 3 + 0, idx] = 1.0
        U[0 * 3 + 0, idx] = -1.0
        V[0 * 10 + k0, idx] = 1.0
        V[0 * 10 + k1, idx] = 1.0
        W[k1 * 2 + 1, idx] = 1.0
        idx += 1

        # M7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[1,k1])
        #     -> C[0,k0] += M7
        U[0 * 3 + 1, idx] = 1.0
        U[1 * 3 + 1, idx] = -1.0
        V[1 * 10 + k0, idx] = 1.0
        V[1 * 10 + k1, idx] = 1.0
        W[k0 * 2 + 0, idx] = 1.0
        idx += 1

    # Part 2: Schoolbook for 2x1 x 1x10
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2, idx] = 1.0
            V[2 * 10 + k, idx] = 1.0
            W[k * 2 + i, idx] = 1.0
            idx += 1

    return U, V, W


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    # Build the rank-55 decomposition.
    U, V, W = _strassen_decomposition(n, m, p)

    # Rigorous verification with several seeds; fall back to schoolbook on failure.
    ok, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=1234)
    if not ok:
        ok2, _ = _verify_decomposition(U, V, W, n, m, p, num_trials=8, seed=9999)
        if not ok2:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Deadline guard
    if time.time() - start_time > DEADLINE:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt to reduce rank below 55 using ALS
    # Try target ranks: 54, 53, 52, ... down to some minimum
    # Use remaining time wisely
    deadline = start_time + DEADLINE
    best_rank = 55
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try reducing from rank 55
    for target_rank in range(54, 29, -1):  # Try 54, 53, ..., 30
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        
        # Try multiple initializations
        found = False
        for trial in range(3):
            if time.time() > deadline - 3:
                break
            
            # Different initialization strategies
            if trial == 0:
                # Use first target_rank columns
                U_init = U[:, :target_rank].copy()
                V_init = V[:, :target_rank].copy()
                W_init = W[:, :target_rank].copy()
            elif trial == 1:
                # Use last target_rank columns
                U_init = U[:, -target_rank:].copy()
                V_init = V[:, -target_rank:].copy()
                W_init = W[:, -target_rank:].copy()
            else:
                # Random subset
                rng = np.random.RandomState(42 + trial)
                cols = rng.choice(55, target_rank, replace=False)
                U_init = U[:, cols].copy()
                V_init = V[:, cols].copy()
                W_init = W[:, cols].copy()
            
            success, U_new, V_new, W_new = _als_reduce_rank(
                U_init, V_init, W_init, target_rank, 
                max_iters=100, tol=1e-12, deadline=deadline - 3
            )
            
            if success:
                # Verify the solution
                ok, _ = _verify_decomposition(U_new, V_new, W_new, n, m, p, num_trials=5, seed=777)
                if ok:
                    best_rank = target_rank
                    best_U, best_V, best_W = U_new, V_new, W_new
                    found = True
                    break  # Stop trying other initializations for this rank
        
        if found:
            # Try to go even lower
            continue
        else:
            # Can't achieve this rank, stop
            break
    
    # Final deadline check
    if time.time() - start_time > DEADLINE:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


# Entry point required by evaluator
if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    A_flat = A.flatten()
    B_flat = B.flatten()
    u_sums = U.T @ A_flat
    v_sums = V.T @ B_flat
    scalars = u_sums * v_sums
    C_flat = W @ scalars
    C_decomp = C_flat.reshape(p, n).T

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
