import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [4, 5, 6]

def _build_block_strassen(n, m, p):
    """Build decomposition using Strassen for 2x2x2 blocks and schoolbook elsewhere."""
    i_blocks = [(0, 2), (2, 2)]
    j_blocks = [(0, 2), (2, 2), (4, 1)]
    k_blocks = [(0, 2), (2, 2), (4, 2)]
    
    total_rank = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * nj * nk
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    offset = 0
    for i0, ni in i_blocks:
        for j0, nj in j_blocks:
            for k0, nk in k_blocks:
                if ni == 2 and nj == 2 and nk == 2:
                    offset = _strassen_2x2x2(U, V, W, i0, j0, k0, offset)
                else:
                    offset = _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset)
    
    return U, V, W

def _strassen_2x2x2(U, V, W, i0, j0, k0, offset):
    n, m, p = 4, 5, 6
    U[i0*m + j0, offset+0] = 1
    U[(i0+1)*m + (j0+1), offset+0] = 1
    V[j0*p + k0, offset+0] = 1
    V[(j0+1)*p + (k0+1), offset+0] = 1
    W[k0*n + i0, offset+0] = 1
    W[(k0+1)*n + i0, offset+0] = 1
    W[k0*n + (i0+1), offset+0] = 1
    W[(k0+1)*n + (i0+1), offset+0] = 1
    
    U[(i0+1)*m + j0, offset+1] = 1
    U[(i0+1)*m + (j0+1), offset+1] = 1
    V[j0*p + k0, offset+1] = 1
    W[k0*n + (i0+1), offset+1] = 1
    W[(k0+1)*n + (i0+1), offset+1] = -1
    
    U[i0*m + j0, offset+2] = 1
    V[j0*p + (k0+1), offset+2] = 1
    V[(j0+1)*p + (k0+1), offset+2] = -1
    W[(k0+1)*n + (i0+1), offset+2] = 1
    
    U[(i0+1)*m + (j0+1), offset+3] = 1
    V[(j0+1)*p + k0, offset+3] = 1
    V[j0*p + k0, offset+3] = -1
    W[k0*n + i0, offset+3] = -1
    W[(k0+1)*n + i0, offset+3] = 1
    
    U[i0*m + j0, offset+4] = 1
    U[i0*m + (j0+1), offset+4] = 1
    V[(j0+1)*p + (k0+1), offset+4] = 1
    W[(k0+1)*n + i0, offset+4] = 1
    W[(k0+1)*n + (i0+1), offset+4] = 1
    
    U[(i0+1)*m + j0, offset+5] = 1
    U[i0*m + j0, offset+5] = -1
    V[j0*p + k0, offset+5] = 1
    V[j0*p + (k0+1), offset+5] = 1
    W[k0*n + i0, offset+5] = 1
    W[k0*n + (i0+1), offset+5] = 1
    
    U[i0*m + (j0+1), offset+6] = 1
    U[(i0+1)*m + (j0+1), offset+6] = -1
    V[(j0+1)*p + k0, offset+6] = 1
    V[(j0+1)*p + (k0+1), offset+6] = 1
    W[k0*n + i0, offset+6] = 1
    W[(k0+1)*n + (i0+1), offset+6] = 1
    
    return offset + 7

def _schoolbook_block(U, V, W, i0, j0, k0, ni, nj, nk, offset):
    n, m, p = 4, 5, 6
    for ii in range(ni):
        for jj in range(nj):
            for kk in range(nk):
                i = i0 + ii
                j = j0 + jj
                k = k0 + kk
                U[i*m + j, offset] = 1
                V[j*p + k, offset] = 1
                W[k*n + i, offset] = 1
                offset += 1
    return offset

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify the tensor decomposition reconstructs matrix multiplication exactly."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    # Check zero constraints (sample)
    rng = np.random.RandomState(42)
    for _ in range(5000):
        a = rng.randint(0, n*m)
        b = rng.randint(0, m*p)
        g = rng.randint(0, p*n)
        i = a // m; j = a % m
        jp = b // p; k = b % p
        kn = g // n; ip = g % n
        if i != ip or j != jp or k != kn:
            val = 0.0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[g, r]
            if abs(val) > tol:
                return False
    return True

def _solve_W_from_UV(U, V, n, m, p, rank):
    """
    Given U (n*m x rank) and V (m*p x rank), solve for W (p*n x rank)
    such that the tensor equation holds.
    
    For each output index (k, i) in W, we need:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
    
    This is a system of linear equations in the entries of W.
    """
    W = np.zeros((p * n, rank))
    
    # For each (k, i) pair, solve for W[k*n+i, :]
    for k in range(p):
        for i in range(n):
            # We need: for all j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is: (U_i^T * V_k) dot W[k*n+i, :] = 1 where U_i is row i of U reshaped
            # Actually: sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Let M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Then: sum_r (sum_j M[j, r]) * W[k*n+i, r] = 1
            # This is underdetermined if rank > 1. We need more constraints.
            
            # Actually, the constraint is for EACH j separately:
            # For each j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # Wait, that's not right either. Let me re-derive.
            
            # The tensor equation is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k'*n+i', r] = delta_{i,i'} * delta_{k,k'}
            
            # For fixed (i, k), the equation for all j is:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  (for the (i,j,k) entry)
            
            # But W[k*n+i, r] appears in this equation for ALL j simultaneously.
            # So we have m equations (one for each j) and rank unknowns (W[k*n+i, r]).
            
            # Let's set up the system: for each j, 
            # sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
            
            # This is: A dot w = 1 where A[j, r] = U[i*m+j, r] * V[j*p+k, r] and w = W[k*n+i, :]
            
            A = np.zeros((m, rank))
            for j in range(m):
                A[j, :] = U[i*m+j, :] * V[j*p+k, :]
            
            # Solve A dot w = ones(m)
            # This is underdetermined if rank > m, overdetermined if rank < m
            try:
                if rank <= m:
                    # Overdetermined or square, use least squares
                    w, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if solution is consistent
                    if np.max(np.abs(A.dot(w) - np.ones(m))) > 1e-6:
                        return None  # No solution
                else:
                    # Underdetermined, find minimum norm solution
                    # Use SVD
                    U_svd, s, Vt = np.linalg.svd(A, full_matrices=False)
                    # Pseudoinverse
                    A_pinv = Vt.T @ np.diag(1.0 / s) @ U_svd.T
                    w = A_pinv @ np.ones(m)
                    if np.max(np.abs(A.dot(w) - np.ones(m))) > 1e-6:
                        return None  # No solution
            except np.linalg.LinAlgError:
                return None
            
            W[k*n+i, :] = w
    
    return W

def _alternating_minimization(U, V, W, n, m, p, rank, max_iters=50, tol=1e-10):
    """
    Alternating minimization: fix two factors, solve for the third.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iters):
        # Solve for W given U, V
        W_new = _solve_W_from_UV(U, V, n, m, p, rank)
        if W_new is None:
            break
        
        # Check convergence
        diff_W = np.max(np.abs(W_new - W))
        W = W_new
        
        if diff_W < tol:
            break
    
    return U, V, W

def _try_rank_reduction(U_init, V_init, W_init, n, m, p, target_rank, max_restarts=10):
    """
    Try to find a rank-target_rank decomposition starting from the given initialization.
    """
    best_U, best_V, best_W = None, None, None
    best_error = np.inf
    
    rng = np.random.RandomState(42)
    
    for restart in range(max_restarts):
        # Initialize with random perturbations
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        # Add small random perturbations
        U += rng.randn(*U.shape) * 0.01
        V += rng.randn(*V.shape) * 0.01
        W += rng.randn(*W.shape) * 0.01
        
        # Run alternating minimization
        U, V, W = _alternating_minimization(U, V, W, n, m, p, target_rank, max_iters=30)
        
        # Check if the decomposition is valid
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
            return U, V, W
    
    return None, None, None

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Build block-Strassen decomposition as initial solution
    U, V, W = _build_block_strassen(n, m, p)
    rank = U.shape[1]
    
    # Verify the initial solution
    if _verify_decomposition(U, V, W, n, m, p):
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    else:
        # Fallback to schoolbook
        schoolbook_rank = n * m * p
        U = np.zeros((n * m, schoolbook_rank))
        V = np.zeros((m * p, schoolbook_rank))
        W = np.zeros((p * n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
        best_rank = schoolbook_rank
    
    # Try to reduce rank using alternating minimization
    if time.time() < deadline - 30:
        for target_rank in range(best_rank - 1, max(best_rank - 10, 50), -1):
            if time.time() > deadline - 10:
                break
            
            U_new, V_new, W_new = _try_rank_reduction(
                best_U, best_V, best_W, n, m, p, target_rank, max_restarts=5
            )
            
            if U_new is not None:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                break  # Found a lower rank, stop
    
    # Return the best solution found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
