"""Matrix multiplication 2x3x6: rank-minimal exact tensor decomposition search.

Strategy:
  - The schoolbook rank is n*m*p = 36.
  - We search for lower-rank exact decompositions with entries in {0, +-1/2, +-1, ...}.
  - We use a randomized construction that fixes one factor (U) to a small structured
    basis, then solves for V and W via linear systems, then snaps to half-integers
    and verifies exactly.
  - We also try a direct "guess and verify" loop over random sparse half-integer
    matrices for U, V, W, checking the bilinear identity.
  - Deadline guard: track elapsed time and return the best valid result.
"""
import time
import random
import numpy as np
import itertools

DIMENSIONS = [2, 3, 6]

# Deadline guard: stop 10 seconds before the 150s wall-clock limit.
DEADLINE_SECONDS = 140.0

def _target_tensor(n, m, p):
    """Return the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    where T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise."""
    nm = n * m
    mp = m * p
    pn = p * n
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W exactly decompose the matrix multiplication tensor.

    The decomposition says: for all (i*m+j, j*p+k, k*n+i),
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1,
    and for all other (a, b, c) triples, the sum should be 0.

    More precisely, the tensor reconstructed is:
    T_recon[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]

    We need T_recon == T (the target tensor).
    """
    rank = U.shape[1]
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    T_target = _target_tensor(n, m, p)
    return np.allclose(T_recon, T_target, atol=tol)

def _snap_to_half_integers(M, tol=1e-6):
    """Snap matrix entries to the nearest half-integer if close."""
    M_snapped = np.round(M * 2) / 2
    if np.allclose(M, M_snapped, atol=tol):
        return M_snapped
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _to_list(M):
    """Convert numpy array to list of lists (or keep as-is if already list)."""
    return M.tolist()

def _find_sub_decomposition(n_s, m_s, p_s, target_rank, start_time, deadline, max_iterations=50000):
    """Find a rank-target_rank decomposition for M_{n_s, m_s, p_s} using random search.
    
    Returns (U, V, W) if found, else None.
    """
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    nm_s = n_s * m_s
    mp_s = m_s * p_s
    pn_s = p_s * n_s
    
    # Precompute target tensor for subproblem
    T_sub = np.zeros((nm_s, mp_s, pn_s), dtype=float)
    for i in range(n_s):
        for j in range(m_s):
            for k in range(p_s):
                T_sub[i * m_s + j, j * p_s + k, k * n_s + i] = 1.0
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Random half-integer matrices
        U_r = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(nm_s)])
        V_r = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(mp_s)])
        W_r = np.array([[random.choice(vals) for _ in range(target_rank)] for _ in range(pn_s)])
        
        # Verify
        T_recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_recon, T_sub, atol=1e-9):
            return U_r, V_r, W_r
    
    return None

def _construct_from_blocks(n, m, p, block_size, sub_decomp_func, start_time, deadline):
    """Construct full decomposition by partitioning p into blocks of size block_size.
    
    sub_decomp_func: function that returns (U_s, V_s, W_s) for the subproblem
    """
    n_sub = n
    m_sub = m
    p_sub = block_size
    
    # Find sub-decomposition
    result = sub_decomp_func(n_sub, m_sub, p_sub, start_time, deadline)
    if result is None:
        return None
    
    U_s, V_s, W_s = result
    sub_rank = U_s.shape[1]
    
    # Number of blocks
    num_blocks = p // block_size
    if num_blocks * block_size != p:
        return None
    
    total_rank = sub_rank * num_blocks
    
    # Construct full matrices
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    for b in range(num_blocks):
        col_offset = b * sub_rank
        
        # U is the same for all blocks
        U_full[:, col_offset:col_offset + sub_rank] = U_s
        
        # V: for block b, j is 0..m-1, k_local is 0..p_sub-1
        # V_full index: j * p + (b * p_sub + k_local)
        for j in range(m_sub):
            for k_local in range(p_sub):
                k_global = b * p_sub + k_local
                V_full[j * p + k_global, col_offset:col_offset + sub_rank] = V_s[j * p_sub + k_local, :]
        
        # W: for block b, k_local is 0..p_sub-1, i is 0..n-1
        # W_full index: (b * p_sub + k_local) * n + i
        for k_local in range(p_sub):
            for i in range(n_sub):
                k_global = b * p_sub + k_local
                W_full[k_global * n + i, col_offset:col_offset + sub_rank] = W_s[k_local * n_sub + i, :]
    
    return U_full, V_full, W_full

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = nm * mp
    best_result = {"U": _to_list(U_sb), "V": _to_list(V_sb), "W": _to_list(W_sb)}

    def elapsed():
        return time.time() - start_time

    def time_up():
        return elapsed() > DEADLINE_SECONDS

    # Strategy 1: Try block decomposition with 3 blocks of 3x2
    # Find rank-6 decomposition for 2x3x2 subproblem
    if not time_up():
        def find_232_decomp(n_s, m_s, p_s, start_time, deadline):
            # Try rank 6 first (schoolbook for 2x3x2 is 12)
            # Actually, let's try ranks from 6 up
            for r in range(6, 13):
                if time.time() > deadline:
                    break
                result = _find_sub_decomposition(n_s, m_s, p_s, r, start_time, deadline, max_iterations=2000)
                if result is not None:
                    return result
            return None
        
        U_full, V_full, W_full = _construct_from_blocks(n, m, p, 2, find_232_decomp, start_time, DEADLINE_SECONDS)
        if U_full is not None:
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                best_rank = U_full.shape[1]
                best_result = {"U": _to_list(U_full), "V": _to_list(V_full), "W": _to_list(W_full)}

    # Strategy 2: Random search with half-integer entries
    # For ranks from lower bound to best_rank-1, try random half-integer matrices.
    
    # Lower bound: The rank of the tensor is at least the max over all
    # "slice ranks". For the matrix multiplication tensor, the rank is
    # at least n*m (fixing k), at least m*p (fixing i), at least p*n (fixing j).
    # n*m = 6, m*p = 18, p*n = 12. So rank >= 18.
    
    max_trials_per_rank = 100
    vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]

    for r in range(18, best_rank):
        if time_up():
            break
        for trial in range(max_trials_per_rank):
            if time_up():
                break
            # Random half-integer matrices
            U_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(nm)])
            V_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(mp)])
            W_r = np.array([[random.choice(vals) for _ in range(r)] for _ in range(pn)])
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_result = {"U": _to_list(U_r), "V": _to_list(V_r), "W": _to_list(W_r)}
                break  # Found a valid decomposition for this rank, move to next rank

    # Strategy 3: Structured approach using linear algebra
    # Fix U to have a specific structure and solve for V, W
    # This exploits the fact that the tensor has a specific index structure
    
    if not time_up():
        # Try to find a decomposition where U is sparse/structured
        # For each rank r, try different structured U matrices
        for r in range(18, best_rank):
            if time_up():
                break
            
            # Try U as identity-like matrix (first r rows are standard basis)
            U_struct = np.zeros((nm, r))
            for col in range(min(r, nm)):
                U_struct[col, col] = 1.0
            
            # Now solve for V and W such that:
            # sum_col U[a,col] * V[b,col] * W[c,col] = T[a,b,c]
            # Since U is identity-like, this means:
            # For a < r: V[b,a] * W[c,a] = T[a,b,c]
            # For a >= r: 0 = T[a,b,c]
            
            # Check if T[a,b,c] = 0 for all a >= r
            T_target = _target_tensor(n, m, p)
            if np.any(T_target[r:, :, :] != 0):
                continue
            
            # For each a < r, we need V[:,a] * W[:,a] = T_target[a, :, :]
            # This is a rank-1 factorization of each slice T_target[a]
            # T_target[a] is an (mp x pn) matrix
            # We need V[:,a] to be (mp x 1) and W[:,a] to be (pn x 1)
            
            valid = True
            V_struct = np.zeros((mp, r))
            W_struct = np.zeros((pn, r))
            
            for a in range(min(r, nm)):
                # Get the slice T_target[a, :, :]
                slice_a = T_target[a, :, :]
                if np.all(slice_a == 0):
                    # V[:,a] and W[:,a] can be anything, set to 0
                    continue
                
                # Check if slice_a has rank 1
                rank_slice = np.linalg.matrix_rank(slice_a, tol=1e-10)
                if rank_slice > 1:
                    valid = False
                    break
                
                # If rank 1, find V[:,a] and W[:,a]
                # Use SVD or direct factorization
                U_svd, s_vals, Vt_svd = np.linalg.svd(slice_a, full_matrices=False)
                if len(s_vals) > 0 and s_vals[0] > 1e-10:
                    # slice_a = U_svd[:,0] * s_vals[0] * Vt_svd[0,:]
                    # Set V[:,a] = U_svd[:,0] * sqrt(s_vals[0])
                    # Set W[:,a] = Vt_svd[0,:] * sqrt(s_vals[0])
                    sqrt_s = np.sqrt(s_vals[0])
                    V_struct[:, a] = U_svd[:, 0] * sqrt_s
                    W_struct[:, a] = Vt_svd[0, :] * sqrt_s
                else:
                    # All zeros
                    V_struct[:, a] = 0
                    W_struct[:, a] = 0
            
            if valid:
                # Snap to half-integers and verify
                U_snapped = _snap_to_half_integers(U_struct)
                V_snapped = _snap_to_half_integers(V_struct)
                W_snapped = _snap_to_half_integers(W_struct)
                
                if U_snapped is not None and V_snapped is not None and W_snapped is not None:
                    if _verify_decomposition(U_snapped, V_snapped, W_snapped, n, m, p):
                        best_rank = r
                        best_result = {"U": _to_list(U_snapped), "V": _to_list(V_snapped), "W": _to_list(W_snapped)}
                        break

    return best_result
