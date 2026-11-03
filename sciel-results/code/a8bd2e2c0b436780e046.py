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

    # Strategy 1: Random search with half-integer entries
    # For ranks from lower bound to best_rank-1, try random half-integer matrices.
    
    # Lower bound: The rank of the tensor is at least the max over all
    # "slice ranks". For the matrix multiplication tensor, the rank is
    # at least n*m (fixing k), at least m*p (fixing i), at least p*n (fixing j).
    # n*m = 6, m*p = 18, p*n = 12. So rank >= 18.
    
    max_trials_per_rank = 200
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

    # Strategy 2: Structured approach using linear algebra
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

    # Strategy 3: Try other structured U matrices
    if not time_up():
        # Try U with different patterns
        for r in range(18, best_rank):
            if time_up():
                break
            
            # Try several structured U patterns
            patterns = []
            
            # Pattern 1: Block diagonal
            U_block = np.zeros((nm, r))
            block_size = max(1, r // 3)
            for block in range(min(3, r)):
                start = block * block_size
                end = min(start + block_size, r)
                for col in range(start, end):
                    if col < nm:
                        U_block[col, col] = 1.0
            
            if np.any(U_block != 0):
                patterns.append(U_block)
            
            # Pattern 2: Sparse random
            for _ in range(10):
                U_sparse = np.zeros((nm, r))
                for col in range(r):
                    if col < nm:
                        U_sparse[col, col] = random.choice([0.5, 1, -0.5, -1])
                if np.any(U_sparse != 0):
                    patterns.append(U_sparse)
            
            for U_struct in patterns:
                if time_up():
                    break
                
                # Check if T[a,b,c] = 0 for all a where U has non-zero rows
                T_target = _target_tensor(n, m, p)
                
                # For each column a, check if the corresponding slice is rank 1
                # when weighted by U
                valid = True
                V_struct = np.zeros((mp, r))
                W_struct = np.zeros((pn, r))
                
                for col in range(r):
                    # Get the weight vector for this column
                    weight = U_struct[:, col]
                    if np.all(weight == 0):
                        continue
                    
                    # The contribution to the tensor from this column is:
                    # T_col[a,b,c] = weight[a] * V_struct[b,col] * W_struct[c,col]
                    # We need: sum_col weight[a] * V_struct[b,col] * W_struct[c,col] = T_target[a,b,c]
                    
                    # This is more complex. Let's use a different approach.
                    # For now, skip this and try a simpler method.
                    valid = False
                    break
                
                if not valid:
                    continue

    # Strategy 4: Use the fact that the tensor is "diagonal" in a certain sense
    # T[i*m+j, j*p+k, k*n+i] = 1
    # This suggests a decomposition where each term corresponds to a (i,j,k) triple
    
    if not time_up():
        # Try to find a decomposition with rank = m*p = 18
        # This is the lower bound from the slice rank
        r_target = 18
        
        # For rank 18, we can try to use the structure where
        # each term corresponds to a (j,k) pair, and we sum over i
        # But this requires combining terms for different i
        
        # Alternative: use a decomposition based on the Kronecker product structure
        # T = sum_{j,k} (sum_i e_{i*m+j} ⊗ e_{j*p+k} ⊗ e_{k*n+i})
        # But this doesn't directly give a rank-1 decomposition
        
        # Let's try a different approach: use the fact that for fixed j,
        # the tensor has a specific structure
        
        # For now, try a more aggressive random search with more trials
        max_trials_aggressive = 500
        for trial in range(max_trials_aggressive):
            if time_up():
                break
            
            U_r = np.array([[random.choice(vals) for _ in range(r_target)] for _ in range(nm)])
            V_r = np.array([[random.choice(vals) for _ in range(r_target)] for _ in range(mp)])
            W_r = np.array([[random.choice(vals) for _ in range(r_target)] for _ in range(pn)])
            
            if _verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r_target
                best_result = {"U": _to_list(U_r), "V": _to_list(V_r), "W": _to_list(W_r)}
                break

    return best_result
