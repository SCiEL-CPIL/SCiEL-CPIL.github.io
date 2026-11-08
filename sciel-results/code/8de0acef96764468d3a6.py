"""Matrix multiplication 2x4 by 4x7 rank-minimal decomposition search."""
import numpy as np
import itertools
from fractions import Fraction
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build tensor from factors
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    for r in range(rank):
        T += np.outer(np.outer(U_arr[:, r], V_arr[:, r]), W_arr[:, r])
    # Compare with true tensor
    T_true = _build_tensor(n, m, p).astype(np.float64)
    return np.allclose(T, T_true, atol=tol)

def _check_half_integer(mat):
    """Check if all entries are integer or half-integer (i.e., 2*x is integer)."""
    for row in mat:
        for val in row:
            if isinstance(val, (int, np.integer)):
                continue
            v = float(val)
            if abs(2 * v - round(2 * v)) > 1e-9:
                return False
    return True

def _to_rational_list(mat):
    """Convert a matrix of floats (that are half-integers) to list of Fractions."""
    result = []
    for row in mat:
        r = []
        for val in row:
            v = float(val)
            # Convert to fraction: multiply by 2, round, divide by 2
            r.append(Fraction(round(2 * v), 2))
        result.append(r)
    return result

def _try_structural_search(n, m, p, time_limit=100.0):
    """
    Try to find a rank-reduced decomposition using structural knowledge.
    Uses a faster approach: exploit the sparse structure of the tensor.
    """
    start_time = time.time()
    T = _build_tensor(n, m, p)
    
    # The tensor T has exactly n*m*p = 56 non-zero entries.
    # Each non-zero entry corresponds to a triple (i,j,k) where:
    #   a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1
    
    # For a rank-r decomposition, we need:
    #   sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Key insight: The tensor is very sparse. We can use this to our advantage.
    
    # Strategy: Try to find a decomposition where each rank-1 term
    # "covers" multiple non-zero entries of T.
    
    # For the 2x2x2 case, Strassen shows rank 7 < 8 is possible.
    # The idea is to have some rank-1 terms that are nonzero at multiple
    # positions of T, effectively combining products.
    
    # For 2x4 x 4x7, let's try to construct a decomposition by:
    # 1. Starting with the schoolbook (rank 56)
    # 2. Trying to merge pairs of terms where possible
    
    # Actually, a more systematic approach:
    # We know the support of T. Let's try to find a low-rank completion.
    
    # Let's try a different strategy: use the fact that for fixed i,
    # the sub-tensor T[i*m+*, *, k*n+i] has a specific structure.
    
    # For fixed i and k, T[a, b, c] = 1 when a = i*m+j, b = j*p+k, c = k*n+i
    # So for fixed (i,k), the non-zero entries form a "diagonal" pattern
    # across the j index.
    
    # Let's try to build a decomposition incrementally.
    
    # First, let's compute a lower bound using matricization ranks.
    T_mat1 = T.reshape(n*m, m*p * p*n)
    T_mat2 = T.reshape(n*m * m*p, p*n)
    T_mat3 = T.reshape(n*m * p*n, m*p)
    
    rank_lb1 = np.linalg.matrix_rank(T_mat1)
    rank_lb2 = np.linalg.matrix_rank(T_mat2)
    rank_lb3 = np.linalg.matrix_rank(T_mat3)
    rank_lower = max(rank_lb1, rank_lb2, rank_lb3)
    
    # Try ranks from lower bound up to schoolbook
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # For efficiency, let's try a greedy approach:
    # Start with schoolbook and try to eliminate terms.
    
    # Actually, let's try a random search with a much faster verification.
    # Key optimization: instead of checking all (a,b) pairs, only check
    # the non-zero entries of T (56 entries).
    
    # Get the support of T
    support = np.argwhere(T != 0)  # shape (56, 3)
    
    for r in range(max(rank_lower, 40), n * m * p + 1):
        if time.time() - start_time > time_limit:
            break
        if r >= best_rank:
            continue
            
        # Try a few random initializations
        found = False
        for attempt in range(100):
            if time.time() - start_time > time_limit:
                break
                
            # Random W with small values to encourage sparsity
            W_rand = np.random.choice([-1, 0, 1], size=(p * n, r))
            # Ensure W has full column rank
            if np.linalg.matrix_rank(W_rand) < r:
                continue
            
            # For each non-zero entry (a,b,c) in T, we need:
            # sum_r U[a,r] * V[b,r] * W[c,r] = 1
            # For zero entries, we need sum_r U[a,r] * V[b,r] * W[c,r] = 0
            
            # This is a system of equations. Let's set up the linear system.
            # Unknowns: U (n*m * r) and V (m*p * r), but they're coupled.
            
            # Alternative: fix W, then for each r, we need:
            #   U[:,r] * V[:,r].T to satisfy the constraints given W[:,r]
            
            # Let's try: for each r, define X_r = U[:,r] * V[:,r].T
            # Then T[a,b,c] = sum_r X_r[a,b] * W[c,r]
            # And X_r must be rank 1.
            
            # For fixed W, we can solve for the "effective" coefficients:
            # For each (a,b): sum_r X_r[a,b] * W[c,r] = T[a,b,c] for all c
            # This means the vector (X_1[a,b], X_2[a,b], ..., X_r[a,b]) 
            # must satisfy: X_vec[a,b] @ W[c,:] = T[a,b,c] for all c
            # If W has full column rank, X_vec[a,b] is uniquely determined:
            # X_vec[a,b] = T[a,b,:] @ W @ (W.T @ W)^{-1}
            
            try:
                WtW = W_rand.T @ W_rand
                WtW_inv = np.linalg.inv(WtW)
            except:
                continue
            
            # Compute X_vec for all (a,b)
            # X_vec[a,b,r] = sum_c T[a,b,c] * W[c,r] * (WtW_inv)[r,s]... 
            # Actually: T[a,b,:] = X_vec[a,b,:] @ W.T
            # So: X_vec[a,b,:] = T[a,b,:] @ W @ (W.T @ W)^{-1}
            
            # T is (n*m, m*p, p*n), W is (p*n, r)
            # T @ W: (n*m, m*p, r)
            TW = np.einsum('abc,cr->abr', T.astype(np.float64), W_rand)
            # X_vec[a,b,r] = TW[a,b,r] @ WtW_inv... no
            # X_vec[a,b,:] = (T[a,b,:] @ W) @ WtW_inv
            X_vec = np.einsum('abr,rs->abs', TW, WtW_inv)
            # X_vec shape: (n*m, m*p, r)
            
            # Now for each r_idx, X_vec[:,:,r_idx] must be rank 1
            ok_all_rank1 = True
            U_sol = np.zeros((n * m, r))
            V_sol = np.zeros((m * p, r))
            
            for r_idx in range(r):
                C_r = X_vec[:, :, r_idx]
                # Check rank 1
                if C_r.size == 0:
                    continue
                # Use SVD for speed
                svals = np.linalg.svd(C_r, compute_uv=False)
                if len(svals) > 1 and svals[1] > 1e-8 * max(svals[0], 1e-12):
                    ok_all_rank1 = False
                    break
                if svals[0] < 1e-12:
                    U_sol[:, r_idx] = 0
                    V_sol[:, r_idx] = 0
                else:
                    U_sv, S_sv, Vt_sv = np.linalg.svd(C_r)
                    U_sol[:, r_idx] = U_sv[:, 0] * np.sqrt(S_sv[0])
                    V_sol[:, r_idx] = Vt_sv[0, :] * np.sqrt(S_sv[0])
            
            if not ok_all_rank1:
                continue
            
            # Verify using only the support (much faster)
            T_reconstructed_support = np.zeros(len(support))
            for idx, (a, b, c) in enumerate(support):
                val = 0.0
                for r_idx in range(r):
                    val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
                T_reconstructed_support[idx] = val
            
            if not np.allclose(T_reconstructed_support, 1.0, atol=1e-6):
                continue
            
            # Also verify a sample of zero entries
            # Get some zero entries to check
            all_indices = np.array([(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)])
            zero_mask = T.reshape(-1)[np.array(all_indices).T.ravel(axis=None)] == 0
            # Actually let's just check a random sample
            np.random.seed(42)
            sample_size = min(200, n*m * m*p * p*n - len(support))
            all_abcs = [(a, b, c) for a in range(n*m) for b in range(m*p) for c in range(p*n)]
            support_set = set(map(tuple, support))
            zero_entries = [abc for abc in all_abcs if abc not in support_set]
            if zero_entries:
                sample = np.random.choice(len(zero_entries), size=sample_size, replace=False)
                ok_zero = True
                for s in sample:
                    a, b, c = zero_entries[s]
                    val = 0.0
                    for r_idx in range(r):
                        val += U_sol[a, r_idx] * V_sol[b, r_idx] * W_rand[c, r_idx]
                    if abs(val) > 1e-6:
                        ok_zero = False
                        break
                if not ok_zero:
                    continue
            
            # Found a valid decomposition! Check half-integer constraint
            all_half_int = all(all_half_int_arr(arr) for arr in [U_sol, V_sol, W_rand])
            if all_half_int:
                best_rank = r
                best_U = U_sol
                best_V = V_sol
                best_W = W_rand
                found = True
                break
            else:
                # Try to round to half-integers and re-verify
                U_round = np.round(U_sol * 2) / 2
                V_round = np.round(V_sol * 2) / 2
                W_round = np.round(W_rand * 2) / 2
                
                # Verify rounded version
                T_recon_round = np.zeros_like(T, dtype=np.float64)
                for r_idx in range(r):
                    T_recon_round += np.outer(np.outer(U_round[:, r_idx], V_round[:, r_idx]), W_round[:, r_idx])
                if np.allclose(T_recon_round, T.astype(np.float64), atol=1e-6):
                    best_rank = r
                    best_U = U_round
                    best_V = V_round
                    best_W = W_round
                    found = True
                    break
        
        if found:
            # Found rank r, continue to see if we can do better
            pass
    
    if best_U is not None:
        U_out = _to_list(best_U)
        V_out = _to_list(best_V)
        W_out = _to_list(best_W)
        return U_out, V_out, W_out
    
    return None

def all_half_int_arr(arr, tol=1e-6):
    """Check if all entries in array are integer or half-integer."""
    for val in arr.flat:
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
            # Round to nearest half-integer
            rv = round(2 * v) / 2.0
            r.append(rv)
        result.append(r)
    return result

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
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

# EVOLVE-BLOCK-END

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Try to find a better decomposition
    try:
        remaining = deadline - time.time()
        if remaining > 5.0:
            result = _try_structural_search(n, m, p, time_limit=remaining)
            if result is not None:
                U, V, W = result
                # Verify
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
    except:
        pass
    
    # Check if we're still within deadline
    if time.time() > deadline - 2.0:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
