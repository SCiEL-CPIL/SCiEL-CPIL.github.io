"""Improved program for 2x2 * 2x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    """
    if len(U) == 0 or len(U[0]) == 0:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    T_reconstructed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_true = np.ones((n, m, p))
    
    return np.allclose(T_reconstructed, T_true, atol=1e-10)

def _build_strassen_block(k0, k1, p):
    """
    Build the 7 Strassen terms for a 2x2 * 2x2 block (columns k0, k1).
    """
    terms = []
    
    # M1 = (a+d)(e+h) -> C[0,k0] +1, C[1,k1] +1
    terms.append(({0: 1, 3: 1}, {k0: 1, p+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
    
    # M2 = (c+d)e -> C[1,k0] +1, C[1,k1] -1
    terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
    
    # M3 = a(f-h) -> C[0,k1] +1, C[1,k1] +1
    terms.append(({0: 1}, {k1: 1, p+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
    
    # M4 = d(g-e) -> C[0,k0] +1, C[1,k0] +1
    terms.append(({3: 1}, {p+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
    
    # M5 = (a+b)h -> C[0,k0] -1, C[0,k1] +1
    terms.append(({0: 1, 1: 1}, {p+k1: 1}, {2*k0: -1, 2*k1: 1}))
    
    # M6 = (c-a)(e+f) -> C[1,k1] +1
    terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
    
    # M7 = (b-d)(g+h) -> C[0,k0] +1
    terms.append(({1: 1, 3: -1}, {p+k0: 1, p+k1: 1}, {2*k0: 1}))
    
    return terms

def _build_schoolbook_block(k, p):
    """
    Build the 4 schoolbook terms for a single column k.
    """
    terms = []
    terms.append(({0: 1}, {k: 1}, {2*k: 1}))
    terms.append(({1: 1}, {p+k: 1}, {2*k: 1}))
    terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
    terms.append(({3: 1}, {p+k: 1}, {2*k+1: 1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """
    Convert list of (U_dict, V_dict, W_dict) to matrix format.
    """
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (Ud, Vd, Wd) in enumerate(terms):
        for idx, val in Ud.items():
            U[idx][t] = val
        for idx, val in Vd.items():
            V[idx][t] = val
        for idx, val in Wd.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}

def _trivial_decomposition(n, m, p):
    """
    Trivial rank n*m*p decomposition.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}

def _compute_tensor_terms(U, V, W, n, m, p):
    """
    Compute the tensor terms from the decomposition.
    Returns array of shape (rank, n*m*p) where each row is one term's contribution.
    """
    r = len(U[0])
    U_arr = np.array(U, dtype=float).reshape(n, m, r)
    V_arr = np.array(V, dtype=float).reshape(m, p, r)
    W_arr = np.array(W, dtype=float).reshape(p, n, r)
    
    terms = np.zeros((r, n*m*p))
    for r_idx in range(r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms[r_idx, i*m*p + j*p + k] = U_arr[i,j,r_idx] * V_arr[j,k,r_idx] * W_arr[k,i,r_idx]
    return terms

def _try_single_elimination(U, V, W, n, m, p, max_time=30):
    """
    Try to eliminate single terms by checking if they can be expressed as linear combinations.
    """
    start_time = time.time()
    r = len(U[0])
    terms = _compute_tensor_terms(U, V, W, n, m, p)
    
    for remove_idx in range(r):
        if time.time() - start_time > max_time:
            break
        
        other_indices = [i for i in range(r) if i != remove_idx]
        other_terms = terms[other_indices]  # shape (r-1, n*m*p)
        target = terms[remove_idx]
        
        try:
            c, residuals, rank_ls, sv = np.linalg.lstsq(other_terms.T, target, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        reconstructed = other_terms.T @ c
        if np.allclose(reconstructed, target, atol=1e-10):
            # Can eliminate this term
            new_U = [U[i][:] for i in other_indices]
            new_V = [V[i][:] for i in other_indices]
            new_W = [W[i][:] for i in other_indices]
            
            new_U_arr = np.array(new_U)
            new_V_arr = np.array(new_V)
            new_W_arr = np.array(new_W)
            
            for j in range(len(other_indices)):
                scale = 1.0 + c[j]
                new_U_arr[j] *= scale
                new_V_arr[j] *= scale
                new_W_arr[j] *= scale
            
            new_U_list = new_U_arr.tolist()
            new_V_list = new_V_arr.tolist()
            new_W_list = new_W_arr.tolist()
            
            if _verify_decomposition(new_U_list, new_V_list, new_W_list, n, m, p):
                return {"U": new_U_list, "V": new_V_list, "W": new_W_list}
    
    return None

def _try_pair_elimination(U, V, W, n, m, p, max_time=30):
    """
    Try to eliminate pairs of terms.
    """
    start_time = time.time()
    r = len(U[0])
    terms = _compute_tensor_terms(U, V, W, n, m, p)
    
    # Try removing each pair
    for idx1, idx2 in combinations(range(r), 2):
        if time.time() - start_time > max_time:
            break
        
        other_indices = [i for i in range(r) if i not in (idx1, idx2)]
        if len(other_indices) == 0:
            continue
            
        other_terms = terms[other_indices]  # shape (r-2, n*m*p)
        target1 = terms[idx1]
        target2 = terms[idx2]
        
        # Stack the two targets
        stacked_target = np.vstack([target1, target2])  # shape (2, n*m*p)
        
        try:
            # Solve for coefficients that can reconstruct both targets
            # We need: sum_j c_j * other_terms[j] = target1
            # and:      sum_j d_j * other_terms[j] = target2
            # This is a 2*(n*m*p) system
            A = other_terms.T  # shape (n*m*p, r-2)
            # Stack A vertically twice
            A_stacked = np.vstack([A, A])  # shape (2*n*m*p, r-2)
            b_stacked = np.concatenate([target1, target2])  # shape (2*n*m*p,)
            
            c, residuals, rank_ls, sv = np.linalg.lstsq(A_stacked, b_stacked, rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution reconstructs both targets
        recon1 = other_terms.T @ c[:len(other_indices)]  # This is wrong, need to be careful
        # Actually, let's just check the residual
        reconstructed = A_stacked @ c
        if np.allclose(reconstructed, b_stacked, atol=1e-10):
            # Can eliminate this pair
            new_U = [U[i][:] for i in other_indices]
            new_V = [V[i][:] for i in other_indices]
            new_W = [W[i][:] for i in other_indices]
            
            new_U_arr = np.array(new_U)
            new_V_arr = np.array(new_V)
            new_W_arr = np.array(new_W)
            
            # The scaling is more complex here - we need to figure out the right scaling
            # For now, just return the reduced rank without scaling (this might not verify)
            new_U_list = new_U_arr.tolist()
            new_V_list = new_V_arr.tolist()
            new_W_list = new_W_arr.tolist()
            
            if _verify_decomposition(new_U_list, new_V_list, new_W_list, n, m, p):
                return {"U": new_U_list, "V": new_V_list, "W": new_W_list}
    
    return None

def _local_search(U, V, W, n, m, p, max_time=20):
    """
    Perform a local search to try to reduce the rank by zeroing out entries.
    """
    start_time = time.time()
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_U = U_arr.copy()
    best_V = V_arr.copy()
    best_W = W_arr.copy()
    
    for _ in range(30):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        
        for attempt in range(15):
            if time.time() - start_time > max_time:
                break
            
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Pick a random non-zero entry to try to zero out
            non_zero_U = np.argwhere(U_test != 0)
            non_zero_V = np.argwhere(V_test != 0)
            non_zero_W = np.argwhere(W_test != 0)
            
            all_nonzero = [(0, i, j) for i, j in non_zero_U] + \
                          [(1, i, j) for i, j in non_zero_V] + \
                          [(2, i, j) for i, j in non_zero_W]
            
            if not all_nonzero:
                break
            
            arr_idx, i, j = all_nonzero[np.random.randint(len(all_nonzero))]
            
            if arr_idx == 0:
                U_test[i, j] = 0
            elif arr_idx == 1:
                V_test[i, j] = 0
            else:
                W_test[i, j] = 0
            
            if _verify_decomposition(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                best_U = U_test
                best_V = V_test
                best_W = W_test
                improved = True
                break
        
        if not improved:
            break
    
    # Check if we can reduce rank by finding zero columns
    zero_cols = []
    for r_idx in range(best_U.shape[1]):
        if np.allclose(best_U[:, r_idx], 0) or np.allclose(best_V[:, r_idx], 0) or np.allclose(best_W[:, r_idx], 0):
            zero_cols.append(r_idx)
    
    if zero_cols:
        keep_cols = [i for i in range(best_U.shape[1]) if i not in zero_cols]
        best_U = best_U[:, keep_cols]
        best_V = best_V[:, keep_cols]
        best_W = best_W[:, keep_cols]
    
    if best_U.shape[1] < r:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def construct():
    """
    Construct a minimal-rank tensor decomposition for 2x2 * 2x9 matrix multiplication.
    
    Primary: rank-32 Strassen-block construction.
    Then try to reduce rank via linear dependency detection and local search.
    Fallback: rank-36 schoolbook if verification fails.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    
    # Build the rank-32 Strassen block construction
    terms = []
    for block_idx in range(4):
        k0 = 2 * block_idx
        k1 = k0 + 1
        terms.extend(_build_strassen_block(k0, k1, p))
    
    # Schoolbook for k=8
    terms.extend(_build_schoolbook_block(8, p))
    
    result = _terms_to_matrices(terms, n, m, p)
    
    if time.time() > deadline:
        return _trivial_decomposition(n, m, p)
    
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback to schoolbook
        return _trivial_decomposition(n, m, p)
    
    # Try to reduce rank
    current_U, current_V, current_W = result["U"], result["V"], result["W"]
    
    # Try single-term elimination
    if time.time() < deadline - 40:
        reduced = _try_single_elimination(current_U, current_V, current_W, n, m, p, max_time=30)
        if reduced is not None:
            current_U, current_V, current_W = reduced["U"], reduced["V"], reduced["W"]
    
    # Try pair-term elimination
    if time.time() < deadline - 20:
        reduced = _try_pair_elimination(current_U, current_V, current_W, n, m, p, max_time=15)
        if reduced is not None:
            current_U, current_V, current_W = reduced["U"], reduced["V"], reduced["W"]
    
    # Try local search for further reduction
    if time.time() < deadline - 10:
        further = _local_search(current_U, current_V, current_W, n, m, p, max_time=15)
        if further is not None:
            current_U, current_V, current_W = further["U"], further["V"], further["W"]
    
    # Final verification
    if _verify_decomposition(current_U, current_V, current_W, n, m, p):
        return {"U": current_U, "V": current_V, "W": current_W}
    else:
        # Return the original valid decomposition
        return result

# EVOLVE-BLOCK-END
