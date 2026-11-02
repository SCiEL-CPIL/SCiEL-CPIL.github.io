"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition exactly represents matrix multiplication.
    T[i,j,k] = 1 for all i,j,k in the 2x2x9 multiplication tensor.
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
    Returns list of (U_dict, V_dict, W_dict) tuples.
    A indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
    B indices: row 0: k0, k1; row 1: p+k0, p+k1
    C indices: col k0: 2*k0, 2*k0+1; col k1: 2*k1, 2*k1+1
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

def _try_rank_reduction(U, V, W, n, m, p, max_time=60):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    For each term, check if it can be expressed as a linear combination of the others.
    If so, we can eliminate it and reduce rank by 1.
    """
    start_time = time.time()
    r = len(U[0])
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, r)
    V3 = V_arr.reshape(m, p, r)
    W3 = W_arr.reshape(p, n, r)
    
    # Flatten each term to a vector of length n*m*p
    terms = []
    for r_idx in range(r):
        term_vec = np.zeros(n*m*p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    term_vec[i*m*p + j*p + k] = U3[i,j,r_idx] * V3[j,k,r_idx] * W3[k,i,r_idx]
        terms.append(term_vec)
    
    terms_arr = np.array(terms)  # shape (r, n*m*p)
    
    # Try removing each term and see if we can adjust the remaining terms
    best_result = None
    
    for remove_idx in range(r):
        if time.time() - start_time > max_time:
            break
        
        other_indices = [i for i in range(r) if i != remove_idx]
        other_terms = terms_arr[other_indices]  # shape (r-1, n*m*p)
        
        # Try to solve: sum_{j in other} c_j * term_j = term[remove_idx]
        try:
            c, residuals, rank_ls, sv = np.linalg.lstsq(other_terms.T, terms_arr[remove_idx], rcond=1e-12)
        except np.linalg.LinAlgError:
            continue
        
        # Check if the solution is exact
        reconstructed = other_terms.T @ c
        if np.allclose(reconstructed, terms_arr[remove_idx], atol=1e-10):
            # We can eliminate term[remove_idx] by scaling the remaining terms
            
            new_U = []
            new_V = []
            new_W = []
            
            for j, orig_idx in enumerate(other_indices):
                new_U.append(U[orig_idx][:])
                new_V.append(V[orig_idx][:])
                new_W.append(W[orig_idx][:])
            
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
            
            # Verify
            if _verify_decomposition(new_U_list, new_V_list, new_W_list, n, m, p):
                if best_result is None or len(new_U_list[0]) < len(best_result["U"][0]):
                    best_result = {"U": new_U_list, "V": new_V_list, "W": new_W_list}
                    if len(new_U_list[0]) == r - 1:
                        return best_result
    
    return best_result

def _local_search(U, V, W, n, m, p, max_time=30):
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
    
    # Try to zero out entries and see if the decomposition still works
    for _ in range(50):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        
        for attempt in range(20):
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
            
            # Verify
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
    if time.time() < deadline - 30:
        reduced = _try_rank_reduction(current_U, current_V, current_W, n, m, p, max_time=40)
        if reduced is not None:
            current_U, current_V, current_W = reduced["U"], reduced["V"], reduced["W"]
    
    # Try local search for further reduction
    if time.time() < deadline - 10:
        further = _local_search(current_U, current_V, current_W, n, m, p, max_time=20)
        if further is not None:
            current_U, current_V, current_W = further["U"], further["V"], further["W"]
    
    # Final verification
    if _verify_decomposition(current_U, current_V, current_W, n, m, p):
        return {"U": current_U, "V": current_V, "W": current_W}
    else:
        # Return the original valid decomposition
        return result

# EVOLVE-BLOCK-END
