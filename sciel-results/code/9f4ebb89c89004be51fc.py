"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import itertools

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # full_tensor[u,v,w] = sum_r U[u,r]*V[v,r]*W[w,r]
    full_tensor = np.einsum('ur,vr,wr->uvw', U, V, W)
    
    # Build expected: for u=i*m+j, v=jp*p+k, w=kp*n+i2
    # valid iff i==i2 and j==jp and k==kp
    # i = u//m, j = u%m, jp = v//p, k = v%p, kp = w//n, i2 = w%n
    
    u_idx = np.arange(n * m)
    i_arr = u_idx // m
    j_arr = u_idx % m
    
    v_idx = np.arange(m * p)
    jp_arr = v_idx // p
    k_arr = v_idx % p
    
    w_idx = np.arange(p * n)
    kp_arr = w_idx // n
    i2_arr = w_idx % n
    
    # Build masks
    # i_arr shape (9,), jp_arr shape (9,), i2_arr shape (9,)
    # Need to check i_arr[u] == i2_arr[w] and j_arr[u] == jp_arr[v] and k_arr[v] == kp_arr[w]
    
    # Create 3D boolean mask
    # mask[u,v,w] = (i_arr[u]==i2_arr[w]) & (j_arr[u]==jp_arr[v]) & (k_arr[v]==kp_arr[w])
    
    mask_i = (i_arr[:, None, None] == i2_arr[None, None, :])  # (9,1,9)
    mask_j = (j_arr[:, None, None] == jp_arr[None, :, None])  # (9,9,1)
    mask_k = (k_arr[None, :, None] == kp_arr[None, None, :])  # (1,9,9)
    
    valid_mask = mask_i & mask_j & mask_k  # (9,9,9)
    
    expected = np.where(valid_mask, 1.0, 0.0)
    
    if np.max(np.abs(full_tensor - expected)) > 1e-9:
        return False
    return True

def _schoolbook():
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_rank26():
    """
    Rank-26 decomposition for 3x3 matrix multiplication.
    Strassen core (7 terms) for the 2x2 block + schoolbook border terms.
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    # M1 = (A00+A11)(B00+B11)
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    # M2 = (A10+A11)B00
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    # M3 = A00(B01-B11)
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M4 = A11(B10-B00)
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M5 = (A00+A01)B11
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    # M6 = (A10-A00)(B00+B01)
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    # M7 = (A01-A11)(B10+B11)
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] (6 terms) ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] (6 terms) ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] (3 terms) ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] (4 terms) ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(base_result, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition by removing linearly dependent terms.
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best = None
    best_rank = cur_rank
    
    # Build term tensors: each term r gives a 27-dim vector
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    # Precompute all term tensors
    all_terms = np.array([term_tensor(r) for r in range(cur_rank)])
    
    # Greedy removal: try removing each term, check if remaining still spans the identity
    removed = set()
    for _ in range(cur_rank - target_rank):
        if time.time() > deadline:
            break
        found = False
        for a in range(cur_rank):
            if a in removed:
                continue
            remaining = [x for x in range(cur_rank) if x != a and x not in removed]
            if len(remaining) < target_rank:
                continue
            
            # Check if sum of remaining term tensors equals the identity tensor
            identity_vec = np.zeros(27)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        identity_vec[i*9 + j*3 + k] = 1.0
            
            s = np.sum(all_terms[remaining], axis=0)
            if np.max(np.abs(s - identity_vec)) < 1e-9:
                removed.add(a)
                found = True
                break
        if not found:
            break
    
    if len(removed) > 0:
        remaining = [x for x in range(cur_rank) if x not in removed]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    return best, best_rank

def _search_rank_reductions(base_result, deadline):
    """
    Search for lower-rank decompositions by trying various strategies:
    1. Greedy term removal (vectorized)
    2. Linear combination search (try to merge pairs of terms)
    3. Systematic subset search for small rank reductions
    """
    import numpy as np
    
    n = m = p = 3
    U = np.array(base_result["U"], dtype=float)
    V = np.array(base_result["V"], dtype=float)
    W = np.array(base_result["W"], dtype=float)
    cur_rank = U.shape[1]
    
    best_result = None
    best_rank = cur_rank
    
    # Build identity tensor vector
    identity_vec = np.zeros(27)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                identity_vec[i*9 + j*3 + k] = 1.0
    
    # Precompute all term tensors
    def term_tensor(r):
        vec = np.zeros(27)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[i*9 + j*3 + k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        return vec
    
    all_terms = np.array([term_tensor(r) for r in range(cur_rank)])
    
    # Strategy 1: Greedy removal (vectorized)
    greedy_result, greedy_rank = _try_reduce_rank(base_result, cur_rank - 10, deadline)
    if greedy_result is not None and greedy_rank < best_rank:
        best_rank = greedy_rank
        best_result = greedy_result
    
    # Strategy 2: Check if any single term is linearly dependent on others
    removable = []
    for r in range(cur_rank):
        if time.time() > deadline:
            break
        others = [s for s in range(cur_rank) if s != r]
        A = np.column_stack([all_terms[s] for s in others])
        b = all_terms[r]
        try:
            sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-7:
                removable.append(r)
        except:
            pass
    
    if removable:
        remaining = [x for x in range(cur_rank) if x not in removable]
        new_rank = len(remaining)
        if new_rank < best_rank:
            best_rank = new_rank
            best_result = {
                "U": U[:, remaining].tolist(),
                "V": V[:, remaining].tolist(),
                "W": W[:, remaining].tolist()
            }
    
    # Strategy 3: Try all pairs of terms to see if one can be expressed as a combination
    # of the other plus a shift that preserves the identity
    if time.time() < deadline - 5:
        for a in range(cur_rank):
            if time.time() > deadline:
                break
            for b in range(a+1, cur_rank):
                if time.time() > deadline:
                    break
                # Try: can we remove term a and adjust term b?
                # Or: is term_a + alpha*term_b still in the span?
                # Simpler: check if removing both and adding their sum works
                # Actually, just check if term_a is in span of all others (already done)
                # Let's try: can we merge two terms into one?
                # If term_a = c * term_b for some scalar c, we could merge them
                # But that only works if they're proportional, which is rare
                pass
    
    return best_result, best_rank

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    # Start with the verified rank-26 construction
    result = _strassen_rank26()
    if not _verify(result["U"], result["V"], result["W"]):
        # Should not happen, but fall back to schoolbook
        return _schoolbook()
    
    # Try to reduce the rank
    if time.time() < deadline - 10:
        reduced, reduced_rank = _search_rank_reductions(result, deadline - 5)
        if reduced is not None:
            if _verify(reduced["U"], reduced["V"], reduced["W"]):
                result = reduced
    
    # Final verification
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook if something went wrong
    return _schoolbook()

# EVOLVE-BLOCK-END
