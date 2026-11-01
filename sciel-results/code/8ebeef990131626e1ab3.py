"""Matrix multiplication 2,2,11 - rank-minimal decomposition via structured search."""
import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def _build_target_tensor():
    """Build the multiplication tensor T of shape (n*m, m*p, p*n)."""
    n, m, p = 2, 2, 11
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify(U, V, W, T=None):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    if T is None:
        T = _build_target_tensor()
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _verify_fast(U, V, W, T):
    """Fast verification using precomputed tensor structure."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(T, T_decomp, atol=1e-9)

def _build_strassen_decomposition():
    """
    Build a decomposition using Strassen for 2x2 blocks.
    For 2x2 * 2x11: use Strassen on pairs of columns.
    
    11 columns = 5 pairs + 1 single column
    Each Strassen block (2x2 * 2x2) has rank 7
    Each single column (2x2 * 2x1) has rank 4 (schoolbook for that column)
    
    Total: 5*7 + 4 = 39
    """
    n, m, p = 2, 2, 11
    num_pairs = p // 2  # 5 pairs
    remainder = p % 2   # 1 remaining column
    
    # For the single column, we need 4 terms (a*e, b*g, c*e, d*g for 2x2 * 1x2... wait)
    # Actually for 2x2 * 2x1: C = A * B where B is 2x1
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> needs 2 terms
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  -> needs 2 terms
    # So rank 4 for the single column
    rank = num_pairs * 7 + remainder * 4
    U = np.zeros((n * m, rank), dtype=np.float64)  # (4, rank)
    V = np.zeros((m * p, rank), dtype=np.float64)  # (22, rank)
    W = np.zeros((p * n, rank), dtype=np.float64)  # (22, rank)
    
    # Strassen terms for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[a*e+b*g, a*f+b*h], [c*e+d*g, c*f+d*h]]
    # 
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    col = 0
    for b in range(num_pairs):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V indices: B[j, k] has index j*p + k
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = p + k0   # B[1, k0]
        v_h = p + k1   # B[1, k1]
        
        # W indices: C[i, k] has index k*n + i
        w_c00 = k0 * n + 0   # C[0, k0]
        w_c01 = k1 * n + 0   # C[0, k1]
        w_c10 = k0 * n + 1   # C[1, k0]
        w_c11 = k1 * n + 1   # C[1, k1]
        
        # U indices: A[i,j] has index i*m + j
        u_a = 0  # A[0,0]
        u_b = 1  # A[0,1]
        u_c = 2  # A[1,0]
        u_d = 3  # A[1,1]
        
        # m1 = (a+d)(e+h)
        U[u_a, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m2 = (c+d)e
        U[u_c, col] = 1.0; U[u_d, col] = 1.0
        V[v_e, col] = 1.0
        W[w_c10, col] = 1.0; W[w_c11, col] = -1.0
        col += 1
        
        # m3 = a(f-h)
        U[u_a, col] = 1.0
        V[v_f, col] = 1.0; V[v_h, col] = -1.0
        W[w_c01, col] = 1.0; W[w_c11, col] = 1.0
        col += 1
        
        # m4 = d(g-e)
        U[u_d, col] = 1.0
        V[v_g, col] = 1.0; V[v_e, col] = -1.0
        W[w_c00, col] = 1.0; W[w_c10, col] = 1.0
        col += 1
        
        # m5 = (a+b)h
        U[u_a, col] = 1.0; U[u_b, col] = 1.0
        V[v_h, col] = 1.0
        W[w_c00, col] = -1.0; W[w_c01, col] = 1.0
        col += 1
        
        # m6 = (c-a)(e+f)
        U[u_c, col] = 1.0; U[u_a, col] = -1.0
        V[v_e, col] = 1.0; V[v_f, col] = 1.0
        W[w_c11, col] = 1.0
        col += 1
        
        # m7 = (b-d)(g+h)
        U[u_b, col] = 1.0; U[u_d, col] = -1.0
        V[v_g, col] = 1.0; V[v_h, col] = 1.0
        W[w_c00, col] = 1.0
        col += 1
    
    # Handle remainder column (column 10)
    if remainder == 1:
        k2 = num_pairs * 2  # = 10
        
        v_e2 = k2        # B[0, 10]
        v_g2 = p + k2    # B[1, 10]
        w_c02 = k2 * n + 0   # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]
        
        # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
        # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
        
        # Term 1: A[0,0]*B[0,10]
        U[0, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 2: A[0,1]*B[1,10]
        U[1, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c02, col] = 1.0
        col += 1
        
        # Term 3: A[1,0]*B[0,10]
        U[2, col] = 1.0
        V[v_e2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
        
        # Term 4: A[1,1]*B[1,10]
        U[3, col] = 1.0
        V[v_g2, col] = 1.0
        W[w_c12, col] = 1.0
        col += 1
    
    return U, V, W

def _build_schoolbook():
    """Build the schoolbook rank-44 decomposition."""
    n, m, p = 2, 2, 11
    rank = n * m * p  # 44
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _try_rank_reduction(U, V, W, T, max_time=60):
    """
    Attempt to reduce rank by finding linearly dependent terms.
    Uses SVD on the tensor matricization to find a minimal basis,
    then greedy single- and double-column removal.
    """
    start_time = time.time()
    current_rank = U.shape[1]
    
    if current_rank <= 1:
        return U, V, W
    
    # Flatten each rank-1 tensor into a vector
    dim_a, dim_b, dim_c = U.shape[0], V.shape[0], W.shape[0]
    M = np.zeros((dim_a * dim_b * dim_c, current_rank), dtype=np.float64)
    for r in range(current_rank):
        M[:, r] = np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r]).flatten()
    
    # Use SVD to find the numerical rank
    try:
        U_svd, s_vals, Vt = np.linalg.svd(M, full_matrices=False)
        tol = max(M.shape) * np.finfo(float).eps * s_vals[0]
        num_rank = np.sum(s_vals > tol)
    except Exception:
        num_rank = current_rank
    
    if num_rank < current_rank:
        # Use SVD right singular vectors to find the best subset of columns
        # The top num_rank right singular vectors span the column space
        # We want to find num_rank original columns that span the same space
        # Strategy: greedily add columns that maximize the minimum singular value
        # of the selected subset
        try:
            Vt_top = Vt[:num_rank, :]  # (num_rank, current_rank)
            # Greedy selection: pick columns that maximize span
            selected = []
            remaining = list(range(current_rank))
            for _ in range(num_rank):
                best_col = -1
                best_score = -1
                for col in remaining:
                    # Score: norm of component of column in orthogonal complement of selected
                    if not selected:
                        score = np.linalg.norm(Vt_top[:, col])
                    else:
                        # Project column onto span of already selected
                        proj_matrix = Vt_top[:, selected]  # (num_rank, len(selected))
                        # Least squares to project
                        try:
                            coeffs, _, _, _ = np.linalg.lstsq(proj_matrix, Vt_top[:, col], rcond=None)
                            residual = Vt_top[:, col] - proj_matrix @ coeffs
                            score = np.linalg.norm(residual)
                        except Exception:
                            score = np.linalg.norm(Vt_top[:, col])
                    if score > best_score:
                        best_score = score
                        best_col = col
                if best_col >= 0:
                    selected.append(best_col)
                    remaining.remove(best_col)
                else:
                    break
            if len(selected) == num_rank:
                U_new = U[:, selected]
                V_new = V[:, selected]
                W_new = W[:, selected]
                if _verify(U_new, V_new, W_new, T) and U_new.shape[1] < current_rank:
                    return U_new, V_new, W_new
        except Exception:
            pass
    
    # Greedy: remove one column at a time, trying schoolbook columns first
    # Schoolbook columns are the last 4 (indices 35-38)
    # Try removing from the end first (schoolbook terms are more likely redundant)
    removal_order = list(range(current_rank - 1, -1, -1))
    for attempt in range(current_rank):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r in removal_order:
            if time.time() - start_time > max_time:
                break
            if r >= current_rank:
                continue
            mask = [i for i in range(current_rank) if i != r]
            U_test = U[:, mask]
            V_test = V[:, mask]
            W_test = W[:, mask]
            if _verify_fast(U_test, V_test, W_test, T):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                # Update removal order
                removal_order = [x if x < r else x - 1 for x in removal_order if x != r]
                removal_order.reverse()
                break
        if not removed:
            break
    
    # Try removing pairs of columns
    for attempt in range(current_rank // 2):
        if time.time() - start_time > max_time:
            break
        removed = False
        for r1 in range(current_rank - 1, -1, -1):
            if time.time() - start_time > max_time:
                break
            for r2 in range(r1 - 1, -1, -1):
                if time.time() - start_time > max_time:
                    break
                mask = [i for i in range(current_rank) if i != r1 and i != r2]
                U_test = U[:, mask]
                V_test = V[:, mask]
                W_test = W[:, mask]
                if _verify_fast(U_test, V_test, W_test, T):
                    U, V, W = U_test, V_test, W_test
                    current_rank -= 2
                    removed = True
                    break
            if removed:
                break
        if not removed:
            break
    
    return U, V, W

def _try_alternative_constructions(T, deadline):
    """Try different block structures to find lower rank."""
    n, m, p = 2, 2, 11
    
    # Try: 4 pairs + 3 single columns? No, that's worse.
    # Try: mix of different block sizes
    
    # Theoretical lower bound: 
    # For 2x2 * 2xp, the rank is at least... 
    # Actually, let's just try to see if we can do better than 39
    
    # One idea: can we share some terms between blocks?
    # The Strassen decomposition is already optimal for each 2x2 block.
    # The question is whether we can reduce the total.
    
    # For now, return None - we'll rely on the main construction
    return None

def _try_extended_block_decomposition():
    """
    Try to find a rank-10 decomposition for 2x2 * 2x3 (the last 3 columns).
    If found, total rank = 4*7 + 10 = 38 instead of 4*7 + 11 = 39.
    Uses a structured search over modified Strassen terms.
    """
    n, m, p = 2, 2, 11
    # Build target tensor for just the last 3 columns (indices 8,9,10)
    # U: (4, rank), V: (6, rank), W: (6, rank) for the 3-column subproblem
    # V indices: B[j,k] for k in {8,9,10} -> local index j*3 + (k-8)
    # W indices: C[i,k] for k in {8,9,10} -> local index (k-8)*2 + i
    
    # Target tensor for 2x2 * 2x3
    T3 = np.zeros((4, 6, 6), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i * 2 + j
                b = j * 3 + k
                c = k * 2 + i
                T3[a, b, c] = 1.0
    
    # Try rank 10: search for a valid decomposition
    # Start from Strassen(2 cols) + schoolbook(1 col) = rank 11
    # Then try to merge one term
    
    # Strassen terms for columns 0,1 (local):
    # m1=(a+d)(e+i), m2=(c+d)e, m3=a(f-h), m4=d(h-e), m5=(a+b)i, m6=(c-a)(e+f), m7=(b-d)(h+i)
    # Schoolbook for column 2: a*g, b*j, c*g, d*j
    
    # Try: can we express c*g using existing terms + 1 new term?
    # Or can we modify m6 to also help with column 2?
    
    # Alternative: use a different 7-term decomposition for the first 2 columns
    # that leaves room for the 3rd column
    
    # Let's try: what if we use a "generalized Strassen" for 3 columns?
    # Terms:
    # t1 = (a+d)(e+i)
    # t2 = (c+d)*e
    # t3 = a*(f-h)
    # t4 = d*(h-e)
    # t5 = (a+b)*i
    # t6 = (c-a)*(e+f)
    # t7 = (b-d)*(h+i)
    # t8 = a*g
    # t9 = b*j
    # t10 = (c+d)*j  -- this gives cj+dj, but we need cg+dj for C[1,2]
    
    # C[1,2] = c*g + d*j. We have t8=a*g, t9=b*j, t10=(c+d)*j=cj+dj.
    # We need cg+dj = (cg+dj). We have cj+dj from t10. We need cg-cj = c(g-j).
    # That would need another term. So rank 10 with this structure doesn't work.
    
    # Try different t10: t10 = c*g
    # Then C[1,2] = t10 + d*j. Need d*j as separate term -> rank 11.
    
    # Try: t8 = (a+c)*g, t9 = (b+d)*j
    # C[0,2] = a*g + b*j = (t8 - c*g) + (t9 - d*j) -- still need c*g and d*j
    # Not helpful.
    
    # Try: modify Strassen to incorporate column 2
    # t1 = (a+d)(e+i+g)
    # t2 = (c+d)*e
    # t3 = a*(f-h)
    # t4 = d*(h-e)
    # t5 = (a+b)*i
    # t6 = (c-a)*(e+f)
    # t7 = (b-d)*(h+i)
    # t8 = a*g
    # t9 = b*j
    # t10 = d*j
    
    # Check C[0,0] = m1+m4-m5+m7 with m1=(a+d)(e+i+g):
    # = (a+d)(e+i+g) + d(h-e) - (a+b)i + (b-d)(h+i)
    # = ae+ai+ag + de+di+dg + dh-de - ai-bi + bh+bi-dh-di
    # = ae + ag + dg + bh - bi + bi ... let me redo this carefully
    # = ae + ai + ag + de + di + dg + dh - de - ai - bi + bh + bi - dh - di
    # = ae + ag + dg + bh
    # But we need C[0,0] = ae + bh. Extra terms: ag + dg. NOT CORRECT.
    
    # This approach doesn't work because modifying m1 introduces unwanted cross terms.
    
    # I think rank 10 for 2x2*2x3 is not achievable with simple modifications.
    # Return None to indicate failure.
    return None

def construct():
    start_time = time.time()
    DEADLINE = 145  # seconds, leave margin before 150s limit
    
    T = _build_target_tensor()
    
    # Try the Strassen-based construction (rank 39)
    U, V, W = _build_strassen_decomposition()
    
    if _verify(U, V, W, T):
        # Try rank reduction
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed - 3
        if remaining_time > 5:
            U_red, V_red, W_red = _try_rank_reduction(U, V, W, T, max_time=min(remaining_time, 100))
            if U_red.shape[1] < U.shape[1]:
                return {'U': U_red.tolist(), 'V': V_red.tolist(), 'W': W_red.tolist()}
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Fall back to schoolbook rank 44
    U, V, W = _build_schoolbook()
    if _verify(U, V, W, T):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    # Should not reach here
    raise RuntimeError("No valid decomposition found")
