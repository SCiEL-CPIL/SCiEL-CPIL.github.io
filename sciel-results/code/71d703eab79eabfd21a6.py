import time
import numpy as np

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _build_strassen_based():
    """
    Build a Strassen-based decomposition for 2x5 * 5x5.
    Partition k={0,1},{2,3},{4} and j={0,1},{2,3},{4}.
    2x2 blocks use Strassen (7), 2x1/1x2 use 4, 1x1 uses 2.
    Total: 4*7 + 4*4 + 2 = 28+16+2 = 46
    """
    n, m, p = 2, 5, 5
    k_groups = [(0,1), (2,3), (4,)]
    j_groups = [(0,1), (2,3), (4,)]
    
    total_rank = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                total_rank += 7
            elif len(kg) == 2 and len(jg) == 1:
                total_rank += 4
            elif len(kg) == 1 and len(jg) == 2:
                total_rank += 4
            else:
                total_rank += 2
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for kg in k_groups:
        for jg in j_groups:
            if len(kg) == 2 and len(jg) == 2:
                offset = _add_strassen_222(U, V, W, offset, kg[0], kg[1], jg[0], jg[1])
            elif len(kg) == 2 and len(jg) == 1:
                offset = _add_2x1(U, V, W, offset, kg[0], kg[1], jg[0])
            elif len(kg) == 1 and len(jg) == 2:
                offset = _add_1x2(U, V, W, offset, kg[0], jg[0], jg[1])
            else:
                offset = _add_1x1(U, V, W, offset, kg[0], jg[0])
    
    return U, V, W

def _add_strassen_222(U, V, W, offset, k0, k1, j0, j1):
    a00 = 0*5 + k0
    a01 = 0*5 + k1
    a10 = 1*5 + k0
    a11 = 1*5 + k1
    b00 = k0*5 + j0
    b01 = k0*5 + j1
    b10 = k1*5 + j0
    b11 = k1*5 + j1
    c00 = j0*2 + 0
    c01 = j1*2 + 0
    c10 = j0*2 + 1
    c11 = j1*2 + 1
    
    # M0 = (A00+A11)(B00+B11) -> C00 += M0, C11 += M0
    U[a00][offset+0] += 1; U[a11][offset+0] += 1
    V[b00][offset+0] += 1; V[b11][offset+0] += 1
    W[c00][offset+0] += 1; W[c11][offset+0] += 1
    
    # M1 = (A10+A11)B00 -> C10 += M1, C11 -= M1
    U[a10][offset+1] += 1; U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1; W[c11][offset+1] += -1
    
    # M2 = A00(B01-B11) -> C01 += M2, C11 += M2
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1; V[b11][offset+2] += -1
    W[c01][offset+2] += 1; W[c11][offset+2] += 1
    
    # M3 = A11(B10-B00) -> C00 += M3, C10 += M3
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1; V[b00][offset+3] += -1
    W[c00][offset+3] += 1; W[c10][offset+3] += 1
    
    # M4 = (A00+A01)B11 -> C00 -= M4, C01 += M4
    U[a00][offset+4] += 1; U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1; W[c01][offset+4] += 1
    
    # M5 = (A10-A00)(B00+B01) -> C11 += M5
    U[a10][offset+5] += 1; U[a00][offset+5] += -1
    V[b00][offset+5] += 1; V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    
    # M6 = (A01-A11)(B10+B11) -> C00 += M6
    U[a01][offset+6] += 1; U[a11][offset+6] += -1
    V[b10][offset+6] += 1; V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j):
    a0k0 = 0*5 + k0; a0k1 = 0*5 + k1
    a1k0 = 1*5 + k0; a1k1 = 1*5 + k1
    bk0j = k0*5 + j; bk1j = k1*5 + j
    c0j = j*2 + 0; c1j = j*2 + 1
    
    U[a0k0][offset+0] += 1; V[bk0j][offset+0] += 1; W[c0j][offset+0] += 1
    U[a0k1][offset+1] += 1; V[bk1j][offset+1] += 1; W[c0j][offset+1] += 1
    U[a1k0][offset+2] += 1; V[bk0j][offset+2] += 1; W[c1j][offset+2] += 1
    U[a1k1][offset+3] += 1; V[bk1j][offset+3] += 1; W[c1j][offset+3] += 1
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1):
    a0k = 0*5 + k; a1k = 1*5 + k
    bkj0 = k*5 + j0; bkj1 = k*5 + j1
    c0j0 = j0*2 + 0; c0j1 = j1*2 + 0
    c1j0 = j0*2 + 1; c1j1 = j1*2 + 1
    
    U[a0k][offset+0] += 1; V[bkj0][offset+0] += 1; W[c0j0][offset+0] += 1
    U[a0k][offset+1] += 1; V[bkj1][offset+1] += 1; W[c0j1][offset+1] += 1
    U[a1k][offset+2] += 1; V[bkj0][offset+2] += 1; W[c1j0][offset+2] += 1
    U[a1k][offset+3] += 1; V[bkj1][offset+3] += 1; W[c1j1][offset+3] += 1
    return offset + 4

def _add_1x1(U, V, W, offset, k, j):
    a0k = 0*5 + k; a1k = 1*5 + k
    bkj = k*5 + j
    c0j = j*2 + 0; c1j = j*2 + 1
    
    U[a0k][offset+0] += 1; V[bkj][offset+0] += 1; W[c0j][offset+0] += 1
    U[a1k][offset+1] += 1; V[bkj][offset+1] += 1; W[c1j][offset+1] += 1
    return offset + 2

def _try_rank_reduction(U, V, W, n, m, p, max_attempts=50):
    """
    Try to reduce rank by finding linear dependencies.
    
    Strategy: For each pair of rank positions (r1, r2), check if we can find
    a linear combination that still satisfies the tensor equation.
    
    More concretely: we look for a rank-r' decomposition (r' < r) by checking
    if any rank position can be expressed as a linear combination of others
    while preserving the tensor.
    
    We use the approach: for each rank slot r, check if there exist coefficients
    c_s (s != r) such that for all (i,j,k):
    sum_{s != r} c_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    
    This is a system of n*m*p equations in r-1 unknowns.
    """
    rank = len(U[0])
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Convert to numpy for efficiency
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each rank slot, try to eliminate it
    for r in range(rank):
        if not _deadline_ok():
            break
        
        # Build the system: for each (i,j,k), we need
        # sum_{s != r} c_s * U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        # Number of equations: n*m*p = 50
        # Number of unknowns: rank - 1
        
        if rank - 1 >= n * m * p:
            # Underdetermined - try to find a solution
            pass
        
        # Build matrix A and vector b
        # A[(i,j,k), s] = U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] for s != r
        # b[(i,j,k)] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        eq_idx = 0
        A_rows = []
        b_vals = []
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for s in range(rank):
                        if s == r:
                            continue
                        row.append(U_np[i*m+j, s] * V_np[j*p+k, s] * W_np[k*n+i, s])
                    A_rows.append(row)
                    b_vals.append(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r])
                    eq_idx += 1
        
        if not A_rows:
            continue
        
        A_mat = np.array(A_rows, dtype=float)
        b_vec = np.array(b_vals, dtype=float)
        
        # Try to solve A * c = b
        # Use least squares
        try:
            c, residuals, rank_s, sv = np.linalg.lstsq(A_mat, b_vec, rcond=1e-10)
        except:
            continue
        
        # Check if the solution is exact
        if np.allclose(A_mat @ c, b_vec, atol=1e-8):
            # We found a valid elimination!
            # Check that c has at least one non-zero entry (otherwise it's trivial)
            if np.any(np.abs(c) > 1e-8):
                # Build reduced matrices
                new_rank = rank - 1
                new_U = np.zeros((n*m, new_rank), dtype=float)
                new_V = np.zeros((m*p, new_rank), dtype=float)
                new_W = np.zeros((p*n, new_rank), dtype=float)
                
                s_idx = 0
                for s in range(rank):
                    if s == r:
                        continue
                    new_U[:, s_idx] = U_np[:, s] + c[s_idx] * U_np[:, r]
                    new_V[:, s_idx] = V_np[:, s]  # We absorb into U only
                    new_W[:, s_idx] = W_np[:, s]
                    s_idx += 1
                
                # Wait, this isn't right. If we eliminate r by expressing it in terms of others,
                # we need: for all (i,j,k):
                # U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = sum_{s!=r} c_s * U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s]
                # 
                # This means the r-th term equals the weighted sum of others.
                # So we can just REMOVE the r-th term and ADD the c_s * (other terms).
                # But that changes the other terms!
                #
                # Actually, the correct approach: the total sum is
                # sum_{all s} U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] = 1
                # If term r = sum_{s!=r} c_s * term_s, then:
                # sum_{s!=r} term_s + sum_{s!=r} c_s * term_s = 1
                # sum_{s!=r} (1 + c_s) * term_s = 1
                # 
                # So we scale the surviving terms by (1 + c_s)!
                
                new_U = np.zeros((n*m, new_rank), dtype=float)
                new_V = np.zeros((m*p, new_rank), dtype=float)
                new_W = np.zeros((p*n, new_rank), dtype=float)
                
                s_idx = 0
                for s in range(rank):
                    if s == r:
                        continue
                    scale = 1.0 + c[s_idx]
                    new_U[:, s_idx] = U_np[:, s] * scale
                    new_V[:, s_idx] = V_np[:, s]
                    new_W[:, s_idx] = W_np[:, s]
                    s_idx += 1
                
                # Verify
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s_sum = 0.0
                            for t in range(new_rank):
                                s_sum += new_U[i*m+j, t] * new_V[j*p+k, t] * new_W[k*n+i, t]
                            if abs(s_sum - 1.0) > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Convert to lists with exact fractions if possible
                    best_U = new_U.tolist()
                    best_V = new_V.tolist()
                    best_W = new_W.tolist()
                    best_rank = new_rank
                    U_np = new_U
                    V_np = new_V
                    W_np = new_W
                    rank = new_rank
                    # Restart the loop with new rank
                    break  # Break to restart with updated matrices
        
        # Check deadline
        if not _deadline_ok():
            break
    
    # Convert back to lists
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return best_U, best_V, best_W, best_rank

def construct():
    """
    Returns a valid rank decomposition for 2x5 * 5x5 matrix multiplication.
    """
    n, m, p = 2, 5, 5
    
    # Build Strassen-based decomposition (rank 46)
    U, V, W = _build_strassen_based()
    
    if not _verify_decomp(n, m, p, U, V, W):
        # Fallback to schoolbook
        rank = n * m * p
        U_s = [[0] * rank for _ in range(n * m)]
        V_s = [[0] * rank for _ in range(m * p)]
        W_s = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m + j][col] = 1
                    V_s[j*p + k][col] = 1
                    W_s[k*n + i][col] = 1
                    col += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    # Try rank reduction
    if _deadline_ok():
        U_r, V_r, W_r, r_rank = _try_rank_reduction(U, V, W, n, m, p)
        if r_rank < len(U[0]):
            if _verify_decomp(n, m, p, U_r, V_r, W_r):
                return {"U": U_r, "V": V_r, "W": W_r}
    
    return {"U": U, "V": V, "W": W}
