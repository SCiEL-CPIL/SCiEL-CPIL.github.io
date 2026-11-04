import numpy as np
import time
from fractions import Fraction

def _strassen_2x2x2():
    U2 = np.zeros((4, 7)); V2 = np.zeros((4, 7)); W2 = np.zeros((4, 7))
    U2[0,0]=1; U2[3,0]=1; V2[0,0]=1; V2[3,0]=1; W2[0,0]=1; W2[3,0]=1
    U2[2,1]=1; U2[3,1]=1; V2[0,1]=1; W2[2,1]=1; W2[3,1]=-1
    U2[0,2]=1; V2[1,2]=1; V2[3,2]=-1; W2[1,2]=1; W2[3,2]=1
    U2[3,3]=1; V2[2,3]=1; V2[0,3]=-1; W2[0,3]=1; W2[2,3]=1
    U2[0,4]=1; U2[1,4]=1; V2[3,4]=1; W2[0,4]=-1; W2[1,4]=1
    U2[2,5]=1; U2[0,5]=-1; V2[0,5]=1; V2[1,5]=1; W2[3,5]=1
    U2[1,6]=1; U2[3,6]=-1; V2[2,6]=1; V2[3,6]=1; W2[0,6]=1
    return U2, V2, W2

def _build(rank=98):
    n, m, p = 4, 4, 8
    U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
    U2, V2, W2 = _strassen_2x2x2()
    A_specs = [
        [(1,0),(1,3)], [(1,2),(1,3)], [(1,0)], [(1,3)],
        [(1,0),(1,1)], [(1,2),(-1,0)], [(1,1),(-1,3)],
    ]
    B_specs = [
        [(1,0),(1,3)], [(1,0)], [(1,1),(-1,3)], [(1,2),(-1,0)],
        [(1,3)], [(1,0),(1,1)], [(1,2),(1,3)],
    ]
    C_specs = [
        [(1,0),(1,3)], [(1,2),(-1,3)], [(1,1),(1,3)], [(1,0),(1,2)],
        [(-1,0),(1,1)], [(1,3)], [(1,0)],
    ]
    for half in range(2):
        for kp in range(7):
            for ms in range(7):
                col = half*49 + kp*7 + ms
                for ca, bi in A_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c
                            U[gi*4+gj, col] += ca * U2[r*2+c, ms]
                for cb, bi in B_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            V[gi*8+gj, col] += cb * V2[r*2+c, ms]
                for cc, bi in C_specs[kp]:
                    bri, bcj = divmod(bi, 2)
                    for r in range(2):
                        for c in range(2):
                            gi = bri*2+r; gj = bcj*2+c + half*4
                            W[gj*4+gi, col] += cc * W2[r*2+c, ms]
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float); W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for k in range(p):
            for j in range(m):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_M(U, V, W, n, m, p):
    rank = U.shape[1]
    N = n*p*m
    M = np.zeros((N, rank))
    idx = 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                M[idx, :] = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                idx += 1
    return M

def _exact_null_space(M_int, max_dim=50):
    """Find null space dimension of integer matrix M_int using exact Gaussian elimination.
    Returns list of null space basis vectors as Fraction lists, or None if too large."""
    rows = len(M_int)
    cols = len(M_int[0]) if rows > 0 else 0
    if cols <= max_dim:
        # Work with the transpose if cols > rows to reduce work
        pass
    
    # Build augmented system: find basis for null space of M
    # Use fraction-based Gaussian elimination
    # M is rows x cols, we want vectors x in F^cols such that Mx = 0
    
    if rows == 0:
        return None
    
    # Make a copy as list of lists of Fractions
    A = [[Fraction(M_int[i][j]) for j in range(cols)] for i in range(rows)]
    
    rank = 0
    pivot_cols = []
    
    for c in range(cols):
        # Find pivot
        pivot_row = -1
        for r in range(rank, rows):
            if A[r][c] != 0:
                pivot_row = r
                break
        if pivot_row == -1:
            continue
        
        # Swap
        A[rank], A[pivot_row] = A[pivot_row], A[rank]
        pivot_cols.append(c)
        
        # Eliminate below
        pivot_val = A[rank][c]
        for r in range(rank + 1, rows):
            if A[r][c] != 0:
                factor = A[r][c] / pivot_val
                for cc in range(c, cols):
                    A[r][cc] -= factor * A[rank][cc]
        rank += 1
        if rank == rows:
            break
    
    # Determine free columns
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(cols) if c not in pivot_set]
    
    if not free_cols:
        return []  # No null space
    
    # Build null space basis: for each free column, set it to 1, solve for pivot columns
    null_basis = []
    for fc in free_cols:
        vec = [Fraction(0)] * cols
        vec[fc] = Fraction(1)
        # Back-substitute: for each pivot row, solve for pivot column
        for r in range(rank - 1, -1, -1):
            pc = pivot_cols[r]
            # A[r][pc] * vec[pc] = -sum_{c>pc} A[r][c] * vec[c]
            s = Fraction(0)
            for c in range(pc + 1, cols):
                if A[r][c] != 0 and vec[c] != 0:
                    s += A[r][c] * vec[c]
            if A[r][pc] != 0:
                vec[pc] = -s / A[r][pc]
        null_basis.append(vec)
    
    return null_basis

def _try_reduce_exact(U, V, W, n, m, p, deadline, max_rounds=50):
    """Use exact rational arithmetic to find and eliminate null space columns."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 20.0:
            break
        if R <= 1:
            break
        
        # Build M with exact integers (multiply by 2 to handle half-integers)
        M_float = _build_M(U, V, W, n, m, p)
        # Convert to integers: multiply by 2
        M_int = (M_float * 2).round().astype(int)
        
        # Verify M is truly integer
        if not np.allclose(M_float * 2, M_int, atol=1e-6):
            # Try to find a common scaling
            break
        
        # Find null space exactly
        null_basis = _exact_null_space(M_int.tolist(), max_dim=120)
        
        if null_basis is None or len(null_basis) == 0:
            break
        
        # For each null vector, try to eliminate a column
        reduced_any = False
        for null_vec in null_basis:
            if time.time() > deadline - 15.0:
                break
            
            c = np.array([float(x) for x in null_vec])
            
            # Find pivot column
            pivot_candidates = np.argsort(-np.abs(c))[:min(5, R)]
            
            for r0 in pivot_candidates:
                if abs(c[r0]) < 1e-12:
                    continue
                
                keep = [q for q in range(R) if q != r0]
                scales = []
                valid = True
                for r in keep:
                    factor = 1.0 + c[r] / c[r0]
                    if abs(factor) < 1e-14:
                        valid = False
                        break
                    if factor >= 0:
                        s_val = factor ** (1.0/3.0)
                    else:
                        s_val = -((-factor) ** (1.0/3.0))
                    if not np.isfinite(s_val):
                        valid = False
                        break
                    scales.append(s_val)
                
                if not valid:
                    continue
                
                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                for i, r in enumerate(keep):
                    new_U[:, i] *= scales[i]
                    new_V[:, i] *= scales[i]
                    new_W[:, i] *= scales[i]
                
                if _verify(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    reduced_any = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if reduced_any:
                break
        
        if not reduced_any:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_reduce_svd(U, V, W, n, m, p, deadline, max_rounds=200):
    """SVD-based single-column reduction (fast path)."""
    U = U.astype(float).copy(); V = V.astype(float).copy(); W = W.astype(float).copy()
    R = U.shape[1]
    best_rank = R
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 25.0:
            break
        if R <= 1:
            break
        
        M = _build_M(U, V, W, n, m, p)
        
        try:
            _, s, Vt = np.linalg.svd(M, full_matrices=True)
        except np.linalg.LinAlgError:
            break
        
        tol = max(1e-8, 1e-6 * max(s) if len(s) > 0 else 1e-8)
        null_idx = np.where(s < tol)[0]
        if len(null_idx) == 0:
            tol2 = max(tol * 10, 1e-5)
            null_idx = np.where(s < tol2)[0]
            if len(null_idx) == 0:
                break
        
        reduced = False
        for ni in null_idx:
            if time.time() > deadline - 20.0:
                break
            c = Vt[ni, :]
            
            pivot_candidates = np.argsort(-np.abs(c))[:min(5, R)]
            
            for r0 in pivot_candidates:
                if abs(c[r0]) < 1e-10:
                    continue
                
                keep = [q for q in range(R) if q != r0]
                scales = []
                valid = True
                for r in keep:
                    factor = 1.0 + c[r] / c[r0]
                    if abs(factor) < 1e-14:
                        valid = False
                        break
                    if factor >= 0:
                        s_val = factor ** (1.0/3.0)
                    else:
                        s_val = -((-factor) ** (1.0/3.0))
                    if not np.isfinite(s_val):
                        valid = False
                        break
                    scales.append(s_val)
                
                if not valid:
                    continue
                
                new_U = U[:, keep].copy()
                new_V = V[:, keep].copy()
                new_W = W[:, keep].copy()
                for i, r in enumerate(keep):
                    new_U[:, i] *= scales[i]
                    new_V[:, i] *= scales[i]
                    new_W[:, i] *= scales[i]
                
                if _verify(new_U, new_V, new_W, n, m, p, tol=1e-4):
                    U, V, W = new_U, new_V, new_W
                    R -= 1
                    reduced = True
                    if R < best_rank:
                        best_rank = R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    break
            
            if reduced:
                break
        
        if not reduced:
            break
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 4, 4, 8
    start = time.time()
    deadline = start + 145.0
    
    U, V, W = _build(98)
    if not _verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Phase 1: Fast SVD-based reduction
    U, V, W, rank1 = _try_reduce_svd(U, V, W, n, m, p, deadline)
    
    # Phase 2: Exact rational reduction for remaining null space
    if time.time() < deadline - 30.0 and rank1 < 98:
        U, V, W, rank2 = _try_reduce_exact(U, V, W, n, m, p, deadline)
    else:
        rank2 = rank1
    
    # Final verification
    if not _verify(U, V, W, n, m, p, tol=1e-4):
        U, V, W = _build(98)
        if not _verify(U, V, W, n, m, p):
            rank = n * m * p
            U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i*m*p + j*p + k
                        U[i*m+j, idx] = 1
                        V[j*p+k, idx] = 1
                        W[k*n+i, idx] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
