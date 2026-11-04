import time
import numpy as np
from fractions import Fraction

_T0 = time.time()
_DEADLINE_S = 110.0

def _deadline_ok():
    return (time.time() - _T0) < _DEADLINE_S

def _verify_decomp(n, m, p, U, V, W):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                if s != 1:
                    return False
    return True

def _schoolbook(n=2, m=5, p=5):
    rank = n * m * p
    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
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
    
    U[a00][offset+0] += 1
    U[a11][offset+0] += 1
    V[b00][offset+0] += 1
    V[b11][offset+0] += 1
    W[c00][offset+0] += 1
    W[c11][offset+0] += 1
    
    U[a10][offset+1] += 1
    U[a11][offset+1] += 1
    V[b00][offset+1] += 1
    W[c10][offset+1] += 1
    W[c11][offset+1] += -1
    
    U[a00][offset+2] += 1
    V[b01][offset+2] += 1
    V[b11][offset+2] += -1
    W[c01][offset+2] += 1
    W[c11][offset+2] += 1
    
    U[a11][offset+3] += 1
    V[b10][offset+3] += 1
    V[b00][offset+3] += -1
    W[c00][offset+3] += 1
    W[c10][offset+3] += 1
    
    U[a00][offset+4] += 1
    U[a01][offset+4] += 1
    V[b11][offset+4] += 1
    W[c00][offset+4] += -1
    W[c01][offset+4] += 1
    
    U[a10][offset+5] += 1
    U[a00][offset+5] += -1
    V[b00][offset+5] += 1
    V[b01][offset+5] += 1
    W[c11][offset+5] += 1
    
    U[a01][offset+6] += 1
    U[a11][offset+6] += -1
    V[b10][offset+6] += 1
    V[b11][offset+6] += 1
    W[c00][offset+6] += 1
    
    return offset + 7

def _add_2x1(U, V, W, offset, k0, k1, j):
    a0k0 = 0*5 + k0
    a0k1 = 0*5 + k1
    a1k0 = 1*5 + k0
    a1k1 = 1*5 + k1
    
    bk0j = k0*5 + j
    bk1j = k1*5 + j
    
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k0][offset+0] += 1
    V[bk0j][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a0k1][offset+1] += 1
    V[bk1j][offset+1] += 1
    W[c0j][offset+1] += 1
    
    U[a1k0][offset+2] += 1
    V[bk0j][offset+2] += 1
    W[c1j][offset+2] += 1
    
    U[a1k1][offset+3] += 1
    V[bk1j][offset+3] += 1
    W[c1j][offset+3] += 1
    
    return offset + 4

def _add_1x2(U, V, W, offset, k, j0, j1):
    a0k = 0*5 + k
    a1k = 1*5 + k
    
    bkj0 = k*5 + j0
    bkj1 = k*5 + j1
    
    c0j0 = j0*2 + 0
    c0j1 = j1*2 + 0
    c1j0 = j0*2 + 1
    c1j1 = j1*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj0][offset+0] += 1
    W[c0j0][offset+0] += 1
    
    U[a0k][offset+1] += 1
    V[bkj1][offset+1] += 1
    W[c0j1][offset+1] += 1
    
    U[a1k][offset+2] += 1
    V[bkj0][offset+2] += 1
    W[c1j0][offset+2] += 1
    
    U[a1k][offset+3] += 1
    V[bkj1][offset+3] += 1
    W[c1j1][offset+3] += 1
    
    return offset + 4

def _add_1x1(U, V, W, offset, k, j):
    a0k = 0*5 + k
    a1k = 1*5 + k
    bkj = k*5 + j
    c0j = j*2 + 0
    c1j = j*2 + 1
    
    U[a0k][offset+0] += 1
    V[bkj][offset+0] += 1
    W[c0j][offset+0] += 1
    
    U[a1k][offset+1] += 1
    V[bkj][offset+1] += 1
    W[c1j][offset+1] += 1
    
    return offset + 2

def _build_strassen_based():
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
    
    U = [[Fraction(0)] * total_rank for _ in range(n * m)]
    V = [[Fraction(0)] * total_rank for _ in range(m * p)]
    W = [[Fraction(0)] * total_rank for _ in range(p * n)]
    
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

def _try_reduce_rank(U, V, W):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    This is a heuristic approach: we look for pairs of terms that can be combined
    or for terms that can be eliminated.
    """
    n, m, p = 2, 5, 5
    rank = len(U[0])
    
    # Convert to numpy for linear algebra
    U_arr = np.array([[float(x) for x in row] for row in U])
    V_arr = np.array([[float(x) for x in row] for row in V])
    W_arr = np.array([[float(x) for x in row] for row in W])
    
    # Try to find if any column in U is a linear combination of others
    # For each column r, check if it's in the span of the others
    for r in range(rank - 1, -1, -1):
        if not _deadline_ok():
            break
        
        # Check if column r is in span of columns 0..r-1 and r+1..rank-1
        if r == 0:
            # Can't check if first column is in span of empty set
            continue
        
        # Build matrix of all columns except r
        cols = list(range(r)) + list(range(r + 1, rank))
        if not cols:
            continue
        
        U_others = U_arr[:, cols]
        U_r = U_arr[:, r]
        
        # Check if U_r is in column space of U_others
        # Use SVD for numerical stability
        try:
            U_svd, s, Vt = np.linalg.svd(U_others, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        
        # Rank of U_others
        tol = 1e-10
        rank_others = np.sum(s > tol)
        
        if rank_others < len(cols):
            # The others are linearly dependent, but we need to check if U_r adds rank
            # Project U_r onto column space of U_others
            # If residual is ~0, then U_r is in span
            if rank_others > 0:
                proj = U_svd[:, :rank_others] @ (U_svd[:, :rank_others].T @ U_r)
                residual = np.linalg.norm(U_r - proj)
                if residual < 1e-8:
                    # U_r is in span of others, can potentially eliminate
                    # But we need to check if the same holds for V and W
                    # This is complex, so we skip for now
                    pass
    
    return U, V, W, rank

def construct():
    n, m, p = 2, 5, 5
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    if not _deadline_ok():
        return {"U": U_school, "V": V_school, "W": W_school}
    
    U, V, W = _build_strassen_based()
    
    if _verify_decomp(n, m, p, U, V, W):
        # Try to reduce rank further
        U_red, V_red, W_red, rank_red = _try_reduce_rank(U, V, W)
        if rank_red < len(U[0]) and _verify_decomp(n, m, p, U_red, V_red, W_red):
            return {"U": U_red, "V": V_red, "W": W_red}
        return {"U": U, "V": V, "W": W}
    
    return {"U": U_school, "V": V_school, "W": W_school}
