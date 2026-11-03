"""Rewritten program for matrix multiplication 3x5x7 rank minimization."""
import time
import itertools
import random
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Create schoolbook decomposition of rank n*m*p."""
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


def _strassen_222():
    """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7."""
    # Strassen's algorithm for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # 
    # The 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C[0,0] = p1 - p4 + p5 + p7
    # C[0,1] = p3 + p5
    # C[1,0] = p2 + p4
    # C[1,1] = p1 - p3 + p6 - p7
    #
    # U[i*m+j, col] where m=2, so U has 4 rows: (0,0),(0,1),(1,0),(1,1)
    # V[j*p+k, col] where p=2, so V has 4 rows: (0,0),(0,1),(1,0),(1,1)
    # W[k*n+i, col] where n=2, so W has 4 rows: (0,0),(0,1),(1,0),(1,1)
    
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Column 0: p1 = (a+d)(e+h)
    # A part: a + d => U[0,0]=1, U[3,0]=1
    # B part: e + h => V[0,0]=1, V[1,1]=1
    # C part: contributes to C[0,0] with +1, C[1,1] with +1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Column 1: p2 = (c+d)e
    # A part: c + d => U[2,1]=1, U[3,1]=1
    # B part: e => V[0,1]=1
    # C part: contributes to C[1,0] with +1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Column 2: p3 = a(f-h)
    # A part: a => U[0,2]=1
    # B part: f - h => V[1,2]=1, V[3,2]=-1
    # C part: contributes to C[0,1] with +1, C[1,1] with -1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = -1
    
    # Column 3: p4 = d(g-e)
    # A part: d => U[3,3]=1
    # B part: g - e => V[2,3]=1, V[0,3]=-1
    # C part: contributes to C[0,0] with -1, C[1,0] with +1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = -1; W[2][3] = 1
    
    # Column 4: p5 = (a+b)h
    # A part: a + b => U[0,4]=1, U[1,4]=1
    # B part: h => V[3,4]=1
    # C part: contributes to C[0,0] with +1, C[0,1] with +1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = 1; W[1][4] = 1
    
    # Column 5: p6 = (c-a)(e+f)
    # A part: c - a => U[2,5]=1, U[0,5]=-1
    # B part: e + f => V[0,5]=1, V[1,5]=1
    # C part: contributes to C[1,1] with +1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Column 6: p7 = (b-d)(g+h)
    # A part: b - d => U[1,6]=1, U[3,6]=-1
    # B part: g + h => V[2,6]=1, V[3,6]=1
    # C part: contributes to C[0,0] with +1, C[1,1] with -1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return U, V, W


def _concat(U1, V1, W1, U2, V2, W2):
    """Concatenate two decompositions (direct sum)."""
    r1 = len(U1[0]) if U1 else 0
    r2 = len(U2[0]) if U2 else 0
    rank = r1 + r2
    
    U = [row[:] + [0]*r2 for row in U1]
    U += [[0]*r1 + row[:] for row in U2]
    
    V = [row[:] + [0]*r2 for row in V1]
    V += [[0]*r1 + row[:] for row in V2]
    
    W = [row[:] + [0]*r2 for row in W1]
    W += [[0]*r1 + row[:] for row in W2]
    
    return U, V, W


def _embed(U_sub, V_sub, W_sub, n, m, p, i0, j0, k0):
    """Embed a sub-decomposition for sub-block starting at (i0, j0, k0) into full n,m,p space.
    
    U_sub has rows indexed by (i,j) relative to sub-block, size n_sub x m_sub
    V_sub has rows indexed by (j,k) relative to sub-block, size m_sub x p_sub
    W_sub has rows indexed by (k,i) relative to sub-block, size p_sub x n_sub
    
    Returns (U, V, W) with full sizes.
    """
    n_sub = len(U_sub) // (len(U_sub[0]) and 1)  # need to figure out sub-block size
    # Actually, let's pass sub-block dimensions explicitly
    pass


def _embed2(U_sub, V_sub, W_sub, n, m, p, ni, mi, pi, i0, j0, k0):
    """Embed a sub-decomposition for sub-block of size ni x mi x pi starting at (i0, j0, k0).
    
    U_sub: (ni*mi) x rank_sub
    V_sub: (mi*pi) x rank_sub  
    W_sub: (pi*ni) x rank_sub
    
    Returns full (n*m) x rank_sub, (m*p) x rank_sub, (p*n) x rank_sub
    """
    rs = len(U_sub[0])
    
    U = [[0]*rs for _ in range(n*m)]
    V = [[0]*rs for _ in range(m*p)]
    W = [[0]*rs for _ in range(p*n)]
    
    for ui in range(ni):
        for uj in range(mi):
            U[(i0+ui)*m + (j0+uj)] = U_sub[ui*mi + uj][:]
    
    for uj in range(mi):
        for uk in range(pi):
            V[(j0+uj)*p + (k0+uk)] = V_sub[uj*pi + uk][:]
    
    for uk in range(pi):
        for ui in range(ni):
            W[(k0+uk)*n + (i0+ui)] = W_sub[uk*ni + ui][:]
    
    return U, V, W


def _decompose(n, m, p):
    """Recursively decompose n x m x p matrix multiplication.
    
    Returns (U, V, W) with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    # Base cases
    if n == 1 or m == 1 or p == 1:
        return _schoolbook(n, m, p)
    
    if n == 2 and m == 2 and p == 2:
        return _strassen_222()
    
    # Try splitting strategies and pick the best
    best_rank = n * m * p
    best_result = None
    
    # Strategy 1: Split n into n1 + n2
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1 = _decompose(n1, m, p)
        U2, V2, W2 = _decompose(n2, m, p)
        
        # Embed both into full space
        Ue1, Ve1, We1 = _embed2(U1, V1, W1, n, m, p, n1, m, p, 0, 0, 0)
        Ue2, Ve2, We2 = _embed2(U2, V2, W2, n, m, p, n2, m, p, n1, 0, 0)
        
        U, V, W = _concat(Ue1, Ve1, We1, Ue2, Ve2, We2)
        rank = len(U[0])
        
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 2: Split m into m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1 = _decompose(n, m1, p)
        U2, V2, W2 = _decompose(n, m2, p)
        
        Ue1, Ve1, We1 = _embed2(U1, V1, W1, n, m, p, n, m1, p, 0, 0, 0)
        Ue2, Ve2, We2 = _embed2(U2, V2, W2, n, m, p, n, m2, p, 0, m1, 0)
        
        U, V, W = _concat(Ue1, Ve1, We1, Ue2, Ve2, We2)
        rank = len(U[0])
        
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    # Strategy 3: Split p into p1 + p2
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1 = _decompose(n, m, p1)
        U2, V2, W2 = _decompose(n, m, p2)
        
        Ue1, Ve1, We1 = _embed2(U1, V1, W1, n, m, p, n, m, p1, 0, 0, 0)
        Ue2, Ve2, We2 = _embed2(U2, V2, W2, n, m, p, n, m, p2, 0, 0, p1)
        
        U, V, W = _concat(Ue1, Ve1, We1, Ue2, Ve2, We2)
        rank = len(U[0])
        
        if rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    
    if best_result is None:
        return _schoolbook(n, m, p)
    
    return best_result


def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct.
    
    The matrix multiplication tensor T[i,j,k] = 1 for all valid (i,j,k)
    (since every product A[i,j]*B[j,k] contributes to C[i,k]).
    
    We check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all (i,j,k).
    """
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if s != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: schoolbook
    best_U, best_V, best_W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    try:
        U, V, W = _decompose(n, m, p)
        rank = len(U[0]) if U else 0
        
        # Verify correctness
        if _verify(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    except Exception:
        pass
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
