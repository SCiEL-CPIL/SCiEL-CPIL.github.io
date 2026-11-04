import sys
sys.setrecursionlimit(10000)

from functools import lru_cache

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START

# Helper to create zero matrices
def zero_mat(rows, cols):
    return [[0] * cols for _ in range(rows)]

# Base case: 1x1x1
def get_base_111():
    return [[[1]], [[1]], [[1]]]

# Base case: 1xm x p
def get_base_1mp(m, p):
    rank = m * p
    U = zero_mat(1, rank)
    V = zero_mat(m * p, rank)
    W = zero_mat(p, rank)
    for j in range(m):
        for k in range(p):
            r = j * p + k
            U[0][r] = 1
            V[j * p + k][r] = 1
            W[k][r] = 1
    return [U, V, W]

# Base case: nx1 x p
def get_base_n1p(n, p):
    rank = n * p
    U = zero_mat(n, rank)
    V = zero_mat(p, rank)
    W = zero_mat(p * n, rank)
    for i in range(n):
        for k in range(p):
            r = i * p + k
            U[i][r] = 1
            V[k][r] = 1
            W[k * n + i][r] = 1
    return [U, V, W]

# Base case: nxm x 1
def get_base_nm1(n, m):
    rank = n * m
    U = zero_mat(n * m, rank)
    V = zero_mat(m, rank)
    W = zero_mat(n, rank)
    for i in range(n):
        for j in range(m):
            r = i * m + j
            U[i * m + j][r] = 1
            V[j][r] = 1
            W[i][r] = 1
    return [U, V, W]

# Strassen 2x2x2
def get_strassen_222():
    rank = 7
    U = zero_mat(4, rank)
    V = zero_mat(4, rank)
    W = zero_mat(4, rank)
    
    # Indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # M3 = A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = -1
    
    return [U, V, W]

@lru_cache(maxsize=None)
def min_rank(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return 1
    if n == 2 and m == 2 and p == 2:
        return 7
    if n == 1:
        return m * p
    if m == 1:
        return n * p
    if p == 1:
        return n * m
    
    best = n * m * p
    # Split n
    for n1 in range(1, n):
        n2 = n - n1
        r = min_rank(n1, m, p) + min_rank(n2, m, p)
        if r < best: best = r
    # Split m
    for m1 in range(1, m):
        m2 = m - m1
        r = min_rank(n, m1, p) + min_rank(n, m2, p)
        if r < best: best = r
    # Split p
    for p1 in range(1, p):
        p2 = p - p1
        r = min_rank(n, m, p1) + min_rank(n, m, p2)
        if r < best: best = r
        
    return best

def build_decomp(n, m, p):
    if n == 1 and m == 1 and p == 1:
        return get_base_111()
    if n == 2 and m == 2 and p == 2:
        return get_strassen_222()
    if n == 1:
        return get_base_1mp(m, p)
    if m == 1:
        return get_base_n1p(n, p)
    if p == 1:
        return get_base_nm1(n, m)
        
    best_rank = min_rank(n, m, p)
    best_split = None
    best_dim = -1
    
    # Check splits to find the one that achieves best_rank
    for n1 in range(1, n):
        n2 = n - n1
        r = min_rank(n1, m, p) + min_rank(n2, m, p)
        if r == best_rank:
            best_split = (n1, n2)
            best_dim = 0
            break
    if best_dim == 0:
        U1, V1, W1 = build_decomp(n1, m, p)
        U2, V2, W2 = build_decomp(n2, m, p)
        
        U = zero_mat(n * m, best_rank)
        V = zero_mat(m * p, best_rank)
        W = zero_mat(p * n, best_rank)
        
        rank1 = min_rank(n1, m, p)
        rank2 = min_rank(n2, m, p)
        
        # Copy U1 to top, U2 to bottom
        for i in range(n1 * m):
            for c in range(rank1):
                U[i][c] = U1[i][c]
        for i in range(n2 * m):
            for c in range(rank2):
                U[n1 * m + i][rank1 + c] = U2[i][c]
                
        # Concatenate V
        for i in range(m * p):
            for c in range(rank1):
                V[i][c] = V1[i][c]
            for c in range(rank2):
                V[i][rank1 + c] = V2[i][c]
                
        # Interleave W
        for k in range(p):
            for i in range(n1):
                for c in range(rank1):
                    W[k * n + i][c] = W1[k * n1 + i][c]
            for i in range(n2):
                for c in range(rank2):
                    W[k * n + n1 + i][rank1 + c] = W2[k * n2 + i][c]
                    
        return [U, V, W]

    for m1 in range(1, m):
        m2 = m - m1
        r = min_rank(n, m1, p) + min_rank(n, m2, p)
        if r == best_rank:
            best_split = (m1, m2)
            best_dim = 1
            break
    if best_dim == 1:
        U1, V1, W1 = build_decomp(n, m1, p)
        U2, V2, W2 = build_decomp(n, m2, p)
        
        U = zero_mat(n * m, best_rank)
        V = zero_mat(m * p, best_rank)
        W = zero_mat(p * n, best_rank)
        
        rank1 = min_rank(n, m1, p)
        rank2 = min_rank(n, m2, p)
        
        # Concatenate U
        for i in range(n * m1):
            for c in range(rank1):
                U[i][c] = U1[i][c]
        for i in range(n * m2):
            for c in range(rank2):
                U[n * m1 + i][rank1 + c] = U2[i][c]
                
        # Interleave V
        for j in range(m1):
            for k in range(p):
                for c in range(rank1):
                    V[j * p + k][c] = V1[j * p + k][c]
        for j in range(m2):
            for k in range(p):
                for c in range(rank2):
                    V[(m1 + j) * p + k][rank1 + c] = V2[j * p + k][c]
                    
        # Concatenate W
        for i in range(p * n):
            for c in range(rank1):
                W[i][c] = W1[i][c]
            for c in range(rank2):
                W[i][rank1 + c] = W2[i][c]
                
        return [U, V, W]

    for p1 in range(1, p):
        p2 = p - p1
        r = min_rank(n, m, p1) + min_rank(n, m, p2)
        if r == best_rank:
            best_split = (p1, p2)
            best_dim = 2
            break
    if best_dim == 2:
        U1, V1, W1 = build_decomp(n, m, p1)
        U2, V2, W2 = build_decomp(n, m, p2)
        
        U = zero_mat(n * m, best_rank)
        V = zero_mat(m * p, best_rank)
        W = zero_mat(p * n, best_rank)
        
        rank1 = min_rank(n, m, p1)
        rank2 = min_rank(n, m, p2)
        
        # Concatenate U
        for i in range(n * m):
            for c in range(rank1):
                U[i][c] = U1[i][c]
            for c in range(rank2):
                U[i][rank1 + c] = U2[i][c]
                
        # Interleave V
        for j in range(m):
            for k in range(p1):
                for c in range(rank1):
                    V[j * p + k][c] = V1[j * p1 + k][c]
            for k in range(p2):
                for c in range(rank2):
                    V[j * p + p1 + k][rank1 + c] = V2[j * p2 + k][c]
                    
        # Concatenate W
        for i in range(p1 * n):
            for c in range(rank1):
                W[i][c] = W1[i][c]
        for i in range(p2 * n):
            for c in range(rank2):
                W[p1 * n + i][rank1 + c] = W2[i][c]
                
        return [U, V, W]
        
    # Fallback (should not happen)
    return build_decomp(n, m, p) # This would recurse infinitely if not careful, but min_rank ensures a split exists

def construct():
    n, m, p = DIMENSIONS
    U, V, W = build_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
