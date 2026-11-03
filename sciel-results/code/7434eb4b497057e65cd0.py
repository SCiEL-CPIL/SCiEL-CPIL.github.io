import numpy as np

DIMENSIONS = [3, 5, 5]

def _trivial_decomp(n, m, p):
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
    return U, V, W

def _strassen_decomp():
    # n=2, m=2, p=2
    # Indices for A (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for B (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices for C (2x2): 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W index is k*n + i.
    # C[0,0] -> 0
    # C[0,1] -> 1
    # C[1,0] -> 2
    # C[1,1] -> 3
    
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B10 - B11)
    # M4 = A11 * (B01 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # M5
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # M6
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _combine_decomps(decomps, n, m, p, split_dim, split_indices):
    # decomps: list of (U, V, W, n_sub, m_sub, p_sub)
    # split_dim: 0 for n, 1 for m, 2 for p
    # split_indices: list of (start, size)
    
    total_rank = sum(d[0].__len__() if isinstance(d[0], list) else d[0].shape[1] for d in decomps)
    # Actually, rank is number of columns.
    total_rank = 0
    for d in decomps:
        U_sub, V_sub, W_sub, *_ = d
        total_rank += len(U_sub[0]) if U_sub else 0

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, (U_sub, V_sub, W_sub, n_sub, m_sub, p_sub) in enumerate(decomps):
        start, size = split_indices[idx]
        rank_sub = len(U_sub[0])
        
        for r in range(rank_sub):
            c = col_offset + r
            # Map U
            for i in range(n_sub):
                for j in range(m_sub):
                    val = U_sub[i * m_sub + j][r]
                    if val != 0:
                        if split_dim == 0:
                            i_global = start + i
                            j_global = j
                            U[i_global * m + j_global][c] = val
                        elif split_dim == 1:
                            i_global = i
                            j_global = start + j
                            U[i_global * m + j_global][c] = val
                        else: # split_dim == 2
                            i_global = i
                            j_global = j
                            U[i_global * m + j_global][c] = val
                            
            # Map V
            for j in range(m_sub):
                for k in range(p_sub):
                    val = V_sub[j * p_sub + k][r]
                    if val != 0:
                        if split_dim == 0:
                            j_global = j
                            k_global = k
                            V[j_global * p + k_global][c] = val
                        elif split_dim == 1:
                            j_global = start + j
                            k_global = k
                            V[j_global * p + k_global][c] = val
                        else: # split_dim == 2
                            j_global = j
                            k_global = start + k
                            V[j_global * p + k_global][c] = val
                            
            # Map W
            for k in range(p_sub):
                for i in range(n_sub):
                    val = W_sub[k * n_sub + i][r]
                    if val != 0:
                        if split_dim == 0:
                            i_global = start + i
                            k_global = k
                            W[k_global * n + i_global][c] = val
                        elif split_dim == 1:
                            i_global = i
                            k_global = k
                            W[k_global * n + i_global][c] = val
                        else: # split_dim == 2
                            i_global = i
                            k_global = start + k
                            W[k_global * n + i_global][c] = val
                            
        col_offset += rank_sub
        
    return U, V, W

def _decompose(n, m, p):
    if n == 1:
        U, V, W = _trivial_decomp(n, m, p)
        return U, V, W, n, m, p
    if m == 1:
        U, V, W = _trivial_decomp(n, m, p)
        return U, V, W, n, m, p
    if p == 1:
        U, V, W = _trivial_decomp(n, m, p)
        return U, V, W, n, m, p
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_decomp()
        return U, V, W, n, m, p
        
    # Try splitting n
    best_rank = float('inf')
    best_result = None
    
    # Split n into 1 and n-1
    n1, n2 = 1, n - 1
    d1 = _decompose(n1, m, p)
    d2 = _decompose(n2, m, p)
    r1 = len(d1[0][0])
    r2 = len(d2[0][0])
    if r1 + r2 < best_rank:
        best_rank = r1 + r2
        splits = [(0, n1), (n1, n2)]
        # Combine
        U, V, W = _combine_decomps([d1, d2], n, m, p, 0, splits)
        best_result = (U, V, W, n, m, p)
        
    # Split n into 2 and n-2
    if n >= 4:
        n1, n2 = 2, n - 2
        d1 = _decompose(n1, m, p)
        d2 = _decompose(n2, m, p)
        r1 = len(d1[0][0])
        r2 = len(d2[0][0])
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            splits = [(0, n1), (n1, n2)]
            U, V, W = _combine_decomps([d1, d2], n, m, p, 0, splits)
            best_result = (U, V, W, n, m, p)

    # Try splitting m
    m1, m2 = 1, m - 1
    d1 = _decompose(n, m1, p)
    d2 = _decompose(n, m2, p)
    r1 = len(d1[0][0])
    r2 = len(d2[0][0])
    if r1 + r2 < best_rank:
        best_rank = r1 + r2
        splits = [(0, m1), (m1, m2)]
        U, V, W = _combine_decomps([d1, d2], n, m, p, 1, splits)
        best_result = (U, V, W, n, m, p)
        
    if m >= 4:
        m1, m2 = 2, m - 2
        d1 = _decompose(n, m1, p)
        d2 = _decompose(n, m2, p)
        r1 = len(d1[0][0])
        r2 = len(d2[0][0])
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            splits = [(0, m1), (m1, m2)]
            U, V, W = _combine_decomps([d1, d2], n, m, p, 1, splits)
            best_result = (U, V, W, n, m, p)

    # Try splitting p
    p1, p2 = 1, p - 1
    d1 = _decompose(n, m, p1)
    d2 = _decompose(n, m, p2)
    r1 = len(d1[0][0])
    r2 = len(d2[0][0])
    if r1 + r2 < best_rank:
        best_rank = r1 + r2
        splits = [(0, p1), (p1, p2)]
        U, V, W = _combine_decomps([d1, d2], n, m, p, 2, splits)
        best_result = (U, V, W, n, m, p)
        
    if p >= 4:
        p1, p2 = 2, p - 2
        d1 = _decompose(n, m, p1)
        d2 = _decompose(n, m, p2)
        r1 = len(d1[0][0])
        r2 = len(d2[0][0])
        if r1 + r2 < best_rank:
            best_rank = r1 + r2
            splits = [(0, p1), (p1, p2)]
            U, V, W = _combine_decomps([d1, d2], n, m, p, 2, splits)
            best_result = (U, V, W, n, m, p)

    if best_result is None:
        # Fallback to trivial
        U, V, W = _trivial_decomp(n, m, p)
        return U, V, W, n, m, p
        
    return best_result

def construct():
    n, m, p = DIMENSIONS
    U, V, W, _, _, _ = _decompose(n, m, p)
    return {"U": U, "V": V, "W": W}
