"""Matrix multiplication 3,5,6 - rank-minimal decomposition using block Strassen."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
_TIMEOUT = 145  # seconds, safety margin before 150s kill

def _strassen_2x2x2():
    """Return U,V,W for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4,7) for A (2x2), V: (4,7) for B (2x2), W: (4,7) for C (2x2).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas for 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1 = (a+d)(e+h), m2 = (c+d)e, m3 = a(f-h), m4 = d(g-e)
    # m5 = (a+b)h, m6 = (c-a)(e+f), m7 = (b-d)(g+h)
    # C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3+m6]]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U rows: 0=a, 1=b, 2=c, 3=d
    # V rows: 0=e, 1=f, 2=g, 3=h
    # W rows: 0=C00, 1=C01, 2=C10, 3=C11
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1  # a+d
    V[0][0] = 1; V[3][0] = 1  # e+h
    W[0][0] = 1; W[3][0] = 1  # C00 += m1, C11 += m1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1  # c+d
    V[0][1] = 1               # e
    W[2][1] = 1               # C10 += m2
    
    # m3 = a(f-h)
    U[0][2] = 1               # a
    V[1][2] = 1; V[3][2] = -1  # f-h
    W[1][2] = 1; W[2][2] = 1  # C01 += m3, C10 += m3
    
    # m4 = d(g-e)
    U[3][3] = 1               # d
    V[2][3] = 1; V[0][3] = -1  # g-e
    W[0][3] = 1; W[2][3] = 1  # C00 += m4, C10 += m4
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1  # a+b
    V[3][4] = 1               # h
    W[0][4] = -1; W[1][4] = 1  # C00 -= m5, C01 += m5
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1  # c-a
    V[0][5] = 1; V[1][5] = 1  # e+f
    W[3][5] = 1               # C11 += m6
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1  # b-d
    V[2][6] = 1; V[3][6] = 1  # g+h
    W[0][6] = 1               # C00 += m7
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Schoolbook decomposition for n x m x p block."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _block_decomposition(n, m, p, n_splits, m_splits, p_splits):
    """Decompose M_{n,m,p} using block structure with Strassen on 2x2x2 blocks.
    
    n_splits: list of block sizes summing to n
    m_splits: list of block sizes summing to m  
    p_splits: list of block sizes summing to p
    """
    all_U = []
    all_V = []
    all_W = []
    
    for ni, mi, pi in zip(n_splits, m_splits, p_splits):
        if ni == 2 and mi == 2 and pi == 2:
            Bu, Bv, Bw = _strassen_2x2x2()
            all_U.append(Bu)
            all_V.append(Bv)
            all_W.append(Bw)
        else:
            Bu, Bv, Bw = _schoolbook_block(ni, mi, pi)
            all_U.append(Bu)
            all_V.append(Bv)
            all_W.append(Bw)
    
    return all_U, all_V, all_W

def _assemble(all_U, all_V, all_W, n, m, p, n_splits, m_splits, p_splits):
    """Assemble block decompositions into full U, V, W matrices."""
    total_rank = sum(len(U) for U in all_U)
    
    # Compute offsets
    n_offsets = [0]
    for s in n_splits[:-1]:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_splits[:-1]:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_splits[:-1]:
        p_offsets.append(p_offsets[-1] + s)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    rank_offset = 0
    for bi, (ni, mi, pi) in enumerate(zip(n_splits, m_splits, p_splits)):
        Bu, Bv, Bw = all_U[bi], all_V[bi], all_W[bi]
        block_rank = len(Bu)
        
        for r in range(block_rank):
            # U: rows indexed by (i_block * m + j) within block, map to global
            for row in range(ni * mi):
                i_local = row // mi
                j_local = row % mi
                i_global = n_offsets[bi] + i_local
                j_global = m_offsets[bi] + j_local
                U[i_global * m + j_global][rank_offset + r] = Bu[row][r]
            
            # V: rows indexed by (j_block * p + k) within block
            for row in range(mi * pi):
                j_local = row // pi
                k_local = row % pi
                j_global = m_offsets[bi] + j_local
                k_global = p_offsets[bi] + k_local
                V[j_global * p + k_global][rank_offset + r] = Bv[row][r]
            
            # W: rows indexed by (k_block * n + i) within block
            for row in range(pi * ni):
                k_local = row // ni
                i_local = row % ni
                k_global = p_offsets[bi] + k_local
                i_global = n_offsets[bi] + i_local
                W[k_global * n + i_global][rank_offset + r] = Bw[row][r]
        
        rank_offset += block_rank
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Try different block splits and pick the one with minimum rank
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook
    
    # Split options
    n_options = [[3], [2,1], [1,2], [1,1,1]]
    m_options = [[5], [3,2], [2,3], [2,2,1], [1,2,2], [1,1,1,1,1]]
    p_options = [[6], [3,3], [2,4], [4,2], [2,2,2], [2,3,1], [1,3,2]]
    
    for ns in n_options:
        for ms in m_options:
            for ps in p_options:
                if len(ns) != len(ms) or len(ms) != len(ps):
                    continue
                if sum(ns) != n or sum(ms) != m or sum(ps) != p:
                    continue
                
                all_U, all_V, all_W = _block_decomposition(n, m, p, ns, ms, ps)
                total_rank = sum(len(U) for U in all_U)
                
                if total_rank < best_rank:
                    U, V, W = _assemble(all_U, all_V, all_W, n, m, p, ns, ms, ps)
                    best_U, best_V, best_W = U, V, W
                    best_rank = total_rank
    
    if best_U is None:
        # Fallback to schoolbook
        best_U, best_V, best_W = _schoolbook_block(n, m, p)
        best_rank = n * m * p
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
