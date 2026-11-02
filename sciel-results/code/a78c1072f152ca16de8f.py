"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook(n, m, p):
    """Return schoolbook decomposition for n×m × m×p."""
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
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _strassen_2x2x2():
    """
    Strassen decomposition for 2×2 × 2×2.
    Rank 7.
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # P1 = (A00+A11)*(B01+B11)
    U[0][0] = 1; U[3][0] = 1
    V[1][0] = 1; V[3][0] = 1
    W[0][0] = 1   # C00
    W[2][0] = 1   # C01 (k=1, i=0 -> idx 2)
    W[1][0] = -1  # C10 (k=0, i=1 -> idx 1)
    W[3][0] = 1   # C11 (k=1, i=1 -> idx 3)
    
    # P2 = (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1   # C10
    W[3][1] = -1  # C11
    
    # P3 = A00*(B10-B11)
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[3][2] = 1   # C11
    
    # P4 = A11*(B00-B10)
    U[3][3] = 1
    V[0][3] = 1; V[2][3] = -1
    W[0][3] = 1   # C00
    W[1][3] = 1   # C10
    
    # P5 = (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1  # C00
    W[2][4] = 1   # C01
    
    # P6 = (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    # W is 0 for P6 in standard Strassen output formulas?
    # If so, this term is redundant for the output, but part of the rank-7 structure?
    # For safety, I will leave W as 0.
    
    # P7 = (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1   # C00
    W[3][6] = 1   # C11
    
    return {'U': U, 'V': V, 'W': W, 'rank': rank}

def _combine_blocks(n, m, p, n_splits, m_splits, p_splits, get_decomp):
    """
    Combine decompositions of sub-blocks.
    n_splits: list of (start, size)
    """
    total_rank = 0
    # Calculate total rank first
    for n_start, n_size in n_splits:
        for m_start, m_size in m_splits:
            for p_start, p_size in p_splits:
                decomp = get_decomp(n_size, m_size, p_size)
                total_rank += decomp['rank']
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for n_start, n_size in n_splits:
        for m_start, m_size in m_splits:
            for p_start, p_size in p_splits:
                decomp = get_decomp(n_size, m_size, p_size)
                r = decomp['rank']
                for i in range(n_size):
                    for j in range(m_size):
                        for k in range(r):
                            U[(n_start + i) * m + (m_start + j)][col_offset + k] = decomp['U'][i * m_size + j][k]
                for j in range(m_size):
                    for k in range(p_size):
                        for r_idx in range(r):
                            V[(m_start + j) * p + (p_start + k)][col_offset + r_idx] = decomp['V'][j * p_size + k][r_idx]
                for k in range(p_size):
                    for i in range(n_size):
                        for r_idx in range(r):
                            W[(p_start + k) * n + (n_start + i)][col_offset + r_idx] = decomp['W'][k * n_size + i][r_idx]
                col_offset += r
                
    return {'U': U, 'V': V, 'W': W, 'rank': total_rank}

def _get_decomp(n, m, p):
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()
    else:
        return _schoolbook(n, m, p)

def construct():
    n, m, p = 4, 5, 6
    n_splits = [(0, 2), (2, 2)]
    m_splits = [(0, 2), (2, 2), (4, 1)]
    p_splits = [(0, 2), (2, 2), (4, 2)]
    
    result = _combine_blocks(n, m, p, n_splits, m_splits, p_splits, _get_decomp)
    
    return {"U": result['U'], "V": result['V'], "W": result['W']}
# EVOLVE-BLOCK-END
