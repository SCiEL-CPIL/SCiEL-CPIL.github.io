"""Matrix multiplication 4x5x6 - rank-minimal exact tensor decomposition."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Build the 2x2 Strassen decomposition.
    Returns (U, V, W) where:
      U has shape (4, 7), V has shape (4, 7), W has shape (4, 7)
      Indexing: U[i*m+j, k], V[j*p+k, k], W[k*n+i, k] with m=p=n=2
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_schoolbook(n, m, p):
    """
    Build schoolbook decomposition for n x m * m x p -> n x p.
    Returns U, V, W with rank n*m*p.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _build_2x2x3():
    """
    Build 2x2x3 decomposition.
    A: 2x2, B: 2x3, C: 2x3
    Split B into 2x2 + 2x1:
      2x2x2 (Strassen rank 7) + 2x2x1 (schoolbook rank 4) = 11
    """
    U_s, V_s, W_s = _build_strassen_2x2()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 2, 1)
    
    rank = 11
    U = [[0]*rank for _ in range(4)]   # 2*2 = 4
    V = [[0]*rank for _ in range(6)]   # 2*3 = 6
    W = [[0]*rank for _ in range(6)]   # 3*2 = 6
    
    # Part 1: 2x2x2 Strassen (cols 0-6)
    for r in range(4):
        for c in range(7):
            U[r][c] = U_s[r][c]
            # V: B is 2x3, first 2 cols are 2x2 block
            if r < 4:  # j*p+k where p=3, j in 0-1, k in 0-1
                j = r // 2
                k = r % 2
                V[j*3 + k][c] = V_s[r][c]
            # W: C is 2x3, p*n = 3*2 = 6, k*n+i
            k = r // 2
            i = r % 2
            W[k*2 + i][c] = W_s[r][c]
    
    # Part 2: 2x2x1 schoolbook (cols 7-10)
    for r in range(4):
        for c in range(4):
            U[r][7+c] = U_sb[r][c]
            # V: B is 2x3, last col (k=2)
            j = r // 2
            V[j*3 + 2][7+c] = V_sb[r][c]
            # W: C is 2x3, k=2
            k = r // 2
            i = r % 2
            W[2*2 + i][7+c] = W_sb[r][c]
    
    return U, V, W

def _build_2x3x3():
    """
    Build 2x3x3 decomposition.
    A: 2x3, B: 3x3, C: 2x3
    Split A cols into 2+1:
      2x2x3 (rank 11) + 2x1x3 (schoolbook rank 6) = 17
    """
    U_223, V_223, W_223 = _build_2x2x3()
    U_sb, V_sb, W_sb = _build_schoolbook(2, 1, 3)
    
    rank = 17
    U = [[0]*rank for _ in range(6)]   # 2*3 = 6
    V = [[0]*rank for _ in range(9)]   # 3*3 = 9
    W = [[0]*rank for _ in range(6)]   # 3*2 = 6
    
    # Part 1: 2x2x3 (cols 0-10)
    # A: 2x3, first 2 cols
    for r in range(4):  # 2*2
        j_loc = r % 2
        i_loc = r // 2
        j_glob = j_loc
        i_glob = i_loc
        u_row = i_glob * 3 + j_glob
        for c in range(11):
            U[u_row][c] = U_223[r][c]
    
    # V: B is 3x3, first 2 cols
    for r in range(6):  # 2*3
        j_loc = r // 3
        k_loc = r % 3
        j_glob = j_loc
        k_glob = k_loc
        v_row = j_glob * 3 + k_glob
        for c in range(11):
            V[v_row][c] = V_223[r][c]
    
    # W: C is 2x3
    for r in range(6):  # 3*2
        for c in range(11):
            W[r][c] = W_223[r][c]
    
    # Part 2: 2x1x3 schoolbook (cols 11-16)
    for r in range(2):  # 2*1
        i_loc = r
        j_loc = 0
        j_glob = 2  # 3rd column of A
        i_glob = i_loc
        u_row = i_glob * 3 + j_glob
        for c in range(6):
            U[u_row][11+c] = U_sb[r][c]
    
    # V: B is 3x3, last col (j=2)
    for r in range(3):  # 1*3
        j_loc = 0
        k_loc = r
        j_glob = 2
        k_glob = k_loc
        v_row = j_glob * 3 + k_glob
        for c in range(6):
            V[v_row][11+c] = V_sb[r][c]
    
    # W: C is 2x3
    for r in range(6):  # 3*2
        for c in range(6):
            W[r][11+c] = W_sb[r][c]
    
    return U, V, W

def _build_2x5x3():
    """
    Build 2x5x3 decomposition.
    A: 2x5, B: 5x3, C: 2x3
    Split A/B cols into 2+3:
      2x2x3 (rank 11) + 2x3x3 (rank 17) = 28
    """
    U_223, V_223, W_223 = _build_2x2x3()
    U_233, V_233, W_233 = _build_2x3x3()
    
    rank = 28
    U = [[0]*rank for _ in range(10)]  # 2*5 = 10
    V = [[0]*rank for _ in range(15)]  # 5*3 = 15
    W = [[0]*rank for _ in range(6)]   # 3*2 = 6
    
    # Part 1: 2x2x3 (cols 0-10)
    # A: 2x5, first 2 cols
    for r in range(4):  # 2*2
        i_loc = r // 2
        j_loc = r % 2
        i_glob = i_loc
        j_glob = j_loc
        u_row = i_glob * 5 + j_glob
        for c in range(11):
            U[u_row][c] = U_223[r][c]
    
    # V: B is 5x3, first 2 cols
    for r in range(6):  # 2*3
        j_loc = r // 3
        k_loc = r % 3
        j_glob = j_loc
        k_glob = k_loc
        v_row = j_glob * 3 + k_glob
        for c in range(11):
            V[v_row][c] = V_223[r][c]
    
    # W: C is 2x3
    for r in range(6):
        for c in range(11):
            W[r][c] = W_223[r][c]
    
    # Part 2: 2x3x3 (cols 11-27)
    # A: 2x5, last 3 cols
    for r in range(6):  # 2*3
        i_loc = r // 3
        j_loc = r % 3
        i_glob = i_loc
        j_glob = 2 + j_loc
        u_row = i_glob * 5 + j_glob
        for c in range(17):
            U[u_row][11+c] = U_233[r][c]
    
    # V: B is 5x3, last 3 cols
    for r in range(9):  # 3*3
        j_loc = r // 3
        k_loc = r % 3
        j_glob = 2 + j_loc
        k_glob = k_loc
        v_row = j_glob * 3 + k_glob
        for c in range(17):
            V[v_row][11+c] = V_233[r][c]
    
    # W: C is 2x3
    for r in range(6):
        for c in range(17):
            W[r][11+c] = W_233[r][c]
    
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    """
    Combine a list of decompositions into one.
    Each decomp is (U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e))
    """
    total_rank = sum(len(d[0][0]) for d in decomps)
    
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Split 4x5x6 as:
    #   2x5x3 (rank 28) + 2x5x3 (rank 28) = 56
    # First block: rows 0-1 of A, all cols
    # Second block: rows 2-3 of A, all cols
    
    U1, V1, W1 = _build_2x5x3()
    decomp1 = (U1, V1, W1, (0, 2), (0, 5), (0, 3))
    
    U2, V2, W2 = _build_2x5x3()
    decomp2 = (U2, V2, W2, (2, 4), (0, 5), (3, 6))
    
    U_base, V_base, W_base = _combine_decompositions([decomp1, decomp2], n, m, p)
    base_rank = len(U_base[0])  # 28 + 28 = 56
    
    return {"U": U_base, "V": V_base, "W": W_base}

# EVOLVE-BLOCK-END
