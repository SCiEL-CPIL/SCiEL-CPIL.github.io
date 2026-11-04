import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 140

def _strassen_2x2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 matrix multiplication using Strassen.
    U: shape (4, 7)
    V: shape (4, 7)
    W: shape (4, 7)
    """
    U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W = [
        [1, 0, 0, 1],
        [0, -1, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    return (U, V, W)

def _winograd_3x3x3():
    """
    Return (U, V, W) for 3x3 * 3x3 matrix multiplication using Winograd.
    Rank 23.
    """
    # This is a known decomposition.
    # Coefficients are integers or halves.
    # I will use a standard Winograd 3x3 implementation.
    # U: 9x23, V: 9x23, W: 9x23
    
    # The 23 multiplications correspond to the 23 non-zero entries in the 
    # transformed space.
    # This is complex to hardcode from memory. 
    # I will use a simpler 27-rank (schoolbook) for 3x3 if I can't recall Winograd exactly,
    # but 23 is significantly better.
    # Let's try to find a compact representation or use a library if available? 
    # No, standard library only.
    
    # Actually, I can implement the 3x3x3 with rank 23 using the coefficients from literature.
    # However, to ensure correctness within the time limit, I might stick to a 
    # slightly higher rank if Winograd is too risky to implement blindly.
    # Rank 27 is schoolbook.
    # Rank 23 is Winograd.
    # Difference is 4.
    # In the total sum, 3x3x3 appears once.
    # So 92 vs 96.
    # Let's try to implement Winograd.
    
    # U, V, W for Winograd 3x3
    # Based on "Winograd's method for matrix multiplication"
    # The transformation matrices are 9x23.
    
    # I will use a pre-computed set of coefficients for Winograd 3x3.
    # If I cannot guarantee correctness, I will fallback to schoolbook 27.
    # Given the risk, I will use schoolbook 27 for 3x3x3 to be safe, 
    # but optimize the structure.
    
    # Wait, 3x3x3 schoolbook is 27.
    # My previous calculation:
    # M334: 23 (Winograd) + 9 + 9 + 3 = 44.
    # If I use 27: 27 + 9 + 9 + 3 = 48.
    # Total rank becomes 56 - 4 + 36 = 96.
    # This is the same as the 4+1 split.
    
    # Let's try to get Winograd right.
    # I will hardcode the U, V, W matrices for Winograd 3x3.
    
    # The coefficients for Winograd 3x3 are:
    # U = [[1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      ...]
    # This is too long to type out and verify mentally.
    
    # I will use the 92 rank strategy but with Schoolbook 27 for 3x3x3?
    # No, that gives 96.
    
    # Let's stick to the 4+1 split with Strassen 64 for 4x4x4?
    # Rank 100.
    
    # Let's stick to the 3+2 split with Strassen 28 for 2x2x2 blocks?
    # Rank 92 with Winograd 23.
    # Rank 96 with Schoolbook 27.
    
    # I will implement the 96 rank version (3+2 split, schoolbook 3x3) as it is safer 
    # to implement correctly than Winograd 23 from memory.
    # 96 is better than 100.
    
    return None # Placeholder

def _schoolbook(m, n, p):
    rank = m * n * p
    U = [[0] * rank for _ in range(m * n)]
    V = [[0] * rank for _ in range(n * p)]
    W = [[0] * rank for _ in range(p * m)]
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j][col] = 1
                V[j * p + k][col] = 1
                W[k * m + i][col] = 1
                col += 1
    return (U, V, W)

def _combine_decompositions(decomps, m, n, p):
    """
    Combine a list of decompositions.
    Each decomp is (U, V, W, offset_m, offset_n, offset_p, m_i, n_i, p_i)
    """
    total_rank = sum(len(U) for U, V, W, *_ in decomps)
    U_total = [[0] * total_rank for _ in range(m * n)]
    V_total = [[0] * total_rank for _ in range(n * p)]
    W_total = [[0] * total_rank for _ in range(p * m)]
    
    rank_offset = 0
    for U_s, V_s, W_s, om, on, op, m_i, n_i, p_i in decomps:
        for r in range(len(U_s)):
            for ii in range(m_i):
                for jj in range(n_i):
                    U_total[(om + ii) * n + (on + jj)][rank_offset + r] = U_s[r][ii * n_i + jj]
            for jj in range(n_i):
                for kk in range(p_i):
                    V_total[(on + jj) * p + (op + kk)][rank_offset + r] = V_s[r][jj * p_i + kk]
            for kk in range(p_i):
                for ii in range(m_i):
                    W_total[(op + kk) * m + (om + ii)][rank_offset + r] = W_s[r][kk * m_i + ii]
        rank_offset += len(U_s)
        
    return (U_total, V_total, W_total)

def construct():
    global _start_time
    _start_time = time.time()
    
    # Dimensions: 4x5 * 5x5
    m, n, p = 4, 5, 5
    
    # Split inner dimension 5 into 3 and 2
    n1, n2 = 3, 2
    
    # A is 4x5 -> A1 (4x3), A2 (4x2)
    # B is 5x5 -> B1 (3x5), B2 (2x5)
    
    decomps = []
    
    # 1. A1 * B1 (4x3 * 3x5)
    # Split B1 (3x5) into B11 (3x4) and B12 (3x1)
    p11, p12 = 4, 1
    
    # 1.1 A1 * B11 (4x3 * 3x4)
    # Split A1 (4x3) into A11 (3x3) and A12 (1x3)
    # Split B11 (3x4) into B111 (3x3) and B112 (3x1)
    m11, m12 = 3, 1
    p111, p112 = 3, 1
    
    # 1.1.1 A11 * B111 (3x3 * 3x3) -> Schoolbook 27
    U111, V111, W111 = _schoolbook(3, 3, 3)
    decomps.append((U111, V111, W111, 0, 0, 0, 3, 3, 3))
    
    # 1.1.2 A11 * B112 (3x3 * 3x1) -> Schoolbook 9
    U112, V112, W112 = _schoolbook(3, 3, 1)
    decomps.append((U112, V112, W112, 0, 0, 3, 3, 3, 1))
    
    # 1.1.3 A12 * B111 (1x3 * 3x3) -> Schoolbook 9
    U121, V121, W121 = _schoolbook(1, 3, 3)
    decomps.append((U121, V121, W121, 3, 0, 0, 1, 3, 3))
    
    # 1.1.4 A12 * B112 (1x3 * 3x1) -> Schoolbook 3
    U122, V122, W122 = _schoolbook(1, 3, 1)
    decomps.append((U122, V122, W122, 3, 0, 3, 1, 3, 1))
    
    # 1.2 A1 * B12 (4x3 * 3x1) -> Schoolbook 12
    U12, V12, W12 = _schoolbook(4, 3, 1)
    decomps.append((U12, V12, W12, 0, 0, 4, 4, 3, 1))
    
    # 2. A2 * B2 (4x2 * 2x5)
    # Split B2 (2x5) into B21 (2x4) and B22 (2x1)
    p21, p22 = 4, 1
    
    # 2.1 A2 * B21 (4x2 * 2x4)
    # Split A2 (4x2) into A21 (2x2) and A22 (2x2)
    # Split B21 (2x4) into B211 (2x2) and B212 (2x2)
    m21, m22 = 2, 2
    p211, p212 = 2, 2
    
    # 2.1.1 A21 * B211 (2x2 * 2x2) -> Strassen 7
    U2111, V2111, W2111 = _strassen_2x2x2()
    decomps.append((U2111, V2111, W2111, 0, 3, 0, 2, 2, 2))
    
    # 2.1.2 A21 * B212 (2x2 * 2x2) -> Strassen 7
    U2112, V2112, W2112 = _strassen_2x2x2()
    decomps.append((U2112, V2112, W2112, 0, 3, 2, 2, 2, 2))
    
    # 2.1.3 A22 * B211 (2x2 * 2x2) -> Strassen 7
    U2211, V2211, W2211 = _strassen_2x2x2()
    decomps.append((U2211, V2211, W2211, 2, 3, 0, 2, 2, 2))
    
    # 2.1.4 A22 * B212 (2x2 * 2x2) -> Strassen 7
    U2212, V2212, W2212 = _strassen_2x2x2()
    decomps.append((U2212, V2212, W2212, 2, 3, 2, 2, 2, 2))
    
    # 2.2 A2 * B22 (4x2 * 2x1) -> Schoolbook 8
    U22, V22, W22 = _schoolbook(4, 2, 1)
    decomps.append((U22, V22, W22, 0, 3, 4, 4, 2, 1))
    
    # Combine
    U, V, W = _combine_decompositions(decomps, m, n, p)
    
    return {"U": U, "V": V, "W": W}
