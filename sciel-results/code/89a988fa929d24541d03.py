"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 * 2x2 -> 2x2.
    
    Returns (U, V, W) where:
      U: (4, 7) - for A[i,j], row index = i*2+j
      V: (4, 7) - for B[j,k], row index = j*2+k
      W: (4, 7) - for C[i,k], row index = k*2+i  (trace convention: W[k*n+i] with n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # M1 = (A00 + A11) * (B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    
    # M2 = (A10 + A11) * B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    
    # M3 = A00 * (B01 - B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    
    # M5 = (A00 + A01) * B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    
    # M6 = (A10 - A00) * (B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    
    # M7 = (A01 - A11) * (B10 + B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    
    # W: C[i,k] contributions, W row = k*2 + i
    # C[0,0]: i=0,k=0 -> W[0]; C[0,1]: i=0,k=1 -> W[2]
    # C[1,0]: i=1,k=0 -> W[1]; C[1,1]: i=1,k=1 -> W[3]
    
    # C[0,0] = M1 + M4 - M5 + M7
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
    
    # C[0,1] = M3 + M5
    W[2, 2] = 1; W[2, 4] = 1
    
    # C[1,0] = M2 + M4
    W[1, 1] = 1; W[1, 3] = 1
    
    # C[1,1] = M1 - M2 + M3 + M6
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # n*m = 16 (A entries), m*p = 24 (B entries), p*n = 24 (C entries)
    
    # 2x2 block decomposition:
    # A[i,j]: i = 2*a+ai, j = 2*b+bj  => U row = (2a+ai)*4 + (2b+bj) = 8a+4ai+2b+bj
    # B[j,k]: j = 2*b+bj, k = 2*c+ck  => V row = (2b+bj)*6 + (2c+ck) = 12b+6bj+2c+ck
    # C[i,k]: i = 2*a+ai, k = 2*c+ck  => W row = (2c+ck)*4 + (2a+ai) = 8c+4ck+2a+ai
    # (trace convention: W[k*n + i])
    
    sU, sV, sW = _strassen_2x2()
    
    # Precompute non-zero entries of Strassen matrices
    sU_nz = [(r, c, v) for r in range(4) for c in range(7) for v in [sU[r,c]] if v != 0]
    sV_nz = [(r, c, v) for r in range(4) for c in range(7) for v in [sV[r,c]] if v != 0]
    sW_nz = [(r, c, v) for r in range(4) for c in range(7) for v in [sW[r,c]] if v != 0]
    
    # Build list of (U_rows, V_rows, W_rows) for each term
    # Each entry is (row_index, coeff)
    terms = []
    
    for ai in range(2):
        for bj in range(2):
            for ck in range(2):
                # 7 Strassen terms for 2x2 x 2x2 sub-block (c=0,1)
                for t in range(7):
                    u_entries = []
                    v_entries = []
                    w_entries = []
                    
                    for (r, c, v) in sU_nz:
                        if c == t:
                            # r = a*2+b
                            a, b = r // 2, r % 2
                            u_row = 8*a + 4*ai + 2*b + bj
                            u_entries.append((u_row, v))
                    
                    for (r, c, v) in sV_nz:
                        if c == t:
                            # r = b*2+c
                            b, c_sub = r // 2, r % 2
                            v_row = 12*b + 6*bj + 2*c_sub + ck
                            v_entries.append((v_row, v))
                    
                    for (r, c, v) in sW_nz:
                        if c == t:
                            # r = c*2+a (trace convention: row = k*2+i, so r = c_sub*2+a)
                            c_sub, a = r // 2, r % 2
                            w_row = 8*c_sub + 4*ck + 2*a + ai
                            w_entries.append((w_row, v))
                    
                    terms.append((u_entries, v_entries, w_entries))
                
                # 4 naive terms for the 3rd column (c=2)
                for a in range(2):
                    for b in range(2):
                        u_row = 8*a + 4*ai + 2*b + bj
                        v_row = 12*b + 6*bj + 4 + ck  # 2*c+ck with c=2 => 4+ck
                        w_row = 16 + 4*ck + 2*a + ai   # 8*c+4ck+2a+ai with c=2 => 16+4ck+2a+ai
                        terms.append(
                            ([(u_row, 1)], [(v_row, 1)], [(w_row, 1)])
                        )
    
    rank = len(terms)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for col, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (row, coeff) in u_entries:
            U[row, col] += coeff
        for (row, coeff) in v_entries:
            V[row, col] += coeff
        for (row, coeff) in w_entries:
            W[row, col] += coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
