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
    
    # Clear block structure:
    # A (4x4): 2x2 blocks of size 2x2. A_block[a,b] for a,b in {0,1}
    # B (4x6): 2 row-blocks x 3 col-blocks, each 2x2. B_block[b,c] for b in {0,1}, c in {0,1,2}
    # C (4x6): 2 row-blocks x 3 col-blocks, each 2x2. C_block[a,c] for a in {0,1}, c in {0,1,2}
    #
    # C_block[a,c] = A_block[a,0] * B_block[0,c] + A_block[a,1] * B_block[1,c]
    #
    # For each (a,c), this is sum of two 2x2*2x2 products.
    # Total: 2*3 = 6 output blocks, each is sum of 2 rank-7 terms = 14 terms? No.
    # We need to decompose the sum, not just add ranks.
    #
    # Alternative: For each a, C_block[a,:] (2x6) = A_block[a,0]*B_block[0,:] + A_block[a,1]*B_block[1,:]
    # where B_block[0,:] is 2x6 and B_block[1,:] is 2x6.
    # This is 2x2 * 2x6 + 2x2 * 2x6 -> 2x6
    # 
    # For 2x2 * 2x6: use Strassen for each 2x2 column pair.
    # 2x6 = 3 pairs of 2x2. Each needs rank 7. Total 21 per 2x2*2x6.
    # Two such terms summed: can't easily share.
    #
    # Simplest correct approach: 12 independent 2x2*2x2 multiplications (a,b,c) in {0,1}^3
    # Each rank 7, total 84. Too high.
    #
    # The current code's approach groups by (ai,bj,ck) giving 4 groups of 9 = 36.
    # Let me just fix the existing code to be correct and verified.
    
    sU, sV, sW = _strassen_2x2()
    
    # The current approach: for each (ai, bj, ck) in {0,1}^3:
    #   7 Strassen terms for the 2x2*2x2 sub-problem on columns 0,1
    #   4 extra terms for column 2
    # Total: 4 * (7+4) = 44
    #
    # But the reported score is 36, suggesting the actual rank is 36.
    # Let me just rebuild the original logic cleanly.
    
    rank_terms = []
    for ai in range(2):
        for bj in range(2):
            for ck in range(2):
                for t in range(7):
                    rank_terms.append(('strassen', ai, bj, ck, t))
                for a in range(2):
                    for b in range(2):
                        rank_terms.append(('extra', ai, bj, ck, a, b))
    
    rank = len(rank_terms)
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    for col, term in enumerate(rank_terms):
        if term[0] == 'strassen':
            _, ai, bj, ck, t = term
            for a in range(2):
                for b in range(2):
                    coeff = sU[a*2 + b, t]
                    if coeff != 0:
                        u_row = (2*a + ai) * 4 + (2*b + bj)
                        U[u_row, col] += coeff
            for b in range(2):
                for c in range(2):
                    coeff = sV[b*2 + c, t]
                    if coeff != 0:
                        v_row = (2*b + bj) * 6 + (2*c + ck)
                        V[v_row, col] += coeff
            for a in range(2):
                for c in range(2):
                    coeff = sW[c*2 + a, t]
                    if coeff != 0:
                        w_row = (2*c + ck) * 4 + (2*a + ai)
                        W[w_row, col] += coeff
        else:
            _, ai, bj, ck, a, b = term
            u_row = (2*a + ai) * 4 + (2*b + bj)
            v_row = (2*b + bj) * 6 + (2*2 + ck)
            w_row = (2*2 + ck) * 4 + (2*a + ai)
            U[u_row, col] = 1
            V[v_row, col] = 1
            W[w_row, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
