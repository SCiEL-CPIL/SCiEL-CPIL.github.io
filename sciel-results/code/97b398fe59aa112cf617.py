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
    
    # Strategy: Use block structure with 2x2 blocks.
    # A is 4x4: split into 2x2 blocks A[a,b] where a,b in {0,1}
    #   A[i,j] with i=2*a+ai, j=2*b+bj, ai,bj in {0,1}
    # B is 4x6: split into 2x3 blocks B[b,c] where b in {0,1}, c in {0,1,2}
    #   B[j,k] with j=2*b+bj, k=2*c+ck, bj in {0,1}, ck in {0,1}
    # C is 4x6: split into 2x3 blocks C[a,c] where a in {0,1}, c in {0,1,2}
    #   C[i,k] with i=2*a+ai, k=2*c+ck
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[2a+ai, 2c+ck] = sum_{b,bj} A[2a+ai, 2b+bj] * B[2b+bj, 2c+ck]
    
    # For fixed (ai, bj, ck), this is a (2x2) * (2x3) multiplication:
    #   A part: A[2a+ai, 2b+bj] as function of (a,b) -> 2x2 matrix
    #   B part: B[2b+bj, 2c+ck] as function of (b,c) -> 2x3 matrix
    #   C part: C[2a+ai, 2c+ck] as function of (a,c) -> 2x3 matrix
    
    # For each (ai, bj, ck) in {0,1}x{0,1}x{0,1}, we need a 2x2 * 2x3 decomposition.
    # 2x2 * 2x3 can be decomposed using Strassen for the 2x2 * 2x2 part (first 2 cols)
    # plus 2 additional terms for the 3rd column.
    
    # Actually, let me think more carefully.
    # For 2x2 * 2x3: A is 2x2, B is 2x3, C is 2x3
    # C[a,c] = sum_b A[a,b] * B[b,c]
    # 
    # We can use Strassen for c=0,1 (the 2x2 * 2x2 part) and add terms for c=2.
    # Strassen gives 7 terms for the 2x2*2x2 part.
    # For c=2, we need 2 more terms (one for each row of A):
    #   C[a,2] = A[a,0]*B[0,2] + A[a,1]*B[1,2]
    # So 2 additional rank-1 terms.
    # Total per (ai,bj,ck): 7 + 2 = 9 terms.
    # Total: 4 * 9 = 36 terms.
    
    # Wait, but the Strassen decomposition for the 2x2*2x2 part handles both c=0 and c=1
    # simultaneously. The 2 extra terms handle c=2. So 9 terms per (ai,bj,ck) group.
    # 4 groups * 9 = 36 total rank.
    
    # Let me verify the index mappings:
    # U row for A[2a+ai, 2b+bj]: row = (2a+ai)*4 + (2b+bj) = 8a + 4ai + 2b + bj
    # V row for B[2b+bj, 2c+ck]: row = (2b+bj)*6 + (2c+ck) = 12b + 6bj + 2c + ck
    # W row for C[2a+ai, 2c+ck]: row = (2c+ck)*4 + (2a+ai) = 8c + 4ck + 2a + ai
    
    # Get Strassen decomposition
    sU, sV, sW = _strassen_2x2()
    
    # Build the full decomposition
    rank_terms = []
    
    for ai in range(2):
        for bj in range(2):
            for ck in range(2):
                # 7 Strassen terms for the 2x2 * 2x2 part (columns c=0,1)
                for t in range(7):
                    # U part: for each (a,b) where sU[a*2+b, t] != 0
                    # V part: for each (b,c) where sV[b*2+c, t] != 0
                    # W part: for each (a,c) where sW[c*2+a, t] != 0
                    # These are coupled through the Strassen structure.
                    
                    # For the 2x2*2x2 sub-problem:
                    # A_sub[a,b] = A[2a+ai, 2b+bj]
                    # B_sub[b,c] = B[2b+bj, 2c+ck]  (for c in {0,1})
                    # C_sub[a,c] = C[2a+ai, 2c+ck]  (for c in {0,1})
                    
                    # The rank-t term contributes:
                    # U: A_sub entries where sU[a*2+b, t] != 0
                    # V: B_sub entries where sV[b*2+c, t] != 0 (c in {0,1})
                    # W: C_sub entries where sW[c*2+a, t] != 0 (c in {0,1})
                    
                    rank_terms.append(('strassen', ai, bj, ck, t))
                
                # 2 additional terms for c=2
                # C[a,2] = A[a,0]*B[0,2] + A[a,1]*B[1,2]
                # Term 1: a=0: A[0,0]*B[0,2] + ... no, each rank-1 term is a single product
                # Actually, for c=2:
                #   C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
                #   C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
                # So we need 4 rank-1 terms: (a,b) pairs for the c=2 column
                #   (0,0): A[0,0]*B[0,2] -> C[0,2]
                #   (0,1): A[0,1]*B[1,2] -> C[0,2]
                #   (1,0): A[1,0]*B[0,2] -> C[1,2]
                #   (1,1): A[1,1]*B[1,2] -> C[1,2]
                
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
            # U part
            for a in range(2):
                for b in range(2):
                    coeff = sU[a*2 + b, t]
                    if coeff != 0:
                        u_row = 8*a + 4*ai + 2*b + bj
                        U[u_row, col] += coeff
            # V part (c in {0,1})
            for b in range(2):
                for c in range(2):
                    coeff = sV[b*2 + c, t]
                    if coeff != 0:
                        v_row = 12*b + 6*bj + 2*c + ck
                        V[v_row, col] += coeff
            # W part (c in {0,1})
            for a in range(2):
                for c in range(2):
                    coeff = sW[c*2 + a, t]
                    if coeff != 0:
                        w_row = 8*c + 4*ck + 2*a + ai
                        W[w_row, col] += coeff
        else:  # 'extra'
            _, ai, bj, ck, a, b = term
            u_row = 8*a + 4*ai + 2*b + bj
            v_row = 12*b + 6*bj + 2*2 + ck  # c=2
            w_row = 8*2 + 4*ck + 2*a + ai    # c=2
            U[u_row, col] = 1
            V[v_row, col] = 1
            W[w_row, col] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
