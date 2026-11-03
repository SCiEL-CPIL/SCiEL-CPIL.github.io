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


def _block_outer_decomp():
    """
    Decompose the outer 2x2 * 2x3 -> 2x3 block multiplication.
    
    Returns a list of (U_coeff, V_coeff, W_coeff) for the outer structure.
    Each entry describes how to combine the 2x2 block entries.
    
    For the 2x2 * 2x2 part (cols 0,1): Strassen (7 terms)
    For the 2x2 * 1 part (col 2): schoolbook (4 terms)
    
    Returns: list of dicts with keys 'U', 'V', 'W' where each is a dict
    mapping (row_block, col_block) -> coefficient
    """
    sU, sV, sW = _strassen_2x2()
    
    terms = []
    
    # 7 Strassen terms for cols 0,1
    for t in range(7):
        U_dict = {}
        V_dict = {}
        W_dict = {}
        for a in range(2):
            for b in range(2):
                coeff = sU[a*2+b, t]
                if coeff != 0:
                    U_dict[(a,b)] = U_dict.get((a,b), 0) + coeff
        for b in range(2):
            for c in range(2):
                coeff = sV[b*2+c, t]
                if coeff != 0:
                    V_dict[(b,c)] = V_dict.get((b,c), 0) + coeff
        for a in range(2):
            for c in range(2):
                coeff = sW[c*2+a, t]
                if coeff != 0:
                    W_dict[(a,c)] = W_dict.get((a,c), 0) + coeff
        terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_dict = {(a,b): 1}
            V_dict = {(b,2): 1}
            W_dict = {(a,2): 1}
            terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
    
    return terms


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Use 2x2 block structure with Kronecker product decomposition.
    # A is 4x4: 2x2 blocks of size 2x2. A[ai][bj] where ai,bj in {0,1}
    #   A[i,j] with i=2*ai+ai2, j=2*bj+bj2, ai2,bj2 in {0,1}
    # B is 4x6: 2x3 blocks of size 2x2. B[bj][ck] where bj in {0,1}, ck in {0,1,2}
    #   B[j,k] with j=2*bj+bj2, k=2*ck+ck2, bj2,ck2 in {0,1}
    # C is 4x6: 2x3 blocks of size 2x2. C[ai][ck] where ai in {0,1}, ck in {0,1,2}
    #   C[i,k] with i=2*ai+ai2, k=2*ck+ck2, ai2,ck2 in {0,1}
    #
    # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]  (block matrix multiplication)
    # Each block product is 2x2 * 2x2 -> 2x2, decomposed via Strassen (rank 7).
    #
    # Outer structure: 2x2 * 2x3 -> 2x3 block multiplication
    # Use 11 outer terms: 7 (Strassen for cols 0,1) + 4 (schoolbook for col 2)
    # Each outer term involves one 2x2*2x2 block product (rank 7)
    # Total rank: 11 * 7 = 77
    
    outer_terms = _block_outer_decomp()
    sU, sV, sW = _strassen_2x2()
    
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            # U part: for each (ai, bj) block position
            for (ai, bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # A[i,j] where i=2*ai+ai2, j=2*bj+bj2
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
            
            # V part: for each (bj, ck) block position
            for (bj, ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # B[j,k] where j=2*bj+bj2, k=2*ck+ck2
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
            
            # W part: for each (ai, ck) block position
            for (ai, ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            # C[i,k] where i=2*ai+ai2, k=2*ck+ck2
                            # Trace convention: W[k*n+i] = W[k*4+i]
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
