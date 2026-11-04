"""Program for matrix multiplication 4x4 x 4x6 tensor decomposition."""
import numpy as np
import time

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


def _decompose_4x4x2():
    """
    Decompose 4x4 * 4x2 -> 4x2 matrix multiplication.
    
    Structure: 2x2 blocks of size 2x2.
    A is 4x4: 2x2 blocks A[ai][bj]
    B is 4x2: 2x1 blocks B[bj][ck] where ck=0 (only one column block)
    C is 4x2: 2x1 blocks C[ai][ck]
    
    For each (ai, bj) pair, we compute A[ai][bj] * B[bj][0] which is 2x2 * 2x2 -> 2x2
    But B only has 1 column block, so B[bj][0] is 2x2.
    
    Actually, let me think more carefully:
    A is 4x4, B is 4x2, C is 4x2.
    Block structure with block size 2:
    A: 2x2 blocks of 2x2
    B: 2x1 blocks of 2x2 (since 4 rows -> 2 blocks, 2 cols -> 1 block)
    C: 2x1 blocks of 2x2
    
    C[ai][0] = sum_{bj} A[ai][bj] * B[bj][0]
    
    Each product A[ai][bj] * B[bj][0] is 2x2 * 2x2 -> 2x2, using Strassen (rank 7).
    For each ai (2 choices) and bj (2 choices), we have one such product.
    Total: 4 products, each rank 7, total rank = 28.
    """
    sU, sV, sW = _strassen_2x2()
    
    # We need 4 terms: (ai, bj) for ai in {0,1}, bj in {0,1}
    # Each term uses Strassen for the 2x2*2x2 product
    # But we need to handle the fact that B and C have only 1 column block
    
    n, m, p = 4, 4, 2
    total_rank = 4 * 7  # 4 block products, each rank 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ai in range(2):
        for bj in range(2):
            base = (ai * 2 + bj) * 7
            for t in range(7):
                col = base + t
                
                # U part: A[ai][bj] block
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff = sU[ai2*2 + bj2, t]
                        if coeff != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff
                
                # V part: B[bj][0] block
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff = sV[bj2*2 + ck2, t]
                        if coeff != 0:
                            v_row = (2*bj + bj2) * 2 + ck2
                            V[v_row, col] += coeff
                
                # W part: C[ai][0] block
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff = sW[ck2*2 + ai2, t]
                        if coeff != 0:
                            w_row = ck2 * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Strategy: Split 4x4x6 into three independent 4x4x2 subproblems.
    # Each subproblem has rank 28 (4 Strassen 2x2*2x2 products).
    # Total rank: 3 * 28 = 84.
    #
    # This is better than the previous approach which used 77 rank with a
    # different block structure. Wait, 84 > 77. Let me reconsider.
    #
    # Actually, the current program achieves rank 77. Let me verify that's correct.
    # The current approach: 11 outer terms * 7 inner = 77.
    # 
    # Let me try a different split. What if we split 4x4x6 as:
    # - First 4 columns: 4x4x4, decomposed as 2x2x2 block Strassen
    # - Last 2 columns: 4x4x2
    #
    # For 4x4x4: 2x2 blocks, 4 block products, each rank 7 = 28
    # For 4x4x2: 4 block products, each rank 7 = 28
    # Total: 56. That's much better!
    
    # Let me implement 4x4x4 first
    def _decompose_4x4x4():
        n2, m2, p2 = 4, 4, 4
        sU, sV, sW = _strassen_2x2()
        
        # 2x2 blocks of 2x2
        # A: 2x2 blocks, B: 2x2 blocks, C: 2x2 blocks
        # C[ai][ck] = sum_bj A[ai][bj] * B[bj][ck]
        # 4 products: (ai, bj, ck) for ai,ck in {0,1}, bj in {0,1}
        # Wait, for each (ai, ck), we sum over bj. But in tensor decomposition,
        # each rank-1 term corresponds to one (ai, bj, ck) triple.
        # So we have 2*2*2 = 8 block products, each rank 7 = 56.
        # Hmm, that's not right either. Let me think again.
        
        # Actually in the Kronecker product approach:
        # The 4x4x4 multiplication tensor can be written as sum over (ai, bj, ck)
        # of A_block[ai,bj] * B_block[bj,ck] * C_block[ai,ck]
        # where each block product is 2x2*2x2->2x2.
        # There are 2*2*2 = 8 such terms, each with rank 7.
        # Total rank = 8 * 7 = 56.
        
        total_rank = 8 * 7
        U = np.zeros((n2*m2, total_rank), dtype=int)
        V = np.zeros((m2*p2, total_rank), dtype=int)
        W = np.zeros((p2*n2, total_rank), dtype=int)
        
        for ai in range(2):
            for bj in range(2):
                for ck in range(2):
                    base = (ai * 4 + bj * 2 + ck) * 7
                    for t in range(7):
                        col = base + t
                        
                        # U part: A[ai][bj]
                        for ai2 in range(2):
                            for bj2 in range(2):
                                coeff = sU[ai2*2 + bj2, t]
                                if coeff != 0:
                                    u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                                    U[u_row, col] += coeff
                        
                        # V part: B[bj][ck]
                        for bj2 in range(2):
                            for ck2 in range(2):
                                coeff = sV[bj2*2 + ck2, t]
                                if coeff != 0:
                                    v_row = (2*bj + bj2) * 4 + (2*ck + ck2)
                                    V[v_row, col] += coeff
                        
                        # W part: C[ai][ck]
                        for ai2 in range(2):
                            for ck2 in range(2):
                                coeff = sW[ck2*2 + ai2, t]
                                if coeff != 0:
                                    w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                                    W[w_row, col] += coeff
        
        return U, V, W
    
    # For 4x4x6, split into 4x4x4 and 4x4x2
    # 4x4x4: rank 56
    # 4x4x2: rank 28
    # Total: 84
    
    # Wait, that's still 84, same as before. Let me reconsider.
    
    # Actually, the key insight from the knowledge is:
    # "Splitting the 4x4x6 problem into three independent 4x4x2 subproblems,
    # each solved by 4 rank-7 Strassen 2x2*2x2 products, gives rank 84"
    
    # But the current program achieves 77. So 84 is worse.
    
    # Let me try to improve on 77. The current approach uses:
    # - 7 Strassen terms for cols 0,1 (2x2 * 2x2 part)
    # - 4 schoolbook terms for col 2 (2x2 * 1 part)
    # Each outer term is multiplied by inner rank 7.
    # Total: 11 * 7 = 77.
    
    # Can we do better? What if we use a different block size?
    # For example, 1x4 blocks or 4x1 blocks?
    
    # Let me try: A is 4x4, B is 4x6, C is 4x6.
    # Use 1x4 blocks: A has 4 rows of 1x4, B has 4 rows of 1x6, C has 4 rows of 1x6.
    # This doesn't help directly.
    
    # What about using the fact that 4x4x6 can be seen as a Kronecker product?
    # 4x4x6 = (2x2x3) * (2x2x2) in some sense?
    
    # Let me try a different approach: use rank-7 Strassen for the 2x2x2 core,
    # and find a better outer decomposition.
    
    # The outer problem is 2x2 * 2x3 -> 2x3.
    # Current: 7 Strassen + 4 schoolbook = 11 terms.
    # Can we decompose 2x2 * 2x3 with fewer terms?
    
    # The tensor rank of 2x2 * 2x3 matrix multiplication is at most 12 (schoolbook).
    # With Strassen for the 2x2x2 part, we get 7 + 4 = 11.
    # Is there a way to get less than 11?
    
    # Actually, we can potentially share some terms between the Strassen and schoolbook parts.
    # But this is complex to implement.
    
    # For now, let's keep the current approach which achieves 77.
    # The task says to combine strategies, so let me use the cleaner implementation
    # from the inspirations with the same 77-rank structure.
    
    sU, sV, sW = _strassen_2x2()
    
    # Outer decomposition: 2x2 * 2x3 -> 2x3
    # 7 Strassen terms for cols 0,1
    # 4 schoolbook terms for col 2
    outer_terms = []
    
    # 7 Strassen terms for the 2x2 * 2x2 part (cols 0,1)
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
        outer_terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
        
    # 4 schoolbook terms for col 2
    for a in range(2):
        for b in range(2):
            U_dict = {(a,b): 1}
            V_dict = {(b,2): 1}
            W_dict = {(a,2): 1}
            outer_terms.append({'U': U_dict, 'V': V_dict, 'W': W_dict})
            
    total_rank = len(outer_terms) * 7
    U = np.zeros((n*m, total_rank), dtype=int)
    V = np.zeros((m*p, total_rank), dtype=int)
    W = np.zeros((p*n, total_rank), dtype=int)
    
    for ot_idx, outer in enumerate(outer_terms):
        for t in range(7):
            col = ot_idx * 7 + t
            
            for (ai, bj), coeff_outer in outer['U'].items():
                for ai2 in range(2):
                    for bj2 in range(2):
                        coeff_inner = sU[ai2*2 + bj2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            u_row = (2*ai + ai2) * 4 + (2*bj + bj2)
                            U[u_row, col] += coeff_outer * coeff_inner
                            
            for (bj, ck), coeff_outer in outer['V'].items():
                for bj2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sV[bj2*2 + ck2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            v_row = (2*bj + bj2) * 6 + (2*ck + ck2)
                            V[v_row, col] += coeff_outer * coeff_inner
                            
            for (ai, ck), coeff_outer in outer['W'].items():
                for ai2 in range(2):
                    for ck2 in range(2):
                        coeff_inner = sW[ck2*2 + ai2, t]
                        if coeff_outer != 0 and coeff_inner != 0:
                            w_row = (2*ck + ck2) * 4 + (2*ai + ai2)
                            W[w_row, col] += coeff_outer * coeff_inner
                            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
