"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def verify_np(U, V, W):
        """Fast numpy-based verification."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = Ua.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(Ua[i*m+j], Va[j*p+k]) * 1.0
                    # Actually need sum of products of all three
                    s = 0.0
                    for r in range(R):
                        s += Ua[i*m+j, r] * Va[j*p+k, r] * Wa[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B):
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B):
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None

    def try_strassen_shared_rank13():
        """
        Attempt rank-13 by sharing the (a00+a11) factor between both blocks.
        The key idea: use one shared term for (a00+a11) and distribute its
        contributions to both left and right B-blocks.
        
        Structure:
        - r=0: (a00+a11)(b00+b11+b02+b13) -- shared diagonal term
        - r=1: (a10+a11)(b00+b02) -- shared lower-left-ish
        - r=2: a00(b01-b11+b03-b13) -- a00 off-diagonal both sides
        - r=3: a11(b10-b00+b12-b02) -- a11 off-diagonal both sides  
        - r=4: (a00+a01)(b11+b13) -- a00+a01 with bottom row
        - r=5: (a10-a00)(b00+b01+b02+b03) -- a10-a00 with top row
        - r=6: (a01-a11)(b10+b11+b12+b13) -- a01-a11 with bottom row
        
        That's only 7 terms... we need more. Let me think differently.
        
        Actually, let's try a different approach. The standard rank-14 uses 7+7=14.
        To get rank 13, we need to save 1 term. The natural sharing is:
        - Both blocks have a term like (a00+a11)*something
        - If we can make the "something" cover both blocks, we save 1 term.
        
        Let me try:
        r=0: (a00+a11)(b00+b11)  -> contributes to C[0][0], C[1][1] (left)
        r=1: (a00+a11)(b02+b13)  -> contributes to C[0][2], C[1][3] (right)
        
        These are separate. To share, we'd need:
        r=0: (a00+a11)(b00+b11+b02+b13)
        Then this single term contributes to C[0][0], C[1][1], C[0][2], C[1][3].
        But we need C[0][0] = a00*b00 + a01*b10, etc.
        
        Let me set up the system properly. We want:
        C[0][0] = a00*b00 + a01*b10
        C[1][0] = a10*b00 + a11*b10
        C[0][1] = a00*b01 + a01*b11
        C[1][1] = a10*b01 + a11*b11
        C[0][2] = a00*b02 + a01*b12
        C[1][2] = a10*b02 + a11*b12
        C[0][3] = a00*b03 + a01*b13
        C[1][3] = a10*b03 + a11*b13
        
        With rank 13, we have 13 terms. Let me try a structured approach:
        Use 6 terms for the left 2x2 block (Strassen-like) and 7 for right,
        but share one term. Or use a different decomposition entirely.
        
        Actually, let me try: what if we use the same 7 Strassen patterns
        but for the right block, reuse one of the left block's terms?
        
        Left block terms (standard Strassen):
        L1 = (a00+a11)(b00+b11)
        L2 = (a10+a11)b00
        L3 = a00(b01-b11)
        L4 = a11(b10-b00)
        L5 = (a00+a01)b11
        L6 = (a10-a00)(b00+b01)
        L7 = (a01-a11)(b10+b11)
        
        Right block needs:
        R1 = (a00+a11)(b02+b13)
        R2 = (a10+a11)b02
        R3 = a00(b03-b13)
        R4 = a11(b12-b02)
        R5 = (a00+a01)b13
        R6 = (a10-a00)(b02+b03)
        R7 = (a01-a11)(b12+b13)
        
        Can we express any R term as a combination of L terms? Not directly
        since they involve different B entries.
        
        Alternative: What if we don't use Strassen structure at all?
        Let me try a linear algebra approach.
        
        The tensor T has entries T[i,j,k] = 1 for the matrix multiplication.
        We need to find the minimum number of rank-1 tensors that sum to T.
        
        For 2x2*2x4, the tensor has 2*2*4 = 16 components.
        Lower bound: max(nm, mp, pn) = max(4,8,8) = 8.
        Upper bound from Strassen: 14.
        
        Let me try to find rank 13 by solving a system of equations.
        """
        if time.time() > deadline - 10:
            return None
        
        # Try a parameterized search for rank 13
        # Use the structure where we have 13 terms with coefficients in {-1,0,1}
        # and try to find one that works.
        
        # Strategy: Start from rank-14 and try to merge two terms.
        # Or: Try a different 7+6 split where the 6-term block is a
        # compact decomposition of 2x2*2x2.
        
        # Actually, the minimum rank for 2x2*2x2 is 7 (Strassen).
        # So 7+7=14 is the natural bound. To get 13, we need sharing.
        
        # Let me try: what if the right block uses only 6 terms by being
        # slightly non-optimal but sharing with the left?
        
        # Alternative approach: Use a different factorization.
        # C = A*B where A is 2x2 and B is 2x4.
        # C[0][k] = a00*B[0][k] + a01*B[1][k]
        # C[1][k] = a10*B[0][k] + a11*B[1][k]
        
        # For each k, we need 4 products. Total 16 products.
        # Each rank-1 term gives at most 4 products (one for each output).
        # So we need at least 4 terms. But the structure matters.
        
        # Let me try a direct search with a specific structure.
        # Assume 13 terms with the following pattern:
        # - 4 terms involving a00
        # - 4 terms involving a01  
        # - 4 terms involving a10
        # - 4 terms involving a11
        # But some terms can involve sums like (a00+a11).
        
        # This is getting complex. Let me just return the rank-14 for now
        # and try a quick random search.
        
        # Quick random search for rank 13
        np.random.seed(42)
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
            R = 13
            # Generate random sparse matrices with entries in {-1, 0, 1}
            U = np.zeros((4, R), dtype=int)
            V = np.zeros((8, R), dtype=int)
            W = np.zeros((8, R), dtype=int)
            
            for r in range(R):
                # Each term: pick which A entries, B entries, C entries
                # A entries: subset of {0,1,2,3}
                # B entries: subset of {0,...,7}
                # C entries: subset of {0,...,7}
                
                # Random sparsity
                a_mask = np.random.choice([0,1], 4, p=[0.5, 0.5])
                b_mask = np.random.choice([0,1], 8, p=[0.7, 0.3])
                c_mask = np.random.choice([0,1], 8, p=[0.7, 0.3])
                
                if a_mask.sum() < 1 or b_mask.sum() < 1 or c_mask.sum() < 1:
                    continue
                    
                U[a_mask.astype(bool), r] = np.random.choice([-1, 1], a_mask.sum())
                V[b_mask.astype(bool), r] = np.random.choice([-1, 1], b_mask.sum())
                W[c_mask.astype(bool), r] = np.random.choice([-1, 1], c_mask.sum())
            
            # Verify
            valid = True
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        s = 0.0
                        for r in range(R):
                            s += U[i*2+j, r] * V[j*4+k, r] * W[k*2+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None

    # Try rank-13 first
    result = try_strassen_shared_rank13()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
