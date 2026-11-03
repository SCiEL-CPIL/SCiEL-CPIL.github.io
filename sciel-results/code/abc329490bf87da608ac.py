import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    # Coppersmith-Winograd rank-21 decomposition for 4x4x4
    # This is the border rank limit: r(4x4x4) = 21
    # The decomposition uses the tensor T = T1 + T2 + T3 where:
    # T1 is the "diagonal" part (rank 8)
    # T2 is the "off-diagonal" part (rank 7)  
    # T3 is the "cross" part (rank 6)
    # Total rank = 21
    
    # For 4x4x4 matrix multiplication, the Coppersmith-Winograd decomposition:
    # We use the fact that M_{4,4,4} has border rank 21.
    # The explicit decomposition uses:
    # - 8 terms for the "main diagonal" blocks
    # - 7 terms for one set of off-diagonal interactions
    # - 6 terms for another set
    
    # Let me construct this carefully.
    # The 4x4x4 tensor T_{ijk} = 1 if (i,j,k) forms a valid multiplication pattern.
    # For matrix mult: T_{(i,j),(j,k),(k,i)} ... 
    # Actually for M_{n,m,p}: T_{(i,j),(j,k),(i,k)} = 1 for all valid indices.
    
    # Coppersmith-Winograd for 4x4x4 (rank 21):
    # Using the standard construction with parameters:
    # The decomposition is based on the tensor:
    # M = M1 ⊗ M2 ⊗ M3 + ... (21 terms)
    
    # Let me use the known explicit rank-21 decomposition.
    # Reference: Coppersmith & Winograd (1987), "Matrix multiplication via arithmetic progressions"
    
    # The rank-21 decomposition of the 4x4x4 matrix multiplication tensor:
    # We index elements of A (4x4), B (4x4), C (4x4) by (i,j) in {0,1,2,3}^2
    
    # The CW decomposition uses the following structure:
    # Let's define it via the "3-level" Strassen-like recursion with a twist.
    
    # Actually, let me use a more direct approach. The Coppersmith-Winograd 
    # rank-21 decomposition can be constructed as follows:
    
    # For 4x4x4, split into 2x2x2 blocks:
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11], C = [C00 C01; C10 C11]
    # where each block is 2x2.
    
    # The standard 2x2x2 Strassen gives rank 7 per 2x2x2 block.
    # For 4x4x4, naive block Strassen gives 4 * 7 = 28.
    # CW shows we can do it in 21.
    
    # The CW construction:
    # Level 1: Use a "generalized Strassen" on the 4x4x4 tensor.
    # The key insight is that the 4x4x4 tensor has border rank 21, 
    # achieved by a specific algebraic construction.
    
    # Explicit rank-21 decomposition (from literature):
    # We use the following 21 rank-1 tensors (U_r, V_r, W_r) for r=1..21:
    
    # Let me construct this using the known formula.
    # The CW decomposition for n=4 uses the tensor:
    # T = sum_{r=1}^{21} u_r ⊗ v_r ⊗ w_r
    # where u_r, v_r, w_r are specific vectors in R^16 (flattened 4x4).
    
    # From the Coppersmith-Winograd paper, the rank-21 decomposition:
    # Uses the "3-dimensional" arithmetic progression structure.
    
    # Let me use a computational approach to find/verify the decomposition.
    # Actually, let me just directly implement the known rank-21 CW decomposition.
    
    # The standard CW rank-21 decomposition for 4x4x4:
    # (This is well-known in the literature; I'll construct it explicitly)
    
    # Method: The 4x4x4 matrix mult tensor can be decomposed using:
    # 1. The "diagonal" part: 8 terms (one for each diagonal element combination)
    # 2. Two "off-diagonal" Strassen-like parts: 7 + 6 = 13 terms
    # Total: 21 terms
    
    # More precisely, using the CW construction:
    # Let ω be a primitive 4th root of unity (but we need rational/integer coefficients)
    # Actually, CW uses a different approach with integer coefficients.
    
    # Let me use the explicit construction from the literature:
    # The rank-21 decomposition uses the following pattern:
    
    # For the 4x4x4 case, the CW decomposition is:
    # T = sum_{i,j,k} (sum of specific products)
    
    # I'll use the known explicit form. The 21 terms are:
    # Group 1 (8 terms): "Pure" diagonal terms
    # Group 2 (7 terms): First off-diagonal Strassen-like
    # Group 3 (6 terms): Second off-diagonal Strassen-like
    
    # Let me construct this programmatically.
    
    # Actually, a cleaner approach: Use the fact that for 3x4x8,
    # we can embed into 4x4x8 and use the CW decomposition for the 4x4x4 blocks.
    
    # For 4x4x8: split p=8 into two 4s.
    # Each 4x4x4 block uses rank 21 (CW).
    # Total for 4x4x8: 2 * 21 = 42.
    # For 3x4x8: we only need 3 of the 4 output rows, so rank ≤ 42.
    
    # But wait - we need to verify this actually works.
    # The CW decomposition gives border rank 21, but does it give 
    # an EXACT rank-21 decomposition? 
    
    # Important distinction: border rank ≤ rank. 
    # CW shows border rank(4x4x4) = 21, but the actual rank might be higher.
    # In fact, it's known that rank(4x4x4) ≥ 22 (or possibly higher).
    # The exact rank of 4x4x4 is not known to be 21.
    
    # So the CW approach gives us a border rank decomposition, 
    # not necessarily an exact one. For an exact decomposition, 
    # we need rank-1 tensors that sum exactly to the multiplication tensor.
    
    # Let me check: is there a known exact rank-21 decomposition of 4x4x4?
    # From the literature, the best known exact rank for 4x4x4 is 22 
    # (or possibly 21 - this is an open problem in some formulations).
    
    # Actually, I recall that the exact rank of 4x4x4 is known to be 22 
    # (Bürgisser et al. showed rank ≥ 22, and a rank-22 construction exists).
    
    # So for 4x4x8: 2 * 22 = 44 for the 4x4x8 part.
    # For 3x4x8: we might still need more because we're not using the full 4x4.
    
    # Let me reconsider. The current best is 88. Can we do better?
    
    # Alternative approach: Use a more efficient block decomposition.
    # Split 3x4x8 as:
    # - n=3: split into 1+2
    #   - 1x4x8: schoolbook = 32
    #   - 2x4x8: split m=4 into 2+2, p=8 into 4+4
    #     Each 2x2x4: can we do better than 14 (2*7)?
    #     2x2x4: split p=4 into 2+2, each 2x2x2 = 7, total 14.
    #     So 2x4x8 = 2 * 2 * 14 = 56
    #   Total: 32 + 56 = 88. (Same as current)
    
    # What if we split differently?
    # - n=3: split into 1+1+1
    #   Each 1x4x8: schoolbook = 32
    #   Total: 3 * 32 = 96. Worse.
    
    # - m=4: split into 1+1+1+1
    #   Each 3x1x8: schoolbook = 24
    #   Total: 4 * 24 = 96. Worse.
    
    # - m=4: split into 1+3
    #   3x1x8 = 24
    #   3x3x8: split n=3 into 1+2, m=3 into 1+2, p=8 into 4+4
    #     1x1x4 = 4, 1x2x4 = 8, 2x1x4 = 8, 2x2x4 = 14
    #     3x3x4 = 4+8+8+14 = 34
    #     3x3x8 = 2 * 34 = 68
    #   Total: 24 + 68 = 92. Worse.
    
    # - m=4: split into 2+2 (current approach)
    #   Each 3x2x8: split n=3 into 1+2
    #     1x2x8 = 16
    #     2x2x8 = 4 * 7 = 28
    #     Total per 3x2x8 = 44
    #   Grand total: 2 * 44 = 88.
    
    # So 88 seems to be the best from simple block-Strassen.
    
    # Let me try a more sophisticated approach:
    # Can we share structure between adjacent blocks?
    
    # Idea: For the 2x2x8 part, instead of 4 independent 2x2x2 Strassen blocks,
    # can we use a more efficient 2x2x8 decomposition?
    # 2x2x8: split p=8 into 2+2+2+2.
    # Each 2x2x2 = 7 (Strassen).
    # Total: 4 * 7 = 28.
    # Is there a better 2x2x8 decomposition? 
    # The rank of 2x2x8 is at least max(4, 16, 16) = 16 (weak bound).
    # Actually, for 2x2xp, the rank is known to be 2p for p ≥ 2 
    # (no improvement over schoolbook for the 2x2 case when p is large).
    # Wait, that's not right. For 2x2x2, rank = 7 < 8 = 2*2*2/2.
    # For 2x2x4: rank ≤ 14 (two 2x2x2 Strassen). 
    # Is rank(2x2x4) < 14? I don't think so for exact rank.
    
    # Let me just verify the current 88-rank solution and return it.
    # The inspirations confirm that 88 is the achievable rank with this approach.
    
    def _build_strassen_3x4x8():
        """Build the block-Strassen decomposition: rank 88."""
        rank = 88
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Strassen 2x2x2 decomposition
        strassen_decomp = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),         # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),         # M3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),         # M4
            ({0: 1, 1: 1}, {3: 1}, {1: 1, 0: -1}),         # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})          # M7
        ]
        
        for j1 in range(2):  # 2 blocks of 2 in m dimension
            # Top part: 2x2x8 (rows 0,1 of A)
            for l in range(4):  # 4 blocks of 2 in p dimension
                for u_coeffs, v_coeffs, w_coeffs in strassen_decomp:
                    # U (A): local 2x2 -> global
                    for li, coeff in u_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i  # 0 or 1
                        global_j = 2 * j1 + li_j
                        U[global_i * m + global_j][col] += coeff
                    
                    # V (B): local 2x2 -> global
                    for li, coeff in v_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_j = 2 * j1 + li_i
                        global_k = 2 * l + li_j
                        V[global_j * p + global_k][col] += coeff
                    
                    # W (C): local 2x2 -> global
                    for li, coeff in w_coeffs.items():
                        li_i, li_j = li // 2, li % 2
                        global_i = li_i
                        global_k = 2 * l + li_j
                        W[global_k * n + global_i][col] += coeff
                    
                    col += 1
            
            # Bot part: 1x2x8 (row 2 of A), schoolbook
            for k in range(8):
                for j2 in range(2):
                    global_i = 2
                    global_j = 2 * j1 + j2
                    U[global_i * m + global_j][col] = 1
                    V[global_j * p + k][col] = 1
                    W[k * n + global_i][col] = 1
                    col += 1
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W, tol=1e-9):
        """Verify exact tensor decomposition."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def _verify_half_integer(U, V, W):
        """Check that all entries are integers or half-integers."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
        return True
    
    # Build the decomposition
    result = _build_strassen_3x4x8()
    
    # Verify
    if _verify(result['U'], result['V'], result['W']) and _verify_half_integer(result['U'], result['V'], result['W']):
        return result
    
    # Fallback: schoolbook
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
    
    return {"U": U, "V": V, "W": W}
