"""Rewritten program for 4x5x6 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0

    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24

    def build_schoolbook():
        """Build the schoolbook rank-120 decomposition."""
        rank = n * m * p  # 120
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        return U, V, W, rank

    def build_block_decomp():
        """Build a block decomposition using Strassen for 2x2x2 sub-blocks.
        
        For 4x5x6: partition into 2x2 blocks in i and k dimensions,
        and 2x2 blocks in j dimension (with one extra column).
        
        Block structure:
        - i: 2 blocks of size 2
        - j: 2 blocks of size 2 + 1 block of size 1
        - k: 3 blocks of size 2
        
        For each (i_block, j_block, k_block) triple:
        - If both i_block and k_block are size 2 and j_block is size 2:
          Use Strassen rank-7 decomposition
        - Otherwise: use schoolbook for that sub-block
        
        This gives rank = 2*2*3*7 + remaining_schoolbook_terms
        """
        # Strassen rank-7 vectors for 2x2x2:
        # We need u_r, v_r, w_r in R^2 such that
        # sum_r u_r[i]*v_r[j]*w_r[k] = 1 for all i,j,k in {0,1}
        
        # Standard Strassen decomposition:
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # M2 = (A10+A11)B00       -> C10 += M2
        # M3 = A00(B01-B11)       -> C01 += M3
        # M4 = A11(B10-B00)       -> C00 += M4, C10 += M4
        # M5 = (A00+A01)B11       -> C01 += M5, C00 -= M5
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        # M7 = (A01-A11)(B10+B11) -> C00 += M7, C11 += M7
        
        # In tensor form, each Strassen "multiply" corresponds to one rank-1 term
        # where the U, V, W vectors encode the linear combinations.
        
        # For M1: (A00+A11)(B00+B11)
        # U vector: [1, 1] (coefficients for A00, A11)
        # V vector: [1, 1] (coefficients for B00, B11)
        # W contributions: C00 gets +1, C11 gets +1
        # But W is a single vector... 
        # 
        # Actually, in the tensor decomposition, each rank-1 term r has:
        # U_r[i] * V_r[j] * W_r[k] summed over r = T[i,j,k]
        # 
        # For Strassen, the 7 terms are:
        # r=0: U=[1,1], V=[1,1], W=[1,0]  -> no, this doesn't work directly
        # 
        # Let me think again. The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
        # We need: sum_r U_r[i] * V_r[j] * W_r[k] = 1
        # 
        # One valid decomposition (from literature):
        # Using the fact that 1 = 1*1*1, we can use:
        # r=0: U=[1,0], V=[1,0], W=[1,0]  (A00*B00 -> C00)
        # r=1: U=[1,0], V=[0,1], W=[0,1]  (A00*B10 -> C01)
        # ... this is just schoolbook (8 terms)
        # 
        # For Strassen rank 7, we need to be clever. The key is that the
        # matrix multiplication tensor for 2x2 has border rank 7, and there
        # exists an exact rank-7 decomposition over the reals.
        # 
        # Here's one explicit rank-7 decomposition:
        # We use the identity:
        # 1 = (x+y)(u+v) - xu - yv + ... no, let me look this up properly.
        # 
        # Actually, for the 2x2x2 multiplication tensor, a rank-7 decomposition
        # can be constructed as follows. Consider:
        # 
        # The seven "products" in Strassen's algorithm:
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)B00
        # P3 = A00(B01-B11)
        # P4 = A11(B10-B00)
        # P5 = (A00+A01)B11
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        # 
        # Each P_r is a product of a linear combination of A entries and a
        # linear combination of B entries. In the tensor decomposition, each
        # P_r corresponds to a rank-1 term where:
        # U_r encodes the coefficients on the A side
        # V_r encodes the coefficients on the B side  
        # W_r encodes how P_r contributes to C entries
        # 
        # But W_r must be a fixed vector, not depending on which C entry.
        # The contribution of P_r to C[i,k] is W_r[k] (in our indexing W[k*n+i]).
        # 
        # For P1 = (A00+A11)(B00+B11):
        # U_1 = [1, 1] (A00 and A11)
        # V_1 = [1, 1] (B00 and B11)
        # P1 contributes to C00 with +1 and C11 with +1
        # In our W indexing: W[k*n+i], so C00 is W[0*2+0]=W[0], C11 is W[1*2+1]=W[3]
        # W_1 = [1, 0, 0, 1]
        
        # Let me define all 7 terms:
        # Using local indices: i,j,k in {0,1}
        # U_r[i], V_r[j], W_r[k*2+i]
        
        strassen_terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # P1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
            ([1, 1], [1, 1], [1, 0, 0, 1]),
            # P2: (A10+A11)B00 -> C10 += 1
            ([0, 1], [1, 0], [0, 1, 0, 0]),
            # P3: A00(B01-B11) -> C01 += 1
            ([1, 0], [0, 1], [0, 0, 1, 0]),
            # P4: A11(B10-B00) -> C00 += 1, C10 += 1
            ([0, 1], [1, 0], [1, 1, 0, 0]),
            # P5: (A00+A01)B11 -> C01 += 1, C00 -= 1
            ([1, 1], [0, 1], [-1, 0, 1, 0]),
            # P6: (A10-A00)(B00+B01) -> C11 += 1
            ([1, -1], [1, 1], [0, 0, 0, 1]),
            # P7: (A01-A11)(B10+B11) -> C00 += 1, C11 += 1
            ([1, -1], [1, 1], [1, 0, 0, 1]),
        ]
        
        # Verify Strassen decomposition
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for U_c, V_c, W_c in strassen_terms:
                        s += U_c[i] * V_c[j] * W_c[k*2+i]
                    if abs(s - 1.0) > 1e-10:
                        # Fallback: use schoolbook for 2x2x2
                        pass
                        break
        
        # Check if Strassen works
        strassen_valid = True
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for U_c, V_c, W_c in strassen_terms:
                        s += U_c[i] * V_c[j] * W_c[k*2+i]
                    if abs(s - 1.0) > 1e-10:
                        strassen_valid = False
                        break
                if not strassen_valid:
                    break
            if not strassen_valid:
                break
        
        rank = 0
        U = [[0] * (n*m*p) for _ in range(nm)]  # Pre-allocate max size
        V = [[0] * (n*m*p) for _ in range(mp)]
        W = [[0] * (n*m*p) for _ in range(pn)]
        
        if strassen_valid:
            # Use Strassen for 2x2x2 blocks, schoolbook for others
            for ib in range(2):  # i blocks: 0, 1 (each size 2)
                for jb in range(3):  # j blocks: 0,1 (size 2), 2 (size 1)
                    for kb in range(3):  # k blocks: 0,1,2 (each size 2)
                        i0 = ib * 2
                        j0 = jb * 2 if jb < 2 else 4
                        k0 = kb * 2
                        j_size = 2 if jb < 2 else 1
                        
                        if ib < 2 and jb < 2 and kb < 3 and j_size == 2:
                            # Use Strassen: 7 rank-1 terms
                            for U_c, V_c, W_c in strassen_terms:
                                for ii in range(2):
                                    for jj in range(2):
                                        for kk in range(2):
                                            a = (i0+ii) * m + (j0+jj)
                                            b = (j0+jj) * p + (k0+kk)
                                            c = (k0+kk) * n + (i0+ii)
                                            U[a][rank] = U_c[ii]
                                            V[b][rank] = V_c[jj]
                                            W[c][rank] = W_c[kk*2+ii]
                                rank += 1
                        else:
                            # Schoolbook for this sub-block
                            i_size = 2
                            for ii in range(i_size):
                                for jj in range(j_size):
                                    for kk in range(2):
                                        a = (i0+ii) * m + (j0+jj)
                                        b = (j0+jj) * p + (k0+kk)
                                        c = (k0+kk) * n + (i0+ii)
                                        U[a][rank] = 1
                                        V[b][rank] = 1
                                        W[c][rank] = 1
                                        rank += 1
        else:
            # Fall back to full schoolbook
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a][rank] = 1
                        V[b][rank] = 1
                        W[c][rank] = 1
                        rank += 1
        
        # Trim to actual rank
        U = [row[:rank] for row in U]
        V = [row[:rank] for row in V]
        W = [row[:rank] for row in W]
        
        return U, V, W, rank

    def verify(U, V, W, rank):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    s = 0.0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True

    # Try block decomposition first
    U_block, V_block, W_block, rank_block = build_block_decomp()
    
    # Verify block decomposition
    if verify(U_block, V_block, W_block, rank_block):
        best_result = {"U": U_block, "V": V_block, "W": W_block}
        best_rank = rank_block
    else:
        # Fall back to schoolbook
        U_school, V_school, W_school, rank_school = build_schoolbook()
        if not verify(U_school, V_school, W_school, rank_school):
            raise ValueError("Schoolbook decomposition failed verification")
        best_result = {"U": U_school, "V": V_school, "W": W_school}
        best_rank = rank_school
    
    return best_result
# EVOLVE-BLOCK-END
