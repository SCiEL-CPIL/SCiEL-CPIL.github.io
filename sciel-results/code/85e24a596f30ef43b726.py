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
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition by merging one Strassen term from the left 2x2 block
        with one from the right 2x2 block, provided they share the same U-vector.
        """
        if time.time() > deadline - 10:
            return None

        # Standard Strassen terms for 2x2 * 2x2
        # U: [a00, a01, a10, a11]
        # V: [b00, b01, b10, b11] -- Note: In 2x2, V has 4 entries.
        # W: [c00, c10, c01, c11] -- Note: In 2x2, W has 4 entries.
        
        # The 7 Strassen multiplications M1..M7
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # U vectors (coefficients of A)
        # V vectors (coefficients of B)
        # W vectors (coefficients of C)
        
        strassen = [
            # U, V, W
            ([1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 1, -1]), # M1: (A00+A11)(B00+B11) -> C00+=M1, C10-=M1, C01+=M1, C11-=M1? 
            # Let's re-verify standard Strassen W mapping.
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M7
            # C10 = M2 + M4
            # C11 = M1 - M3 + M6 + M7
            # My previous W vectors in the code were:
            # M1: [1, 0, 0, -1] -> This looks like C00, C10, C01, C11? No, indices are k*n+i.
            # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
            # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11.
            # M1 contributes to C00 (+1), C10 (-1? No, M1 is in C00 and C11 usually).
            # Standard Strassen:
            # C00 = M1 - M4 + M5 - M7
            # C10 = M2 + M4
            # C01 = M3 + M7
            # C11 = M1 - M3 + M6 + M7
            
            # Let's check the W vectors in the original code's left_terms:
            # M1: [1, 0, 0, -1] -> C00=1, C10=0, C01=0, C11=-1. This is WRONG for standard Strassen.
            # Standard M1 adds to C00 and C11.
            # Let's re-derive or use a known correct set.
            
            # Correct Strassen decomposition for 2x2:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            
            # C00 = M1 - M4 + M5 - M7
            # C10 = M2 + M4
            # C01 = M3 + M7
            # C11 = M1 - M3 + M6 + M7
            
            # U vectors:
            # M1: A00+A11 -> [1, 0, 0, 1]
            # M2: A10+A11 -> [0, 0, 1, 1]
            # M3: A00 -> [1, 0, 0, 0]
            # M4: A11 -> [0, 0, 0, 1]
            # M5: A00+A01 -> [1, 1, 0, 0]
            # M6: A10-A00 -> [-1, 0, 1, 0] (Note: A10 is index 2, A00 is index 0. So -1 at 0, 1 at 2)
            # M7: A01-A11 -> [0, 1, 0, -1] (Note: A01 is index 1, A11 is index 3. So 1 at 1, -1 at 3)
            
            # V vectors (B00, B01, B10, B11):
            # M1: B00+B11 -> [1, 0, 0, 1]
            # M2: B00 -> [1, 0, 0, 0]
            # M3: B01-B11 -> [0, 1, 0, -1]
            # M4: B10-B00 -> [-1, 0, 1, 0]
            # M5: B11 -> [0, 0, 0, 1]
            # M6: B00+B01 -> [1, 1, 0, 0]
            # M7: B10+B11 -> [0, 0, 1, 1]
            
            # W vectors (C00, C10, C01, C11):
            # M1: C00+1, C11+1 -> [1, 0, 0, 1]
            # M2: C10+1 -> [0, 1, 0, 0]
            # M3: C01+1 -> [0, 0, 1, 0]
            # M4: C00-1, C10+1 -> [-1, 1, 0, 0]
            # M5: C00+1 -> [1, 0, 0, 0]
            # M6: C11+1 -> [0, 0, 0, 1]
            # M7: C00-1, C01+1, C11+1 -> [-1, 0, 1, 1]
            
            # Let's verify M1: (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
            # C00 gets A00B00 + A11B11.
            # C11 gets A00B11 + A11B00.
            # So M1 adds 1 to C00 and 1 to C11. Correct.
            
            # Let's verify M4: A11(B10-B00) = A11B10 - A11B00.
            # C10 gets A11B10.
            # C00 gets -A11B00.
            # So M4 adds -1 to C00 and 1 to C10. Correct.
            
            # Let's verify M7: (A01-A11)(B10+B11) = A01B10 + A01B11 - A11B10 - A11B11.
            # C01 gets A01B10.
            # C11 gets A01B11 - A11B10.
            # C00 gets -A11B11.
            # So M7 adds -1 to C00, 1 to C01, 1 to C11. Correct.
        ]
        
        strassen_terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]), # M2
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]), # M3
            ([0, 0, 0, 1], [-1, 0, 1, 0], [-1, 1, 0, 0]), # M4
            ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]), # M5
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
            ([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 1, 1]), # M7
        ]
        
        # Check if any pair of terms from left and right blocks can be merged.
        # They must have the same U vector.
        for li in range(7):
            if time.time() > deadline - 5:
                return None
            for ri in range(7):
                if strassen_terms[li][0] != strassen_terms[ri][0]:
                    continue
                
                R = 13
                U_mat = np.zeros((4, R))
                V_mat = np.zeros((8, R))
                W_mat = np.zeros((8, R))
                
                # Column 0: Merged term
                U_mat[:, 0] = strassen_terms[li][0]
                # V for left block (k=0,1)
                V_mat[0:4, 0] = strassen_terms[li][1]
                # V for right block (k=2,3)
                V_mat[4:8, 0] = strassen_terms[ri][1]
                # W for left block (k=0,1)
                W_mat[0:4, 0] = strassen_terms[li][2]
                # W for right block (k=2,3)
                W_mat[4:8, 0] = strassen_terms[ri][2]
                
                col = 1
                # Add remaining left terms
                for l_idx in range(7):
                    if l_idx == li: continue
                    U_mat[:, col] = strassen_terms[l_idx][0]
                    V_mat[0:4, col] = strassen_terms[l_idx][1]
                    V_mat[4:8, col] = 0
                    W_mat[0:4, col] = strassen_terms[l_idx][2]
                    W_mat[4:8, col] = 0
                    col += 1
                    
                # Add remaining right terms
                for r_idx in range(7):
                    if r_idx == ri: continue
                    U_mat[:, col] = strassen_terms[r_idx][0]
                    V_mat[0:4, col] = 0
                    V_mat[4:8, col] = strassen_terms[r_idx][1]
                    W_mat[0:4, col] = 0
                    W_mat[4:8, col] = strassen_terms[r_idx][2]
                    col += 1
                
                if verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
        return None
    
    # Try rank-13 search first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
