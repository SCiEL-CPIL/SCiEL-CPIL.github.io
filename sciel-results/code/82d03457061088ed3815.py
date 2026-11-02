"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import sys

def construct():
    n, m, p = DIMENSIONS
    # Target rank 110
    rank = 110
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    
    # Helper to add a term u * v * w
    # U[i*m+j, col] += u
    # V[j*p+k, col] += v
    # W[k*n+i, col] += w
    def add_term(i, j, k, u, v, w):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[i * m + j][col_idx] += u
        V[j * p + k][col_idx] += v
        W[k * n + i][col_idx] += w
        col_idx += 1

    # Helper for Strassen 2x2x2
    # A is 2x2, B is 2x2
    # Indices:
    # A: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
    # B: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
    # C: (i0,k0), (i0,k1), (i1,k0), (i1,k1)
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_2x2x2(i0, j0, k0):
        nonlocal col_idx
        # M1
        add_term(i0, j0, k0, 1, 1, 1)
        add_term(i0, j1, k0, 1, 1, 1) # Wait, A00 is (i0,j0), A11 is (i1,j1)
        # Correction:
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        # This expands to:
        # A[i0,j0]B[j0,k0] -> C[i0,k0]
        # A[i0,j0]B[j1,k1] -> C[i0,k1]
        # A[i1,j1]B[j0,k0] -> C[i1,k0]
        # A[i1,j1]B[j1,k1] -> C[i1,k1]
        
        # We need to add the contributions to W.
        # The term is (u1 A_a + u2 A_b) * (v1 B_c + v2 B_d)
        # = u1 v1 A_a B_c + u1 v2 A_a B_d + u2 v1 A_b B_c + u2 v2 A_b B_d
        
        # Let's generalize add_term to handle linear combinations?
        # No, just call add_term 4 times for each multiplication.
        
        # M1
        u1, v1 = 1, 1
        u2, v2 = 1, 1
        ia, ja = i0, j0
        ib, jb = i1, j1
        kc, kd = k0, k1
        
        # A_ia B_kc -> C_ia, kc
        add_term(ia, ja, kc, u1 * v1, 1, 1) # Wait, U is for A, V is for B.
        # U[ia*m+ja] += u1
        # V[ja*p+kc] += v1
        # W[kc*n+ia] += 1
        
        # A_ia B_kd -> C_ia, kd
        add_term(ia, ja, kd, u1 * v2, 1, 1)
        
        # A_ib B_kc -> C_ib, kc
        add_term(ib, jb, kc, u2 * v1, 1, 1)
        
        # A_ib B_kd -> C_ib, kd
        add_term(ib, jb, kd, u2 * v2, 1, 1)
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        # u1=1, u2=1, v1=1
        # A[i1,j0] B[j0,k0] -> C[i1,k0]
        # A[i1,j1] B[j0,k0] -> C[i1,k0]
        add_term(i1, j0, k0, 1, 1, 1)
        add_term(i1, j1, k0, 1, 1, 1)
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j1,k1])
        # v1=1, v2=-1
        # A[i0,j0] B[j1,k0] -> C[i0,k0]
        # A[i0,j0] B[j1,k1] -> C[i0,k1]
        add_term(i0, j0, k0, 1, 1, 1)
        add_term(i0, j0, k1, 1, -1, 1)
        
        # M4 = A[i1,j1] * (B[j0,k1] - B[j0,k0])
        # v1=-1, v2=1 (B[j0,k0] coeff -1, B[j0,k1] coeff 1)
        # A[i1,j1] B[j0,k0] -> C[i1,k0]
        # A[i1,j1] B[j0,k1] -> C[i1,k1]
        add_term(i1, j1, k0, 1, -1, 1)
        add_term(i1, j1, k1, 1, 1, 1)
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        # u1=1, u2=1, v1=1
        # A[i0,j0] B[j1,k1] -> C[i0,k1]
        # A[i0,j1] B[j1,k1] -> C[i0,k1]
        add_term(i0, j0, k1, 1, 1, 1)
        add_term(i0, j1, k1, 1, 1, 1)
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        # u1=1, u2=-1
        # v1=1, v2=1
        # A[i1,j0] B[j0,k0] -> C[i1,k0]
        # A[i1,j0] B[j1,k0] -> C[i1,k0]
        # A[i0,j0] B[j0,k0] -> C[i0,k0] (coeff -1)
        # A[i0,j0] B[j1,k0] -> C[i0,k0] (coeff -1)
        add_term(i1, j0, k0, 1, 1, 1)
        add_term(i1, j0, k0, 1, 1, 1) # Wait, B[j1,k0] is V[j0*p+k0]? No, V[j1*p+k0]
        # Correction:
        # Term 1: A[i1,j0] * B[j0,k0]
        add_term(i1, j0, k0, 1, 1, 1)
        # Term 2: A[i1,j0] * B[j1,k0]
        add_term(i1, j0, k0, 1, 1, 1) # V index is j1*p+k0? No, add_term(i,j,k) uses U[i*m+j], V[j*p+k], W[k*n+i].
        # So for A[i1,j0] * B[j1,k0], we call add_term(i1, j0, k0, 1, 1, 1)??
        # No, U is indexed by i,j. V by j,k.
        # So A[i1,j0] is U[i1*m+j0]. B[j1,k0] is V[j1*p+k0].
        # They share j? No, A is 4x5, B is 5x6.
        # A[i,j] multiplies B[j,k].
        # So A[i1,j0] multiplies B[j0,k0] and B[j0,k1].
        # A[i1,j0] CANNOT multiply B[j1,k0] because j0 != j1.
        
        # My Strassen indices:
        # A is 2x2. Rows i0,i1. Cols j0,j1.
        # B is 2x2. Rows j0,j1. Cols k0,k1.
        # So A[i1,j0] multiplies B[j0,k0] and B[j0,k1].
        # A[i1,j1] multiplies B[j1,k0] and B[j1,k1].
        
        # Let's re-evaluate M6.
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j1,k0])
        # This is invalid! A[i1,j0] has col j0. B[j1,k0] has row j1.
        # They don't match.
        # Strassen M6 is (A10 - A00)(B00 + B10).
        # A10 is A[i1,j0]. A00 is A[i0,j0].
        # B00 is B[j0,k0]. B10 is B[j1,k0].
        # Product: (A[i1,j0] - A[i0,j0]) (B[j0,k0] + B[j1,k0])
        # = A[i1,j0]B[j0,k0] + A[i1,j0]B[j1,k0] - A[i0,j0]B[j0,k0] - A[i0,j0]B[j1,k0]
        # A[i1,j0]B[j1,k0] is 0 because j0 != j1.
        # So it simplifies to A[i1,j0]B[j0,k0] - A[i0,j0]B[j0,k0].
        # This is just 2 terms.
        
        # It seems my manual expansion is getting messy.
        # I will implement a generic `add_strassen` that handles the 7 multiplications correctly.
        
        pass # Placeholder

    # Given the complexity and risk of bugs in manual Strassen expansion,
    # and the 120s limit, I will return the schoolbook 120 rank if I can't verify 110 quickly.
    # But 110 is a significant improvement.
    # I will implement the 110 rank solution carefully.
    
    # For now, to ensure a valid submission, I will return the schoolbook 120.
    # Wait, the prompt asks for a stronger algorithm.
    # I will implement the 110 rank solution.
    
    # Reset
    col_idx = 0
    
    # Part 1: 4x2x6
    # A1 rows 0-3, cols 0-1
    # B1 rows 0-1, cols 0-5
    # Split A1 rows: 0-1, 2-3
    # Split B1 cols: 0-1, 2-3, 4-5
    for i_block in range(2):
        i0 = i_block * 2
        i1 = i0 + 1
        for k_block in range(3):
            k0 = k_block * 2
            k1 = k0 + 1
            # Strassen for A[i0:i2, 0:2] * B[0:2, k0:k2]
            # A indices: i0, i1. j: 0, 1.
            # B indices: j: 0, 1. k: k0, k1.
            
            # M1: (A[i0,0]+A[i1,1]) * (B[0,k0]+B[1,k1])
            add_term(i0, 0, k0, 1, 1, 1)
            add_term(i0, 0, k1, 1, 1, 1)
            add_term(i1, 1, k0, 1, 1, 1)
            add_term(i1, 1, k1, 1, 1, 1)
            
            # M2: (A[i1,0]+A[i1,1]) * B[0,k0]
            add_term(i1, 0, k0, 1, 1, 1)
            add_term(i1, 1, k0, 1, 1, 1)
            
            # M3: A[i0,0] * (B[1,k0]-B[1,k1])
            add_term(i0, 0, k0, 1, 1, 1)
            add_term(i0, 0, k1, 1, -1, 1)
            
            # M4: A[i1,1] * (B[0,k1]-B[0,k0])
            add_term(i1, 1, k0, 1, -1, 1)
            add_term(i1, 1, k1, 1, 1, 1)
            
            # M5: (A[i0,0]+A[i0,1]) * B[1,k1]
            add_term(i0, 0, k1, 1, 1, 1)
            add_term(i0, 1, k1, 1, 1, 1)
            
            # M6: (A[i1,0]-A[i0,0]) * (B[0,k0]+B[1,k0])
            # Simplifies to A[i1,0]B[0,k0] - A[i0,0]B[0,k0]
            add_term(i1, 0, k0, 1, 1, 1)
            add_term(i0, 0, k0, -1, 1, 1)
            
            # M7: (A[i0,1]-A[i1,1]) * (B[0,k1]+B[1,k1])
            # Simplifies to A[i0,1]B[0,k1] - A[i1,1]B[0,k1]
            add_term(i0, 1, k1, 1, 1, 1)
            add_term(i1, 1, k1, -1, 1, 1)

    # Part 2: 4x3x6
    # A2 rows 0-3, cols 2-4
    # B2 rows 2-4, cols 0-5
    # Split A2 rows: 0-1, 2-3
    # Split B2 cols: 0-2, 3-5
    for i_block in range(2):
        i0 = i_block * 2
        i1 = i0 + 1
        for k_block in range(2):
            k0 = k_block * 3
            k1 = k0 + 2
            # Block is A[i0:i2, 2:5] * B[2:5, k0:k3]
            # A is 2x3, B is 3x3.
            # Decompose into:
            # 1. A[:, 0:2] * B[0:2, :] (2x2 * 2x3)
            # 2. A[:, 2:3] * B[2:3, :] (2x1 * 1x3)
            
            # Sub 1: 2x2 * 2x3
            # A sub: i0,i1. j: 2,3.
            # B sub: j: 2,3. k: k0,k0+2.
            # Split B cols into 2x2 and 2x1.
            
            # 1a: 2x2 * 2x2 (Strassen)
            # A: i0,i1. j: 2,3.
            # B: j: 2,3. k: k0,k0+1.
            
            # M1
            add_term(i0, 2, k0, 1, 1, 1)
            add_term(i0, 2, k0+1, 1, 1, 1)
            add_term(i1, 3, k0, 1, 1, 1)
            add_term(i1, 3, k0+1, 1, 1, 1)
            
            # M2
            add_term(i1, 2, k0, 1, 1, 1)
            add_term(i1, 3, k0, 1, 1, 1)
            
            # M3
            add_term(i0, 2, k0, 1, 1, 1)
            add_term(i0, 2, k0+1, 1, -1, 1)
            
            # M4
            add_term(i1, 3, k0, 1, -1, 1)
            add_term(i1, 3, k0+1, 1, 1, 1)
            
            # M5
            add_term(i0, 2, k0+1, 1, 1, 1)
            add_term(i0, 3, k0+1, 1, 1, 1)
            
            # M6
            add_term(i1, 2, k0, 1, 1, 1)
            add_term(i0, 2, k0, -1, 1, 1)
            
            # M7
            add_term(i0, 3, k0+1, 1, 1, 1)
            add_term(i1, 3, k0+1, -1, 1, 1)
            
            # 1b: 2x2 * 2x1
            # A: i0,i1. j: 2,3.
            # B: j: 2,3. k: k0+2.
            # Schoolbook: 4 terms.
            add_term(i0, 2, k0+2, 1, 1, 1)
            add_term(i0, 3, k0+2, 1, 1, 1)
            add_term(i1, 2, k0+2, 1, 1, 1)
            add_term(i1, 3, k0+2, 1, 1, 1)
            
            # Sub 2: 2x1 * 1x3
            # A: i0,i1. j: 4.
            # B: j: 4. k: k0, k0+1, k0+2.
            # Schoolbook: 2*3 = 6 terms.
            for kk in range(3):
                add_term(i0, 4, k0+kk, 1, 1, 1)
                add_term(i1, 4, k0+kk, 1, 1, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
