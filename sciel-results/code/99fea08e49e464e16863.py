"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # New Design: Padded Strassen
    # Pad n=3 to n_pad=4.
    # Compute 4x4x8 multiplication using Strassen on 4x4x4 blocks.
    # 4x4x4 uses 2x2x2 Strassen blocks.
    # Rank of 4x4x4 is 4 * 7 = 28? No.
    # Strassen on 4x4 uses 7 multiplications of 2x2 matrices.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank for 4x4x4 is 7 * 7 = 49.
    # We have two 4x4x4 blocks for the p=8 dimension.
    # Total rank = 2 * 49 = 98.
    
    rank = 98
    
    # U shape: (n*m, rank) -> (12, 98)
    # V shape: (m*p, rank) -> (32, 98)
    # W shape: (p*n, rank) -> (24, 98)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_rank_unit(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W[idx][col] += val
        col += 1

    # Strassen 2x2x2 factors
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    def add_strassen_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """
        Adds Strassen decomposition for a 2x2 A block, 2x2 B block, 2x2 C block.
        Indices are (row, col) in the local 2x2 block.
        """
        u_a = (a_r[0]) * m + (a_c[0])
        u_b = (a_r[0]) * m + (a_c[1])
        u_c = (a_r[1]) * m + (a_c[0])
        u_d = (a_r[1]) * m + (a_c[1])
        
        v_e = (b_r[0]) * p + (b_c[0])
        v_f = (b_r[0]) * p + (b_c[1])
        v_g = (b_r[1]) * p + (b_c[0])
        v_h = (b_r[1]) * p + (b_c[1])
        
        # W index is k*n + i. Here k is the col of B (which is row of C? No, B is m x p).
        # B is m x p. A is n x m. C is n x p.
        # Wait, the problem says:
        # U: (n*m, rank) -> A is n x m.
        # V: (m*p, rank) -> B is m x p.
        # W: (p*n, rank) -> C is p x n?
        # "output factor follows the official trace convention W[k*n+i,r]"
        # This implies C is indexed by k (col of B?) and i (row of A?).
        # Standard: C = A B. A is n x m, B is m x p. C is n x p.
        # W[k*n+i] implies C is p x n?
        # If C is n x p, then index is i*p + k.
        # If C is p x n, then index is k*n + i.
        # The prompt says W[k*n+i]. This suggests the output matrix is transposed or the indices are swapped.
        # Let's assume the standard multiplication tensor but with W indexed by (k, i) where k is the column of B and i is the row of A.
        # This corresponds to C^T or just a different flattening.
        # The condition is sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = delta_{j, j'}?
        # If we use this convention, my Strassen implementation must map C_{i, k} to W_{k*n + i}.
        
        w_c11 = (b_c[0]) * n + (c_r[0]) # k=b_c[0], i=c_r[0]
        w_c12 = (b_c[1]) * n + (c_r[0]) # k=b_c[1], i=c_r[0]
        w_c21 = (b_c[0]) * n + (c_r[1]) # k=b_c[0], i=c_r[1]
        w_c22 = (b_c[1]) * n + (c_r[1]) # k=b_c[1], i=c_r[1]
        
        # M1
        add_rank_unit({u_a: 1, u_d: 1}, {v_e: 1, v_h: 1}, {w_c11: 1, w_c22: 1})
        # M2
        add_rank_unit({u_c: 1, u_d: 1}, {v_e: 1}, {w_c21: 1, w_c22: -1})
        # M3
        add_rank_unit({u_a: 1}, {v_f: 1, v_h: -1}, {w_c12: 1, w_c22: 1})
        # M4
        add_rank_unit({u_d: 1}, {v_g: 1, v_e: -1}, {w_c11: 1, w_c21: 1})
        # M5
        add_rank_unit({u_a: 1, u_b: 1}, {v_h: 1}, {w_c11: -1, w_c12: 1})
        # M6
        add_rank_unit({u_c: 1, u_a: -1}, {v_e: 1, v_f: 1}, {w_c22: 1})
        # M7
        add_rank_unit({u_b: 1, u_d: -1}, {v_g: 1, v_h: 1}, {w_c11: 1})

    # We need to build 4x4x4 Strassen.
    # A is 4x4, B is 4x4, C is 4x4.
    # Split A into 4 2x2 blocks: A11, A12, A21, A22.
    # Split B into 4 2x2 blocks: B11, B12, B21, B22.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # M1 = (A11+A22)(B11+B22)
    # This is a 2x2x2 multiplication of (A11+A22) and (B11+B22).
    # We need to represent the sum A11+A22 in U.
    # U for M1 will have coefficients for A11 and A22.
    
    # Let's implement the 4x4x4 Strassen by adding the 7 2x2x2 multiplications.
    # Each 2x2x2 multiplication contributes 7 rank units.
    # Total 49 rank units per 4x4x4 block.
    
    def add_strassen_4x4(a_base_r, a_base_c, b_base_r, b_base_c, c_base_r, c_base_c):
        """
        a_base_r: row start for A (0 or 2)
        a_base_c: col start for A (0 or 2)
        b_base_r: row start for B (0 or 2)
        b_base_c: col start for B (0 or 2)
        c_base_r: row start for C (0 or 2)
        c_base_c: col start for C (0 or 2)
        """
        # M1: (A11+A22)(B11+B22) -> C11, C22
        # A11 is at (a_base_r, a_base_c)
        # A22 is at (a_base_r+2, a_base_c+2)
        # B11 is at (b_base_r, b_base_c)
        # B22 is at (b_base_r+2, b_base_c+2)
        
        # We cannot just call add_strassen_2x2 because the inputs are sums.
        # We must manually construct the U, V, W for the 7 factors.
        
        # Factor 1: M1 = (A11+A22)(B11+B22)
        # U: A11=1, A22=1
        # V: B11=1, B22=1
        # W: C11=1, C22=1
        
        # Indices for A11: rows a_base_r..a_base_r+1, cols a_base_c..a_base_c+1
        # Indices for A22: rows a_base_r+2..a_base_r+3, cols a_base_c+2..a_base_c+3
        
        u_M1 = {}
        for dr in range(2):
            for dc in range(2):
                u_M1[(a_base_r+dr)*m + (a_base_c+dc)] = 1
                u_M1[(a_base_r+2+dr)*m + (a_base_c+2+dc)] = 1
                
        v_M1 = {}
        for dr in range(2):
            for dc in range(2):
                v_M1[(b_base_r+dr)*p + (b_base_c+dc)] = 1
                v_M1[(b_base_r+2+dr)*p + (b_base_c+2+dc)] = 1
                
        w_M1 = {}
        # C11 is at (c_base_r, c_base_c)
        # C22 is at (c_base_r+2, c_base_c+2)
        for dr in range(2):
            for dc in range(2):
                # W index: k*n + i. k is col of C, i is row of C.
                # C11: i=c_base_r+dr, k=c_base_c+dc
                w_M1[(c_base_c+dc)*n + (c_base_r+dr)] = 1
                # C22: i=c_base_r+2+dr, k=c_base_c+2+dc
                w_M1[(c_base_c+2+dc)*n + (c_base_r+2+dr)] = 1
                
        add_rank_unit(u_M1, v_M1, w_M1)
        
        # Factor 2: M2 = (A21+A22)B11
        # U: A21=1, A22=1
        # V: B11=1
        # W: C21=1, C22=-1
        
        u_M2 = {}
        for dr in range(2):
            for dc in range(2):
                u_M2[(a_base_r+2+dr)*m + (a_base_c+dc)] = 1
                u_M2[(a_base_r+2+dr)*m + (a_base_c+2+dc)] = 1
                
        v_M2 = {}
        for dr in range(2):
            for dc in range(2):
                v_M2[(b_base_r+dr)*p + (b_base_c+dc)] = 1
                
        w_M2 = {}
        for dr in range(2):
            for dc in range(2):
                w_M2[(c_base_c+dc)*n + (c_base_r+2+dr)] = 1
                w_M2[(c_base_c+2+dc)*n + (c_base_r+2+dr)] = -1
                
        add_rank_unit(u_M2, v_M2, w_M2)
        
        # Factor 3: M3 = A11(B12-B22)
        # U: A11=1
        # V: B12=1, B22=-1
        # W: C12=1, C22=1
        
        u_M3 = {}
        for dr in range(2):
            for dc in range(2):
                u_M3[(a_base_r+dr)*m + (a_base_c+dc)] = 1
                
        v_M3 = {}
        for dr in range(2):
            for dc in range(2):
                v_M3[(b_base_r+dr)*p + (b_base_c+2+dc)] = 1
                v_M3[(b_base_r+2+dr)*p + (b_base_c+2+dc)] = -1
                
        w_M3 = {}
        for dr in range(2):
            for dc in range(2):
                w_M3[(c_base_c+2+dc)*n + (c_base_r+dr)] = 1
                w_M3[(c_base_c+2+dc)*n + (c_base_r+2+dr)] = 1
                
        add_rank_unit(u_M3, v_M3, w_M3)
        
        # Factor 4: M4 = A22(B21-B11)
        # U: A22=1
        # V: B21=1, B11=-1
        # W: C11=1, C21=1
        
        u_M4 = {}
        for dr in range(2):
            for dc in range(2):
                u_M4[(a_base_r+2+dr)*m + (a_base_c+2+dc)] = 1
                
        v_M4 = {}
        for dr in range(2):
            for dc in range(2):
                v_M4[(b_base_r+2+dr)*p + (b_base_c+dc)] = 1
                v_M4[(b_base_r+dr)*p + (b_base_c+dc)] = -1
                
        w_M4 = {}
        for dr in range(2):
            for dc in range(2):
                w_M4[(c_base_c+dc)*n + (c_base_r+dr)] = 1
                w_M4[(c_base_c+dc)*n + (c_base_r+2+dr)] = 1
                
        add_rank_unit(u_M4, v_M4, w_M4)
        
        # Factor 5: M5 = (A11+A12)B22
        # U: A11=1, A12=1
        # V: B22=1
        # W: C11=-1, C12=1
        
        u_M5 = {}
        for dr in range(2):
            for dc in range(2):
                u_M5[(a_base_r+dr)*m + (a_base_c+dc)] = 1
                u_M5[(a_base_r+dr)*m + (a_base_c+2+dc)] = 1
                
        v_M5 = {}
        for dr in range(2):
            for dc in range(2):
                v_M5[(b_base_r+2+dr)*p + (b_base_c+2+dc)] = 1
                
        w_M5 = {}
        for dr in range(2):
            for dc in range(2):
                w_M5[(c_base_c+dc)*n + (c_base_r+dr)] = -1
                w_M5[(c_base_c+2+dc)*n + (c_base_r+dr)] = 1
                
        add_rank_unit(u_M5, v_M5, w_M5)
        
        # Factor 6: M6 = (A21-A11)(B11+B12)
        # U: A21=1, A11=-1
        # V: B11=1, B12=1
        # W: C22=1
        
        u_M6 = {}
        for dr in range(2):
            for dc in range(2):
                u_M6[(a_base_r+2+dr)*m + (a_base_c+dc)] = 1
                u_M6[(a_base_r+dr)*m + (a_base_c+dc)] = -1
                
        v_M6 = {}
        for dr in range(2):
            for dc in range(2):
                v_M6[(b_base_r+dr)*p + (b_base_c+dc)] = 1
                v_M6[(b_base_r+dr)*p + (b_base_c+2+dc)] = 1
                
        w_M6 = {}
        for dr in range(2):
            for dc in range(2):
                w_M6[(c_base_c+2+dc)*n + (c_base_r+2+dr)] = 1
                
        add_rank_unit(u_M6, v_M6, w_M6)
        
        # Factor 7: M7 = (A12-A22)(B21+B22)
        # U: A12=1, A22=-1
        # V: B21=1, B22=1
        # W: C11=1
        
        u_M7 = {}
        for dr in range(2):
            for dc in range(2):
                u_M7[(a_base_r+dr)*m + (a_base_c+2+dc)] = 1
                u_M7[(a_base_r+2+dr)*m + (a_base_c+2+dc)] = -1
                
        v_M7 = {}
        for dr in range(2):
            for dc in range(2):
                v_M7[(b_base_r+2+dr)*p + (b_base_c+dc)] = 1
                v_M7[(b_base_r+2+dr)*p + (b_base_c+2+dc)] = 1
                
        w_M7 = {}
        for dr in range(2):
            for dc in range(2):
                w_M7[(c_base_c+dc)*n + (c_base_r+dr)] = 1
                
        add_rank_unit(u_M7, v_M7, w_M7)

    # We need two 4x4x4 blocks for p=8.
    # Block 1: B cols 0-3.
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    # Block 2: B cols 4-7.
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # We only need n=3, so we ignore the 4th row of U and W.
    # U is already sliced to 12 rows in the initialization?
    # No, U is initialized to 12 rows.
    # But my add_strassen_4x4 writes to U indices up to (3)*m + 3 = 15.
    # U has only 12 rows (indices 0-11).
    # This will cause an IndexError.
    
    # I must initialize U with 16 rows, then slice it at the end.
    # Or, I must ensure I don't write to rows 12-15.
    # The Strassen algorithm uses all 4 rows of A.
    # Since we are padding A, the 4th row of A is 0.
    # So the terms involving A_row3 should be ignored?
    # No, the decomposition is for the 4x4x4 tensor.
    # If we just drop the 4th row of U, does it still work?
    # The condition is sum_r U_{(i,j), r} V_{(j,k), r} W_{(k,i), r} = delta_{j, j'} for i<3.
    # If we use the full 4x4x4 decomposition, it satisfies the condition for i=0..3.
    # If we remove the rows of U corresponding to i=3, we are effectively setting A_{3, j} = 0.
    # But the decomposition is linear in A.
    # The terms in the sum are $\sum_r (\sum_j U_{i,j}^r A_{i,j}) (\sum_j V_{j,k}^r B_{j,k}) W_{k,i}^r$.
    # If we set $A_{3, j} = 0$, the term $\sum_j U_{3,j}^r A_{3,j}$ becomes 0.
    # But $U_{3,j}^r$ is part of the vector $U_r$.
    # If we remove the row from U, we are changing the vector $U_r$.
    # This breaks the decomposition.
    
    # Correct approach:
    # 1. Build the full 4x4x8 decomposition (U 16x98, V 32x98, W 32x98).
    # 2. Slice U to 12x98.
    # 3. Slice W to 24x98 (remove rows where i=3).
    # 4. Return.
    
    # Let's re-initialize U and W with padded sizes.
    U_pad = [[0] * rank for _ in range(4 * m)]
    W_pad = [[0] * rank for _ in range(p * 4)]
    
    # Re-run the construction with U_pad and W_pad.
    # I need to modify add_rank_unit to write to U_pad and W_pad.
    
    col = 0
    def add_rank_unit_pad(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            raise Exception("Rank overflow")
        for idx, val in u_coeffs.items():
            U_pad[idx][col] += val
        for idx, val in v_coeffs.items():
            V[idx][col] += val
        for idx, val in w_coeffs.items():
            W_pad[idx][col] += val
        col += 1
        
    # Replace add_rank_unit with add_rank_unit_pad in the helper functions?
    # No, I'll just call add_strassen_4x4 which uses add_rank_unit.
    # I need to make add_strassen_4x4 use the pad versions.
    
    # To keep it simple, I will define a local add_rank_unit that writes to the global U, V, W
    # but I will initialize U and W with padded sizes, then slice them before returning.
    
    U = U_pad
    W = W_pad
    
    add_strassen_4x4(0, 0, 0, 0, 0, 0)
    add_strassen_4x4(0, 0, 0, 4, 0, 4)
    
    # Slice U to n*m = 12 rows
    U = U[:n*m]
    
    # Slice W to p*n = 24 rows
    # W_pad is p*4 = 32 rows.
    # Indices are k*4 + i.
    # We want i=0,1,2.
    W_new = []
    for k in range(p):
        for i in range(n):
            W_new.append(W_pad[k*4 + i])
    W = W_new
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
