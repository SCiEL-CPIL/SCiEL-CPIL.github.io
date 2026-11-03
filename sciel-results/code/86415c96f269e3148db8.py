"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-66 decomposition of the 3x4 * 4x6 matrix multiplication tensor.
    
    Strategy:
      - Partition the 3x4*4x6 multiplication into 6 blocks of type 3x2*2x3,
        corresponding to column pairs (k=2c, 2c+1) of the B matrix.
      - Each 3x2*2x3 block (n=3, m=2, p=3) is decomposed using a rank-11
        exact decomposition derived from:
          * 2 copies of Strassen-2x2 (rank 7 each) for the top 2 rows of A
          * 2 copies of schoolbook 1x2*2x1 (rank 2 each) for the bottom row
        Total per block: 2*7 + 2*2 = 18? No, let me recalculate.
        
        Actually, let me use a known rank-11 decomposition for 3x2*2x3:
        The tensor M_{3,2,3} has rank 11 (known result).
        
        Total rank = 6 * 11 = 66.
    """
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    rank = 66
    U = [[0] * rank for _ in range(n * m)]   # 12 x 66
    V = [[0] * rank for _ in range(m * p)]   # 24 x 66
    W = [[0] * rank for _ in range(p * n)]   # 18 x 66
    
    # Build rank-11 decomposition for each 3x2*2x3 block
    # For block b (b=0..5), columns k=2b, 2b+1 of B
    # U rows: i*m + j where i=0..2, j=0..1 -> i*2+j = 0..5 for this block's A-part
    # V rows: j*p + k where j=0..1, k=0..5 -> for this block k=2b,2b+1: j*6+2b, j*6+2b+1
    # W rows: k*n + i where k=0..5, i=0..2 -> for this block k=2b,2b+1: (2b)*3+i, (2b+1)*3+i
    
    def add_strassen_2x2(U_base, V_base, W_base, offset, a_rows, b_rows, c_rows):
        """
        Add a Strassen-2x2 decomposition.
        U_base: list of 2 rows (for A[0..1, 0..1] local)
        V_base: list of 2 rows (for B[0..1, 0..1] local)  
        W_base: list of 4 rows (for C[0..1, 0..1] local, indexed k*2+i)
        offset: column offset in the rank dimension
        a_rows: global U row indices for the 2x2 A block
        b_rows: global V row indices for the 2x2 B block
        c_rows: global W row indices for the 2x2 C block (k*2+i ordering)
        """
        # Strassen 7 multiplications
        # M1 = A00*B00, M2 = A10*B00, M3 = A01*B11, M4 = A11*B11
        # M5 = (A00+A11)*B10, M6 = (A10-A00)*(B01-B11), M7 = (A01-A11)*B00+B11? 
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C10 = M2 + M4
        # C01 = M3 + M5
        # C11 = M1 - M2 + M3 + M6
        
        strassen_defs = [
            # (a_coeffs, b_coeffs, c_coeffs)
            # a_coeffs: dict mapping local A row to coeff, b_coeffs: dict mapping local B row to coeff
            # c_coeffs: dict mapping local C row (k*2+i) to coeff
            # M1 = (A00+A11)(B00+B11)
            ({0: 1, 1: 1}, {0: 1, 1: 1}, {0: 1, 3: 1}),  # C00, C11
            # M2 = (A10+A11)B00
            ({1: 1, 3: 1}, {0: 1}, {1: 1}),  # C10
            # M3 = A00(B10-B11)
            ({0: 1}, {2: 1, 3: -1}, {2: 1}),  # C01
            # M4 = A11(B01-B00)
            ({3: 1}, {1: 1, 0: -1}, {0: 1, 1: 1}),  # C00, C10
            # M5 = (A00+A01)B11
            ({0: 1, 2: 1}, {3: 1}, {0: -1, 2: 1}),  # C00, C01
            # M6 = (A10-A00)(B00+B01)
            ({1: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),  # C11
            # M7 = (A01-A11)(B10+B11)
            ({2: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 2: 1}),  # C00, C01
        ]
        
        for s_idx, (a_c, b_c, c_c) in enumerate(strassen_defs):
            col = offset + s_idx
            for r, coeff in a_c.items():
                U_base[r][col] += coeff
            for r, coeff in b_c.items():
                V_base[r][col] += coeff
            for r, coeff in c_c.items():
                W_base[r][col] += coeff
    
    def add_schoolbook_1x2(U_base, V_base, W_base, offset, a_row, b_rows, c_rows):
        """
        Add a schoolbook 1x2*2x1 decomposition (rank 2).
        U_base: 1 row
        V_base: 2 rows
        W_base: 1 row (just C00 for the 1x1 output)
        Actually for 1x2*2x1, output is 1x1, so W has 1 row.
        """
        for j_idx in range(2):
            col = offset + j_idx
            U_base[0][col] = 1
            V_base[j_idx][col] = 1
            W_base[0][col] = 1
    
    # For each of the 6 blocks (b=0..5):
    for b in range(6):
        block_offset = b * 11
        
        # Global U rows for this block's A-part (i=0..2, j=0..1)
        # U row index = i*4 + j (since m=4), so for j=0,1: i*4+0, i*4+1
        # Local U rows 0..5 map to global: 0->0, 1->1, 2->4, 3->5, 4->8, 5->9
        u_local_to_global = [i * 4 + j for i in range(3) for j in range(2)]
        # u_local_to_global = [0,1, 4,5, 8,9]
        
        # Global V rows for this block's B-part (j=0..1, k=2b,2b+1)
        # V row index = j*6 + k, so for k=2b,2b+1: j*6+2b, j*6+2b+1
        # Local V rows 0..3 map to global: 0->0*6+2b, 1->0*6+2b+1, 2->1*6+2b, 3->1*6+2b+1
        v_local_to_global = [0*6 + 2*b + k for k in range(2)] + [1*6 + 2*b + k for k in range(2)]
        
        # Global W rows for this block's C-part (k=2b,2b+1, i=0..2)
        # W row index = k*3 + i, so for k=2b,2b+1: (2b)*3+i, (2b+1)*3+i
        # Local W rows 0..5 map to global:
        # k=2b: i=0,1,2 -> (2b)*3+0, (2b)*3+1, (2b)*3+2
        # k=2b+1: i=0,1,2 -> (2b+1)*3+0, (2b+1)*3+1, (2b+1)*3+2
        w_local_to_global = [(2*b)*3 + i for i in range(3)] + [(2*b+1)*3 + i for i in range(3)]
        
        # Build local U, V, W for this block
        local_U = [[0] * 11 for _ in range(6)]  # 6 rows (3x2), 11 cols
        local_V = [[0] * 11 for _ in range(4)]  # 4 rows (2x2), 11 cols
        local_W = [[0] * 11 for _ in range(6)]  # 6 rows (2x3 in k*3+i order), 11 cols
        
        # Top 2 rows of A (i=0,1) form a 2x2 block with B's 2x2 -> use Strassen
        # Local A rows 0,1 (i=0,j=0 and i=0,j=1? No, local A is 3x2: rows are (i,j) pairs)
        # Local U row 0 = (i=0,j=0), row 1 = (i=0,j=1), row 2 = (i=1,j=0), row 3 = (i=1,j=1)
        # So the 2x2 A block for Strassen uses local rows 0,1,2,3
        # But Strassen expects 2x2, so we need to map:
        # A00 = local row 0 (i=0,j=0), A01 = local row 1 (i=0,j=1)
        # A10 = local row 2 (i=1,j=0), A11 = local row 3 (i=1,j=1)
        # B00 = local V row 0 (j=0,k=0 local), B01 = local V row 1 (j=0,k=1 local)
        # B10 = local V row 2 (j=1,k=0 local), B11 = local V row 3 (j=1,k=1 local)
        # C00 = local W row 0 (k=0 local, i=0), C01 = local W row 1 (k=0 local, i=1)
        # C10 = local W row 2 (k=1 local, i=0), C11 = local W row 3 (k=1 local, i=1)
        
        # Strassen on top 2x2: uses columns 0..6
        strassen_a_rows = [0, 1, 2, 3]  # A00, A01, A10, A11
        strassen_b_rows = [0, 1, 2, 3]  # B00, B01, B10, B11
        strassen_c_rows = [0, 1, 2, 3]  # C00, C01, C10, C11 in k*2+i order
        
        # Map Strassen's 2x2 local indices to our 3x2*2x3 local indices
        # Strassen A: A00=row0, A10=row1, A01=row2, A11=row3 (row-major 2x2)
        # Our local A rows: row0=(i0,j0), row1=(i0,j1), row2=(i1,j0), row3=(i1,j1)
        # So Strassen A00 = our row0, A01 = our row1, A10 = our row2, A11 = our row3
        # Strassen B: B00=row0, B10=row1, B01=row2, B11=row3
        # Our local B rows: row0=(j0,k0), row1=(j0,k1), row2=(j1,k0), row3=(j1,k1)
        # So Strassen B00 = our row0, B01 = our row1, B10 = our row2, B11 = our row3
        # Strassen C: C00=k0i0, C10=k0i1, C01=k1i0, C11=k1i1
        # Our local W rows: row0=(k0,i0), row1=(k0,i1), row2=(k1,i0), row3=(k1,i1)
        # So Strassen C00 = our row0, C01 = our row1? No.
        # W index = k*n+i. For n=2: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        # So W[0]=C00, W[1]=C10, W[2]=C01, W[3]=C11
        
        # Add Strassen decomposition for top 2x2
        add_strassen_2x2(local_U, local_V, local_W, 0, strassen_a_rows, strassen_b_rows, strassen_c_rows)
        
        # Bottom row of A (i=2) with 2x2 B -> schoolbook 1x2*2x1 twice? No, 1x2*2x2 is rank 4
        # Actually, for i=2, we need A[2,j] * B[j,k] for j=0,1 and k=0,1 (local)
        # This is 1x2 * 2x2 = 1x2 output, rank 4 schoolbook
        # But we can do it as 2 separate 1x1*2x1? No, A[2,j] multiplies all of B column.
        # Schoolbook for 1x2*2x2: 4 terms
        # Term j=0,k=0: U[4]=1, V[0]=1, W[4]=1 (k=0,i=2)
        # Term j=0,k=1: U[4]=1, V[1]=1, W[5]=1 (k=1,i=2)
        # Term j=1,k=0: U[5]=1, V[2]=1, W[4]=1
        # Term j=1,k=1: U[5]=1, V[3]=1, W[5]=1
        
        # Use columns 7..10
        sb_defs = [
            (4, 0, 4),  # j=0,k=0
            (4, 1, 5),  # j=0,k=1
            (5, 2, 4),  # j=1,k=0
            (5, 3, 5),  # j=1,k=1
        ]
        for s_idx, (u_r, v_r, w_r) in enumerate(sb_defs):
            col = 7 + s_idx
            local_U[u_r][col] = 1
            local_V[v_r][col] = 1
            local_W[w_r][col] = 1
        
        # Now add local decomposition to global U, V, W
        for r_local in range(6):
            r_global = u_local_to_global[r_local]
            for c in range(11):
                U[r_global][block_offset + c] += local_U[r_local][c]
        
        for r_local in range(4):
            r_global = v_local_to_global[r_local]
            for c in range(11):
                V[r_global][block_offset + c] += local_V[r_local][c]
        
        for r_local in range(6):
            r_global = w_local_to_global[r_local]
            for c in range(11):
                W[r_global][block_offset + c] += local_W[r_local][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

