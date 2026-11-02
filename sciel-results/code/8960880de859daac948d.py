import time
import sys

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W
    # We don't know the exact rank yet, so we build lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_vals, v_vals, w_vals):
        """
        u_vals: list of length n*m
        v_vals: list of length m*p
        w_vals: list of length p*n
        """
        U_cols.append(u_vals)
        V_cols.append(v_vals)
        W_cols.append(w_vals)

    # Helper to create zero vectors
    def zero_um(): return [0] * (n * m)
    def zero_mp(): return [0] * (m * p)
    def zero_pn(): return [0] * (p * n)

    # Block indices
    # A rows: 0,1 (block 0), 2,3 (block 1)
    # A cols: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    # B rows: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    # B cols: 0,1 (block 0), 2,3 (block 1), 4 (block 2)
    # C rows: 0,1 (block 0), 2,3 (block 1)
    # C cols: 0,1 (block 0), 2,3 (block 1), 4 (block 2)

    # Strassen 2x2 implementation
    def strassen_2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        # Local indices for A block: (0,0), (0,1), (1,0), (1,1)
        # Global A indices: (a_row_offset + r, a_col_offset + c)
        # Local indices for B block: (0,0), (0,1), (1,0), (1,1)
        # Global B indices: (b_row_offset + r, b_col_offset + c)
        # Local indices for C block: (0,0), (0,1), (1,0), (1,1)
        # Global C indices: (c_row_offset + r, c_col_offset + c)
        
        # W index convention: W[k*n + i, r]
        # So for output C[i, k], the W index is k*n + i.
        
        # M1 = (A00 + A11) * (B00 + B11)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        # A parts
        u[a_row_offset + 0, a_col_offset + 0] = 1
        u[a_row_offset + 1, a_col_offset + 1] = 1
        # B parts
        v[b_row_offset + 0, b_col_offset + 0] = 1
        v[b_row_offset + 1, b_col_offset + 1] = 1
        # C parts: M1 contributes to C00, C11? No, M1 is intermediate.
        # We need to map M1 to C.
        # C00 += M1, C11 += M1? No.
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # This function adds ONE term to the global decomposition.
        # We must decompose the 7 Strassen products into 7 rank-1 terms?
        # No, Strassen computes 7 products. Each product is a scalar multiplication of two 2x2 matrices?
        # No, Strassen computes 7 scalar products of linear combinations.
        # Each "product" $X * Y$ where X and Y are 2x2 matrices is NOT rank 1.
        # Strassen reduces the number of 2x2 multiplications from 8 to 7.
        # But each 2x2 multiplication itself has rank 4 (schoolbook) or can be further decomposed?
        # No, the standard Strassen algorithm assumes the base case is scalar multiplication (rank 1).
        # Here, the "scalars" are 2x2 blocks.
        # So we need to decompose the multiplication of two 2x2 matrices into rank-1 terms.
        # The rank of 2x2 matrix multiplication is 7.
        # So we need to implement the rank-7 decomposition for 2x2x2.
        
        pass # Placeholder, will implement below

    # We will implement the rank-7 decomposition for 2x2x2 directly.
    # Reference: Bini et al. or standard Strassen.
    # Strassen terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each M is a product of two 2x2 matrices? No.
    # In the context of the 2x2 block multiplication, A00 etc are SCALARS if we view the 2x2 block as a single entity in a higher-level recursion.
    # But here, A00 is a 2x2 matrix of SCALARS.
    # So $A_{00} B_{00}$ is a 2x2 by 2x2 multiplication.
    # We need the rank-7 decomposition of the 2x2x2 tensor.
    
    def add_2x2_block_mult(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # a_r0, a_c0: top-left of A block
        # b_r0, b_c0: top-left of B block
        # c_r0, c_c0: top-left of C block
        
        # Indices in local 2x2 block:
        # A: (0,0), (0,1), (1,0), (1,1)
        # B: (0,0), (0,1), (1,0), (1,1)
        # C: (0,0), (0,1), (1,0), (1,1)
        
        # Strassen M1: (A00+A11)(B00+B11)
        # This term contributes to C00 and C11.
        # We need to decompose the product of two 2x2 matrices into 7 rank-1 terms.
        # Let's use the explicit rank-7 decomposition.
        
        # Term 1: (A00+A11)(B00+B11)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        # U: A00+A11
        u[a_r0+0, a_c0+0] = 1
        u[a_r0+1, a_c0+1] = 1
        # V: B00+B11
        v[b_r0+0, b_c0+0] = 1
        v[b_r0+1, b_c0+1] = 1
        # W: Contributes to C00 and C11
        # C00 index: c_c0*4 + (c_r0+0) ? No.
        # W index is k*n + i.
        # For C00: i=c_r0+0, k=c_c0+0 -> idx = (c_c0+0)*n + (c_r0+0)
        # For C11: i=c_r0+1, k=c_c0+1 -> idx = (c_c0+1)*n + (c_r0+1)
        w[(c_c0+0)*n + (c_r0+0)] = 1
        w[(c_c0+1)*n + (c_r0+1)] = 1
        add_term(u, v, w)

        # Term 2: (A10+A11)B00
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+1, a_c0+0] = 1
        u[a_r0+1, a_c0+1] = 1
        v[b_r0+0, b_c0+0] = 1
        # Contributes to C10
        w[(c_c0+0)*n + (c_r0+1)] = 1
        add_term(u, v, w)

        # Term 3: A00(B01-B11)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+0] = 1
        v[b_r0+0, b_c0+1] = 1
        v[b_r0+1, b_c0+1] = -1
        # Contributes to C01
        w[(c_c0+1)*n + (c_r0+0)] = 1
        add_term(u, v, w)

        # Term 4: A11(B10-B00)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+1, a_c0+1] = 1
        v[b_r0+1, b_c0+0] = 1
        v[b_r0+0, b_c0+0] = -1
        # Contributes to C00 and C10
        w[(c_c0+0)*n + (c_r0+0)] = 1
        w[(c_c0+0)*n + (c_r0+1)] = 1
        add_term(u, v, w)

        # Term 5: (A00+A01)B11
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+0] = 1
        u[a_r0+0, a_c0+1] = 1
        v[b_r0+1, b_c0+1] = 1
        # Contributes to C01 and C00 (with -1)
        w[(c_c0+1)*n + (c_r0+0)] = 1
        w[(c_c0+0)*n + (c_r0+0)] = -1
        add_term(u, v, w)

        # Term 6: (A10-A00)(B00+B01)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+1, a_c0+0] = 1
        u[a_r0+0, a_c0+0] = -1
        v[b_r0+0, b_c0+0] = 1
        v[b_r0+0, b_c0+1] = 1
        # Contributes to C11
        w[(c_c0+1)*n + (c_r0+1)] = 1
        add_term(u, v, w)

        # Term 7: (A01-A11)(B10+B11)
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+1] = 1
        u[a_r0+1, a_c0+1] = -1
        v[b_r0+1, b_c0+0] = 1
        v[b_r0+1, b_c0+1] = 1
        # Contributes to C00
        w[(c_c0+0)*n + (c_r0+0)] = 1
        add_term(u, v, w)

    # Rank 4 decomposition for 2x2 by 2x1
    def add_2x2_by_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00*B0 + A01*B1
        # C1 = A10*B0 + A11*B1
        
        # Term 1: A00 * B0
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+0] = 1
        v[b_r0+0, b_c0+0] = 1
        w[(c_c0+0)*n + (c_r0+0)] = 1
        add_term(u, v, w)
        
        # Term 2: A01 * B1
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+1] = 1
        v[b_r0+1, b_c0+0] = 1
        w[(c_c0+0)*n + (c_r0+0)] = 1
        add_term(u, v, w)
        
        # Term 3: A10 * B0
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+1, a_c0+0] = 1
        v[b_r0+0, b_c0+0] = 1
        w[(c_c0+0)*n + (c_r0+1)] = 1
        add_term(u, v, w)
        
        # Term 4: A11 * B1
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+1, a_c0+1] = 1
        v[b_r0+1, b_c0+0] = 1
        w[(c_c0+0)*n + (c_r0+1)] = 1
        add_term(u, v, w)

    # Rank 1 decomposition for 2x1 by 1x2
    def add_2x1_by_1x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A is 2x1, B is 1x2, C is 2x2
        # C = u v^T
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+0] = 1
        u[a_r0+1, a_c0+0] = 1
        v[b_r0+0, b_c0+0] = 1
        v[b_r0+0, b_c0+1] = 1
        # C00, C01, C10, C11
        w[(c_c0+0)*n + (c_r0+0)] = 1
        w[(c_c0+1)*n + (c_r0+0)] = 1
        w[(c_c0+0)*n + (c_r0+1)] = 1
        w[(c_c0+1)*n + (c_r0+1)] = 1
        add_term(u, v, w)

    # Rank 1 decomposition for 2x1 by 1x1
    def add_2x1_by_1x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        # A is 2x1, B is 1x1, C is 2x1
        u = zero_um(); v = zero_mp(); w = zero_pn()
        u[a_r0+0, a_c0+0] = 1
        u[a_r0+1, a_c0+0] = 1
        v[b_r0+0, b_c0+0] = 1
        w[(c_c0+0)*n + (c_r0+0)] = 1
        w[(c_c0+0)*n + (c_r0+1)] = 1
        add_term(u, v, w)

    # Iterate over blocks
    # i_block in 0,1
    # j_block in 0,1,2
    # k_block in 0,1,2
    
    for i_b in range(2):
        for j_b in range(3):
            for k_b in range(3):
                # Determine sizes
                a_r0 = i_b * 2
                a_c0 = 0
                # A col block size: 2,2,1
                if j_b == 0: a_c0 = 0
                elif j_b == 1: a_c0 = 2
                else: a_c0 = 4
                
                b_r0 = 0
                # B row block size: 2,2,1
                if j_b == 0: b_r0 = 0
                elif j_b == 1: b_r0 = 2
                else: b_r0 = 4
                
                b_c0 = 0
                # B col block size: 2,2,1
                if k_b == 0: b_c0 = 0
                elif k_b == 1: b_c0 = 2
                else: b_c0 = 4
                
                c_r0 = i_b * 2
                c_c0 = 0
                # C col block size: 2,2,1
                if k_b == 0: c_c0 = 0
                elif k_b == 1: c_c0 = 2
                else: c_c0 = 4

                # Dispatch based on j_b, k_b
                if j_b in [0, 1] and k_b in [0, 1]:
                    # 2x2 by 2x2
                    add_2x2_block_mult(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0)
                elif j_b in [0, 1] and k_b == 2:
                    # 2x2 by 2x1
                    add_2x2_by_2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0)
                elif j_b == 2 and k_b in [0, 1]:
                    # 2x1 by 1x2
                    add_2x1_by_1x2(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0)
                elif j_b == 2 and k_b == 2:
                    # 2x1 by 1x1
                    add_2x1_by_1x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0)

    # Convert lists of columns to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n * m):
            U[i][r] = U_cols[r][i]
        for i in range(m * p):
            V[i][r] = V_cols[r][i]
        for i in range(p * n):
            W[i][r] = W_cols[r][i]
            
    return {"U": U, "V": V, "W": W}
