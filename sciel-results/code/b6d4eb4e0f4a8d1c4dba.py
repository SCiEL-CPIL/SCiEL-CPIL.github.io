"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based block decomposition for 2x5x5.
    # Split A (2x5) into blocks: A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B (5x5) into blocks:
    #   B_{11} (2x2), B_{12} (2x2), B_{13} (2x1)
    #   B_{21} (2x2), B_{22} (2x2), B_{23} (2x1)
    #   B_{31} (1x2), B_{32} (1x2), B_{33} (1x1)
    #
    # C = A * B (2x5)
    # C_{11} (2x2) = A1*B11 + A2*B21 + A3*B31
    # C_{12} (2x2) = A1*B12 + A2*B22 + A3*B32
    # C_{13} (2x1) = A1*B13 + A2*B23 + A3*B33
    #
    # For the 2x2x2 multiplications, use Strassen (rank 7).
    # For 2x2x1, 2x1x2, 2x1x1, use schoolbook.
    #
    # Count:
    # 4 instances of 2x2x2 -> 4*7 = 28
    # 2 instances of 2x2x1 -> 2*4 = 8
    # 2 instances of 2x1x2 -> 2*4 = 8
    # 1 instance of 2x1x1 -> 1*2 = 2
    # Total rank = 46

    n, m, p = DIMENSIONS
    rank = 46

    # U: shape (n*m, rank) = (10, 46)
    # V: shape (m*p, rank) = (25, 46)
    # W: shape (p*n, rank) = (10, 46)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    # Helper to add a rank-1 term
    def add_term(u_row, u_val, v_row, v_val, w_row, w_val):
        nonlocal col
        U[u_row][col] = u_val
        V[v_row][col] = v_val
        W[w_row][col] = w_val
        col += 1

    # Helper for Strassen 2x2x2
    # A indices: a(0), b(1), c(2), d(3) -> rows 0..3 in A block
    # B indices: e(0), f(1), g(2), h(3) -> cols 0..3 in B block
    # C indices: c11(0), c12(1), c21(2), c22(3)
    def strassen_222(a_base, b_base, c_base, offset):
        """
        a_base: row offset in A for the 2x2 block (0 or 2)
        b_base: col offset in B for the 2x2 block (0, 2, or 4)
        c_base: col offset in C for the 2x2 block (0 or 2)
        offset: starting column index
        """
        # A rows: a_base+0, a_base+1
        # B cols: b_base+0, b_base+1
        # C cols: c_base+0, c_base+1
        
        # A entries: a(A_{0,0}), b(A_{0,1}), c(A_{1,0}), d(A_{1,1})
        # relative to a_base, b_base
        a_r = a_base
        b_r = a_base + 1
        # But wait, A is 2x5, so rows 0,1. The 2x2 blocks are in columns.
        # A1 = A[:, 0:2], A2 = A[:, 2:4], A3 = A[:, 4:5]
        # So for A1: rows 0,1; cols 0,1. In U, row = i*m + j.
        # A[i,j] is at U[i*5+j].
        
        # Let's redefine: 
        # For a 2x2x2 block at A rows [ra, ra+1], A cols [ca, ca+1], B rows [rb, rb+1], B cols [cb, cb+1]
        # Wait, the block structure is:
        # A is 2x5. We split columns into 2,2,1.
        # B is 5x5. We split rows into 2,2,1 and cols into 2,2,1.
        # C is 2x5. We split columns into 2,2,1.
        
        # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2), A3 = A[:, 4:5] (2x1)
        # B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:4] (2x2), B13 = B[0:2, 4:5] (2x1)
        # B21 = B[2:4, 0:2] (2x2), B22 = B[2:4, 2:4] (2x2), B23 = B[2:4, 4:5] (2x1)
        # B31 = B[4:5, 0:2] (1x2), B32 = B[4:5, 2:4] (1x2), B33 = B[4:5, 4:5] (1x1)
        
        # C11 = A1*B11 + A2*B21 + A3*B31
        # C12 = A1*B12 + A2*B22 + A3*B32
        # C13 = A1*B13 + A2*B23 + A3*B33
        
        # For a 2x2x2 multiplication: A_block (2x2) * B_block (2x2) -> C_block (2x2)
        # A_block rows: i in [0,1], cols: j in [ca, ca+1]
        # B_block rows: j' in [rb, rb+1], cols: k in [cb, cb+1]
        # Note: the inner dimension j must match. 
        # A1 has cols 0,1. B11 has rows 0,1. Match.
        # A2 has cols 2,3. B21 has rows 2,3. Match.
        # A3 has col 4. B31 has row 4. Match.
        
        # So for Strassen on A1*B11:
        # A rows: 0,1. A cols: 0,1.
        # B rows: 0,1. B cols: 0,1.
        # C rows: 0,1. C cols: 0,1.
        
        # U row for A[i,j] is i*5+j.
        # V row for B[j,k] is j*5+k.
        # W row for C[k,i] is k*2+i. (Trace convention: W[k*n+i, r] for C[i,k])
        
        # Let's implement Strassen for a specific 2x2x2 block.
        # a = A[0,ca], b = A[0,ca+1], c = A[1,ca], d = A[1,ca+1]
        # e = B[rb,cb], f = B[rb,cb+1], g = B[rb+1,cb], h = B[rb+1,cb+1]
        
        a_row0 = 0 * 5 + ca
        a_row1 = 0 * 5 + (ca + 1)
        a_row2 = 1 * 5 + ca
        a_row3 = 1 * 5 + (ca + 1)
        
        e_row0 = rb * 5 + cb
        e_row1 = rb * 5 + (cb + 1)
        e_row2 = (rb + 1) * 5 + cb
        e_row3 = (rb + 1) * 5 + (cb + 1)
        
        c_row00 = 0 * 2 + 0  # C[0,0] -> W[0]
        c_row01 = 1 * 2 + 0  # C[0,1] -> W[2]
        c_row10 = 0 * 2 + 1  # C[1,0] -> W[1]
        c_row11 = 1 * 2 + 1  # C[1,1] -> W[3]
        
        # But we need to offset the C columns by c_base.
        # C is 2x5. C[i,k] -> W[k*2+i].
        # If C block is at cols [c_base, c_base+1], then:
        # C[i, c_base] -> W[(c_base)*2 + i]
        # C[i, c_base+1] -> W[(c_base+1)*2 + i]
        
        # So:
        # C[0, c_base] -> W[c_base*2 + 0]
        # C[1, c_base] -> W[c_base*2 + 1]
        # C[0, c_base+1] -> W[(c_base+1)*2 + 0]
        # C[1, c_base+1] -> W[(c_base+1)*2 + 1]
        
        c_w00 = c_base * 2 + 0
        c_w10 = c_base * 2 + 1
        c_w01 = (c_base + 1) * 2 + 0
        c_w11 = (c_base + 1) * 2 + 1
        
        # Strassen:
        # P1 = (a+d)(e+h)
        # P2 = (c+d)e
        # P3 = a(f-h)
        # P4 = d(g-e)
        # P5 = (a+b)h
        # P6 = (c-a)(e+f)
        # P7 = (b-d)(g+h)
        #
        # C11 = P1 + P4 - P5 - P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P7
        
        # We need to add 7 rank-1 terms.
        # Each term is a linear combination of A entries, a linear combination of B entries,
        # and contributes to a linear combination of C entries.
        
        # Term 1: (a+d)(e+h)
        # U: a_row0 += 1, a_row3 += 1
        # V: e_row0 += 1, e_row3 += 1
        # W: c_w00 += 1, c_w01 += 1, c_w10 += 1, c_w11 += -1
        U[a_row0][offset] = 1
        U[a_row3][offset] = 1
        V[e_row0][offset] = 1
        V[e_row3][offset] = 1
        W[c_w00][offset] = 1
        W[c_w10][offset] = 1
        W[c_w01][offset] = 1
        W[c_w11][offset] = -1
        offset += 1
        
        # Term 2: (c+d)e
        U[a_row2][offset] = 1
        U[a_row3][offset] = 1
        V[e_row0][offset] = 1
        W[c_w10][offset] = 1
        W[c_w11][offset] = -1
        offset += 1
        
        # Term 3: a(f-h)
        U[a_row0][offset] = 1
        V[e_row1][offset] = 1
        V[e_row3][offset] = -1
        W[c_w01][offset] = 1
        W[c_w11][offset] = 1
        offset += 1
        
        # Term 4: d(g-e)
        U[a_row3][offset] = 1
        V[e_row2][offset] = 1
        V[e_row0][offset] = -1
        W[c_w10][offset] = 1
        W[c_w11][offset] = 1
        offset += 1
        
        # Term 5: (a+b)h
        U[a_row0][offset] = 1
        U[a_row1][offset] = 1
        V[e_row3][offset] = 1
        W[c_w10][offset] = -1
        W[c_w01][offset] = 1
        offset += 1
        
        # Term 6: (c-a)(e+f)
        U[a_row2][offset] = 1
        U[a_row0][offset] = -1
        V[e_row0][offset] = 1
        V[e_row1][offset] = 1
        W[c_w01][offset] = 1
        W[c_w11][offset] = 1
        offset += 1
        
        # Term 7: (b-d)(g+h)
        U[a_row1][offset] = 1
        U[a_row3][offset] = -1
        V[e_row2][offset] = 1
        V[e_row3][offset] = 1
        W[c_w11][offset] = 1
        W[c_w00][offset] = -1
        offset += 1
        
        return offset

    # Helper for 2x2x1 (A 2x2, B 2x1 -> C 2x1)
    def schoolbook_221(a_ca, b_rb, c_cb, offset):
        # A rows 0,1; cols a_ca, a_ca+1
        # B rows b_rb, b_rb+1; col c_cb
        # C rows 0,1; col c_cb
        # C[0,c_cb] = A[0,a_ca]*B[b_rb,c_cb] + A[0,a_ca+1]*B[b_rb+1,c_cb]
        # C[1,c_cb] = A[1,a_ca]*B[b_rb,c_cb] + A[1,a_ca+1]*B[b_rb+1,c_cb]
        
        # Term 1: A[0,a_ca]*B[b_rb,c_cb] -> C[0,c_cb]
        U[0*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 0][offset] = 1
        offset += 1
        
        # Term 2: A[0,a_ca+1]*B[b_rb+1,c_cb] -> C[0,c_cb]
        U[0*5 + a_ca+1][offset] = 1
        V[(b_rb+1)*5 + c_cb][offset] = 1
        W[c_cb*2 + 0][offset] = 1
        offset += 1
        
        # Term 3: A[1,a_ca]*B[b_rb,c_cb] -> C[1,c_cb]
        U[1*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 1][offset] = 1
        offset += 1
        
        # Term 4: A[1,a_ca+1]*B[b_rb+1,c_cb] -> C[1,c_cb]
        U[1*5 + a_ca+1][offset] = 1
        V[(b_rb+1)*5 + c_cb][offset] = 1
        W[c_cb*2 + 1][offset] = 1
        offset += 1
        
        return offset

    # Helper for 2x1x2 (A 2x1, B 1x2 -> C 2x2)
    def schoolbook_212(a_ca, b_rb, c_cb, offset):
        # A rows 0,1; col a_ca
        # B row b_rb; cols c_cb, c_cb+1
        # C rows 0,1; cols c_cb, c_cb+1
        # C[i, c_cb] = A[i,a_ca]*B[b_rb,c_cb]
        # C[i, c_cb+1] = A[i,a_ca]*B[b_rb,c_cb+1]
        
        # Term 1: A[0,a_ca]*B[b_rb,c_cb] -> C[0,c_cb]
        U[0*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 0][offset] = 1
        offset += 1
        
        # Term 2: A[1,a_ca]*B[b_rb,c_cb] -> C[1,c_cb]
        U[1*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 1][offset] = 1
        offset += 1
        
        # Term 3: A[0,a_ca]*B[b_rb,c_cb+1] -> C[0,c_cb+1]
        U[0*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb+1][offset] = 1
        W[(c_cb+1)*2 + 0][offset] = 1
        offset += 1
        
        # Term 4: A[1,a_ca]*B[b_rb,c_cb+1] -> C[1,c_cb+1]
        U[1*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb+1][offset] = 1
        W[(c_cb+1)*2 + 1][offset] = 1
        offset += 1
        
        return offset

    # Helper for 2x1x1 (A 2x1, B 1x1 -> C 2x1)
    def schoolbook_211(a_ca, b_rb, c_cb, offset):
        # A rows 0,1; col a_ca
        # B row b_rb; col c_cb
        # C rows 0,1; col c_cb
        # C[0,c_cb] = A[0,a_ca]*B[b_rb,c_cb]
        # C[1,c_cb] = A[1,a_ca]*B[b_rb,c_cb]
        
        # Term 1: A[0,a_ca]*B[b_rb,c_cb] -> C[0,c_cb]
        U[0*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 0][offset] = 1
        offset += 1
        
        # Term 2: A[1,a_ca]*B[b_rb,c_cb] -> C[1,c_cb]
        U[1*5 + a_ca][offset] = 1
        V[b_rb*5 + c_cb][offset] = 1
        W[c_cb*2 + 1][offset] = 1
        offset += 1
        
        return offset

    # Now assemble the full decomposition.
    
    # C11 = A1*B11 + A2*B21 + A3*B31
    # A1: ca=0, B11: rb=0, cb=0. C11: c_base=0.
    col = strassen_222(a_ca=0, b_rb=0, c_cb=0, offset=col)
    # A2: ca=2, B21: rb=2, cb=0. C11: c_base=0.
    col = strassen_222(a_ca=2, b_rb=2, c_cb=0, offset=col)
    # A3: ca=4, B31: rb=4, cb=0. C11: c_base=0.
    col = schoolbook_212(a_ca=4, b_rb=4, c_cb=0, offset=col)
    
    # C12 = A1*B12 + A2*B22 + A3*B32
    # A1: ca=0, B12: rb=0, cb=2. C12: c_base=2.
    col = strassen_222(a_ca=0, b_rb=0, c_cb=2, offset=col)
    # A2: ca=2, B22: rb=2, cb=2. C12: c_base=2.
    col = strassen_222(a_ca=2, b_rb=2, c_cb=2, offset=col)
    # A3: ca=4, B32: rb=4, cb=2. C12: c_base=2.
    col = schoolbook_212(a_ca=4, b_rb=4, c_cb=2, offset=col)
    
    # C13 = A1*B13 + A2*B23 + A3*B33
    # A1: ca=0, B13: rb=0, cb=4. C13: c_base=4.
    col = schoolbook_221(a_ca=0, b_rb=0, c_cb=4, offset=col)
    # A2: ca=2, B23: rb=2, cb=4. C13: c_base=4.
    col = schoolbook_221(a_ca=2, b_rb=2, c_cb=4, offset=col)
    # A3: ca=4, B33: rb=4, cb=4. C13: c_base=4.
    col = schoolbook_211(a_ca=4, b_rb=4, c_cb=4, offset=col)
    
    # Check if col == rank
    # It should be 46.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
