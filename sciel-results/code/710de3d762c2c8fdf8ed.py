"""Matrix multiplication 4x5 by 5x6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    U = []
    V = []
    W = []

    def add_term(u, v, w):
        U.append(u)
        V.append(v)
        W.append(w)

    # Strassen's 7-term decomposition for 2x2 x 2x2 -> 2x2
    # C = A * B
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For a 2x2 block with row indices (r1, r2) for A, col indices (c1, c2) for A (=row for B)
    # and col indices (d1, d2) for B, the 7 terms are:
    # Each term is a product of a sum/difference of A entries times a sum/difference of B entries
    # U vector: coefficient for each A entry (A[r1][c1], A[r1][c2], A[r2][c1], A[r2][c2])
    # V vector: coefficient for each B entry (B[c1][d1], B[c1][d2], B[c2][d1], B[c2][d2])
    # W vector: which C entries it contributes to, and with what coefficient

    def strassen_222(r1, r2, c1, c2, d1, d2):
        """
        Returns list of 7 (u, v, w) tuples for Strassen decomposition of
        A[r1:r2+1][c1:c2+1] * B[c1:c2+1][d1:d2+1]
        where A is n x m, B is m x p, C is n x p.
        u has length n*m, v has length m*p, w has length p*n.
        Indices: A row r, col c -> u index r*m + c
                 B row c, col d -> v index c*p + d
                 C row r, col d -> w index d*n + r  (W[k*n+i, r] convention: W[col * n + row])
        Wait, the convention is W[k*n+i, r]. Let me re-read.
        "The output factor follows the official trace convention W[k*n+i,r]."
        So W is indexed by [k*n+i, r] where k is the column of C, i is the row of C.
        So W index = k * n + i where C[i][k] is the output entry.
        Actually, C is n x p. C[i][k] where i in 0..n-1, k in 0..p-1.
        W index = k * n + i.
        """
        # A entries: (r1,c1), (r1,c2), (r2,c1), (r2,c2)
        # B entries: (c1,d1), (c1,d2), (c2,d1), (c2,d2)
        # C entries: (r1,d1), (r1,d2), (r2,d1), (r2,d2)
        
        terms = []
        # M1 = (A[r1][c1] + A[r2][c2]) * (B[c1][d1] + B[c2][d2])
        u1 = [0]*(n*m); u1[r1*m+c1] = 1; u1[r2*m+c2] = 1
        v1 = [0]*(m*p); v1[c1*p+d1] = 1; v1[c2*p+d2] = 1
        w1 = [0]*(p*n); w1[d1*n+r1] = 1; w1[d2*n+r2] = 1; w1[d2*n+r1] = 1; w1[d1*n+r2] = 1
        terms.append((u1, v1, w1))
        
        # M2 = (A[r2][c1] + A[r2][c2]) * B[c1][d1]
        u2 = [0]*(n*m); u2[r2*m+c1] = 1; u2[r2*m+c2] = 1
        v2 = [0]*(m*p); v2[c1*p+d1] = 1
        w2 = [0]*(p*n); w2[d1*n+r2] = 1
        terms.append((u2, v2, w2))
        
        # M3 = A[r1][c1] * (B[c2][d1] - B[c2][d2])
        u3 = [0]*(n*m); u3[r1*m+c1] = 1
        v3 = [0]*(m*p); v3[c2*p+d1] = 1; v3[c2*p+d2] = -1
        w3 = [0]*(p*n); w3[d1*n+r1] = 1; w3[d2*n+r1] = -1
        terms.append((u3, v3, w3))
        
        # M4 = A[r2][c2] * (B[c1][d2] - B[c1][d1])
        u4 = [0]*(n*m); u4[r2*m+c2] = 1
        v4 = [0]*(m*p); v4[c1*p+d2] = 1; v4[c1*p+d1] = -1
        w4 = [0]*(p*n); w4[d2*n+r2] = 1; w4[d1*n+r2] = -1
        terms.append((u4, v4, w4))
        
        # M5 = (A[r1][c1] + A[r1][c2]) * B[c2][d2]
        u5 = [0]*(n*m); u5[r1*m+c1] = 1; u5[r1*m+c2] = 1
        v5 = [0]*(m*p); v5[c2*p+d2] = 1
        w5 = [0]*(p*n); w5[d2*n+r1] = 1
        terms.append((u5, v5, w5))
        
        # M6 = (A[r2][c1] - A[r1][c1]) * (B[c1][d1] + B[c1][d2])
        u6 = [0]*(n*m); u6[r2*m+c1] = 1; u6[r1*m+c1] = -1
        v6 = [0]*(m*p); v6[c1*p+d1] = 1; v6[c1*p+d2] = 1
        w6 = [0]*(p*n); w6[d2*n+r2] = 1; w6[d1*n+r2] = 1
        terms.append((u6, v6, w6))
        
        # M7 = (A[r1][c2] - A[r2][c2]) * (B[c2][d1] + B[c2][d2])
        u7 = [0]*(n*m); u7[r1*m+c2] = 1; u7[r2*m+c2] = -1
        v7 = [0]*(m*p); v7[c2*p+d1] = 1; v7[c2*p+d2] = 1
        w7 = [0]*(p*n); w7[d1*n+r1] = 1; w7[d2*n+r1] = 1
        terms.append((u7, v7, w7))
        
        return terms

    def schoolbook(n_sb, m_sb, p_sb):
        """Schoolbook decomposition for n_sb x m_sb times m_sb x p_sb."""
        terms = []
        for i in range(n_sb):
            for j in range(m_sb):
                for k in range(p_sb):
                    u = [0]*(n_sb*m_sb); u[i*m_sb+j] = 1
                    v = [0]*(m_sb*p_sb); v[j*p_sb+k] = 1
                    w = [0]*(p_sb*n_sb); w[k*n_sb+i] = 1
                    terms.append((u, v, w))
        return terms

    # Now let's think about the structure.
    # A is 4x5, B is 5x6, C is 4x6.
    # We can split this into blocks.
    
    # Strategy: Use Strassen where possible.
    # Split A into rows 0-1 and 2-3 (each 2x5), B into cols 0-1, 2-3, 4-5 (each 5x2).
    # But Strassen needs square blocks.
    
    # Alternative: Pad to 4x4 x 4x6 or use a combination.
    
    # Let me think of a cleaner decomposition.
    # M(4,5,6) can be bounded by splitting:
    # Split A into A1 (2x5) and A2 (2x5)
    # Split B into B1 (5x2), B2 (5x2), B3 (5x2)
    # C1 = A1*B1 (2x2), C2 = A1*B2 (2x2), C3 = A1*B3 (2x2)
    # C4 = A2*B1 (2x2), C5 = A2*B2 (2x2), C6 = A2*B3 (2x2)
    # Each 2x2 product can use Strassen with 7 mults.
    # Total: 6 * 7 = 42? But the intermediate sums share entries.
    
    # Actually, the standard way: if we have A (n1+n2) x m, B (m) x (p1+p2+p3),
    # and we compute (n1 x p1), (n1 x p2), (n1 x p3), (n2 x p1), (n2 x p2), (n2 x p3)
    # each as a separate matrix multiply, we get sum of ranks.
    # M(2,5,2) for each block.
    
    # M(2,5,2): split B into two 5x1 parts? Or A into 1x5 parts?
    # M(2,5,2) <= M(2,2,2) + M(2,3,2) or similar.
    # Actually M(2,5,2) schoolbook = 20. Strassen-like:
    # Split A into 1+1 rows, B into 1+1 cols? No, B is 5x2.
    # Split A into 2 blocks of 1 row, B into 5 blocks of 1 col, C into 2x2.
    # That's schoolbook.
    
    # Let me try: M(2,5,2). 
    # A is 2x5, B is 5x2. C is 2x2.
    # C[0][0] = sum_j A[0][j]*B[j][0]
    # C[0][1] = sum_j A[0][j]*B[j][1]
    # C[1][0] = sum_j A[1][j]*B[j][0]
    # C[1][1] = sum_j A[1][j]*B[j][1]
    
    # Can we use Strassen? Strassen is for 2x2x2. Here we have 2x5x2.
    # We can view this as: for each of the 5 intermediate columns, we have a 2x1 x 1x2 product.
    # That's 5 separate 2x1x2 products, each of rank 4 (schoolbook) or...
    # M(2,1,2) = 4 schoolbook. Can we do better? M(2,1,2) is a rank-4 tensor, minimal rank is 4.
    # So M(2,5,2) >= 5 * 4 = 20? No, that's not right because they share structure.
    # Actually M(2,5,2) schoolbook = 2*5*2 = 20.
    
    # Let me try a different approach. Let's use the known bounds.
    # M(4,4,4) <= 49 (Strassen)
    # M(4,5,6) <= ?
    
    # One approach: pad A to 4x6 and B to 6x6, compute 4x6 x 6x6, take top 4x5 part... no.
    
    # Let me just do: split into 2x2 blocks where possible.
    # Rows 0,1 and 2,3 of A. Cols 0,1 and 2,3 and 4,5 of B... but B has 5 rows.
    
    # Let me split differently:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    # B = [[B1], [B2]] where B1 is 2x6, B2 is 3x6
    # C = A11*B1 + A12*B2 (top 2 rows) and A21*B1 + A22*B2 (bottom 2 rows)
    
    # A11*B1: 2x2 times 2x6. M(2,2,6).
    # A12*B2: 2x3 times 3x6. M(2,3,6).
    # A21*B1: 2x2 times 2x6. M(2,2,6).
    # A22*B2: 2x3 times 3x6. M(2,3,6).
    
    # M(2,2,6): split into 3 blocks of 2x2x2, each Strassen 7. Total 21.
    # Or: M(2,2,6) <= M(2,2,2) + M(2,2,2) + M(2,2,2) = 21.
    # M(2,3,6): split into M(2,1,6) + M(2,2,6) = ?
    # M(2,1,6) = 12 schoolbook (2*1*6).
    # M(2,2,6) = 21 as above.
    # So M(2,3,6) <= 12 + 21 = 33.
    
    # Total: 21 + 33 + 21 + 33 = 108.
    # That's worse than 120? No, 108 < 120. Good.
    
    # Can we do better on M(2,3,6)?
    # M(2,3,6): A is 2x3, B is 3x6, C is 2x6.
    # Split B into two 3x3 blocks: M(2,3,3) + M(2,3,3).
    # M(2,3,3): A 2x3, B 3x3, C 2x3.
    # Split A into 1x3 + 1x3: M(1,3,3) + M(1,3,3) = 2 * 9 = 18.
    # Or M(2,3,3): split into M(2,1,3) + M(2,2,3) = 6 + M(2,2,3).
    # M(2,2,3) = 12 schoolbook or M(2,2,2)+M(2,2,1) = 7+4=11.
    # So M(2,3,3) <= 6 + 11 = 17.
    # M(2,3,6) <= 2 * 17 = 34. Slightly worse than 33.
    
    # Let me try M(2,2,6) better.
    # M(2,2,6): A 2x2, B 2x6, C 2x6.
    # Strassen for the 2x2 part gives 7 mults for 2x2x2.
    # For 2x2x6, we can think of it as 3 separate 2x2x2 multiplies (cols 0-1, 2-3, 4-5 of B).
    # But the Strassen intermediate sums are the same for all three!
    # M1 = (A11+A22)(B11+B22), etc.
    # For B with 6 columns, B11+B22 means B[0][2k]+B[1][2k+1] for k=0,1,2.
    # So we still need 7 "multiplies" but each involves a sum of 2 B entries times a sum of 2 A entries.
    # Each such term contributes to all 6 columns of C.
    # So M(2,2,6) can be done in 7 mults? No, each "multiply" is between a scalar (from A sum) and a vector of length 6 (from B sums).
    # In the tensor decomposition, each rank-1 term is a single multiplication.
    # So M(2,2,6) needs at least 2*2*6/2 = 12? No.
    # Schoolbook is 24. Strassen gives 7 * 3 = 21 if we do it as three separate 2x2x2.
    # But we can do better: the 7 Strassen formulas work for any number of columns.
    # M1 = (A[0][0]+A[1][1]) * (B[0][j]+B[1][j]) for j=0..5 -> this is 6 separate products? No.
    # In the tensor rank sense, (A[0][0]+A[1][1]) * (B[0][j]+B[1][j]) is a sum of 2*2 = 4 rank-1 tensors for each j.
    # So M1 contributes 4*6 = 24 rank-1 terms? No, that's schoolbook.
    
    # I'm overcomplicating. Let me just implement the 108-rank decomposition.
    
    def strassen_222_global(r1, r2, c1, c2, d1, d2):
        """Strassen for A[r1:r2+1][c1:c2+1] * B[c1:c2+1][d1:d2+1] in global n,m,p coords."""
        terms = []
        # M1
        u = [0]*(n*m); u[r1*m+c1]=1; u[r2*m+c2]=1
        v = [0]*(m*p); v[c1*p+d1]=1; v[c2*p+d2]=1
        w = [0]*(p*n); w[d1*n+r1]=1; w[d2*n+r2]=1; w[d2*n+r1]=1; w[d1*n+r2]=1
        terms.append((u,v,w))
        # M2
        u = [0]*(n*m); u[r2*m+c1]=1; u[r2*m+c2]=1
        v = [0]*(m*p); v[c1*p+d1]=1
        w = [0]*(p*n); w[d1*n+r2]=1
        terms.append((u,v,w))
        # M3
        u = [0]*(n*m); u[r1*m+c1]=1
        v = [0]*(m*p); v[c2*p+d1]=1; v[c2*p+d2]=-1
        w = [0]*(p*n); w[d1*n+r1]=1; w[d2*n+r1]=-1
        terms.append((u,v,w))
        # M4
        u = [0]*(n*m); u[r2*m+c2]=1
        v = [0]*(m*p); v[c1*p+d2]=1; v[c1*p+d1]=-1
        w = [0]*(p*n); w[d2*n+r2]=1; w[d1*n+r2]=-1
        terms.append((u,v,w))
        # M5
        u = [0]*(n*m); u[r1*m+c1]=1; u[r1*m+c2]=1
        v = [0]*(m*p); v[c2*p+d2]=1
        w = [0]*(p*n); w[d2*n+r1]=1
        terms.append((u,v,w))
        # M6
        u = [0]*(n*m); u[r2*m+c1]=1; u[r1*m+c1]=-1
        v = [0]*(m*p); v[c1*p+d1]=1; v[c1*p+d2]=1
        w = [0]*(p*n); w[d2*n+r2]=1; w[d1*n+r2]=1
        terms.append((u,v,w))
        # M7
        u = [0]*(n*m); u[r1*m+c2]=1; u[r2*m+c2]=-1
        v = [0]*(m*p); v[c2*p+d1]=1; v[c2*p+d2]=1
        w = [0]*(p*n); w[d1*n+r1]=1; w[d2*n+r1]=1
        terms.append((u,v,w))
        return terms

    # M(2,2,6): A rows 0-1, B cols 0-1, C cols 0-5
    # Split into three 2x2x2 blocks: cols (0,1), (2,3), (4,5)
    for d_start in [0, 2, 4]:
        for u,v,w in strassen_222_global(0, 1, 0, 1, d_start, d_start+1):
            add_term(u, v, w)
    
    # M(2,3,6): A rows 0-1, B cols 2-4, C cols 0-5
    # Split B col 2 separately (1 col), then cols 3-4 (2 cols)
    # M(2,1,6): schoolbook
    for i in range(2):
        for k in range(6):
            u = [0]*(n*m); u[i*m+2]=1
            v = [0]*(m*p); v[2*p+k]=1
            w = [0]*(p*n); w[k*n+i]=1
            add_term(u,v,w)
    # M(2,2,6): A rows 0-1, B cols 3-4, C cols 0-5
    for d_start in [0, 2, 4]:
        for u,v,w in strassen_222_global(0, 1, 3, 4, d_start, d_start+1):
            add_term(u,v,w)
    
    # M(2,2,6) for A rows 2-3, B cols 0-1
    for d_start in [0, 2, 4]:
        for u,v,w in strassen_222_global(2, 3, 0, 1, d_start, d_start+1):
            add_term(u,v,w)
    
    # M(2,3,6) for A rows 2-3, B cols 2-4
    for i in range(2,4):
        for k in range(6):
            u = [0]*(n*m); u[i*m+2]=1
            v = [0]*(m*p); v[2*p+k]=1
            w = [0]*(p*n); w[k*n+i]=1
            add_term(u,v,w)
    for d_start in [0, 2, 4]:
        for u,v,w in strassen_222_global(2, 3, 3, 4, d_start, d_start+1):
            add_term(u,v,w)
    
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
