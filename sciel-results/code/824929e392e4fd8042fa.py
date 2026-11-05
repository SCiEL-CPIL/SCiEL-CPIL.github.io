"""
Rank-36 exact tensor decomposition for M_{2,4,5} matrix multiplication.

Construction:
  - Split the 4x5 right factor into two 4x3 sub-problems: M_{2,4,3} and M_{2,4,2}.
  - M_{2,4,3}: use Strassen-style decomposition for the first 4x3 block (rank 9) +
    schoolbook for the remaining 4x0 (rank 0). Actually, let me reconsider.

  The known result from the knowledge: rank-36 via Strassen+Trivial blocks.
  
  Let me think about this more carefully.
  
  M_{2,4,5}: multiply 2x4 by 4x5 to get 2x5.
  
  We can view this as:
    C = A * B where A is 2x4, B is 4x5.
  
  Split B into B1 (4x3) and B2 (4x2).
    C1 = A * B1 (2x3)
    C2 = A * B2 (2x2)
  
  For M_{2,4,3}: rank can be at most 2*4*3 = 24, but we can do better.
  For M_{2,2,2}: rank is 7 (Strassen).
  
  Actually, let's use a different split.
  
  M_{2,4,5} can be decomposed using:
  - Treat the 4x5 part as two 4x2.5... no, must be integer dimensions.
  
  Let me try: split the middle dimension (size 4) into 2+2.
    A = [A1 | A2] where A1, A2 are 2x2
    B = [B1; B2] where B1, B2 are 2x5
    C = A1*B1 + A2*B2
    
  This is a sum of two M_{2,2,5} problems.
  If r(M_{2,2,5}) = 18, then total rank = 36.
  
  So the key question: what is r(M_{2,2,5})?
  
  Schoolbook: 2*2*5 = 20.
  Can we do 18?
  
  M_{2,2,5}: multiply 2x2 by 2x5.
  C[i,j] = sum_k A[i,k] * B[k,j]
  
  This is equivalent to two independent 2x2 * 2x1 multiplications... no, 
  it's one 2x2 times five 2x1 vectors.
  
  Actually, M_{2,2,5} tensor: T_{i,k,j} = delta_{i,k} (for each of 5 j's)
  No wait, T_{i,j,k} = 1 if... let me think.
  
  The matrix multiplication tensor M_{m,n,p} has entries T_{i,j,k} = 1 if i,k pair for row i, and j is column.
  
  For M_{2,2,5}: indices (i, j, k) where i in {0,1}, j in {0,1,2,3,4}, k in {0,1}.
  T_{i,j,k} = 1 always (it's the identity in the middle dimension).
  
  Rank of M_{2,2,5}:
  Upper bounds:
  - Schoolbook: 20
  - Can we use Strassen-like?
  
  Actually, M_{2,2,5} = M_{2,2,4} ⊕ M_{2,2,1}
  r(M_{2,2,4}) ≤ 14 (from previous analysis: two Strassen blocks of size 2x2x2)
  Wait, M_{2,2,4}: 2x2 * 2x4.
  Split the 4 columns into two pairs: M_{2,2,2} ⊕ M_{2,2,2} = 7+7 = 14.
  r(M_{2,2,1}) = 4 (schoolbook: 2*2*1 = 4)
  So r(M_{2,2,5}) ≤ 14 + 4 = 18.
  
  Yes! So r(M_{2,2,5}) ≤ 18, giving r(M_{2,4,5}) ≤ 36.
  
  Let me verify M_{2,2,1} = 4:
  C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
  C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
  This needs 4 terms: each output is a product of two inputs, and they share terms.
  Actually:
  C[0,0] = A00*B00 + A01*B10
  C[1,0] = A10*B00 + A11*B10
  
  We can write this as:
  t0 = A00*B00
  t1 = A01*B10
  t2 = A10*B00
  t3 = A11*B10
  C[0,0] = t0 + t1
  C[1,0] = t2 + t3
  
  That's 4 rank-1 terms. Can we do 3? 
  The rank of M_{2,2,1} is actually 4 (it's the tensor product of M_{2,1,1} and M_{1,2,1}, both rank 1, so... no).
  
  Actually M_{2,2,1} is the 2x2 matrix multiplication into a single column. This is equivalent to computing Ax where A is 2x2 and x is 2x1.
  The rank of this tensor is 4 (you need all 4 products).
  
  So r(M_{2,2,5}) ≤ 14 + 4 = 18. 
  
  Now let's construct the rank-36 decomposition.
  
  Step 1: Split A (2x4) into A1 (2x2) and A2 (2x2), columns 0-1 and 2-3.
  Step 2: Split B (4x5) into B1 (2x5) and B2 (2x5), rows 0-1 and 2-3.
  Step 3: C = A1*B1 + A2*B2
  
  For each M_{2,2,5} factor (e.g., A1*B1):
  - Split B1 (2x5) into B1a (2x4) and B1b (2x1)
  - C1a = A1 * B1a: use two Strassen blocks (rank 14)
  - C1b = A1 * B1b: use schoolbook (rank 4)
  - Total: 18
  
  Same for A2*B2: another 18.
  
  Total: 36.
  
  Now let's detail the Strassen decomposition for M_{2,2,2}:
  Given P (2x2) and Q (2x2), compute R = P*Q using 7 multiplications:
    m0 = (P00+P11)(Q00+Q11)
    m1 = (P10+P11)Q00
    m2 = P00(Q01-Q11)
    m3 = P11(Q10-Q00)
    m4 = (P00+P01)Q11
    m5 = (P10-P00)(Q00+Q01)
    m6 = (P01-P11)(Q10+Q11)
    
    R00 = m0 + m3 - m4 + m6
    R01 = m2 + m4
    R10 = m1 + m3
    R11 = m0 - m1 + m2 + m5
  
  Each "multiplication" is a rank-1 term in the tensor decomposition.
  
  For M_{2,2,4} using two Strassen blocks:
  Split Q (2x4) into Qa (2x2) and Qb (2x2).
  R = P*Qa + P*Qb
  
  Strassen block 1 (for P*Qa): 7 terms
  Strassen block 2 (for P*Qb): 7 terms
  Total: 14 terms
  
  For M_{2,2,1} schoolbook: 4 terms
  
  Total for M_{2,2,5}: 14 + 4 = 18 terms
  
  For M_{2,4,5}: 2 * 18 = 36 terms
  
  Now I need to carefully construct the U, V, W matrices.
  
  U has shape (n*m, rank) = (8, 36)
  V has shape (m*p, rank) = (20, 36)
  W has shape (p*n, rank) = (10, 36)
  
  The trace convention: W[k*n+i, r]
  
  For a rank-1 term corresponding to multiplication of A[i,j] * B[j,k] contributing to C[i,k]:
  U[i*m+j, r] = coefficient for A
  V[j*p+k, r] = coefficient for B  
  W[k*n+i, r] = coefficient for C
  
  Let me index:
  n=2, m=4, p=5
  
  U: 8x36, row index = i*m+j = i*4+j, i in {0,1}, j in {0,1,2,3}
  V: 20x36, row index = j*p+k = j*5+k, j in {0,1,2,3}, k in {0,1,2,3,4}
  W: 10x36, row index = k*n+i = k*2+i, k in {0,1,2,3,4}, i in {0,1}
  
  Block 1: A1 * B1 where A1 = A[:,0:2], B1 = B[0:2,:]
  Block 2: A2 * B2 where A2 = A[:,2:4], B2 = B[2:4,:]
  
  For Block 1 (A1*B1), columns j in {0,1} for A, rows j in {0,1} for B:
  - A1 indices: i in {0,1}, j in {0,1} -> U rows: i*4+j
  - B1 indices: j in {0,1}, k in {0,1,2,3,4} -> V rows: j*5+k
  
  Sub-block 1a: A1 * B1a where B1a = B1[0:2,0:4] (first 4 columns)
    Strassen block 1a-1: A1 * B1a[:,0:2] (columns 0,1 of B1)
    Strassen block 1a-2: A1 * B1a[:,2:4] (columns 2,3 of B1)
  
  Sub-block 1b: A1 * B1b where B1b = B1[0:2,4:5] (column 4 of B1)
    Schoolbook: 4 terms
  
  Similarly for Block 2.
  
  Let me code this up.
  
  For Strassen on A (2x2) * B (2x2) -> C (2x2):
  The 7 rank-1 terms:
  
  Term 0: m0 = (A00+A11)(B00+B11)
    U contribution: A00 coeff 1, A11 coeff 1
    V contribution: B00 coeff 1, B11 coeff 1
    W contribution: C00 coeff 1, C01 coeff 0, C10 coeff 0, C11 coeff 0 (partially)
    Actually W: R00 gets +m0, R11 gets +m0
    
  Let me be more careful. For each of the 7 multiplications, I create one rank-1 term.
  
  For m0 = (A00+A11)*(B00+B11):
    U: A00->1, A11->1
    V: B00->1, B11->1
    W: C00->1, C11->1 (since R00 += m0, R11 += m0)
  
  For m1 = (A10+A11)*B00:
    U: A10->1, A11->1
    V: B00->1
    W: C10->1 (R10 += m1), C11->-1 (R11 -= m1)
  
  For m2 = A00*(B01-B11):
    U: A00->1
    V: B01->1, B11->-1
    W: C01->1 (R01 += m2), C11->1 (R11 += m2)
  
  For m3 = A11*(B10-B00):
    U: A11->1
    V: B10->1, B00->-1
    W: C00->1 (R00 += m3), C10->1 (R10 += m3)
  
  For m4 = (A00+A01)*B11:
    U: A00->1, A01->1
    V: B11->1
    W: C01->1 (R01 += m4), C00->-1 (R00 -= m4)
  
  For m5 = (A10-A00)*(B00+B01):
    U: A10->1, A00->-1
    V: B00->1, B01->1
    W: C11->1 (R11 += m5)
  
  For m6 = (A01-A11)*(B10+B11):
    U: A01->1, A11->-1
    V: B10->1, B11->1
    W: C00->1 (R00 += m6), C11->-1 (R11 -= m6)
  
  Let me verify R00 = m0 + m3 - m4 + m6:
    m0: +1 to C00
    m3: +1 to C00
    m4: -1 to C00
    m6: +1 to C00
    Total: 1+1-1+1 = 2? That's wrong.
  
  Wait, I think I made errors. Let me redo Strassen carefully.
  
  Standard Strassen:
  m1 = (a11+a22)(b11+b22)
  m2 = (a21+a22)b11
  m3 = a11(b12-b22)
  m4 = a22(b21-b11)
  m5 = (a11+a12)b22
  m6 = (a21-a11)(b11+b12)
  m7 = (a12-a22)(b21+b22)
  
  c11 = m1 + m4 - m5 + m7
  c12 = m3 + m5
  c21 = m2 + m4
  c22 = m1 - m2 + m3 + m6
  
  Using 1-indexed, with a11=A[0,0], etc.
  
  So for each term, the W coefficients are:
  m1: C00->+1, C11->+1
  m2: C10->+1, C11->-1
  m3: C01->+1, C11->+1
  m4: C00->+1, C10->+1
  m5: C01->+1, C00->-1
  m6: C11->+1
  m7: C00->+1, C11->-1
  
  Let me verify:
  C00 = m1 + m4 - m5 + m7: +1 +1 -1 +1 = 2? No...
  
  Hmm, m1 contributes +1 to C00, m4 contributes +1 to C00, m5 contributes -1 to C00, m7 contributes +1 to C00.
  Sum: 1+1-1+1 = 2. That's not right.
  
  Oh wait, I think the formula is:
  c11 = m1 + m4 - m5 + m7
  
  But m1 has C00 coefficient +1 and C11 coefficient +1.
  So m1 contributes +1 to C00.
  m4 has C00 coefficient +1 and C10 coefficient +1.
  So m4 contributes +1 to C00.
  m5 has C01 coefficient +1 and C00 coefficient -1.
  So m5 contributes -1 to C00.
  m7 has C00 coefficient +1 and C11 coefficient -1.
  So m7 contributes +1 to C00.
  
  Total C00 = 1+1-1+1 = 2. That's wrong; should be 1.
  
  I must have the Strassen formula wrong. Let me look it up properly.
  
  Correct Strassen:
  p1 = (a11 + a22)(b11 + b22)
  p2 = (a21 + a22) * b11
  p3 = a11 * (b12 - b22)
  p4 = a22 * (b21 - b11)
  p5 = (a11 + a12) * b22
  p6 = (a21 - a11) * (b11 + b12)
  p7 = (a12 - a22) * (b21 + b22)
  
  c11 = p1 - p5 + p6 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 - p6
  
  Let me verify with a simple case. If A = I, B = I:
  p1 = (1+0)(1+0) = 1
  p2 = (0+0)*1 = 0
  p3 = 1*(0-0) = 0
  p4 = 0*(0-1) = 0
  p5 = (1+0)*0 = 0
  p6 = (0-1)*(1+0) = -1
  p7 = (0-0)*(0+0) = 0
  
  c11 = 1 - 0 + (-1) + 0 = 0. But should be 1. Wrong.
  
  Hmm, let me try yet another formulation.
  
  p1 = (a11 + a22)(b11 + b22)
  p2 = (a21 + a22)b11
  p3 = a11(b12 - b22)
  p4 = a22(b21 - b11)
  p5 = (a11 + a12)b22
  p6 = (a21 - a11)(b11 + b12)
  p7 = (a12 - a22)(b21 + b22)
  
  c11 = p1 - p5 + p6 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 - p6
  
  With A=I, B=I:
  p1 = 1, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = -1, p7 = 0
  c11 = 1 - 0 + (-1) + 0 = 0. Still wrong.
  
  I think the issue is with p6. Let me recompute:
  a21 - a11 = 0 - 1 = -1
  b11 + b12 = 1 + 0 = 1
  p6 = -1 * 1 = -1
  
  c11 = p1 - p5 + p6 + p7 = 1 - 0 + (-1) + 0 = 0.
  
  But c11 should be a11*b11 + a12*b21 = 1*1 + 0*0 = 1.
  
  So this formula is wrong. Let me try:
  c11 = p1 - p5 + p7
  c12 = p3 + p5
  c21 = p2 + p4
  c22 = p1 - p2 + p3 - p6 + p7? No...
  
  OK let me just derive it from scratch or use a verified source.
  
  The correct Strassen decomposition (0-indexed):
  
  m0 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
  m1 = (A[1,0]+A[1,1])*B[0,0]
  m2 = A[0,0]*(B[0,1]-B[1,1])
  m3 = A[1,1]*(B[1,0]-B[0,0])
  m4 = (A[0,0]+A[0,1])*B[1,1]
  m5 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
  m6 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
  
  C[0,0] = m0 + m3 - m4 + m6
  C[0,1] = m2 + m4
  C[1,0] = m1 + m3
  C[1,1] = m0 - m1 + m2 + m5
  
  Test with A=I, B=I:
  m0 = (1+0)(1+0) = 1
  m1 = (0+0)*1 = 0
  m2 = 1*(0-0) = 0
  m3 = 0*(0-1) = 0
  m4 = (1+0)*0 = 0
  m5 = (0-1)*(1+0) = -1
  m6 = (0-0)*(0+0) = 0
  
  C[0,0] = 1+0-0+0 = 1 ✓
  C[0,1] = 0+0 = 0 ✓
  C[1,0] = 0+0 = 0 ✓
  C[1,1] = 1-0+0+(-1) = 0 ✓
  
  Now let's test A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
  Expected C = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  
  m0 = (1+4)*(5+8) = 5*13 = 65
  m1 = (3+4)*5 = 35
  m2 = 1*(6-8) = -2
  m3 = 4*(7-5) = 8
  m4 = (1+2)*8 = 24
  m5 = (3-1)*(5+6) = 2*11 = 22
  m6 = (2-4)*(7+8) = -2*15 = -30
  
  C[0,0] = 65+8-24+(-30) = 19 ✓
  C[0,1] = -2+24 = 22 ✓
  C[1,0] = 35+8 = 43 ✓
  C[1,1] = 65-35+(-2)+22 = 50 ✓
  
  So the Strassen formulas are correct. Now the W coefficients for each m:
  
  m0: C[0,0]->+1, C[1,1]->+1
  m1: C[1,0]->+1, C[1,1]->-1
  m2: C[0,1]->+1, C[1,1]->+1
  m3: C[0,0]->+1, C[1,0]->+1
  m4: C[0,1]->+1, C[0,0]->-1
  m5: C[1,1]->+1
  m6: C[0,0]->+1, C[1,1]->-1
  
  Now I can build the full rank-36 decomposition.
  
  Let me code it up:
  """

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    
    def add_term(r, u_dict, v_dict, w_dict):
        """Add a rank-1 term at column r.
        u_dict: {(i,j): coeff} for A[i,j], mapped to U[i*m+j]
        v_dict: {(j,k): coeff} for B[j,k], mapped to V[j*p+k]
        w_dict: {(k,i): coeff} for C[i,k], mapped to W[k*n+i]
        """
        for (i, j), coeff in u_dict.items():
            if coeff != 0:
                U[i * m + j][r] += coeff
        for (j, k), coeff in v_dict.items():
            if coeff != 0:
                V[j * p + k][r] += coeff
        for (k, i), coeff in w_dict.items():
            if coeff != 0:
                W[k * n + i][r] += coeff
    
    def strassen_block(col_start, a_row_offset, b_row_offset, b_col_offset, b_col_end):
        """
        Add Strassen decomposition for A[a_row_offset:a_row_offset+2, :] * B[b_row_offset:b_row_offset+2, b_col_offset:b_col_end]
        where B sub-matrix is 2x2 (b_col_end = b_col_offset+2).
        Returns next column index.
        """
        # A sub-block: rows a_row_offset to a_row_offset+1 (2 rows), all 4 columns? No.
        # Actually, for M_{2,2,2}, A is 2x2 and B is 2x2.
        # In our context, A1 is 2x2 (i in {0,1}, j in {a_row_offset, a_row_offset+1})
        # B sub is 2x2 (j in {b_row_offset, b_row_offset+1}, k in {b_col_offset, b_col_offset+1})
        
        a_r0 = a_row_offset
        a_r1 = a_row_offset + 1
        b_r0 = b_row_offset
        b_r1 = b_row_offset + 1
        b_c0 = b_col_offset
        b_c1 = b_col_offset + 1
        
        col = col_start
        
        # m0 = (A[a_r0,j0] + A[a_r1,j1]) * (B[b_r0,k0] + B[b_r1,k1])
        # where j0=a_r0, j1=a_r1 (since A is 2x2, column indices are the same as row indices in the sub-block)
        # Wait, I need to be more careful. In the sub-problem M_{2,2,2}, A has 2 rows and 2 cols.
        # The "row offset" for A refers to which columns of the original A we're using.
        # Let me redefine: 
        #   A_sub[i][j] = A[i][a_col_offset + j] for i in {0,1}, j in {0,1}
        #   B_sub[i][j] = B[b_row_offset + i][b_col_offset + j] for i in {0,1}, j in {0,1}
        #   C_sub[i][j] = C[i][b_col_offset + j] for i in {0,1}, j in {0,1}
        
        a_c0 = a_row_offset  # first column of A sub-block
        a_c1 = a_row_offset + 1  # second column of A sub-block
        
        # m0
        u = {(0, a_c0): 1, (1, a_c1): 1}
        v = {(b_r0, b_c0): 1, (b_r1, b_c1): 1}
        w = {(b_c0, 0): 1, (b_c1, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        # m1 = (A[1,a_c0] + A[1,a_c1]) * B[b_r0, b_c0]
        u = {(1, a_c0): 1, (1, a_c1): 1}
        v = {(b_r0, b_c0): 1}
        w = {(b_c0, 1): 1, (b_c1, 1): -1}
        add_term(col, u, v, w)
        col += 1
        
        # m2 = A[0,a_c0] * (B[b_r0,b_c1] - B[b_r1,b_c1])
        u = {(0, a_c0): 1}
        v = {(b_r0, b_c1): 1, (b_r1, b_c1): -1}
        w = {(b_c1, 0): 1, (b_c1, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        # m3 = A[1,a_c1] * (B[b_r1,b_c0] - B[b_r0,b_c0])
        u = {(1, a_c1): 1}
        v = {(b_r1, b_c0): 1, (b_r0, b_c0): -1}
        w = {(b_c0, 0): 1, (b_c0, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        # m4 = (A[0,a_c0] + A[0,a_c1]) * B[b_r1,b_c1]
        u = {(0, a_c0): 1, (0, a_c1): 1}
        v = {(b_r1, b_c1): 1}
        w = {(b_c1, 0): 1, (b_c0, 0): -1}
        add_term(col, u, v, w)
        col += 1
        
        # m5 = (A[1,a_c0] - A[0,a_c0]) * (B[b_r0,b_c0] + B[b_r0,b_c1])
        u = {(1, a_c0): 1, (0, a_c0): -1}
        v = {(b_r0, b_c0): 1, (b_r0, b_c1): 1}
        w = {(b_c1, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        # m6 = (A[0,a_c1] - A[1,a_c1]) * (B[b_r1,b_c0] + B[b_r1,b_c1])
        u = {(0, a_c1): 1, (1, a_c1): -1}
        v = {(b_r1, b_c0): 1, (b_r1, b_c1): 1}
        w = {(b_c0, 0): 1, (b_c1, 1): -1}
        add_term(col, u, v, w)
        col += 1
        
        return col
    
    def schoolbook_2x2x1(col, a_c0, a_c1, b_r0, b_r1, b_c):
        """
        Schoolbook for A[0:2, a_c0:a_c1+1] * B[b_r0:b_r1+1, b_c:b_c+1] -> C[0:2, b_c]
        4 terms.
        """
        # C[0,b_c] = A[0,a_c0]*B[b_r0,b_c] + A[0,a_c1]*B[b_r1,b_c]
        # C[1,b_c] = A[1,a_c0]*B[b_r0,b_c] + A[1,a_c1]*B[b_r1,b_c]
        
        # Term 1: A[0,a_c0]*B[b_r0,b_c] -> C[0,b_c]
        u = {(0, a_c0): 1}
        v = {(b_r0, b_c): 1}
        w = {(b_c, 0): 1}
        add_term(col, u, v, w)
        col += 1
        
        # Term 2: A[0,a_c1]*B[b_r1,b_c] -> C[0,b_c]
        u = {(0, a_c1): 1}
        v = {(b_r1, b_c): 1}
        w = {(b_c, 0): 1}
        add_term(col, u, v, w)
        col += 1
        
        # Term 3: A[1,a_c0]*B[b_r0,b_c] -> C[1,b_c]
        u = {(1, a_c0): 1}
        v = {(b_r0, b_c): 1}
        w = {(b_c, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        # Term 4: A[1,a_c1]*B[b_r1,b_c] -> C[1,b_c]
        u = {(1, a_c1): 1}
        v = {(b_r1, b_c): 1}
        w = {(b_c, 1): 1}
        add_term(col, u, v, w)
        col += 1
        
        return col
    
    col = 0
    
    # Block 1: A1 (cols 0,1 of A) * B1 (rows 0,1 of B)
    #   Sub-block 1a-1: A1 * B1[0:2, 0:2] (Strassen, 7 terms)
    col = strassen_block(col, a_row_offset=0, b_row_offset=0, b_col_offset=0, b_col_end=2)
    #   Sub-block 1a-2: A1 * B1[0:2, 2:4] (Strassen, 7 terms)
    col = strassen_block(col, a_row_offset=0, b_row_offset=0, b_col_offset=2, b_col_end=4)
    #   Sub-block 1b: A1 * B1[0:2, 4:5] (schoolbook 2x2x1, 4 terms)
    col = schoolbook_2x2x1(col, a_c0=0, a_c1=1, b_r0=0, b_r1=1, b_c=4)
    
    # Block 2: A2 (cols 2,3 of A) * B2 (rows 2,3 of B)
    #   Sub-block 2a-1: A2 * B2[2:4, 0:2] (Strassen, 7 terms)
    col = strassen_block(col, a_row_offset=2, b_row_offset=2, b_col_offset=0, b_col_end=2)
    #   Sub-block 2a-2: A2 * B2[2:4, 2:4] (Strassen, 7 terms)
    col = strassen_block(col, a_row_offset=2, b_row_offset=2, b_col_offset=2, b_col_end=4)
    #   Sub-block 2b: A2 * B2[2:4, 4:5] (schoolbook 2x2x1, 4 terms)
    col = schoolbook_2x2x1(col, a_c0=2, a_c1=3, b_r0=2, b_r1=3, b_c=4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
