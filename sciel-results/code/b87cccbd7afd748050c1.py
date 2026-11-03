"""
Rank-minimal exact tensor decomposition for 2x4 @ 4x7 matrix multiplication.

Strategy:
- n=2, m=4, p=7
- Split the 4-column input into two 2-column halves: A1 (2x2), A2 (2x2)
- Split the 4-column B into two 2-row halves: B1 (2x7), B2 (2x7)
- Apply Strassen to each 2x2 @ 2x7 block (7 multiplications each)
- For the second block (A2 @ B2), note that B2 = [row2..row3 of B]
  and we also need the "cross" terms. Actually, let's think more carefully.

C = A @ B where A is 2x4, B is 4x7.
C = A1 @ B1 + A2 @ B2 where:
  A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x7)
  A2 = A[:, 2:4] (2x2), B2 = B[2:4, :] (2x7)

Each 2x2 @ 2x7 product: Strassen gives 7 scalar multiplications of (2x2 sums) times (2x7 sums),
but actually Strassen for 2x2 @ 2xn uses 7 multiplications where each "multiplication" is between
a 2x2 matrix (from A side, which is a scalar sum) and a 2xn matrix (from B side).
Wait, no. Strassen works on matrices. For 2x2 @ 2xn, we can still apply the same 7 multiplications
where M1..M7 are scalars (from A side linear combos) times... no.

Actually Strassen decomposes 2x2 @ 2x2 into 7 scalar multiplications. For 2x2 @ 2xn, we can
view B as n separate 2x2 blocks? No, that doesn't work directly.

Let me reconsider. The standard approach:
- C[i,k] = sum_j A[i,j] * B[j,k] for i in 0,1; j in 0,1,2,3; k in 0..6

Using Strassen on the j=0,1 block and j=2,3 block:

For block 1 (j=0,1):
  M1 = (A[0,0]+A[1,1]) * (B[0,0..6]+B[1,0..6])  -- no, this isn't right for rectangular.

Actually, Strassen for 2x2 @ 2xn:
  Let A_block = [[a, b], [c, d]] (2x2), B_block = [[e, f, ..., s], [g, h, ..., t]] (2xn)
  
  M1 = (a+d)(e+g+...+s) -- this gives a vector, not a scalar.

Hmm, Strassen's algorithm naturally extends to 2x2 @ 2xn by treating each column independently
or by noting that the 7 "multiplications" each produce a 1xn vector result.

So for 2x2 @ 2xn:
  M1 = (a+d) * (e+g)  [for each column, so a 1xn vector]
  M2 = (c+d) * e      [1xn vector]
  M3 = a * (f-h)      [1xn vector]
  M4 = d * (g-e)      [1xn vector]
  M5 = (a+c) * t      [1xn vector]
  M6 = (b-d) * (g+t)  [1xn vector]
  M7 = (c-a) * (e+s)  [1xn vector]

Each M_i is a scalar (from A) times a 1xn vector (from B), giving a 1xn vector.
Then the 2xn result is:
  result[0,:] = M1 + M4 - M5 + M7
  result[1,:] = M2 + M1 - M3 - M6

So each 2x2 @ 2xn costs 7 "scalar-vector multiplications", which in tensor terms means 7 rank-1 terms
where the A-side is a scalar linear combo and the B-side is a vector linear combo.

In the tensor decomposition framework:
- U has shape (n*m, rank) = (8, rank)
- V has shape (m*p, rank) = (28, rank)
- W has shape (p*n, rank) = (14, rank)

For each Strassen multiplication, we have one rank-1 term where:
- U side: a linear combo of A entries (scalar)
- V side: a linear combo of B entries (vector over k)
- W side: contributes to C[i,k]

Let me work this out explicitly.

For A1 @ B1 (using columns 0,1 of A and rows 0,1 of B):
  a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
  For each k: e=B[0,k], f=B[1,k]

  M1 = (a+d)(e+f)
  M2 = (c+d)e
  M3 = a(f-h) -- wait, h is B[1,k], so M3 = a*(B[1,k] - B[1,k])? No.

Let me redo Strassen properly for 2x2 @ 2xn:
  A = [[a, b], [c, d]], B = [[b0, b1, ..., b6], [c0, c1, ..., c6]]
  
  M1 = (a+d)(b0+c0, b1+c1, ..., b6+c6)
  M2 = (c+d)(b0, b1, ..., b6)
  M3 = a(c0-b0, c1-b1, ..., c6-b6)  -- wait, standard Strassen:
  
  Standard Strassen for 2x2:
  P1 = (a+d)(e+g)
  P2 = (c+d)e
  P3 = a(f-h)
  P4 = d(g-e)
  P5 = (a+c)h
  P6 = (b-d)(g+h)
  P7 = (c-a)(e+f)

  Where A=[[a,b],[c,d]], B=[[e,f],[g,h]]

  So for 2x2 @ 2xn, with B having rows [b0..b6] and [c0..c6]:
  P1 = (a+d) * [b0+c0, b1+c1, ..., b6+c6]
  P2 = (c+d) * [b0, b1, ..., b6]
  P3 = a * [c0-b0, c1-b1, ..., c6-b6]
  P4 = d * [c0-b0, c1-b1, ..., c6-b6]  -- wait no, P4 = d*(g-e) where g is row2, e is row1
       = d * [c0-b0, c1-b1, ..., c6-b6]
  P5 = (a+c) * [c0, c1, ..., c6]
  P6 = (b-d) * [c0+b0, c1+b1, ..., c6+b6]
  P7 = (c-a) * [b0+c0, b1+c1, ..., b6+c6]

  Result:
  R[0,:] = P1 + P4 - P5 + P7
  R[1,:] = P2 + P1 - P3 - P6

Now in tensor terms, each P_i corresponds to 1 rank-1 term (since the A-side is a scalar linear combo
and the B-side is a linear combo producing a vector, and W distributes to C[i,k]).

For P1 of block 1 (A1@B1):
  A-side scalar: A[0,0] + A[1,1]
  B-side: for each k, B[0,k] + B[1,k]
  This contributes to C[0,k] with coeff +1 and C[1,k] with coeff +1 (from R[0,:] and R[1,:] formulas)

Wait, P1 appears in both R[0,:] and R[1,:] with coefficient +1.
So W[0*n+0, c] = 1 (C[0,k]) and W[1*n+0, c] = 1 (C[1,k]) for all k? No, W is indexed by (k*n+i).

W[k*n + i, c]: this is the coefficient for C[i,k] from term c.

For P1 of block 1: contributes +1 to C[0,k] for all k (0..6) and +1 to C[1,k] for all k.
So W[k*2+0, c] = 1 and W[k*2+1, c] = 1 for all k=0..6.

U side: U[0*4+0, c] = 1 (A[0,0]), U[1*4+1, c] = 1 (A[1,1]) -- wait, U is indexed by [i*m+j] = [i*4+j]
  A[0,0] -> index 0*4+0 = 0
  A[1,1] -> index 1*4+1 = 5
  So U[0, c] = 1, U[5, c] = 1

V side: V is indexed by [j*p+k] = [j*7+k]
  B[0,k] -> index 0*7+k = k
  B[1,k] -> index 1*7+k = 7+k
  So V[k, c] = 1 and V[7+k, c] = 1 for all k=0..6

This is one rank-1 term. Let me enumerate all 14 terms (7 for each block).

Actually, let me just code this up directly.
"""
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We'll build rank-1 terms. Each term has U_row (scalar linear combo of A), 
    # V_row (vector linear combo of B), W_row (distribution to C).
    
    terms = []  # list of (U_dict, V_dict, W_dict)
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """u_coeffs: dict {index: coeff}, v_coeffs: dict {index: coeff}, w_coeffs: dict {index: coeff}"""
        terms.append((u_coeffs, v_coeffs, w_coeffs))
    
    # Block 1: A1 = A[:, 0:2], B1 = B[0:2, :]
    # A1 entries: A[0,0]@idx0, A[0,1]@idx1, A[1,0]@idx4, A[1,1]@idx5
    # B1 entries: B[j,k] @ idx (j*7+k) for j=0,1
    
    a00 = 0; a01 = 1; a10 = 4; a11 = 5  # U indices for A[0,0],A[0,1],A[1,0],A[1,1]
    
    # P1 = (a+d)(b0+c0, ...) -> U: A[0,0]+A[1,1], V: B[0,k]+B[1,k], W: +1 to C[0,k] and C[1,k]
    u = {a00: 1, a11: 1}
    v = {}
    for k in range(p):
        v[k] = 1       # B[0,k]
        v[7+k] = 1     # B[1,k]
    w = {}
    for k in range(p):
        w[k*n+0] = 1   # C[0,k]
        w[k*n+1] = 1   # C[1,k]
    add_term(u, v, w)
    
    # P2 = (c+d)*e -> U: A[1,0]+A[1,1], V: B[0,k], W: +1 to C[1,k]
    u = {a10: 1, a11: 1}
    v = {}
    for k in range(p):
        v[k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = 1
    add_term(u, v, w)
    
    # P3 = a*(f-h) -> U: A[0,0], V: B[1,k]-B[0,k], W: -1 to C[1,k]
    u = {a00: 1}
    v = {}
    for k in range(p):
        v[k] = -1
        v[7+k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = -1
    add_term(u, v, w)
    
    # P4 = d*(g-e) -> U: A[1,1], V: B[1,k]-B[0,k], W: +1 to C[0,k]
    u = {a11: 1}
    v = {}
    for k in range(p):
        v[k] = -1
        v[7+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = 1
    add_term(u, v, w)
    
    # P5 = (a+c)*h -> U: A[0,0]+A[1,0], V: B[1,k], W: -1 to C[0,k]
    u = {a00: 1, a10: 1}
    v = {}
    for k in range(p):
        v[7+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = -1
    add_term(u, v, w)
    
    # P6 = (b-d)*(g+h) -> U: A[0,1]-A[1,1], V: B[1,k]+B[0,k], W: -1 to C[1,k]
    u = {a01: 1, a11: -1}
    v = {}
    for k in range(p):
        v[k] = 1
        v[7+k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = -1
    add_term(u, v, w)
    
    # P7 = (c-a)*(e+f) -> U: A[1,0]-A[0,0], V: B[0,k]+B[1,k], W: +1 to C[0,k]
    u = {a10: 1, a00: -1}
    v = {}
    for k in range(p):
        v[k] = 1
        v[7+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = 1
    add_term(u, v, w)
    
    # Block 2: A2 = A[:, 2:4], B2 = B[2:4, :]
    # A2 entries: A[0,2]@idx2, A[0,3]@idx3, A[1,2]@idx6, A[1,3]@idx7
    # B2 entries: B[j,k] @ idx (j*7+k) for j=2,3
    
    b00 = 2; b01 = 3; b10 = 6; b11 = 7  # U indices for A[0,2],A[0,3],A[1,2],A[1,3]
    
    # P1 = (a+d)(b0+c0, ...)
    u = {b00: 1, b11: 1}
    v = {}
    for k in range(p):
        v[14+k] = 1   # B[2,k]
        v[21+k] = 1   # B[3,k]
    w = {}
    for k in range(p):
        w[k*n+0] = 1
        w[k*n+1] = 1
    add_term(u, v, w)
    
    # P2 = (c+d)*e
    u = {b10: 1, b11: 1}
    v = {}
    for k in range(p):
        v[14+k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = 1
    add_term(u, v, w)
    
    # P3 = a*(f-h)
    u = {b00: 1}
    v = {}
    for k in range(p):
        v[14+k] = -1
        v[21+k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = -1
    add_term(u, v, w)
    
    # P4 = d*(g-e)
    u = {b11: 1}
    v = {}
    for k in range(p):
        v[14+k] = -1
        v[21+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = 1
    add_term(u, v, w)
    
    # P5 = (a+c)*h
    u = {b00: 1, b10: 1}
    v = {}
    for k in range(p):
        v[21+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = -1
    add_term(u, v, w)
    
    # P6 = (b-d)*(g+h)
    u = {b01: 1, b11: -1}
    v = {}
    for k in range(p):
        v[14+k] = 1
        v[21+k] = 1
    w = {}
    for k in range(p):
        w[k*n+1] = -1
    add_term(u, v, w)
    
    # P7 = (c-a)*(e+f)
    u = {b10: 1, b00: -1}
    v = {}
    for k in range(p):
        v[14+k] = 1
        v[21+k] = 1
    w = {}
    for k in range(p):
        w[k*n+0] = 1
    add_term(u, v, w)
    
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, coeff in u_dict.items():
            U[idx][c] = coeff
        for idx, coeff in v_dict.items():
            V[idx][c] = coeff
        for idx, coeff in w_dict.items():
            W[idx][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
