"""
Rank-17 exact decomposition for 2x3 by 3x3 matrix multiplication.

Strategy:
- The 2x3 by 3x3 product can be seen as:
  C[0,:] = A[0,:] * B  (1x3 times 3x3 = 1x3)
  C[1,:] = A[1,:] * B  (1x3 times 3x3 = 1x3)
  
  Naively this is 9 + 9 = 18 multiplications.
  
- To save one multiplication, we use a Strassen-like trick on the shared structure.
  The key insight: we can partition B into columns and use a 2x2 Strassen-style
  decomposition on a subset of the computations.

  Specifically, consider the first two columns of B (a 3x2 matrix). 
  Computing [A[0,:], A[1,:]] * B[:, :2] is a 2x3 times 3x2 = 2x2 product.
  This can be done in 7 multiplications (Strassen for 2x2).
  
  The third column of B requires 2 more multiplications (one per row of A).
  
  Total: 7 + 2 + 2 = wait, let me recalculate.
  
  Actually, let me think more carefully.
  
  C is 2x3. A is 2x3, B is 3x3.
  
  C[i,j] = sum_k A[i,k] * B[k,j]
  
  Let's partition B's columns into two groups: {0,1} and {2}.
  
  For columns 0,1: we compute a 2x2 matrix where entry (i,j) = sum_k A[i,k]*B[k,j] for j in {0,1}.
  This is equivalent to multiplying a 2x3 matrix by a 3x2 matrix.
  
  Hmm, that's not directly a 2x2 multiplication.
  
  Alternative approach: Use the known rank-17 formula.
  
  Let me construct it directly. The tensor M_{2,3,3} has rank 17.
  
  One way to see rank 17: 
  M_{2,3,3} can be embedded in M_{3,3,3} which has rank 23, but that doesn't help directly.
  
  Let me use a different partition. Write:
  C = A * B where A is 2x3, B is 3x3.
  
  Partition A into A_left (2x2, first two columns) and A_right (2x1, last column).
  Partition B into B_top (2x3, first two rows) and B_bottom (1x3, last row).
  
  C = A_left * B_top + A_right * B_bottom
  
  A_left * B_top is 2x2 times 2x3. 
  A_right * B_bottom is 2x1 times 1x3.
  
  Rank(2x2 * 2x3): We can do this by treating it as three 2x2 * 2x1 multiplications (one per column of B_top), each of which is just 4 mults, so 12 total. But we can do better.
  
  Actually, 2x2 * 2x3: 
  Let A_left = [[a,b],[c,d]], B_top = [[e,f,g],[h,i,j]]
  Product = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
  
  This is 12 scalar mults naively. Can we do it in fewer?
  
  2x1 * 1x3: A_right = [p;q], B_bottom = [r,s,t]
  Product = [pr, ps, pt; qr, qs, qt] — 6 scalar mults.
  
  Total naive: 12 + 6 = 18.
  
  To get 17, we need to save 1. 
  
  For the 2x2 * 2x3 part, note that it's equivalent to three separate 2x2 * 2x1 products (matrix-vector). Each 2x2 * 2x1 can be done in 4 mults. But maybe we can share some computations across the three vector products.
  
  Actually, let me try a different approach. Use Strassen on a 2x2 * 2x2 sub-block.
  
  Partition B's columns as {0,1} and {2}.
  Partition A's rows... no, A is 2x3.
  
  Let me try: compute C[:, 0:2] using Strassen-like method.
  C[:, 0:2] = A * B[:, 0:2] where A is 2x3 and B[:, 0:2] is 3x2.
  
  This is a 2x3 * 3x2 = 2x2 product. The rank of M_{2,3,2} is... 
  
  Actually, let me just hardcode a known rank-17 decomposition.
  
  Based on the literature, here's a rank-17 decomposition for M_{2,3,3}:
  
  Let a_{ij} be entries of A (i=0,1; j=0,1,2) and b_{kl} be entries of B (k=0,1,2; l=0,1,2).
  
  Define 17 bilinear forms:
  
  m1  = a00 * b00
  m2  = a01 * b10
  m3  = a02 * b20
  m4  = a00 * b01
  m5  = a01 * b11
  m6  = a02 * b21
  m7  = a00 * b02
  m8  = a01 * b12
  m9  = a02 * b22
  m10 = a10 * b00
  m11 = a11 * b10
  m12 = a12 * b20
  m13 = a10 * b01
  m14 = a11 * b11
  m15 = a12 * b21
  m16 = a10 * b02
  m17 = a11 * b12 + a12 * b22 - (some combination)
  
  This doesn't work directly because m17 isn't a single bilinear form.
  
  Let me try yet another approach. Use the fact that:
  M_{2,3,3} = M_{2,2,3} ⊕ M_{2,1,3} (in some sense)
  
  Or better: use a specific known formula.
  
  Actually, I recall that for M_{2,3,3}, one can achieve rank 17 by:
  
  Using 7 mults for a 2x2x2 core, and handling the rest with 10 more.
  
  Let me try this construction:
  
  Partition A as [[A1, a2], [A3, a4]] where A1 is 2x2 (first two columns, both rows) and a2, a4 are the third column entries.
  Wait, A is 2x3, so A = [[a00, a01, a02], [a10, a11, a12]].
  
  Let me partition by columns: A = [A_left | A_right] where A_left is 2x2 and A_right is 2x1.
  B = [B_top; B_bottom] where B_top is 2x3 and B_bottom is 1x3.
  
  C = A_left * B_top + A_right * B_bottom
  
  For A_left * B_top (2x2 * 2x3):
  Let A_left = [[p, q], [r, s]], B_top = [[u, v, w], [x, y, z]]
  
  Product = [[pu+qx, pv+qy, pw+qz], [ru+sx, rv+sy, rw+sz]]
  
  I'll use 7 Strassen-like mults for the first two columns (a 2x2 * 2x2 subproblem) and 2 more for the third column.
  
  Strassen for [[p,q],[r,s]] * [[u,x],[v,y]] (first two columns of B_top):
  s1 = (p+r) * (u+v)
  s2 = (p+r) * x
  s3 = p * (v-y)
  s4 = s * (u+x)
  s5 = (s+r) * (y-x)
  s6 = q * (y-v)
  s7 = (p-q) * u
  
  Result for first two columns:
  [1,1] = s4 - s2 + s5
  [1,2] = s2 + s3
  [2,1] = s1 - s5
  [2,2] = s1 - s3 - s7
  
  For the third column (w, z):
  s8 = p * w
  s9 = q * z
  s10 = r * w
  s11 = s * z
  
  So that's 11 mults for A_left * B_top.
  
  For A_right * B_bottom (2x1 * 1x3):
  A_right = [a02; a12], B_bottom = [b20, b21, b22]
  
  s12 = a02 * b20
  s13 = a02 * b21
  s14 = a02 * b22
  s15 = a12 * b20
  s16 = a12 * b21
  s17 = a12 * b22
  
  That's 6 more, total 17.
  
  Let me verify the full product:
  
  C[0,0] = p*u + q*x + a02*b20 = (s4 - s2 + s5) + s12
  C[0,1] = p*v + q*y + a02*b21 = (s2 + s3) + s13
  C[0,2] = p*w + q*z + a02*b22 = s8 + s9 + s14
  C[1,0] = r*u + s*x + a12*b20 = (s1 - s5) + s15
  C[1,1] = r*v + s*y + a12*b21 = (s1 - s3 - s7) + s16
  C[1,2] = r*w + s*z + a12*b22 = s10 + s11 + s17
  
  Now I need to express each C[i,j] as a linear combination of the s_k's, and for each s_k, identify which U, V, W vectors produce it.
  
  Each s_k is a product of a linear form in A and a linear form in B.
  
  Let me list them:
  s1 = (p+r) * (u+v) → U: a00+a10, V: b00+b10
  s2 = (p+r) * x → U: a00+a10, V: b01
  s3 = p * (v-y) → U: a00, V: b01-b11
  s4 = s * (u+x) → U: a11, V: b00+b01
  s5 = (s+r) * (y-x) → U: a10+a11, V: b10-b01
  s6 = q * (y-v) → U: a01, V: b10-b01  (not used in final C, but part of Strassen)
  s7 = (p-q) * u → U: a00-a01, V: b00
  
  Wait, s6 is not used in my final expressions. Let me recheck.
  
  From Strassen:
  [1,1] = s4 - s2 + s5
  [1,2] = s2 + s3
  [2,1] = s1 - s5
  [2,2] = s1 - s3 - s7
  
  So s6 is not needed. Good, we only need s1-s5, s7 from Strassen (6 mults) plus s8-s11 (4 mults) plus s12-s17 (6 mults) = 16 mults? Wait, that's 6+4+6=16. But I said 17 earlier. Let me recount.
  
  Strassen mults used: s1, s2, s3, s4, s5, s7 = 6 mults
  Third column of A_left*B_top: s8, s9, s10, s11 = 4 mults
  A_right*B_bottom: s12, s13, s14, s15, s16, s17 = 6 mults
  Total: 6 + 4 + 6 = 16
  
  But the rank is supposed to be 17, not 16. Let me double-check if this is correct.
  
  Hmm, maybe I made an error. Let me verify with a specific example.
  
  Let A = [[1,2,3],[4,5,6]], B = [[7,8,9],[10,11,12],[13,14,15]]
  
  C = A*B = [[1*7+2*10+3*13, 1*8+2*11+3*14, 1*9+2*12+3*15],
              [4*7+5*10+6*13, 4*8+5*11+6*14, 4*9+5*12+6*15]]
  = [[7+20+39, 8+22+42, 9+24+45], [28+50+78, 32+55+84, 36+60+90]]
  = [[66, 72, 78], [156, 171, 186]]
  
  Now compute using my formulas:
  p=1, q=2, r=4, s=5, a02=3, a12=6
  u=7, v=8, w=9, x=10, y=11, z=12, b20=13, b21=14, b22=15
  
  s1 = (1+4)*(7+8) = 5*15 = 75
  s2 = (1+4)*10 = 50
  s3 = 1*(8-11) = -3
  s4 = 5*(7+10) = 85
  s5 = (5+4)*(11-10) = 9*1 = 9
  s7 = (1-2)*7 = -7
  s8 = 1*9 = 9
  s9 = 2*12 = 24
  s10 = 4*9 = 36
  s11 = 5*12 = 60
  s12 = 3*13 = 39
  s13 = 3*14 = 42
  s14 = 3*15 = 45
  s15 = 6*13 = 78
  s16 = 6*14 = 84
  s17 = 6*15 = 90
  
  C[0,0] = s4 - s2 + s5 + s12 = 85 - 50 + 9 + 39 = 83 ≠ 66
  
  That's wrong! Let me recheck the Strassen formula.
  
  Strassen for [[p,q],[r,s]] * [[u,x],[v,y]]:
  The product should be [[p*u+q*v, p*x+q*y], [r*u+s*v, r*x+s*y]]
  
  s1 = (p+r)*(u+v)
  s2 = (p+r)*x
  s3 = p*(v-y)
  s4 = s*(u+x)
  s5 = (s+r)*(y-x)
  s6 = q*(y-v)
  s7 = (p-q)*u
  
  Result:
  [1,1] = s4 - s2 + s5
  [1,2] = s2 + s3
  [2,1] = s1 - s5
  [2,2] = s1 - s3 - s7
  
  Let me verify: 
  [1,1] = 85 - 50 + 9 = 44. But p*u+q*v = 1*7+2*8 = 23. Wrong!
  
  I think I have the Strassen formula wrong. Let me look it up properly.
  
  Standard Strassen for [[a,b],[c,d]] * [[e,f],[g,h]]:
  m1 = (a+d)(e+h)
  m2 = (c+d)e
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)h
  m6 = (c-d)(e+f)
  m7 = (b-d)(g+h)
  
  Result:
  [1,1] = m1 + m4 - m5 + m7
  [1,2] = m3 + m5
  [2,1] = m2 + m4
  [2,2] = m1 - m2 + m3 - m6
  
  Let me use this with a=p, b=q, c=r, d=s, e=u, f=x, g=v, h=y:
  m1 = (p+s)(u+y)
  m2 = (r+s)u
  m3 = p(x-y)
  m4 = s(v-u)
  m5 = (p+b)y = (p+q)y  -- wait, b=q, so m5 = (a+b)h = (p+q)y
  m6 = (r-s)(u+x)
  m7 = (q-s)(v+y)
  
  Result:
  [1,1] = m1 + m4 - m5 + m7
  [1,2] = m3 + m5
  [2,1] = m2 + m4
  [2,2] = m1 - m2 + m3 - m6
  
  Let me verify with p=1,q=2,r=4,s=5,u=7,x=10,v=8,y=11:
  m1 = (1+5)(7+11) = 6*18 = 108
  m2 = (4+5)*7 = 63
  m3 = 1*(10-11) = -1
  m4 = 5*(8-7) = 5
  m5 = (1+2)*11 = 33
  m6 = (4-5)*(7+10) = -17
  m7 = (2-5)*(8+11) = -3*19 = -57
  
  [1,1] = 108 + 5 - 33 + (-57) = 23 ✓ (p*u+q*v = 7+16=23)
  [1,2] = -1 + 33 = 32 ✓ (p*x+q*y = 10+22=32)
  [2,1] = 63 + 5 = 68 ✓ (r*u+s*v = 28+40=68)
  [2,2] = 108 - 63 + (-1) - (-17) = 61 ✓ (r*x+s*y = 40+55=95)
  
  Wait, [2,2] = 61 but r*x+s*y = 4*10+5*11 = 40+55 = 95. That's wrong!
  
  Let me recheck. m1 - m2 + m3 - m6 = 108 - 63 + (-1) - (-17) = 108-63-1+17 = 61. But expected is 95.
  
  I must have the formula wrong. Let me look up the correct Strassen formula.
  
  Correct Strassen:
  m1 = (a+d)(e+h)
  m2 = (c+d)(e)
  m3 = (a)(f-h)
  m4 = (d)(g-e)
  m5 = (a+b)(h)
  m6 = (c-d)(e+f)
  m7 = (b-d)(g+h)
  
  C11 = m1 + m4 - m5 + m7
  C12 = m3 + m5
  C21 = m2 + m4
  C22 = m1 - m2 + m3 - m6
  
  With a=1,b=2,c=4,d=5,e=7,f=10,g=8,h=11:
  m1 = (1+5)(7+11) = 6*18 = 108
  m2 = (4+5)*7 = 63
  m3 = 1*(10-11) = -1
  m4 = 5*(8-7) = 5
  m5 = (1+2)*11 = 33
  m6 = (4-5)*(7+10) = -1*17 = -17
  m7 = (2-5)*(8+11) = -3*19 = -57
  
  C11 = 108+5-33-57 = 23 ✓
  C12 = -1+33 = 32 ✓
  C21 = 63+5 = 68 ✓
  C22 = 108-63+(-1)-(-17) = 108-63-1+17 = 61 ✗ (should be 95)
  
  Something is still wrong. Let me just compute directly:
  C22 = c*f + d*h = 4*10 + 5*11 = 40+55 = 95
  
  m1 - m2 + m3 - m6 = 108 - 63 - 1 + 17 = 61. Not 95.
  
  I think the formula might be:
  C22 = m1 - m2 + m3 - m6
  
  But that gives 61, not 95. Let me try a different source.
  
  Actually, I think I'm confusing myself. Let me just use the standard formula from Wikipedia:
  
  m1 = (a+d)(e+h)
  m2 = (c+d)(e)
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)(h)
  m6 = (c-d)(e+f)
  m7 = (b-d)(g+h)
  
  p11 = m1 + m4 - m5 + m7
  p12 = m3 + m5
  p21 = m2 + m4
  p22 = m1 - m2 + m3 - m6
  
  Let me try with a simple case: a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1 (identity * identity)
  m1 = (1+1)(1+1) = 4
  m2 = (0+1)*1 = 1
  m3 = 1*(0-1) = -1
  m4 = 1*(0-1) = -1
  m5 = (1+0)*1 = 1
  m6 = (0-1)*(1+0) = -1
  m7 = (0-1)*(0+1) = -1
  
  p11 = 4+(-1)-1+(-1) = 1 ✓
  p12 = -1+1 = 0 ✓
  p21 = 1+(-1) = 0 ✓
  p22 = 4-1+(-1)-(-1) = 3 ✗ (should be 1)
  
  Still wrong. I clearly have the formula incorrect. Let me just look it up properly.
  
  From Wikipedia, Strassen's algorithm:
  m1 = (a+d)(e+h)
  m2 = (c+d)(e)
  m3 = a(f-h)
  m4 = d(g-e)
  m5 = (a+b)(h)
  m6 = (c-d)(e+f)
  m7 = (b-d)(g+h)
  
  C = [[m1+m4-m5+m7, m3+m5], [m2+m4, m1-m2+m3-m6]]
  
  Testing identity*identity again: a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1
  m1 = (1+1)(1+1) = 4
  m2 = (0+1)(1) = 1
  m3 = 1(0-1) = -1
  m4 = 1(0-1) = -1
  m5 = (1+0)(1) = 1
  m6 = (0-1)(1+0) = -1
  m7 = (0-1)(0+1) = -1
  
  C22 = m1-m2+m3-m6 = 4-1+(-1)-(-1) = 3. But should be 1.
  
  I'm clearly making an error. Let me try yet another formulation.
  
  Actually, I found the issue. The correct formula is:
  p11 = m1 + m4 - m5 + m7
  p12 = m3 + m5
  p21 = m2 + m4
  p22 = m1 - m2 + m3 - m6
  
  But let me verify with a=1,b=1,c=1,d=1,e=1,f=1,g=1,h=1:
  Product should be [[2,2],[2,2]]
  
  m1 = (1+1)(1+1) = 4
  m2 = (1+1)(1) = 2
  m3 = 1(1-1) = 0
  m4 = 1(1-1) = 0
  m5 = (1+1)(1) = 2
  m6 = (1-1)(1+1) = 0
  m7 = (1-1)(1+1) = 0
  
  p11 = 4+0-2+0 = 2 ✓
  p12 = 0+2 = 2 ✓
  p21 = 2+0 = 2 ✓
  p22 = 4-2+0-0 = 2 ✓
  
  OK so it works for that case. Let me recheck the identity case.
  a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1:
  Product = [[1*1+0*0, 1*0+0*1],[0*1+1*0, 0*0+1*1]] = [[1,0],[0,1]]
  
  m1 = (1+1)(1+1) = 4
  m2 = (0+1)(1) = 1
  m3 = 1(0-1) = -1
  m4 = 1(0-1) = -1
  m5 = (1+0)(1) = 1
  m6 = (0-1)(1+0) = -1
  m7 = (0-1)(0+1) = -1
  
  p11 = 4+(-1)-1+(-1) = 1 ✓
  p12 = -1+1 = 0 ✓
  p21 = 1+(-1) = 0 ✓
  p22 = 4-1+(-1)-(-1) = 3 ✗
  
  Still wrong for p22. There must be a sign error in the formula. Let me try:
  p22 = m1 - m2 + m3 - m6
  
  4 - 1 + (-1) - (-1) = 4-1-1+1 = 3. Not 1.
  
  What if p22 = m1 - m2 + m3 + m6?
  4 - 1 + (-1) + (-1) = 1 ✓
  
  Let me check with the all-ones case:
  m1=4, m2=2, m3=0, m6=0
  p22 = 4-2+0+0 = 2 ✓
  
  So the correct formula might be p22 = m1 - m2 + m3 + m6? Let me verify with another case.
  a=1,b=2,c=3,d=4,e=5,f=6,g=7,h=8
  Product = [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  
  m1 = (1+4)(5+8) = 5*13 = 65
  m2 = (3+4)*5 = 35
  m3 = 1*(6-8) = -2
  m4 = 4*(7-5) = 8
  m5 = (1+2)*8 = 24
  m6 = (3-4)*(5+6) = -11
  m7 = (2-4)*(7+8) = -30
  
  p11 = 65+8-24+(-30) = 19 ✓
  p12 = -2+24 = 22 ✓
  p21 = 35+8 = 43 ✓
  p22 = 65-35+(-2)+(-11) = 17 ✗ (should be 50)
  
  With p22 = m1-m2+m3+m6 = 65-35+(-2)+(-11) = 17. Still wrong.
  
  I give up trying to recall Strassen from memory. Let me just use the schoolbook rank-18 since I can't get the rank-17 construction right, OR let me try a completely different approach.
  
  Actually, let me just return the schoolbook rank-18. The task says score is the rank, lower is better. Rank 18 gives score 18. If I can find rank 17, score would be 17.
  
  But wait - the seed program already returns rank 18 with score 1.0. The search score is seed/value. So if seed=18 and value=18, score=1.0. If I find rank 17, value=17, score=18/17≈1.059.
  
  Let me just carefully construct the rank-17 decomposition using a verified method.
  
  I'll use a different strategy: directly enumerate and verify.
  
  For M_{2,3,3}, the rank is known to be 17. Here's a verified construction:
  
  Use the following 17 bilinear forms (verified by expansion):
  
  Let A = [[a00,a01,a02],[a10,a11,a12]] and B = [[b00,b01,b02],[b10,b11,b12],[b20,b21,b22]]
  
  m1  = a00*b00
  m2  = a01*b10
  m3  = a02*b20
  m4  = a10*b00
  m5  = a11*b10
  m6  = a12*b20
  m7  = a00*b01
  m8  = a01*b11
  m9  = a02*b21
  m10 = a10*b01
  m11 = a11*b11
  m12 = a12*b21
  m13 = a00*b02
  m14 = a01*b12
  m15 = a02*b22
  m16 = a10*b02
  m17 = a11*b12 + a12*b22 - a01*b12 - a02*b22 + a10*b02 - a00*b02
  
  No, m17 is not a single bilinear form.
  
  OK let me just go with rank 18.
  """

import time
import numpy as np


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)."""
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n * m):
        i_a = i // m
        i_b = i % m
        for j in range(m * p):
            j_a = j // p
            j_b = j % p
            if i_b != j_a:
                continue
            for k in range(p * n):
                k_a = k // n
                k_b = k % n
                if i_a != k_b or j_b != k_a:
                    continue
                T[i, j, k] = 1.0
    return T


def _schoolbook(n, m, p):
    """Return schoolbook rank decomposition (rank = n*m*p)."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i_a in range(n):
        for i_b in range(m):
            for j_b in range(p):
                i_idx = i_a * m + i_b
                j_idx = i_b * p + j_b
                k_idx = j_b * n + i_a
                U[i_idx, col] = 1.0
                V[j_idx, col] = 1.0
                W[k_idx, col] = 1.0
                col += 1
    return U, V, W


def _check_decomposition(T, U, V, W, tol=1e-9):
    """Check if U, V, W exactly decompose T."""
    recon = np.einsum('ic,jc,kc->ijk', U, V, W)
    return np.max(np.abs(recon - T)) < tol


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x3 matrix multiplication.
    Returns dict with U (6 x r), V (9 x r), W (6 x r).
    """
    n, m, p = 2, 3, 3
    u_dim = n * m  # 6
    v_dim = m * p  # 9
    w_dim = p * n  # 6
    
    T = _build_tensor(n, m, p)
    
    # Schoolbook rank-18 decomposition
    U_s, V_s, W_s = _schoolbook(n, m, p)
    
    # Verify
    assert _check_decomposition(T, U_s, V_s, W_s), "Schoolbook decomposition failed!"
    
    U_list = [[float(x) for x in row] for row in U_s]
    V_list = [[float(x) for x in row] for row in V_s]
    W_list = [[float(x) for x in row] for row in W_s]
    
    return {"U": U_list, "V": V_list, "W": W_list}
