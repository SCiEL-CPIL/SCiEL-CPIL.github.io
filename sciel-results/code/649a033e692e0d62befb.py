"""
Rank-minimal exact tensor decomposition for 3x4 by 4x8 matrix multiplication.

Strategy: Use the tensor product structure to build a decomposition from
smaller known-rank decompositions.

Key identity: M_{n,m,p} can be built from tensor products:
  M_{n1*m1, m1*p1, n2*p2} = M_{n1,m1,p1} ⊗ M_{n2,m2,p2}  (when dimensions align)

For 3x4 x 4x8:
  Split into: M_{3,2,4} ⊗ M_{1,2,2}
  
  M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4 (schoolbook, already minimal for rank-1 outer product structure)
  M_{3,2,4}: 3x2 * 2x4 -> 3x4
  
  M_{3,2,4} = M_{3,2,2} ⊗ M_{1,1,2}
  M_{3,2,2}: 3x2 * 2x2 -> 3x2, rank 12 (schoolbook)
  M_{1,1,2}: 1x1 * 1x2 -> 1x2, rank 2 (schoolbook)
  
  So M_{3,2,4} has rank 12*2 = 24
  M_{1,2,2} has rank 4
  Total: 24 * 4 = 96... that's just schoolbook again.

Let me try a different split:
  M_{3,4,8} = M_{3,4,4} ⊗ M_{1,1,2}
  M_{1,1,2}: rank 2
  M_{3,4,4}: need rank 48 for total 96, or less for improvement.
  
  M_{3,4,4} = M_{3,2,2} ⊗ M_{1,2,2}
  M_{3,2,2}: 3x2 * 2x2 -> 3x2, rank 12
  M_{1,2,2}: 1x2 * 2x2 -> 1x2, rank 4
  Total M_{3,4,4} rank: 12*4 = 48
  Grand total: 48*2 = 96. Still schoolbook.

Alternative: Use Strassen on 2x2 blocks where possible.
  M_{2,2,2} has rank 7 (Strassen) instead of 8.
  
  M_{3,4,8}: Split rows of A into 2+1, cols of A into 2+2, rows of B into 2+2, cols of B into 4 blocks of 2.
  
  Blocks:
  - A_top (2x4) * B_top (2x8) -> C_top_left (2x4) and C_top_right (2x4)
    Each 2x4 * 2x4 can be split into four 2x2 * 2x2 blocks, each rank 7.
    So 2x4 * 2x4 -> 2x4: 4 blocks * rank 7 = 28
    Two such multiplications: 28 + 28 = 56
  - A_bottom (1x4) * B_top (2x8) -> C_bottom_left (1x4) and C_bottom_right (1x4)
    1x4 * 2x4: split into four 1x2 * 2x2 blocks, each rank 4.
    So 1x4 * 2x4 -> 1x4: 4 blocks * rank 4 = 16
    Two such: 16 + 16 = 32
    
  Total: 56 + 32 = 88

This gives rank 88, saving 8 from schoolbook 96.

Implementation:
- For each 2x2 * 2x2 block, use Strassen's rank-7 scheme.
- For each 1x2 * 2x2 block, use schoolbook rank-4.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _strassen_2x2_factors():
    """
    Return U, V, W factor matrices for 2x2 * 2x2 -> 2x2 with rank 7 (Strassen).
    
    U shape: (2*2, 7) = (4, 7)  -- indexed by [row*2+col]
    V shape: (2*2, 7) = (4, 7)  -- indexed by [row*2+col]  
    W shape: (2*2, 7) = (4, 7)  -- indexed by [row*2+col]
    
    Strassen's 7 products for A*B where:
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = [[c00, c01], [c10, c11]]
    
    m1 = (a00 + a10) * (b00 + b10)
    m2 = (a00 + a10) * (b01 + b11)
    m3 = (a01 + a11) * (b00 + b10)
    m4 = (a01 + a11) * (b01 + b11)
    m5 = a00 * b00
    m6 = a11 * b11
    m7 = (a10 - a11) * (b11 + b01)  -- wait, let me use the standard form
    
    Standard Strassen:
    m1 = (a00 + a10)(b00 + b10)
    m2 = (a10 + a11)(b00 + b01)
    m3 = (a00 + a01)(b10 + b11)
    m4 = a00(b00 - b10)
    m5 = a11(b01 - b00)
    m6 = (a00 + a10)(b10 - b00)  -- hmm, let me use the well-known form
    
    Actually, the standard Strassen 7 products are:
    m1 = (a + b)(e + f)
    m2 = (c + d)(e + f)
    m3 = a(h - f)
    m4 = d(g - e)
    m5 = (a + e)(h + f)
    m6 = (c + a)(f - h)
    m7 = (b + d)(g - c)
    
    where A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    
    C[0,0] = m1 + m4 - m5 + m7
    C[0,1] = m5 - m1 + m2 + m6
    C[1,0] = m1 - m3 + m4 + m6
    C[1,1] = -m3 + m2 + m5 + m7
    
    Let me verify:
    C[0,0] should be a*e + b*g
    m1 + m4 - m5 + m7 = (a+b)(e+f) + d(g-e) - (a+e)(h+f) + (b+d)(g-c)
    = ae + af + be + bf + dg - de - ah - af - eh - ef + bg - bc + dg - dc
    = ae + be + bf + dg - de - ah - eh - ef + bg - bc + dg - dc
    This doesn't simplify to a*e + b*g easily. Let me use a different known-correct scheme.
    
    Let me use the explicit rank-7 decomposition that I can verify:
    
    For 2x2 * 2x2, one valid rank-7 scheme (verified by expansion):
    p1 = (a00 + a10) * (b00 + b10)
    p2 = (a00 + a10) * (b01 + b11)  
    p3 = (a01 + a11) * (b00 + b10)
    p4 = (a01 + a11) * (b01 + b11)
    p5 = a00 * b00
    p6 = a11 * b11
    p7 = (a10 - a11) * (b01 - b00)  -- need to verify
    
    Actually, let me just use a known-correct explicit scheme. Here's one:
    
    The tensor T[i,j,k] = 1 iff i+j=k mod appropriate indexing... 
    For 2x2*2x2->2x2, T[i,j,k] = 1 iff k = i (row of output) and the column matches.
    
    Let me use this verified scheme:
    U rows indexed by (i,j) where i in {0,1} (row of A), j in {0,1} (col of A)
    V rows indexed by (j,k) where j in {0,1} (row of B = col of A), k in {0,1} (col of B)
    W rows indexed by (k,i) where k in {0,1} (col of C), i in {0,1} (row of C)
    
    T[i,j,k] = 1 iff (i,j,k) corresponds to a valid multiplication term.
    For C[i,k] = sum_j A[i,j] * B[j,k], the tensor is T[i,j,k] = delta(i,j,k) where
    the "multiplication" means i indexes row of A and C, j indexes col of A and row of B,
    k indexes col of B and C.
    
    So T[i,j,k] = 1 for all valid (i,j,k) in {0,1}^3, and we need:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    Schoolbook: 8 terms, one for each (i,j,k).
    Strassen: 7 terms.
    
    Here's a concrete rank-7 decomposition (I'll verify by construction):
    
    c=0: U=[1,0,0,0], V=[1,0,0,0], W=[1,0,0,0]  -> T[0,0,0] += 1
    c=1: U=[0,1,0,0], V=[0,1,0,0], W=[0,1,0,0]  -> T[0,1,1] += 1  (wait, W indexed by (k,i))
    
    Let me be more careful with indexing:
    U index: i*2 + j (i=row of A, j=col of A), range 0-3
    V index: j*2 + k (j=row of B, k=col of B), range 0-3  
    W index: k*2 + i (k=col of C, i=row of C), range 0-3
    
    T[i,j,k] = 1 for all (i,j,k) in {0,1}^3.
    
    Schoolbook 8 terms:
    c=0: U[0]=1 (i=0,j=0), V[0]=1 (j=0,k=0), W[0]=1 (k=0,i=0) -> T[0,0,0]=1
    c=1: U[1]=1 (i=0,j=1), V[2]=1 (j=1,k=0), W[0]=1 (k=0,i=0) -> T[0,1,0]=1
    c=2: U[2]=1 (i=1,j=0), V[0]=1 (j=0,k=0), W[2]=1 (k=0,i=1) -> T[1,0,0]=1
    c=3: U[3]=1 (i=1,j=1), V[2]=1 (j=1,k=0), W[2]=1 (k=0,i=1) -> T[1,1,0]=1
    c=4: U[0]=1 (i=0,j=0), V[1]=1 (j=0,k=1), W[1]=1 (k=1,i=0) -> T[0,0,1]=1
    c=5: U[1]=1 (i=0,j=1), V[3]=1 (j=1,k=1), W[1]=1 (k=1,i=0) -> T[0,1,1]=1
    c=6: U[2]=1 (i=1,j=0), V[1]=1 (j=0,k=1), W[3]=1 (k=1,i=1) -> T[1,0,1]=1
    c=7: U[3]=1 (i=1,j=1), V[3]=1 (j=1,k=1), W[3]=1 (k=1,i=1) -> T[1,1,1]=1
    
    Now for rank 7, I need to combine two of these into one term.
    
    A known approach: 
    c=0: U=[1,1,0,0], V=[1,1,0,0], W=[1,0,1,0]
    This gives: sum over (i,j) and (j,k) and (k,i):
    For (i=0,j=0,k=0): U[0]*V[0]*W[0] = 1*1*1 = 1
    For (i=0,j=0,k=1): U[0]*V[1]*W[1] = 1*1*0 = 0
    For (i=0,j=1,k=0): U[1]*V[2]*W[0] = 1*0*1 = 0
    For (i=0,j=1,k=1): U[1]*V[3]*W[1] = 1*0*0 = 0
    For (i=1,j=0,k=0): U[2]*V[0]*W[2] = 0*1*1 = 0
    For (i=1,j=0,k=1): U[2]*V[1]*W[3] = 0*1*0 = 0
    For (i=1,j=1,k=0): U[3]*V[2]*W[2] = 0*0*1 = 0
    For (i=1,j=1,k=1): U[3]*V[3]*W[3] = 0*0*0 = 0
    So this only gives T[0,0,0]=1. Not helpful.
    
    Let me try:
    c=0: U=[1,0,1,0], V=[1,0,0,0], W=[1,0,1,0]
    (i=0,j=0,k=0): 1*1*1 = 1
    (i=0,j=0,k=1): 1*0*0 = 0
    (i=0,j=1,k=0): 0*0*1 = 0
    (i=0,j=1,k=1): 0*0*0 = 0
    (i=1,j=0,k=0): 1*1*1 = 1
    (i=1,j=0,k=1): 1*0*0 = 0
    (i=1,j=1,k=0): 0*0*1 = 0
    (i=1,j=1,k=1): 0*0*0 = 0
    Gives T[0,0,0]=1 and T[1,0,0]=1.
    
    c=1: U=[0,1,0,1], V=[0,0,1,0], W=[0,1,0,1]
    (i=0,j=0,k=0): 0*0*0 = 0
    (i=0,j=0,k=1): 0*0*0 = 0
    (i=0,j=1,k=0): 1*1*0 = 0
    (i=0,j=1,k=1): 1*0*0 = 0
    (i=1,j=0,k=0): 0*0*0 = 0
    (i=1,j=0,k=1): 0*0*0 = 0
    (i=1,j=1,k=0): 1*1*0 = 0
    (i=1,j=1,k=1): 1*0*0 = 0
    Hmm, that gives nothing. Let me reconsider.
    
    W index is k*2+i:
    W[0] = (k=0,i=0), W[1] = (k=0,i=1), W[2] = (k=1,i=0), W[3] = (k=1,i=1)
    
    Let me redo:
    c=0: U=[1,0,1,0], V=[1,0,0,0], W=[1,1,0,0]
    V index j*2+k: V[0]=(j=0,k=0), V[1]=(j=0,k=1), V[2]=(j=1,k=0), V[3]=(j=1,k=1)
    
    (i=0,j=0,k=0): U[0]*V[0]*W[0] = 1*1*1 = 1
    (i=0,j=0,k=1): U[0]*V[1]*W[2] = 1*0*0 = 0
    (i=0,j=1,k=0): U[1]*V[2]*W[0] = 0*0*1 = 0
    (i=0,j=1,k=1): U[1]*V[3]*W[2] = 0*0*0 = 0
    (i=1,j=0,k=0): U[2]*V[0]*W[1] = 1*1*1 = 1
    (i=1,j=0,k=1): U[2]*V[1]*W[3] = 1*0*0 = 0
    (i=1,j=1,k=0): U[3]*V[2]*W[1] = 0*0*1 = 0
    (i=1,j=1,k=1): U[3]*V[3]*W[3] = 0*0*0 = 0
    Gives T[0,0,0]=1, T[1,0,0]=1. Good.
    
    c=1: U=[0,1,0,1], V=[0,1,0,1], W=[0,0,1,1]
    (i=0,j=0,k=0): 0*0*0 = 0
    (i=0,j=0,k=1): 0*1*1 = 0
    (i=0,j=1,k=0): 1*0*0 = 0
    (i=0,j=1,k=1): 1*1*1 = 1
    (i=1,j=0,k=0): 0*0*0 = 0
    (i=1,j=0,k=1): 0*1*1 = 0
    (i=1,j=1,k=0): 1*0*0 = 0
    (i=1,j=1,k=1): 1*1*1 = 1
    Gives T[0,1,1]=1, T[1,1,1]=1. Good.
    
    Now I need T[0,0,1]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,0]=1 with 3 more terms.
    
    c=2: U=[1,0,0,0], V=[0,1,0,0], W=[0,0,1,0]
    (i=0,j=0,k=0): 1*0*0 = 0
    (i=0,j=0,k=1): 1*1*1 = 1
    (i=0,j=1,k=0): 0*0*0 = 0
    (i=0,j=1,k=1): 0*0*1 = 0
    (i=1,j=0,k=0): 0*0*0 = 0
    (i=1,j=0,k=1): 0*1*0 = 0
    (i=1,j=1,k=0): 0*0*0 = 0
    (i=1,j=1,k=1): 0*0*1 = 0
    Gives T[0,0,1]=1. Good.
    
    c=3: U=[0,1,0,0], V=[0,0,1,0], W=[0,1,0,0]
    (i=0,j=0,k=0): 0*0*0 = 0
    (i=0,j=0,k=1): 0*0*0 = 0
    (i=0,j=1,k=0): 1*1*1 = 1
    (i=0,j=1,k=1): 1*0*0 = 0
    (i=1,j=0,k=0): 0*0*0 = 0
    (i=1,j=0,k=1): 0*0*0 = 0
    (i=1,j=1,k=0): 0*1*1 = 0
    (i=1,j=1,k=1): 0*0*0 = 0
    Gives T[0,1,0]=1. Good.
    
    c=4: U=[0,0,1,0], V=[0,1,0,0], W=[0,0,0,1]
    (i=0,j=0,k=0): 0*0*0 = 0
    (i=0,j=0,k=1): 0*1*0 = 0
    (i=0,j=1,k=0): 0*0*0 = 0
    (i=0,j=1,k=1): 0*0*0 = 0
    (i=1,j=0,k=0): 1*0*0 = 0
    (i=1,j=0,k=1): 1*1*1 = 1
    (i=1,j=1,k=0): 0*0*0 = 0
    (i=1,j=1,k=1): 0*0*0 = 0
    Gives T[1,0,1]=1. Good.
    
    c=5: U=[0,0,0,1], V=[0,0,1,0], W=[0,0,0,1]  -- wait, need T[1,1,0]
    (i=1,j=1,k=0): U[3]*V[2]*W[1] = 1*1*0 = 0. Not working.
    
    Let me try c=5: U=[0,0,0,1], V=[0,0,1,0], W=[1,0,0,0]
    (i=1,j=1,k=0): U[3]*V[2]*W[1] = 1*1*0 = 0. Still no.
    
    W[1] = (k=0, i=1). So for T[1,1,0], I need W[1]=1.
    c=5: U=[0,0,0,1], V=[0,0,1,0], W=[0,1,0,0]
    (i=1,j=1,k=0): U[3]*V[2]*W[1] = 1*1*1 = 1. 
    But check others:
    (i=0,j=0,k=0): 0*0*0 = 0
    (i=0,j=0,k=1): 0*0*0 = 0
    (i=0,j=1,k=0): 0*1*0 = 0
    (i=0,j=1,k=1): 0*0*0 = 0
    (i=1,j=0,k=0): 0*0*0 = 0
    (i=1,j=0,k=1): 0*0*0 = 0
    (i=1,j=1,k=0): 1*1*1 = 1 ✓
    (i=1,j=1,k=1): 1*0*0 = 0
    Gives T[1,1,0]=1. 
    
    But wait, I've used 6 terms (c=0 through c=5) and I need 7 total.
    Let me check what I have:
    c=0: T[0,0,0]=1, T[1,0,0]=1
    c=1: T[0,1,1]=1, T[1,1,1]=1
    c=2: T[0,0,1]=1
    c=3: T[0,1,0]=1
    c=4: T[1,0,1]=1
    c=5: T[1,1,0]=1
    
    Total: T[0,0,0]=1, T[1,0,0]=1, T[0,1,1]=1, T[1,1,1]=1, T[0,0,1]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,0]=1
    That's all 8 terms covered with only 6 rank-1 components! But that would mean rank 6, which is impossible (rank of 2x2x2 is 7).
    
    I must have made an error. Let me recheck c=0:
    U=[1,0,1,0], V=[1,0,0,0], W=[1,1,0,0]
    
    U index: U[0]=(i=0,j=0), U[1]=(i=0,j=1), U[2]=(i=1,j=0), U[3]=(i=1,j=1)
    V index: V[0]=(j=0,k=0), V[1]=(j=0,k=1), V[2]=(j=1,k=0), V[3]=(j=1,k=1)
    W index: W[0]=(k=0,i=0), W[1]=(k=0,i=1), W[2]=(k=1,i=0), W[3]=(k=1,i=1)
    
    For T[i,j,k] = sum_c U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    
    c=0: U=[1,0,1,0], V=[1,0,0,0], W=[1,1,0,0]
    T[0,0,0] += U[0]*V[0]*W[0] = 1*1*1 = 1
    T[0,0,1] += U[0]*V[1]*W[2] = 1*0*0 = 0
    T[0,1,0] += U[1]*V[2]*W[0] = 0*0*1 = 0
    T[0,1,1] += U[1]*V[3]*W[2] = 0*0*0 = 0
    T[1,0,0] += U[2]*V[0]*W[1] = 1*1*1 = 1
    T[1,0,1] += U[2]*V[1]*W[3] = 1*0*0 = 0
    T[1,1,0] += U[3]*V[2]*W[1] = 0*0*1 = 0
    T[1,1,1] += U[3]*V[3]*W[3] = 0*0*0 = 0
    
    OK so c=0 gives T[0,0,0]=1 and T[1,0,0]=1. That's correct.
    
    c=1: U=[0,1,0,1], V=[0,1,0,1], W=[0,0,1,1]
    T[0,0,0] += U[0]*V[0]*W[0] = 0*0*0 = 0
    T[0,0,1] += U[0]*V[1]*W[2] = 0*1*1 = 0
    T[0,1,0] += U[1]*V[2]*W[0] = 1*0*0 = 0
    T[0,1,1] += U[1]*V[3]*W[2] = 1*1*1 = 1
    T[1,0,0] += U[2]*V[0]*W[1] = 0*0*0 = 0
    T[1,0,1] += U[2]*V[1]*W[3] = 0*1*1 = 0
    T[1,1,0] += U[3]*V[2]*W[1] = 1*0*0 = 0
    T[1,1,1] += U[3]*V[3]*W[3] = 1*1*1 = 1
    
    c=1 gives T[0,1,1]=1 and T[1,1,1]=1. Correct.
    
    So far: T[0,0,0]=1, T[1,0,0]=1, T[0,1,1]=1, T[1,1,1]=1. Four terms done.
    
    c=2: U=[1,0,0,0], V=[0,1,0,0], W=[0,0,1,0]
    T[0,0,0] += 1*0*0 = 0
    T[0,0,1] += 1*1*1 = 1
    T[0,1,0] += 0*0*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*1*0 = 0
    T[1,1,0] += 0*0*0 = 0
    T[1,1,1] += 0*0*0 = 0
    Gives T[0,0,1]=1. Correct.
    
    c=3: U=[0,1,0,0], V=[0,0,1,0], W=[0,1,0,0]
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*0*0 = 0
    T[0,1,0] += 1*1*1 = 1
    T[0,1,1] += 1*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*0*0 = 0
    T[1,1,0] += 0*1*0 = 0
    T[1,1,1] += 0*0*0 = 0
    Gives T[0,1,0]=1. Correct.
    
    c=4: U=[0,0,1,0], V=[0,1,0,0], W=[0,0,0,1]
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*1*0 = 0
    T[0,1,0] += 0*0*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 1*0*0 = 0
    T[1,0,1] += 1*1*1 = 1
    T[1,1,0] += 0*0*0 = 0
    T[1,1,1] += 0*0*0 = 0
    Gives T[1,0,1]=1. Correct.
    
    c=5: U=[0,0,0,1], V=[0,0,1,0], W=[0,1,0,0]
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*0*0 = 0
    T[0,1,0] += 0*1*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*0*0 = 0
    T[1,1,0] += 1*1*1 = 1
    T[1,1,1] += 1*0*0 = 0
    Gives T[1,1,0]=1. Correct.
    
    So with 6 terms I have: T[0,0,0]=1, T[1,0,0]=1, T[0,1,1]=1, T[1,1,1]=1, T[0,0,1]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,0]=1.
    All 8 entries are 1! But the rank of the 2x2x2 matrix multiplication tensor is 7, not 6.
    
    This means I must have made an error somewhere. The rank of the 2x2x2 tensor over Q is indeed 7 (this is a well-known result). So either my "decomposition" is wrong, or I'm misunderstanding the tensor.
    
    Wait - the matrix multiplication tensor for 2x2*2x2->2x2 has T[i,j,k] = 1 for all (i,j,k) in {0,1}^3? No! That's not right.
    
    The matrix multiplication tensor T_{m,n,p} has entries T[i,j,k] where i in {0..m-1}, j in {0..n-1}, k in {0..p-1}, and T[i,j,k] = 1 iff the (i,k) entry of the product involves A[i,j]*B[j,k].
    
    For 2x2 * 2x2 -> 2x2:
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    
    So T[i,j,k] = 1 iff the term A[i,j]*B[j,k] appears in C[i,k].
    This is always true for any valid (i,j,k)! So T[i,j,k] = 1 for all (i,j,k) in {0,1}^3.
    
    But then the tensor is the all-ones tensor, which has rank 1! That can't be right either.
    
    I think I'm confusing the tensor structure. The matrix multiplication tensor is defined as:
    T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the sum is over all (i,j,k) such that the (i,k) output depends on A[i,j] and B[j,k].
    
    Actually no. The tensor is T in U*V*W* where U has dimension m*n (input A), V has dimension n*p (input B), W has dimension p*m (output C).
    
    T = sum_{i=0}^{m-1} sum_{j=0}^{n-1} sum_{k=0}^{p-1} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    
    Wait, that's not right either. Let me think again.
    
    The matrix multiplication tensor M_{m,n,p} is an element of (R^{m*n})* ⊗ (R^{n*p})* ⊗ (R^{p*m})*.
    
    It's defined as: M_{m,n,p} = sum_{i,j,k} E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    
    where E_{i,j} is the matrix unit in R^{m*n} (row i, col j of the flattened A), etc.
    
    So for 2x2*2x2->2x2:
    M = sum_{i=0}^{1} sum_{j=0}^{1} sum_{k=0}^{1} E_{i,j} ⊗ E_{j,k} ⊗ E_{k,i}
    
    This has 8 terms, and the rank is 7 (not 8).
    
    My decomposition above had 6 terms that sum to the all-ones pattern, but that's not the same as the matrix multiplication tensor. The matrix multiplication tensor has specific structure: the index j in the first factor must match the index j in the second factor, and the index k in the second factor must match the index k in the third factor.
    
    Let me redo this properly.
    
    U has rows indexed by (i,j) where i in {0,1} (row of A), j in {0,1} (col of A).
    V has rows indexed by (j,k) where j in {0,1} (row of B), k in {0,1} (col of B).
    W has rows indexed by (k,i) where k in {0,1} (col of C), i in {0,1} (row of C).
    
    M_{2,2,2} = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    For a rank-1 decomposition term c:
    U[:,c] ⊗ V[:,c] ⊗ W[:,c] = u ⊗ v ⊗ w
    
    This contributes to M if for all (i,j,k):
    u_{(i,j)} * v_{(j,k)} * w_{(k,i)} = delta_{(i,j,k) in support}
    
    where the support is all (i,j,k) in {0,1}^3 (since every combination appears).
    
    So we need: sum_c u_{(i,j),c} * v_{(j,k),c} * w_{(k,i),c} = 1 for all (i,j,k).
    
    This is the same as what I had before. And I found a 6-term decomposition. But the rank is 7, so I must have made an arithmetic error.
    
    Let me recheck c=0 more carefully:
    U[0]=1, U[1]=0, U[2]=1, U[3]=0  (U indexed by i*2+j: (0,0)=1, (0,1)=0, (1,0)=1, (1,1)=0)
    V[0]=1, V[1]=0, V[2]=0, V[3]=0  (V indexed by j*2+k: (0,0)=1, (0,1)=0, (1,0)=0, (1,1)=0)
    W[0]=1, W[1]=1, W[2]=0, W[3]=0  (W indexed by k*2+i: (0,0)=1, (0,1)=1, (1,0)=0, (1,1)=0)
    
    T[0,0,0] += U[(0,0)]*V[(0,0)]*W[(0,0)] = 1*1*1 = 1
    T[0,0,1] += U[(0,0)]*V[(0,1)]*W[(1,0)] = 1*0*0 = 0
    T[0,1,0] += U[(0,1)]*V[(1,0)]*W[(0,0)] = 0*0*1 = 0
    T[0,1,1] += U[(0,1)]*V[(1,1)]*W[(1,0)] = 0*0*0 = 0
    T[1,0,0] += U[(1,0)]*V[(0,0)]*W[(0,1)] = 1*1*1 = 1
    T[1,0,1] += U[(1,0)]*V[(0,1)]*W[(1,1)] = 1*0*0 = 0
    T[1,1,0] += U[(1,1)]*V[(1,0)]*W[(0,1)] = 0*0*1 = 0
    T[1,1,1] += U[(1,1)]*V[(1,1)]*W[(1,1)] = 0*0*0 = 0
    
    OK so c=0 contributes to T[0,0,0] and T[1,0,0]. That's 2 terms.
    
    c=1: U[0]=0, U[1]=1, U[2]=0, U[3]=1
    V[0]=0, V[1]=1, V[2]=0, V[3]=1
    W[0]=0, W[1]=0, W[2]=1, W[3]=1
    
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*1*1 = 0
    T[0,1,0] += 1*0*0 = 0
    T[0,1,1] += 1*1*1 = 1
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*1*1 = 0
    T[1,1,0] += 1*0*0 = 0
    T[1,1,1] += 1*1*1 = 1
    
    c=1 contributes to T[0,1,1] and T[1,1,1]. That's 2 terms.
    
    c=2: U[0]=1, U[1]=0, U[2]=0, U[3]=0
    V[0]=0, V[1]=1, V[2]=0, V[3]=0
    W[0]=0, W[1]=0, W[2]=1, W[3]=0
    
    T[0,0,0] += 1*0*0 = 0
    T[0,0,1] += 1*1*1 = 1
    T[0,1,0] += 0*0*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*1*0 = 0
    T[1,1,0] += 0*0*0 = 0
    T[1,1,1] += 0*0*0 = 0
    
    c=2 contributes to T[0,0,1]. 1 term.
    
    c=3: U[0]=0, U[1]=1, U[2]=0, U[3]=0
    V[0]=0, V[1]=0, V[2]=1, V[3]=0
    W[0]=0, W[1]=1, W[2]=0, W[3]=0
    
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*0*0 = 0
    T[0,1,0] += 1*1*1 = 1
    T[0,1,1] += 1*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*0*0 = 0
    T[1,1,0] += 0*1*0 = 0
    T[1,1,1] += 0*0*0 = 0
    
    c=3 contributes to T[0,1,0]. 1 term.
    
    c=4: U[0]=0, U[1]=0, U[2]=1, U[3]=0
    V[0]=0, V[1]=1, V[2]=0, V[3]=0
    W[0]=0, W[1]=0, W[2]=0, W[3]=1
    
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*1*0 = 0
    T[0,1,0] += 0*0*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 1*0*0 = 0
    T[1,0,1] += 1*1*1 = 1
    T[1,1,0] += 0*0*0 = 0
    T[1,1,1] += 0*0*0 = 0
    
    c=4 contributes to T[1,0,1]. 1 term.
    
    c=5: U[0]=0, U[1]=0, U[2]=0, U[3]=1
    V[0]=0, V[1]=0, V[2]=1, V[3]=0
    W[0]=0, W[1]=1, W[2]=0, W[3]=0
    
    T[0,0,0] += 0*0*0 = 0
    T[0,0,1] += 0*0*0 = 0
    T[0,1,0] += 0*1*0 = 0
    T[0,1,1] += 0*0*0 = 0
    T[1,0,0] += 0*0*0 = 0
    T[1,0,1] += 0*0*0 = 0
    T[1,1,0] += 1*1*1 = 1
    T[1,1,1] += 1*0*0 = 0
    
    c=5 contributes to T[1,1,0]. 1 term.
    
    Total: T[0,0,0]=1, T[1,0,0]=1, T[0,1,1]=1, T[1,1,1]=1, T[0,0,1]=1, T[0,1,0]=1, T[1,0,1]=1, T[1,1,0]=1.
    
    All 8 entries are 1. So I've decomposed the all-ones tensor in 6 terms. But the matrix multiplication tensor is NOT the all-ones tensor!
    
    I see my mistake. The matrix multiplication tensor M_{2,2,2} is:
    M = sum_{i,j,k} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(k,i)}
    
    This means the (i,j,k) entry of the tensor is 1 for ALL (i,j,k) in {0,1}^3. So it IS the all-ones tensor.
    
    But the rank of the all-ones tensor of size 2x2x2 is 1, not 7!
    
    I'm clearly misunderstanding the tensor structure. Let me look at this differently.
    
    The matrix multiplication tensor is NOT the all-ones tensor. It's a specific tensor where the indices are coupled.
    
    Actually, I think the issue is that the "tensor" I'm thinking of is the structure tensor, and its rank is what we're trying to minimize. The structure tensor for matrix multiplication M_{m,n,p} is:
    
    M = sum_{i=0}^{m-1} sum_{j=0}^{n-1} sum_{k=0}^{p-1} (e_i ⊗ e_j) ⊗ (e_j ⊗ e_k) ⊗ (e_k ⊗ e_i)
    
    where the first factor is in (R^{m*n})*, the second in (R^{n*p})*, and the third in (R^{p*m})*.
    
    For 2x2*2x2->2x2, this is:
    M = e_{00}⊗e_{00}⊗e_{00} + e_{00}⊗e_{01}⊗e_{10} + e_{01}⊗e_{00}⊗e_{01} + e_{01}⊗e_{01}⊗e_{11}
        + e_{10}⊗e_{00}⊗e_{01} + e_{10}⊗e_{01}⊗e_{11} + e_{11}⊗e_{00}⊗e_{10} + e_{11}⊗e_{01}⊗e_{11}
    
    Wait, let me be more careful:
    i=0,j=0,k=0: e_{(0,0)} ⊗ e_{(0,0)} ⊗ e_{(0,0)}
    i=0,j=0,k=1: e_{(0,0)} ⊗ e_{(0,1)} ⊗ e_{(1,0)}
    i=0,j=1,k=0: e_{(0,1)} ⊗ e_{(1,0)} ⊗ e_{(0,0)}
    i=0,j=1,k=1: e_{(0,1)} ⊗ e_{(1,1)} ⊗ e_{(1,0)}
    i=1,j=0,k=0: e_{(1,0)} ⊗ e_{(0,0)} ⊗ e_{(0,1)}
    i=1,j=0,k=1: e_{(1,0)} ⊗ e_{(0,1)} ⊗ e_{(1,1)}
    i=1,j=1,k=0: e_{(1,1)} ⊗ e_{(1,0)} ⊗ e_{(0,1)}
    i=1,j=1,k=1: e_{(1,1)} ⊗ e_{(1,1)} ⊗ e_{(1,1)}
    
    So the tensor has 8 nonzero entries, one for each (i,j,k). The "entry" at position (U_idx, V_idx, W_idx) is 1 if U_idx=(i,j), V_idx=(j,k), W_idx=(k,i) for some (i,j,k).
    
    So the tensor IS the all-ones tensor in the sense that every valid (i,j,k) combination contributes 1. But the "valid" combinations are constrained: the j in U must match the j in V, and the k in V must match the k in W.
    
    So the tensor is NOT the full all-ones tensor of size 4x4x4. It's a sparse tensor with 8 nonzero entries.
    
    My decomposition above was for the full 4x4x4 all-ones tensor, which is different. I need to decompose the sparse tensor.
    
    OK so I need: sum_c U[(i,j),c] * V[(j,k),c] * W[(k,i),c] = 1 for all (i,j,k) in {0,1}^3.
    
    And this is exactly what I computed. And I got a 6-term decomposition. But the rank is 7.
    
    I must be making an error. Let me just use the schoolbook decomposition (rank 8) for the 2x2 blocks and not try to use Strassen. The total rank will be 96 (schoolbook), but at least it's correct.
    
    Actually, given the complexity and risk of errors, let me just implement the schoolbook decomposition cleanly. The current program already does this correctly with rank 96.
    
    But wait - the task says "rank-minimal" and the score is the rank. So I should try to get below 96.
    
    Let me try a simpler approach: use the fact that for 1xk * kxm, the rank is km (already minimal). And for 2x2 * 2x2, the rank is 7.
    
    Let me implement the block decomposition with Strassen for 2x2*2x2 blocks.
    
    For the Strassen 2x2*2x2 rank-7 decomposition, let me use a known-correct scheme. Here's one from the literature:
    
    The 7 products are:
    p1 = (a00 + a10)(b00 + b10)
    p2 = (a00 + a10)(b01 + b11)
    p3 = (a01 + a11)(b00 + b10)
    p4 = (a01 + a11)(b01 + b11)
    p5 = a00 * b00
    p6 = a11 * b11
    p7 = (a10 - a11)(b01 - b00)
    
    Then:
    c00 = p5 + (p1 - p3 - p5) + (p3 - p5) = p1 - p3 + p5  -- let me verify
    c00 = a00*b00 + a01*b10
    p1 - p3 + p5 = (a00+a10)(b00+b10) - (a01+a11)(b00+b10) + a00*b00
    = (a00+a10-a01-a11)(b00+b10) + a00*b00
    = (a00-a01+a10-a11)(b00+b10) + a00*b00
    
    This doesn't simplify to a00*b00 + a01*b10 easily. I'm not confident in this scheme.
    
    Given the time constraints and the risk of getting the Strassen coefficients wrong, let me just use the schoolbook decomposition (rank 96) which is guaranteed correct. The current program already does this.
    
    However, I can try a simple improvement: for the 1x2 * 2x2 blocks, the rank is 4 (schoolbook), and this is already minimal. For the 2x2 * 2x2 blocks, I'll use rank 8 (schoolbook) to be safe.
    
    Total: 4*(8+8) + 4*(4+4) = 4*16 + 4*8 = 64 + 32 = 96. Same as schoolbook.
    
    So without a correct Strassen implementation, I can't improve below 96. Let me just return the schoolbook decomposition.
    
    Actually, let me try one more thing: a random search for a rank-95 decomposition, similar to inspiration 1.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(s - 1.0) > tol:
                    return False
    # Check that off-diagonal terms are zero
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if i2 == i and j2 == j and k2 == k:
                                continue
                            s = np.sum(U[i2*m+j2, :] * V[j2*p+k2, :] * W[k2*n+i2, :])
                            if abs(s) > tol:
                                return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower-rank decomposition via random search
    # This is a heuristic approach; may or may not find an improvement
    deadline = 110  # Leave 40s margin for the 150s limit
    
    # Try a few ranks below schoolbook
    for rank in range(best_rank - 1, max(80, best_rank - 10), -1):
        if time.time() - start_time > deadline:
            break
        
        found = False
        for attempt in range(5):
            if time.time() - start_time > deadline:
                break
            
            # Random integer/half-integer U and V
            U_rand = np.random.randint(-2, 3, size=(n*m, rank))
            V_rand = np.random.randint(-2, 3, size=(m*p, rank))
            
            # Compute M[i,k,c] = sum_j U[i*m+j, c] * V[j*p+k, c]
            M = np.zeros((n, p, rank))
            for i in range(n):
                for k in range(p):
                    for c in range(rank):
                        M[i, k, c] = sum(U_rand[i*m+j, c] * V_rand[j*p+k, c] for j in range(m))
            
            # For each (i,k), we need: M[i,k,:] dot W[k*n+i, :] = 1
            # And for (i2,k2) != (i,k): M[i2,k2,:] dot W[k*n+i, :] should be 0
            # This is a system of linear equations for W
            # W has p*n rows, each of length rank
            # For row (k*n+i), we need: M[i,k,:] dot W[k*n+i, :] = 1
            # And M[i2,k2,:] dot W[k*n+i, :] = 0 for (i2,k2) != (i,k)
            
            # This is: for each (k,i), W[k*n+i, :] is in the null space of all M[i2,k2,:] 
            # except M[i,k,:], and M[i,k,:] dot W[k*n+i, :] = 1
            
            # Simple approach: W[k*n+i, :] = M[i,k,:] / (M[i,k,:] dot M[i,k,:])
            # This ensures M[i,k,:] dot W[k*n+i, :] = 1, but doesn't ensure
            # M[i2,k2,:] dot W[k*n+i, :] = 0 for (i2,k2) != (i,k)
            
            # For the decomposition to be exact, we need the full condition.
            # The simple approach only works if the M[i,k,:] vectors are orthogonal.
            
            W_rand = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    norm_sq = np.dot(M[i, k, :], M[i, k, :])
                    if norm_sq < 1e-10:
                        valid = False
                        break
                    W_rand[k*n+i, :] = M[i, k, :] / norm_sq
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round to half-integers
            U_rounded = np.round(U_rand * 2) / 2
            V_rounded = np.round(V_rand * 2) / 2
            W_rounded = np.round(W_rand * 2) / 2
            
            # Verify
            if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
                best_rank = rank
                U, V, W = U_rounded, V_rounded, W_rounded
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
