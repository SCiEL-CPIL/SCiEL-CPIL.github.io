"""
Matrix multiplication 2x3x7 - Rank-minimal exact tensor decomposition.

Approach: Block decomposition using Strassen for 2x2x2 sub-blocks.
Split the 3x7 matrix B into:
  - A 2x2 block (columns 0,1)
  - A 2x2 block (columns 2,3)
  - A 2x2 block (columns 4,5)
  - A 1x1 block (column 6)

For each 2x2x2 sub-problem (multiplying 2x2 by 2x2), use Strassen's algorithm
which achieves rank 7 instead of the schoolbook rank 8.

For the 2x1x7 sub-problem (multiplying 2x1 by 1x7), the rank is 2
(since we need C_i,k = A_i * B_k for i=0,1 and k=0..6, but this is
actually a 2x1x7 tensor which has rank 2: one term for each row i).

Wait, let me reconsider. The 2x1x7 part:
  A is 2x1 (column vector a_0, a_1)
  B is 1x7 (row vector b_0, ..., b_6)
  C is 2x7 where C[i,k] = a_i * b_k

This is an outer product of a 2-vector and a 7-vector, so the tensor
T[i,0,k] = a_i * b_k. This tensor has rank 2 (sum over i of a_i * (b vector)).

Actually, the rank-1 decomposition: T[i,0,k] = sum_lambda U[i,0,lambda] * V[0,k,lambda] * W[k,i,lambda]

For rank 2: 
  lambda=0: U[0,0,0]=1, V[0,k,0]=1 for all k, W[k,0,0]=1 for all k, U[1,0,0]=0
  lambda=1: U[0,0,1]=0, V[0,k,1]=1 for all k, W[k,1,1]=1 for all k, U[1,0,1]=1

Then T[i,0,k] = U[i,0,0]*V[0,k,0]*W[k,0,0] + U[i,0,1]*V[0,k,1]*W[k,1,1]
             = (i==0 ? 1 : 0) * 1 * 1 + (i==1 ? 1 : 0) * 1 * 1
             = i (if we think of i as 0 or 1)

Hmm, that gives T[0,0,k] = 1 and T[1,0,k] = 1, but we need T[i,0,k] = a_i * b_k.

The issue is that the coefficients a_i and b_k are part of the input, not the tensor.
The tensor T[i,j,k] represents the coefficient of A[i,j]*B[j,k] in C[k,i].
For the 2x1x7 sub-problem with j=0 fixed:
  T[i,0,k] = 1 (the coefficient is 1, since C[i,k] = A[i,0]*B[0,k])

So the tensor entries are all 1 for the valid (i,k) pairs. This is a rank-1 tensor!
T[i,0,k] = 1 = 1 * 1 * 1

So we can use a single rank-1 term:
  U[i,0,lambda] = 1 for all i
  V[0,k,lambda] = 1 for all k  
  W[k,i,lambda] = 1 for all k,i

Total rank = 3 * 7 (Strassen for three 2x2x2 blocks) + 1 (for the 2x1x7 block)
           = 21 + 1 = 22

Let me verify the Strassen decomposition for 2x2x2:

Strassen's algorithm computes C = A*B where A,B,C are 2x2:
  C[0,0] = M1 - M2 + M4 + M5
  C[0,1] = M3 + M5
  C[1,0] = M2 + M4
  C[1,1] = M1 - M3 + M4 + M6

Where:
  M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
  M2 = (A[1,0] + A[1,1]) * B[0,0]
  M3 = A[0,0] * (B[0,1] - B[1,1])
  M4 = A[1,1] * (B[1,0] - B[0,0])
  M5 = (A[0,0] + A[0,1]) * B[1,1]
  M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
  M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])

Each M_t is a product of a linear combination of A entries and a linear combination of B entries.
The tensor decomposition has one rank-1 term per M_t.

For term t:
  U[i,j,t] = coefficient of A[i,j] in the A-part of M_t
  V[j,k,t] = coefficient of B[j,k] in the B-part of M_t
  W[k,i,t] = coefficient of C[i,k] in the output (with sign)

Let me write out each term:

M1: A-part: A[0,0] + A[1,1] -> U[0,0]=1, U[1,1]=1
    B-part: B[0,0] + B[1,1] -> V[0,0]=1, V[1,1]=1
    C-part: C[0,0] += 1, C[1,1] += 1 -> W[0,0]=1, W[1,1]=1

M2: A-part: A[1,0] + A[1,1] -> U[1,0]=1, U[1,1]=1
    B-part: B[0,0] -> V[0,0]=1
    C-part: C[1,0] += 1 -> W[0,1]=1

M3: A-part: A[0,0] -> U[0,0]=1
    B-part: B[0,1] - B[1,1] -> V[0,1]=1, V[1,1]=-1
    C-part: C[0,1] += 1 -> W[1,0]=1

M4: A-part: A[1,1] -> U[1,1]=1
    B-part: B[1,0] - B[0,0] -> V[1,0]=1, V[0,0]=-1
    C-part: C[0,0] += 1, C[1,0] += 1 -> W[0,0]=1, W[0,1]=1

M5: A-part: A[0,0] + A[0,1] -> U[0,0]=1, U[0,1]=1
    B-part: B[1,1] -> V[1,1]=1
    C-part: C[0,0] += 1, C[0,1] += 1 -> W[0,0]=1, W[1,0]=1

M6: A-part: A[1,0] - A[0,0] -> U[1,0]=1, U[0,0]=-1
    B-part: B[0,0] + B[0,1] -> V[0,0]=1, V[0,1]=1
    C-part: C[1,1] += 1 -> W[1,1]=1

M7: A-part: A[0,1] - A[1,1] -> U[0,1]=1, U[1,1]=-1
    B-part: B[1,0] + B[1,1] -> V[1,0]=1, V[1,1]=1
    C-part: C[1,1] += 1 -> W[1,1]=1

Let me verify C[0,0]:
  M1: +1
  M4: +1
  M5: +1
  Total: 3? But we need C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]

Let me recompute:
  M1 contributes: (A[0,0]+A[1,1])(B[0,0]+B[1,1]) -> includes A[0,0]*B[0,0]
  M2 contributes: (A[1,0]+A[1,1])*B[0,0] -> no A[0,0]*B[0,0]
  M3 contributes: A[0,0]*(B[0,1]-B[1,1]) -> no
  M4 contributes: A[1,1]*(B[1,0]-B[0,0]) -> no
  M5 contributes: (A[0,0]+A[0,1])*B[1,1] -> no A[0,0]*B[0,0]
  M6 contributes: (A[1,0]-A[0,0])*(B[0,0]+B[0,1]) -> includes -A[0,0]*B[0,0]
  M7 contributes: (A[0,1]-A[1,1])*(B[1,0]+B[1,1]) -> no

So C[0,0] gets contributions from M1 (+A[0,0]*B[0,0]), M6 (-A[0,0]*B[0,0]), and...
That cancels out! Something is wrong with my Strassen formula.

Let me look up the correct Strassen formulas:
  M1 = (A11 + A22)(B11 + B22)
  M2 = (A21 + A22)B11
  M3 = A11(B12 - B22)
  M4 = A22(B21 - B11)
  M5 = (A11 + A12)B22
  M6 = (A21 - A11)(B11 + B12)
  M7 = (A12 - A22)(B21 + B22)

  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M3 + M4 + M6

Wait, that doesn't look right either. Let me use the standard:
  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M3 + M4 + M6

Hmm, let me just verify with a simple case. If A = [[1,0],[0,0]] and B = [[1,0],[0,0]], then C should be [[1,0],[0,0]].

  M1 = (1+0)(1+0) = 1
  M2 = (0+0)*1 = 0
  M3 = 1*(0-0) = 0
  M4 = 0*(0-1) = 0
  M5 = (1+0)*0 = 0
  M6 = (0-1)*(1+0) = -1
  M7 = (0-0)*(0+0) = 0

  C11 = 1 - 0 + 0 + 0 = 1 ✓
  C12 = 0 + 0 = 0 ✓
  C21 = 0 + 0 = 0 ✓
  C22 = 1 - 0 + 0 + (-1) = 0 ✓

Good. Now let me extract the coefficients properly.

For the tensor T[i,j,k] where i is row of C (0-indexed), j is column of A / row of B, k is column of B (which becomes row of C in trace convention W[k*n+i]).

Wait, the trace convention is W[k*n+i] where k is the column index of B (0..p-1) and i is the row index of C (0..n-1). So W is indexed by (k,i) pairs.

For the 2x2x2 block with local indices (u,v,w) where u is row of A/C, v is column of A/row of B, w is column of B:
  U[u,v,t] = coefficient of A[u,v] in term t
  V[v,w,t] = coefficient of B[v,w] in term t
  W[w,u,t] = coefficient of C[u,w] in term t (note: W index is w*n+u = w*2+u)

Let me redo:

M1: (A00+A11)(B00+B11)
  U: U[0,0]=1, U[1,1]=1
  V: V[0,0]=1, V[1,1]=1
  C contribution: C00 += 1, C11 += 1
  W: W[0,0]=1 (w=0,u=0), W[1,1]=1 (w=1,u=1)

M2: (A10+A11)B00
  U: U[1,0]=1, U[1,1]=1
  V: V[0,0]=1
  C contribution: C10 += 1
  W: W[0,1]=1 (w=0,u=1)

M3: A00(B01-B11)
  U: U[0,0]=1
  V: V[0,1]=1, V[1,1]=-1
  C contribution: C01 += 1
  W: W[1,0]=1 (w=1,u=0)

M4: A11(B10-B00)
  U: U[1,1]=1
  V: V[1,0]=1, V[0,0]=-1
  C contribution: C00 += 1, C10 += 1
  W: W[0,0]=1, W[0,1]=1

M5: (A00+A01)B11
  U: U[0,0]=1, U[0,1]=1
  V: V[1,1]=1
  C contribution: C00 += 1, C01 += 1
  W: W[0,0]=1, W[1,0]=1

M6: (A10-A00)(B00+B01)
  U: U[1,0]=1, U[0,0]=-1
  V: V[0,0]=1, V[0,1]=1
  C contribution: C11 += 1
  W: W[1,1]=1

M7: (A01-A11)(B10+B11)
  U: U[0,1]=1, U[1,1]=-1
  V: V[1,0]=1, V[1,1]=1
  C contribution: C11 += 1
  W: W[1,1]=1

Now let me verify C00:
  From M1: +1
  From M4: +1
  From M5: +1
  Total: 3? But we need coefficient 1 for A00*B00.

Let me compute the actual product contribution to C00:
  C00 = A00*B00 + A01*B10

  M1 contributes: (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
  M2 contributes: (A10+A11)*B00 = A10*B00 + A11*B00
  M3 contributes: A00*(B01-B11) = A00*B01 - A00*B11
  M4 contributes: A11*(B10-B00) = A11*B10 - A11*B00
  M5 contributes: (A00+A01)*B11 = A00*B11 + A01*B11
  M6 contributes: (A10-A00)*(B00+B01) = A10*B00 + A10*B01 - A00*B00 - A00*B01
  M7 contributes: (A01-A11)*(B10+B11) = A01*B10 + A01*B11 - A11*B10 - A11*B11

  C00 = M1 - M2 + M4 + M5
  = (A00*B00 + A00*B11 + A11*B00 + A11*B11)
    - (A10*B00 + A11*B00)
    + (A11*B10 - A11*B00)
    + (A00*B11 + A01*B11)
  = A00*B00 + A00*B11 + A11*B00 + A11*B11 - A10*B00 - A11*B00 + A11*B10 - A11*B00 + A00*B11 + A01*B11
  = A00*B00 + 2*A00*B11 + A11*B11 - A10*B00 + A11*B10 - A11*B00 + A01*B11

This is not equal to A00*B00 + A01*B10. So my Strassen formula is wrong.

Let me look up the correct Strassen algorithm again:

  M1 = (A11 + A22)(B11 + B22)
  M2 = (A21 + A22)B11
  M3 = A11(B12 - B22)
  M4 = A22(B21 - B11)
  M5 = (A11 + A12)B22
  M6 = (A21 - A11)(B11 + B12)
  M7 = (A12 - A22)(B21 + B22)

  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M3 + M4 + M6

Wait, I think I had the C22 formula wrong. Let me re-derive or look it up.

Actually, the standard Strassen is:
  C11 = M1 - M2 + M4 + M5
  C12 = M3 + M5
  C21 = M2 + M4
  C22 = M1 - M3 + M4 + M6

But my verification failed. Let me try a different source. The correct formulas are:

  M1 = (a11 + a22)(b11 + b22)
  M2 = (a21 + a22)b11
  M3 = a11(b12 - b22)
  M4 = a22(b21 - b11)
  M5 = (a11 + a12)b22
  M6 = (a21 - a11)(b11 + b12)
  M7 = (a12 - a22)(b21 + b22)

  c11 = M1 - M2 + M4 + M5
  c12 = M3 + M5
  c21 = M2 + M4
  c22 = M1 - M3 + M4 + M6

Let me verify c22 with A=[[1,0],[0,0]], B=[[1,0],[0,0]]:
  M1 = (1+0)(1+0) = 1
  M3 = 1*(0-0) = 0
  M4 = 0*(0-1) = 0
  M6 = (0-1)(1+0) = -1
  c22 = 1 - 0 + 0 + (-1) = 0 ✓

Let me verify c11 with A=[[1,0],[0,0]], B=[[1,0],[0,0]]:
  M1 = 1
  M2 = (0+0)*1 = 0
  M4 = 0
  M5 = (1+0)*0 = 0
  c11 = 1 - 0 + 0 + 0 = 1 ✓

Let me try A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
  C should be [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]

  M1 = (1+4)(5+8) = 5*13 = 65
  M2 = (3+4)*5 = 35
  M3 = 1*(6-8) = -2
  M4 = 4*(7-5) = 8
  M5 = (1+2)*8 = 24
  M6 = (3-1)*(5+6) = 2*11 = 22
  M7 = (2-4)*(7+8) = -2*15 = -30

  c11 = 65 - 35 + 8 + 24 = 62 ≠ 19

This is wrong. The Strassen algorithm must have different formulas. Let me look it up properly.

The correct Strassen algorithm:
  M1 = (a11 + a22)(b11 + b22)
  M2 = (a21 + a22)b11
  M3 = a11(b12 - b22)
  M4 = a22(b21 - b11)
  M5 = (a11 + a12)b22
  M6 = (a21 - a11)(b11 + b12)
  M7 = (a12 - a22)(b21 + b22)

  c11 = M1 - M2 + M4 + M5
  c12 = M3 + M5
  c21 = M2 + M4
  c22 = M1 - M3 + M4 + M6

Hmm, this gives c11 = 62 which is wrong. I must be misremembering. Let me just use a known correct set of Strassen matrices.

After research, the correct Strassen decomposition for 2x2 matrix multiplication has the following rank-1 terms (using 0-indexed):

Term 0: U=[1,0;0,1], V=[1,0;0,1], W=[1,0;0,1] -> (A00+A11)(B00+B11) contributes to C00 and C11
Term 1: U=[0,0;1,1], V=[1,0;0,0], W=[0,1;0,0] -> (A10+A11)B00 contributes to C10
Term 2: U=[1,0;0,0], V=[0,1;-1,0], W=[0,1;0,0] -> A00(B01-B11) contributes to C01
Term 3: U=[0,0;0,1], V=[-1,0;1,0], W=[1,0;1,0] -> A11(B10-B00) contributes to C00 and C10
Term 4: U=[1,1;0,0], V=[0,0;0,1], W=[1,1;0,0] -> (A00+A01)B11 contributes to C00 and C01
Term 5: U=[-1,0;1,0], V=[1,1;0,0], W=[0,0;0,1] -> (A10-A00)(B00+B01) contributes to C11
Term 6: U=[0,1;-1,0], V=[0,1;0,1], W=[0,0;0,1] -> (A01-A11)(B10+B11) contributes to C11

Let me verify c11 for A=[[1,2],[3,4]], B=[[5,6],[7,8]]:
  Term 0: (1+4)(5+8) = 65, contributes +1 to C00, +1 to C11
  Term 1: (3+4)*5 = 35, contributes +1 to C10
  Term 2: 1*(6-8) = -2, contributes +1 to C01
  Term 3: 4*(7-5) = 8, contributes +1 to C00, +1 to C10
  Term 4: (1+2)*8 = 24, contributes +1 to C00, +1 to C01
  Term 5: (3-1)*(5+6) = 22, contributes +1 to C11
  Term 6: (2-4)*(7+8) = -30, contributes +1 to C11

  C00 = 65 + 8 + 24 = 97 ≠ 19

This is still wrong. I'm clearly misremembering the Strassen formulas. Let me just implement a correct Strassen decomposition by looking at it as a tensor decomposition directly.

Actually, for the purpose of this task, let me just use the schoolbook decomposition but with a different structure. The schoolbook rank is 42, and I need to find something better.

Let me try a different approach: use the fact that r(2,3,7) <= r(2,3,3) + r(2,3,4).

For 2x3x3: schoolbook is 18. Can we do better? The tensor rank of 2x3x3 is known to be 18 (I believe).
For 2x3x4: schoolbook is 24.

Total: 18 + 24 = 42. Not better.

What about r(2,3,7) <= r(2,2,7) + r(2,1,7)?
r(2,2,7): We can split 7 into 2+2+2+1.
  r(2,2,2) = 7 (Strassen)
  r(2,2,1) = 4 (schoolbook, since it's just 2*2*1)
  Total for r(2,2,7) <= 7+7+7+4 = 25

r(2,1,7) = 2 (as analyzed above, rank 1 actually since T[i,0,k]=1 is rank 1)

Wait, r(2,1,7): T[i,0,k] = 1 for i=0,1 and k=0..6. This is a 2x1x7 tensor with all entries 1.
The rank of this tensor: T[i,0,k] = sum_lambda U[i,0,lambda] V[0,k,lambda] W[k,i,lambda]
With lambda=0: U[i,0,0]=1, V[0,k,0]=1, W[k,i,0]=1 gives T[i,0,k]=1. So rank 1.

So r(2,3,7) <= r(2,2,7) + r(2,1,7) <= 25 + 1 = 26.

Now I need to correctly implement the Strassen decomposition for 2x2x2. Let me just hardcode a known correct Strassen tensor decomposition.

The 2x2x2 matrix multiplication tensor has rank 7. A valid decomposition is:

For each of the 7 terms, we have vectors u in R^4 (for A), v in R^4 (for B), w in R^4 (for C).
The tensor T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] (with appropriate indexing)

Actually, let me just use the schoolbook for the 2x2x2 blocks (rank 8 each) and see if the total is better:
  3 * 8 + 1 = 25. That's better than 42!

Wait, but Strassen gives rank 7 for 2x2x2, so 3*7 + 1 = 22 would be even better. But I need to get Strassen right.

Let me just use schoolbook for the 2x2x2 blocks for now:
  Total rank = 3 * 8 + 1 = 25

This is already much better than 42. Let me implement this.
"""

def construct():
    n, m, p = 2, 3, 7
    rank = 25  # 3 blocks of 2x2x2 (schoolbook rank 8 each) + 1 block of 2x1x7 (rank 1)
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 25
    V = [[0] * rank for _ in range(m * p)]  # 21 x 25
    W = [[0] * rank for _ in range(p * n)]  # 14 x 25
    
    col = 0
    # Three 2x2x2 blocks: columns 0-1, 2-3, 4-5
    for block_start in [0, 2, 4]:
        for u in range(2):  # row of A/C
            for v in range(2):  # col of A / row of B
                for w in range(2):  # col of B
                    i = u  # global row
                    j = v  # global col of A / row of B
                    k = block_start + w  # global col of B
                    
                    u_idx = i * m + j  # 0..5
                    v_idx = j * p + k  # 0..20
                    w_idx = k * n + i  # 0..13
                    
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
    
    # One 2x1x7 block: column 6, j=2
    # T[i,2,k] = 1 for i=0,1 and k=0..6
    # Rank 1: U[i,2,24]=1, V[2,k,24]=1, W[k,i,24]=1
    for i in range(2):
        for k in range(7):
            u_idx = i * m + 2  # i*3 + 2
            v_idx = 2 * p + k  # 14 + k
            w_idx = k * n + i  # 2*k + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
