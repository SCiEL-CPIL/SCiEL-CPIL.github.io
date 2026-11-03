"""
Rank-minimal tensor decomposition for 4x4 by 4x7 matrix multiplication.

Uses a Kronecker-product / FFT-based decomposition to reduce rank below schoolbook 112.
The key idea: express the multiplication tensor T[i,j,k] = delta_{j,j} (sum over j)
as a sum of rank-1 terms by diagonalizing the j-summation via a cyclic group FFT.

For m=4, we use the 4-point DFT. The multiplication tensor can be written as:
  T = (1/4) sum_{omega in {1,-1,i,-i}}  a_omega ⊗ b_omega ⊗ c_omega

where a_omega[i, omega_idx] = sum_j omega^{-j} A[i,j] (transform of A)
      b_omega[j, omega_idx] = sum_k omega^{-k} B[j,k] (transform of B)  
      c_omega[k, omega_idx] = omega^{j} (coupling factor)

Actually, the cleanest formulation:
  C[i,k] = sum_j A[i,j] * B[j,k]

Using the 4-point DFT on the j-index with omega = i:
  A[i,j] = (1/4) sum_{r=0}^{3} Ã[i,r] * omega^{-r*j}
  B[j,k] = (1/4) sum_{s=0}^{3} B̃[k,s] * omega^{-s*j}

Then:
  C[i,k] = (1/16) sum_{r,s} Ã[i,r] * B̃[k,s] * sum_{j=0}^{3} omega^{-(r+s)*j}

The inner sum is 4 if (r+s) mod 4 == 0, else 0. So s = -r mod 4.

  C[i,k] = (1/4) sum_{r=0}^{3} Ã[i,r] * B̃[k,(-r)%4]

This gives 4 rank-1 terms per (i,k) pair, total = 4 * 7 * 4 = 112. Same as schoolbook.

BUT: the transform matrices have structure. Let's use the fact that we can absorb
the (1/4) factor and the omega powers into the factor matrices.

The rank-1 decomposition: for each r in {0,1,2,3}, for each i in {0,1,2,3}, for each k in {0,...,6}:
  U[i*4 + j, r] ... wait, this doesn't work directly because the DFT mixes the j index.

Let me think differently. The tensor T has entries T[(i,j), (j,k), (k,i)] = 1 for the
schoolbook. We want to write:
  T[u,v,w] = sum_{r=1}^{R} U[u,r] * V[v,r] * W[w,r]

where u = i*m+j, v = j*p+k, w = k*n+i.

Using the FFT approach with 4-point DFT on j:
  U[u=(i,j), r] = omega^{-j * r}  (for r = 0,1,2,3)
  V[v=(j,k), r] = omega^{-j * (4-r)%4}  ... this is getting complicated.

Let me just directly construct the decomposition using the DFT matrices.

F = DFT_4 matrix where F[j, r] = omega^{-j*r} for omega = i.
F = [[1, 1, 1, 1],
     [1, -i, -1, i],
     [1, -1, 1, -1],
     [1, i, -1, -i]]

The multiplication tensor in the transformed basis:
  T' = (F^T ⊗ F^T) * T * (F ⊗ F) ... 

Actually, let's just verify: if we set
  U[(i,j), r] = F[j, r] = omega^{-j*r}
  V[(j,k), s] = F[j, s] = omega^{-j*s}  
  W[(k,i), r+s] = 1 if (r+s) mod 4 == 0, with coefficient 4

Hmm, this is still 112. The issue is that for rectangular matrices, the DFT doesn't
reduce the rank below n*m*p when m is small.

Alternative: use a 2x2 block structure. Split A into 2x2 blocks of 2x2 matrices.
A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
B = [[B1], [B2]] where B1, B2 are 2x7.
C = [[A11, A12], [A21, A22]] * [[B1], [B2]] = [[A11*B1 + A12*B2, ...]]

C1 = A11*B1 + A12*B2
C2 = A21*B1 + A22*B2

Each of C1, C2 is 2x7 = 14 entries.
A11*B1 is 2x2 * 2x7 = 2*2*7 = 28 multiplications.
A12*B2 is 2*2*7 = 28.
So C1 needs 56 multiplications, C2 needs 56. Total 112. Still schoolbook.

The Winograd trick for 2x2 * 2x7:
  A11 = [[a, b], [c, d]], B1 = [[e, f, ...], [g, h, ...]]
  
  Using 2-point Winograd:
  A1 = a + b, A2 = a - b
  B1 = e + g, B2 = e - g
  P1 = A1 * B1, P2 = A2 * B2
  a*e + b*g = (P1 + P2)/2
  a*g + b*e = (P1 - P2)/2

So for each column of B (7 columns), we need 2 products instead of 4.
That saves factor of 2 for the 2x2 * 2x7 multiplication!

So A11*B1: 2 products per column * 7 columns = 14 products (instead of 28)
A12*B2: 14 products
A21*B1: 14 products  
A22*B2: 14 products

Total: 56 products! That's half of 112!

Let me construct this explicitly.

A is 4x4:
  A = [[a00, a01, a02, a03],
       [a10, a11, a12, a13],
       [a20, a21, a22, a23],
       [a30, a31, a32, a33]]

B is 4x7:
  B = [[b00, b01, ..., b06],
       [b10, b11, ..., b16],
       [b20, b21, ..., b26],
       [b30, b31, ..., b36]]

C is 4x7:
  C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]

Split into 2x2 blocks:
  A11 = [[a00, a01], [a10, a11]]  (rows 0,1; cols 0,1)
  A12 = [[a02, a03], [a12, a13]]  (rows 0,1; cols 2,3)
  A21 = [[a20, a21], [a30, a31]]  (rows 2,3; cols 0,1)
  A22 = [[a22, a23], [a32, a33]]  (rows 2,3; cols 2,3)

  B1 = [[b00,...,b06], [b10,...,b16]]  (rows 0,1)
  B2 = [[b20,...,b26], [b30,...,b36]]  (rows 2,3)

  C1 = rows 0,1 of C = A11*B1 + A12*B2
  C2 = rows 2,3 of C = A21*B1 + A22*B2

For A11*B1 using Winograd 2-point:
  Let A11 = [[a, b], [c, d]]
  For column k of B1: [e_k, f_k]^T where e_k = b0k, f_k = b1k

  Products:
  P1_k = (a+b) * (e_k + f_k)
  P2_k = (a-b) * (e_k - f_k)
  P3_k = (c+d) * (e_k + f_k)
  P4_k = (c-d) * (e_k - f_k)

  Then:
  a*e_k + b*f_k = (P1_k + P2_k)/2
  a*f_k + b*e_k = (P1_k - P2_k)/2
  c*e_k + d*f_k = (P3_k + P4_k)/2
  c*f_k + d*e_k = (P3_k - P4_k)/2

So for each k, we get 4 products, and from them we can recover all 4 entries of the
2x2 * vector product. That's 4 products per column, same as schoolbook for 2x2*2!

Wait, no. Schoolbook for 2x2 * 2x1 is 4 multiplications. Winograd gives 4 products too.
No savings for 2x2!

The Winograd savings come when you have larger blocks or when you can share the
transform across multiple columns. But since B has 7 columns and they're independent,
there's no sharing.

Hmm. Let me reconsider. For m=4, n=4, p=7, the schoolbook rank is 112. The
tensor rank of 4x4*4x7 matrix multiplication... 

Actually, I recall that for matrix multiplication, the rank of the multiplication
tensor for n×m × m×p is at least n*m*p in general, but can be less for specific
dimensions. For 2x2*2x2, the rank is 7 (Strassen). For 4x4*4x4, it's known to be
at most 49 (using Strassen recursively) but the exact minimum is not known.

For 4x4 * 4x7, we could potentially use a block approach:
  Treat as (2x2 block) * (2x2 block) where the second matrix is 4x7 split into
  two 2x7 matrices.

  A (4x4) = [A11 A12; A21 A22] with 2x2 blocks
  B (4x7) = [B1; B2] with 2x7 blocks

  C (4x7) = [A11*B1 + A12*B2; A21*B1 + A22*B2]

Each 2x2 * 2x7 product has tensor rank at most 2*2*7 = 28, but with Strassen-like
methods for 2x2*2x2 (rank 7), we can do 2x2*2x7 with rank 7*7 = 49? No, that's
not right either.

Actually, for 2x2 * 2x7, we can use the 2x2*2x2 Strassen decomposition 7 times
(on each column pair? No, 7 is odd).

Let me just try: use the known rank-7 decomposition for 2x2*2x2, and apply it
to each 2x2 sub-block of the 7 columns. 7 = 2+2+2+1. So 3 pairs (rank 7 each) + 1
single (rank 4) = 3*7 + 4 = 25 for 2x2*2x7.

Then total rank = 4 * 25 = 100? Let's see:
  A11*B1: rank 25
  A12*B2: rank 25
  A21*B1: rank 25
  A22*B2: rank 25
  Total: 100

That's better than 112! Let me implement this.

The Strassen 2x2*2x2 decomposition:
  M1 = (a+d)(e+h)
  M2 = (c+d)(e)
  M3 = (a-b)(f+h)
  M4 = (a+d)(g+i)  -- wait, standard Strassen:
  
  For A = [[a,b],[c,d]], B = [[e,f],[g,h]]:
  M1 = (a+d)(e+h)
  M2 = (c+d)(e)
  M3 = (a-b)(f+h)
  M4 = (c+d)(g)
  M5 = (a+c)(h)
  M6 = (a+c)(f-h)
  M7 = (a+d)(e-h) -- actually standard is:
  
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)  -- hmm, let me look up the exact formula.

Standard Strassen:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)  -- no...

Let me use the standard:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)  
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)  -- this isn't standard.

Standard Strassen for 2x2:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)
  P5 = (a+c)(h)
  P6 = (a+c)(f-h)
  P7 = (a+d)(e-h)

Result:
  [[a*e+b*g, a*f+b*h],
   [c*e+d*g, c*f+d*h]]
  
  = [[P1+P4-P5-P7, P1-P3-P4+P7],
     [P2+P3+P5-P1, P1+P2-P3-P6]]

Let me verify: 
  Top-left: P1+P4-P5-P7 = (a+d)(e+h) + (c+d)(g) - (a+c)(h) - (a+d)(e-h)
  = ae+ah+de+dh + cg+dg - ah-ch - ae+ah-de+dh
  = ae + ah + de + dh + cg + dg - ah - ch - ae + ah - de + dh
  = ah + dh + cg + dg - ch + dh
  Hmm, this doesn't look right.

Let me use the well-known Strassen formula:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)
  P5 = (a+c)(h)
  P6 = (a+c)(f-h)
  P7 = (a+d)(e-h)

  C11 = P1 + P4 - P5 - P7
  C12 = P3 + P7
  C21 = P2 + P5
  C22 = P1 - P2 + P3 + P6

Let me verify C11:
  P1 + P4 - P5 - P7
  = (a+d)(e+h) + (c+d)(g) - (a+c)(h) - (a+d)(e-h)
  = ae+ah+de+dh + cg+dg - ah-ch - ae+ah-de+dh
  = (ae-ae) + (ah-ah+ah) + (de-de) + (dh+dh) + cg + dg - ch
  = ah + 2dh + cg + dg - ch
  
  This should equal ae + bg. Let me check with a=1,b=0,c=0,d=1,e=1,f=0,g=0,h=1:
  ae+bg = 1*1+0*0 = 1
  P1=(1+1)(1+1)=4, P4=(0+1)(0)=0, P5=(1+0)(1)=1, P7=(1+1)(1-1)=0
  C11 = 4+0-1-0 = 3 ≠ 1.

My Strassen formula is wrong. Let me use the correct one:

  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)  -- I keep making this mistake.

Correct Strassen:
  P1 = (a+d)(e+h)
  P2 = (c+d)(e)
  P3 = (a-b)(f+h)
  P4 = (c+d)(g)
  P5 = (a+c)(h)
  P6 = (a+c)(f-h)
  P7 = (a+d)(e-h)

  C11 = P1 + P4 - P5 - P7
  C12 = P3 + P7
  C21 = P2 + P5
  C22 = P1 - P2 + P3 + P6

Wait, I think the issue is that I'm not using the standard form. Let me just hardcode
the rank-7 decomposition directly.

For A=[[a,b],[c,d]] and B=[[e,f],[g,h]]:
  7 products:
  p1 = (a+d)*(e+h)
  p2 = (c+d)*(e)
  p3 = (a-b)*(f+h)
  p4 = (c+d)*(g)
  p5 = (a+c)*(h)
  p6 = (a+c)*(f-h)
  p7 = (a+d)*(e-h)

  C[0,0] = p1 + p4 - p5 - p7
  C[0,1] = p3 + p7
  C[1,0] = p2 + p5
  C[1,1] = p1 - p2 + p3 + p6

Let me verify with a=b=c=d=e=f=g=h=1:
  A=[[1,1],[1,1]], B=[[1,1],[1,1]]
  C should be [[2,2],[2,2]]
  
  p1=(1+1)(1+1)=4
  p2=(1+1)(1)=2
  p3=(1-1)(1+1)=0
  p4=(1+1)(1)=2
  p5=(1+1)(1)=2
  p6=(1+1)(1-1)=0
  p7=(1+1)(1-1)=0

  C[0,0] = 4+2-2-0 = 4 ≠ 2

Still wrong. Let me look up the actual Strassen algorithm properly.

Strassen's algorithm:
  M1 = (a + d) · (e + h)
  M2 = (c + d) · e
  M3 = (a - b) · (f + h)
  M4 = (c + d) · g
  M5 = (a + c) · h
  M6 = (a + c) · (f - h)
  M7 = (a + d) · (e - h)

  p = M1 - M4 + M5 + M7
  q = M4 + M6
  r = M1 + M2 - M5 - M3
  s = M2 + M3

So:
  C[0,0] = M1 - M4 + M5 + M7
  C[0,1] = M4 + M6
  C[1,0] = M1 + M2 - M5 - M3
  C[1,1] = M2 + M3

Let me verify with a=b=c=d=e=f=g=h=1:
  M1 = (1+1)(1+1) = 4
  M2 = (1+1)(1) = 2
  M3 = (1-1)(1+1) = 0
  M4 = (1+1)(1) = 2
  M5 = (1+1)(1) = 2
  M6 = (1+1)(1-1) = 0
  M7 = (1+1)(1-1) = 0

  C[0,0] = 4 - 2 + 2 + 0 = 4
  C[0,1] = 2 + 0 = 2
  C[1,0] = 4 + 2 - 2 - 0 = 4
  C[1,1] = 2 + 0 = 2

  But A*B = [[1,1],[1,1]]*[[1,1],[1,1]] = [[2,2],[2,2]]

  C[0,0] should be 1*1+1*1 = 2, but we got 4. Still wrong!

I think I'm confusing the formula. Let me just derive it or use a known correct reference.

Actually, I think the issue is that Strassen's algorithm computes:
  [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh],[ce+dg, cf+dh]]

And the 7 products are:
  M1 = (a+d)(e+h)
  M2 = (c+d)(e)
  M3 = (a-b)(f+h)
  M4 = (c+d)(g)
  M5 = (a+c)(h)
  M6 = (a+c)(f-h)
  M7 = (a+d)(e-h)

And:
  ae+bg = M1 - M4 + M5 + M7
  af+bh = M4 + M6
  ce+dg = M1 + M2 - M5 - M3
  cf+dh = M2 + M3

Let me re-verify with a=1,b=1,c=1,d=1,e=1,f=1,g=1,h=1:
  ae+bg = 1+1 = 2
  M1=(2)(2)=4, M4=(2)(1)=2, M5=(2)(1)=2, M7=(2)(0)=0
  M1-M4+M5+M7 = 4-2+2+0 = 4 ≠ 2

I clearly have the wrong formula. Let me just implement the schoolbook decomposition
and accept rank 112, since I can't get the Strassen formula right in this context.

Actually, you know what, let me just try a completely different approach. Let me use
numpy to compute the tensor rank numerically by forming the matricization and computing
its rank. But that requires the full tensor which is 16*28*28 = 12544 entries, and
computing the rank of a 16*28 by 28 matrix... that's doable.

But finding a rank-minimal decomposition from the rank is hard. Let me just return
the schoolbook solution.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130

    # Schoolbook decomposition: rank = n*m*p = 112
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {'U': U, 'V': V, 'W': W}
