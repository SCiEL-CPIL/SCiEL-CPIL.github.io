"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s wall-clock

    # Known optimal rank for 4x4x4 matrix multiplication is 49 (Strassen-based).
    # We'll construct the rank-49 decomposition using the standard Strassen
    # 2x2x2 blocks composed appropriately for 4x4x4.
    
    # For 4x4x4, we can use a composition: split into 2x2 blocks of 2x2 matrices.
    # Each 2x2x2 multiplication uses Strassen rank 7.
    # The composition gives rank 7^3 = 343? No, that's not right.
    
    # Actually, for 4x4x4, the known optimal is 49. This comes from the 
    # "Strassen composition" where we view 4x4 as 2x2 blocks of 2x2.
    # The tensor rank of the 4x4x4 multiplication is 49.
    
    # Let me construct this explicitly. The standard approach:
    # 4x4 matrix multiplication can be seen as multiplying 2x2 blocks where
    # each block is 2x2. Using Strassen for the 2x2 block multiplication (rank 7)
    # and then composing...
    
    # Actually, the rank-49 decomposition is well-known. Let me construct it
    # using the standard formula.
    
    # The 4x4x4 matrix multiplication tensor has rank 49. One construction:
    # Split A (4x4) into 4x4 blocks... no.
    
    # Let me think differently. The standard rank-49 construction for 4x4x4:
    # Use the fact that M_4 = M_2 ⊗ M_2 in some sense, but with Strassen.
    
    # Actually, the simplest way: the rank of M_4 is 49, and it can be 
    # constructed from the rank-7 Strassen scheme for M_2.
    
    # For M_2 (2x2x2), Strassen gives rank 7 with:
    # U, V, W matrices of size 4x7 each.
    
    # For M_4, we can use a "composition" that gives rank 7^2 = 49.
    # The idea: view 4x4 matrices as 2x2 blocks of 2x2 matrices.
    # Then (AB)_{ij} = sum_k A_{ik} B_{kj} where each entry is a 2x2 product.
    # Using Strassen for each 2x2 block product... but that's not quite right
    # because the block structure is more complex.
    
    # Let me use a direct construction. The rank-49 decomposition of M_4:
    # 
    # I'll use the known explicit construction. The key insight is that
    # M_4 can be decomposed using 49 terms where each term is a rank-1
    # tensor.
    
    # Standard construction using Kronecker products:
    # If M_2 has a rank-7 decomposition with factors U2, V2, W2 (each 4x7),
    # then M_4 = M_2 ⊗ M_2 has rank at most 7*7 = 49 with factors:
    # U4 = U2 ⊗ U2 (size 16x49), V4 = V2 ⊗ V2 (size 16x49), W4 = W2 ⊗ W2 (size 16x49)
    
    # Wait, that's not quite right either. Let me be more careful.
    
    # The matrix multiplication tensor M_n has entries M_n[i,j,k] = delta[i,k]*delta[j,?]...
    # Actually M_n[i,j,k] where i indexes rows of A, j indexes cols of A / rows of B,
    # k indexes cols of B. The product C = AB has C[i,k] = sum_j A[i,j] B[j,k].
    
    # For the Kronecker product structure: if we view 4x4 matrices as 2x2 blocks
    # of 2x2 matrices, then:
    # (AB)_{(i1,i2),(j1,j2)} = sum_{k1,k2} A_{(i1,i2),(k1,k2)} B_{(k1,k2),(j1,j2)}
    
    # This is exactly the tensor product M_2 ⊗ M_2 in the appropriate sense.
    
    # So if M_2 = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r (rank-7 decomposition),
    # then M_4 = sum_{r,s=1}^{7} (u_r ⊗ u_s) ⊗ (v_r ⊗ v_s) ⊗ (w_r ⊗ w_s)
    
    # This gives rank 49.
    
    # Now I need the explicit Strassen rank-7 decomposition of M_2.
    # M_2[i,j,k] = 1 if i=k (for the A factor) and j is the middle index...
    
    # Let me define the convention clearly:
    # U has shape (n*m, rank) = (4, rank) for M_2
    # V has shape (m*p, rank) = (4, rank) for M_2  
    # W has shape (p*n, rank) = (4, rank) for M_2
    # The decomposition says: for all i,j,k: sum_r U[i,r] * V[j,r] * W[k,r] = delta[i,k] * (1 if j is the "middle" index)
    
    # Wait, I need to be careful about the indexing. The task says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # And W[k*n+i, r] is the output factor.
    
    # So for M_n: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # The identity is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta[i,k]... 
    # No wait, that's not right either. Let me re-read.
    
    # "exactly decompose the matrix multiplication tensor"
    # The matrix multiplication tensor T[i,j,k] for C=AB is:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
    # Actually T[i,j,k] = delta[i,k]... no.
    
    # The tensor is: T[i, j, k] where i is row of A (and C), j is col of A / row of B,
    # k is col of B (and C). T[i,j,k] = 1 always? No...
    
    # Actually the matrix multiplication tensor is defined as:
    # T[i, j, k] = delta[i, k]... that doesn't make sense.
    
    # Let me think again. The bilinear map is: (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T has entries T[i, j, k] such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 if the indices match properly, i.e., T[i,j,k] = delta[i,k]... 
    # No, T[i,j,k] = 1 for all valid i,j,k in the sum. Actually:
    # C[i,k] = sum_j A[i,j] B[j,k], so the tensor entry T[i,j,k] = 1 for all i,j,k.
    # Wait no, that would mean every entry is 1, which is a rank-1 tensor.
    
    # I'm confusing myself. Let me use the standard definition:
    # The matrix multiplication tensor M_{n,m,p} is a 3-tensor of dimensions n x m x p
    # (or some permutation) where M[i,j,k] = 1 if and only if the indices are "compatible".
    # For square n x n x n: M[i,j,k] = delta[i,k]... 
    
    # Actually, I think the convention here is:
    # U has rows indexed by (i,j) pairs where i in [0,n), j in [0,m) -> n*m rows
    # V has rows indexed by (j,k) pairs where j in [0,m), k in [0,p) -> m*p rows  
    # W has rows indexed by (k,i) pairs where k in [0,p), i in [0,n) -> p*n rows
    # And the decomposition is: for all (i,j,k): sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
    # if and only if... hmm, this should equal the matrix multiplication tensor.
    
    # The matrix multiplication tensor T[i,j,k] for C=AB:
    # T[i,j,k] = 1 if i is row of A/C, j is col of A/row of B, k is col of B/C
    # and the "connection" is that A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for all valid (i,j,k)? That can't be right because then rank would be 1.
    
    # OH WAIT. I think I've been overthinking this. The tensor is:
    # T[i,j,k] where the three modes are:
    # Mode 1: input A index (i,j) - row i, col j of A
    # Mode 2: input B index (j,k) - row j, col k of B  
    # Mode 3: output C index (i,k) - row i, col k of C
    # And T[(i,j),(j,k),(i,k)] = 1, T[(i,j),(j',k),(i',k)] = 0 if j != j' or i != i'
    
    # So in the flattened form:
    # U row index = i*m + j (A input)
    # V row index = j*p + k (B input)
    # W row index = k*n + i (C output)
    # And the constraint is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # for all i,j,k (since each (i,j,k) triple uniquely determines one product term)
    
    # Wait, but that means for EVERY (i,j,k), the sum equals 1. And for "wrong" combinations
    # (where the indices don't match), it should equal 0. But in this formulation, every
    # (i,j,k) is a valid triple... 
    
    # I think the correct interpretation is:
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = delta[i,i'] * delta[j,j'] * delta[k,k']
    # No, that's not right either.
    
    # Let me just look at the schoolbook code:
    # For each (i,j,k): U[i*m+j][col] = 1, V[j*p+k][col] = 1, W[k*n+i][col] = 1
    # This means for the specific (i,j,k) triple, U[i*m+j] * V[j*p+k] * W[k*n+i] = 1
    # And for other combinations, at least one factor is 0.
    
    # So the condition is: for all (i,j,k) and (i',j',k'):
    # sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r] = 1 if (i,j,k)=(i',j',k') else 0
    # 
    # Wait, that's not quite right. Let me re-examine.
    # In the schoolbook, each (i,j,k) gets its own column. So:
    # sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k''*n+i', r] 
    # = 1 if there exists a column r where U[i*m+j,r]=1, V[j'*p+k',r]=1, W[k''*n+i',r]=1
    # In schoolbook, column (i,j,k) has U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    # So the sum = 1 iff (i,j,k) = (i',j',k')... 
    # Specifically: U[i*m+j, col(i,j,k)] = 1 iff (i,j) matches, 
    # V[j'*p+k', col(i,j,k)] = 1 iff (j',k') = (j,k),
    # W[k''*n+i', col(i,j,k)] = 1 iff (k'',i') = (k,i)
    # So all three match iff (i,j,k) = (i',j',k').
    
    # So the condition is:
    # sum_r U[a, r] * V[b, r] * W[c, r] = 1 if (a,b,c) is a "valid" triple, 0 otherwise
    # where valid means: a = i*m+j, b = j*p+k, c = k*n+i for some (i,j,k)
    # i.e., the j-index in a matches the j-index in b, and the k-index in b matches the k-index in c,
    # and the i-index in c matches the i-index in a.
    
    # OK so now for the Kronecker product construction:
    # For M_2: U2 has shape (4,7), V2 has shape (4,7), W2 has shape (4,7)
    # For M_4: U4 = U2 ⊗ U2 has shape (16,49), etc.
    
    # The Kronecker product of the decompositions:
    # If M_2 = sum_r u_r ⊗ v_r ⊗ w_r, then M_4 = sum_{r,s} (u_r⊗u_s) ⊗ (v_r⊗v_s) ⊗ (w_r⊗w_s)
    
    # For the Kronecker product to work correctly, I need:
    # U4[a4, rs] = U2[a2, r] * U2[b2, s] where a4 = a2*4 + b2 (Kronecker ordering)
    # Similarly for V and W.
    
    # Now I need the explicit Strassen decomposition of M_2.
    # M_2: n=m=p=2, so U2, V2, W2 each have shape (4,7).
    # 
    # Strassen's 7 multiplications for 2x2 matrix multiplication:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # Now I need to express this in the tensor decomposition form.
    # A indices: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d -> U2 rows: 0,1,2,3
    # B indices: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h -> V2 rows: 0,1,2,3
    # C indices: (0,0), (0,1), (1,0), (1,1) -> W2 rows: 0,1,2,3
    #   where C row index = k*n+i, so C[0,0]=row0, C[0,1]=row1, C[1,0]=row2, C[1,1]=row3
    
    # For each Strassen multiplication M_r, it's a product of a linear combo of A entries
    # and a linear combo of B entries, contributing to specific C entries.
    
    # M1 = (a+d)(e+h): 
    #   A-part: a+d = U2[0,1]*A[0,0] + U2[3,1]*A[1,1] -> U2[0,1]=1, U2[3,1]=1
    #   B-part: e+h = V2[0,1]*B[0,0] + V2[3,1]*B[1,1] -> V2[0,1]=1, V2[3,1]=1
    #   C-part: contributes to C[0,0] (coeff +1), C[1,1] (coeff +1)
    #   W2[0,1]=1, W2[3,1]=1
    
    # M2 = (c+d)e:
    #   A-part: c+d -> U2[2,2]=1, U2[3,2]=1
    #   B-part: e -> V2[0,2]=1
    #   C-part: C[1,0] (coeff +1) -> W2[2,2]=1
    
    # M3 = a(f-h):
    #   A-part: a -> U2[0,3]=1
    #   B-part: f-h -> V2[1,3]=1, V2[3,3]=-1
    #   C-part: C[0,1] (coeff +1), C[1,1] (coeff -1) -> W2[1,3]=1, W2[3,3]=-1
    
    # M4 = d(g-e):
    #   A-part: d -> U2[3,4]=1
    #   B-part: g-e -> V2[2,4]=1, V2[0,4]=-1
    #   C-part: C[0,0] (coeff +1), C[1,0] (coeff +1) -> W2[0,4]=1, W2[2,4]=1
    
    # M5 = (a+b)h:
    #   A-part: a+b -> U2[0,5]=1, U2[1,5]=1
    #   B-part: h -> V2[3,5]=1
    #   C-part: C[0,0] (coeff -1), C[0,1] (coeff +1) -> W2[0,5]=-1, W2[1,5]=1
    
    # M6 = (c-a)(e+f):
    #   A-part: c-a -> U2[2,6]=1, U2[0,6]=-1
    #   B-part: e+f -> V2[0,6]=1, V2[1,6]=1
    #   C-part: C[1,1] (coeff +1) -> W2[3,6]=1
    
    # M7 = (b-d)(g+h):
    #   A-part: b-d -> U2[1,7]=1, U2[3,7]=-1
    #   B-part: g+h -> V2[2,7]=1, V2[3,7]=1
    #   C-part: C[0,0] (coeff +1), C[0,1] (coeff +1) -> W2[0,7]=1, W2[1,7]=1
    
    # Let me build U2, V2, W2 as 4x7 matrices:
    U2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    V2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    W2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    
    # Column 0 (M1): U2: row0=1, row3=1; V2: row0=1, row3=1; W2: row0=1, row3=1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # Column 1 (M2): U2: row2=1, row3=1; V2: row0=1; W2: row2=1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # Column 2 (M3): U2: row0=1; V2: row1=1, row3=-1; W2: row1=1, row3=-1
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = -1
    
    # Column 3 (M4): U2: row3=1; V2: row2=1, row0=-1; W2: row0=1, row2=1
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Column 4 (M5): U2: row0=1, row1=1; V2: row3=1; W2: row0=-1, row1=1
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Column 5 (M6): U2: row2=1, row0=-1; V2: row0=1, row1=1; W2: row3=1
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Column 6 (M7): U2: row1=1, row3=-1; V2: row2=1, row3=1; W2: row0=1, row1=1
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[1][6] = 1
    
    # Verify M_2 decomposition:
    # For each (i,j,k) in {0,1}^3, check sum_r U2[i*2+j, r] * V2[j*2+k, r] * W2[k*2+i, r] == 1
    # And for "wrong" combinations, it should be 0.
    
    # Let me verify a few:
    # (i=0,j=0,k=0): sum_r U2[0,r]*V2[0,r]*W2[0,r]
    # r=0: 1*1*1=1, r=1: 0, r=2: 1*0*0=0, r=3: 0*(-1)*1=0, r=4: 1*0*(-1)=0, r=5: -1*1*0=0, r=6: 0*0*1=0
    # Sum = 1 ✓
    
    # (i=0,j=1,k=0): sum_r U2[1,r]*V2[2,r]*W2[0,r]
    # r=0: 0, r=1: 0, r=2: 0, r=3: 0*1*1=0, r=4: 1*0*(-1)=0, r=5: 0, r=6: 1*1*1=1
    # Sum = 1 ✓
    
    # (i=0,j=0,k=1): sum_r U2[0,r]*V2[1,r]*W2[2,r]
    # r=0: 0, r=1: 0, r=2: 1*1*0=0, r=3: 0, r=4: 0, r=5: -1*1*0=0, r=6: 0
    # Sum = 0... but it should be 1!
    
    # Hmm, let me recheck. (i=0,j=0,k=1) means A[0,0]*B[0,1] contributes to C[0,1].
    # C[0,1] = M3+M5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh
    # So A[0,0]=a contributes with coeff 1, A[0,1]=b contributes with coeff 1.
    # For (i=0,j=0,k=1): the contribution is from A[0,0]*B[0,1] = a*f
    # In C[0,1], the coefficient of a*f is 1. So the sum should be 1.
    
    # Let me recompute: U2[0,r]*V2[1,r]*W2[2,r] for r=0..6:
    # r=0: U2[0,0]=1, V2[1,0]=0, W2[2,0]=0 -> 0
    # r=1: U2[0,1]=0, V2[1,1]=0, W2[2,1]=1 -> 0
    # r=2: U2[0,2]=1, V2[1,2]=1, W2[2,2]=0 -> 0
    # r=3: U2[0,3]=0, V2[1,3]=0, W2[2,3]=1 -> 0
    # r=4: U2[0,4]=1, V2[1,4]=0, W2[2,4]=0 -> 0
    # r=5: U2[0,5]=-1, V2[1,5]=1, W2[2,5]=0 -> 0
    # r=6: U2[0,6]=0, V2[1,6]=0, W2[2,6]=0 -> 0
    # Sum = 0. But it should be 1!
    
    # I think I made an error. Let me re-examine the Strassen formula.
    # C[0,1] = M3 + M5 where M3 = a(f-h), M5 = (a+b)h
    # C[0,1] = af - ah + ah + bh = af + bh
    # The term a*f comes from M3: a * (f-h), specifically the f part.
    # In M3: A-part is a (U2[0,2]=1), B-part is f-h (V2[1,2]=1, V2[3,2]=-1)
    # C-part: M3 contributes to C[0,1] with coeff +1 and C[1,1] with coeff -1
    # So W2[1,2]=1 (C[0,1]), W2[3,2]=-1 (C[1,1])
    
    # For (i=0,j=0,k=1): U2[0*2+0, r] * V2[0*2+1, r] * W2[1*2+0, r] = U2[0,r]*V2[1,r]*W2[2,r]
    # W2[2,r] is the coefficient for C[1,0], not C[0,1]!
    
    # I think I have the W indexing wrong. W[k*n+i, r] where k is the column of B/C and i is the row of A/C.
    # For n=2: W[k*2+i, r]
    # C[0,1]: i=0, k=1 -> W[1*2+0, r] = W[2, r]
    # C[1,1]: i=1, k=1 -> W[1*2+1, r] = W[3, r]
    
    # So for M3 contributing to C[0,1] (i=0,k=1): W[2,2] should be +1
    # And C[1,1] (i=1,k=1): W[3,2] should be -1
    
    # But I had W2[1,2]=1 and W2[3,2]=-1. That's wrong!
    # W2[1,2] corresponds to k=0,i=1 -> C[1,0], not C[0,1].
    
    # Let me redo the W indexing:
    # W index = k*n + i where k = col of B (0 or 1), i = row of A (0 or 1)
    # C[i,k] -> W[k*2+i]
    # C[0,0] -> W[0], C[0,1] -> W[2], C[1,0] -> W[1], C[1,1] -> W[3]
    
    # OK so I had the W indexing wrong. Let me redo:
    
    U2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    V2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    W2 = [[0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0]]
    
    # M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
    # W[0,0]=1 (C[0,0]), W[3,0]=1 (C[1,1])
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (c+d)e -> C[1,0]+=1
    # W[1,1]=1 (C[1,0])
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # M3 = a(f-h) -> C[0,1]+=1, C[1,1]-=1
    # W[2,2]=1 (C[0,1]), W[3,2]=-1 (C[1,1])
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = -1
    
    # M4 = d(g-e) -> C[0,0]+=1, C[1,0]+=1
    # W[0,3]=1 (C[0,0]), W[1,3]=1 (C[1,0])
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5 = (a+b)h -> C[0,0]-=1, C[0,1]+=1
    # W[0,4]=-1 (C[0,0]), W[2,4]=1 (C[0,1])
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6 = (c-a)(e+f) -> C[1,1]+=1
    # W[3,5]=1 (C[1,1])
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7 = (b-d)(g+h) -> C[0,0]+=1, C[0,1]+=1
    # W[0,6]=1 (C[0,0]), W[2,6]=1 (C[0,1])
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[2][6] = 1
    
    # Now verify (i=0,j=0,k=1): U2[0,r]*V2[1,r]*W2[2,r]
    # r=0: 1*0*0=0, r=1: 0*0*0=0, r=2: 1*1*1=1, r=3: 0*0*0=0, r=4: 1*0*1=0, r=5: -1*1*0=0, r=6: 0*0*1=0
    # Sum = 1 ✓
    
    # Now build M_4 using Kronecker product:
    # U4[a4, rs] = U2[a2, r] * U2[b2, s] where a4 = a2*4 + b2
    # V4[b4, rs] = V2[b2, r] * V2[c2, s] where b4 = b2*4 + c2
    # W4[c4, rs] = W2[c2, r] * W2[d2, s] where c4 = c2*4 + d2
    
    # For M_4: n=m=p=4
    # U4 shape: (16, 49), V4 shape: (16, 49), W4 shape: (16, 49)
    
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    for r in range(7):
        for s in range(7):
            col = r*7 + s
            for a2 in range(4):
                for b2 in range(4):
                    a4 = a2*4 + b2
                    U4[a4][col] = U2[a2][r] * U2[b2][s]
            for b2 in range(4):
                for c2 in range(4):
                    b4 = b2*4 + c2
                    V4[b4][col] = V2[b2][r] * V2[c2][s]
            for c2 in range(4):
                for d2 in range(4):
                    c4 = c2*4 + d2
                    W4[c4][col] = W2[c2][r] * W2[d2][s]
    
    # Verify the M_4 decomposition:
    # For all (i,j,k) in {0,1,2,3}^3:
    # sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] should equal 1
    # For "wrong" combinations (where indices don't align), it should equal 0.
    
    # Let me verify a few cases:
    # (i=0,j=0,k=0): U4[0,r]*V4[0,r]*W4[0,r]
    # This should be 1.
    
    # (i=0,j=1,k=0): U4[1,r]*V4[4,r]*W4[0,r] should be 1
    
    # Let me do a quick numerical check:
    U4_np = np.array(U4, dtype=float)
    V4_np = np.array(V4, dtype=float)
    W4_np = np.array(W4, dtype=float)
    
    # Check all valid (i,j,k) triples
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                u_row = U4_np[i*4+j]
                v_row = V4_np[j*4+k]
                w_row = W4_np[k*4+i]
                val = np.dot(u_row * v_row, w_row)
                max_err = max(max_err, abs(val - 1))
    
    # Check some invalid triples
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for i2 in range(4):
                    if i2 == i:
                        continue
                    u_row = U4_np[i*4+j]
                    v_row = V4_np[j*4+k]
                    w_row = W4_np[k*4+i2]
                    val = np.dot(u_row * v_row, w_row)
                    max_err = max(max_err, abs(val))
    
    # Also check with wrong j
    for i in range(4):
        for j in range(4):
            for k in range(4):
                for j2 in range(4):
                    if j2 == j:
                        continue
                    u_row = U4_np[i*4+j2]
                    v_row = V4_np[j*4+k]
                    w_row = W4_np[k*4+i]
                    val = np.dot(u_row * v_row, w_row)
                    max_err = max(max_err, abs(val))
    
    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U4, "V": V4, "W": W4}
# EVOLVE-BLOCK-END
