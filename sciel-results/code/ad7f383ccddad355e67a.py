"""Rewritten program: rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy: Use the 2x2 block partition of the 4x4 multiplication tensor.
The 4x4 matrix multiplication can be viewed as multiplying two 2x2 block matrices
where each block is 2x2. This gives a 2x2x2 tensor of 2x2 matrices.

The key insight: The rank of 2x2x2 matrix multiplication is 7 (Strassen).
If we treat the 4x4 multiplication as a 2x2 block multiplication where each
"element" is a 2x2 matrix, we can use Strassen's algorithm at the block level.

However, this only works if the block interactions are independent. For a general
4x4 matrix multiplication, the blocks are NOT independent - there are cross terms.

A better approach: The 4x4 matrix multiplication tensor has a known rank.
The best known upper bound for 4x4 matrix multiplication rank is 49 (from 
applying Strassen recursively or via block methods).

Actually, let me reconsider. The standard result:
- 2x2 matrix multiplication: rank 7 (Strassen)
- 4x4 matrix multiplication: rank <= 49 (by applying Strassen to 2x2 blocks of 2x2)

But we need to be careful about the indexing convention. The output W follows
W[k*n+i, r] where n=m=p=4.

Let me construct the rank-49 decomposition using the block Strassen approach.

For 4x4 multiplication, partition into 2x2 blocks:
A = [[A11, A12], [A21, A22]] where each Aij is 2x2
B = [[B11, B12], [B21, B22]] where each Bij is 2x2

C = A*B = [[C11, C12], [C21, C22]] where each Cij is 2x2

Using Strassen at the block level:
M1 = (A11 + A22)(B11 + B22)
M2 = (A21 + A22)B11
M3 = A11(B12 - B22)
M4 = A22(B21 - B11)
M5 = (A11 + A12)B22
M6 = (A21 - A11)(B11 + B12)
M7 = (A12 - A22)(B21 + B22)

C11 = M1 + M4 - M5 + M7
C12 = M3 + M5
C21 = M2 + M4
C22 = M1 - M2 + M3 + M6

Each Mi is a product of two 2x2 matrices, which itself has rank 7.
So total rank = 7 * 7 = 49.

Now I need to map this to the U, V, W matrices with the correct indexing.

U has shape (n*m, rank) = (16, 49)
V has shape (m*p, rank) = (16, 49)  
W has shape (p*n, rank) = (16, 49)

The tensor identity is:
sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = (A*B)[i,k]

Wait, let me re-read the convention. The task says:
"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So for the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
This gives sum over c of U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = (A*B)[i,k]

Actually the standard tensor decomposition for matrix multiplication T_{i,j,k} means:
(A*B)[i,k] = sum_j A[i,j]*B[j,k]

So the tensor is T[i,j,k] = delta where the contraction is over j.

In the factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Wait, that doesn't look right dimensionally. Let me re-read.

"U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"

So U is indexed by (i*m+j) which ranges over 0..15 for n=m=4
V is indexed by (j*p+k) which ranges over 0..15 for m=p=4
W is indexed by (k*n+i) which ranges over 0..15 for p=n=4

The product sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal (A*B)[i,k]

For this to work as a tensor decomposition of matrix multiplication, we need:
(A*B)[i,k] = sum_j A[i,j] B[j,k]

So the factorization must satisfy:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta_{i,k} in some sense...

No wait. The tensor for matrix multiplication is:
T[i,j,k] such that (A*B)[i,k] = sum_j A[i,j] T[i,j,k] B[j,k]... 

Actually no. The standard way: matrix multiplication is a bilinear map.
C = A*B means C[i,k] = sum_j A[i,j] * B[j,k]

As a trilinear form or tensor: the multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j] and B[j,k].

More precisely, the tensor T of matrix multiplication satisfies:
T[i,j,k] = 1 (it's the identity in the sense that C[i,k] = sum_j A[i,j] B[j,k])

The rank decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]... 

Hmm, I think the indexing in the problem is:
U has rows indexed by (i,j) pairs -> row i*m+j
V has rows indexed by (j,k) pairs -> row j*p+k  
W has rows indexed by (k,i) pairs -> row k*n+i

And the identity is: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = delta where...

Actually I think the correct interpretation is that this decomposes the matrix multiplication tensor where:
(A*B)[i,k] = sum_j sum_r A[i,j] * B[j,k] * (something)

No. Let me think again. In tensor decomposition of matrix multiplication:
The multiplication tensor T is a 3-tensor with T[i,j,k] representing the coefficient.
For standard matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (it's just the structure).

The rank-r decomposition means:
T[i,j,k] = sum_{r=1}^{r} U[i,j,r] * V[j,k,r] * W[i,k,r]... 

But the indices don't match. Let me look at the schoolbook example more carefully.

Schoolbook: for each (i,j,k), we have one term with U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1.

So T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

For schoolbook, this equals 1 for all i,j,k (since each (i,j,k) has exactly one c where all three are 1).

And indeed (A*B)[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_c A[i,j] * B[j,k] * (U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c])

Wait that's not right either because we'd be summing over c too.

I think the correct interpretation is:
The matrix multiplication tensor is defined such that:
(A*B)[i,k] = sum_j A[i,j] * B[j,k]

And the tensor T[i,j,k] = 1 (all ones) represents this.

The rank decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.

This is a rank-1 decomposition of the all-ones tensor, which has rank 1... but that's not right because the schoolbook uses rank 64.

I'm confused. Let me just verify with the schoolbook: it has rank 64 = 4*4*4, and for each (i,j,k) triple, there's exactly one column c where U, V, W all have 1. So sum_r ... = 1 for each (i,j,k). That checks out.

So I need to find U, V, W of rank r such that for all i in [0,4), j in [0,4), k in [0,4):
sum_{r=0}^{r-1} U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1

Let me implement the rank-49 Strassen-based decomposition.

Actually, I realize I should just implement a known correct decomposition. Let me use the recursive Strassen approach properly.

For 4x4, I'll use the fact that it can be decomposed using 7 Strassen multiplications at the 2x2 block level, each of which is a 2x2x2 multiplication of rank 7.

Let me code this up carefully.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # Try to construct a rank-49 decomposition using block Strassen
    # Partition 4x4 matrices into 2x2 blocks of 2x2 submatrices
    # Use Strassen at the block level (7 multiplications of 2x2 matrices)
    # Each 2x2 multiplication uses rank 7 (Strassen)
    # Total rank: 7 * 7 = 49
    
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen formulas for 2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For the block level (4x4 as 2x2 blocks of 2x2):
    # A = [[A00, A01], [A10, A11]] where each is 2x2
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = A*B
    
    # Each block multiplication Mx = X*Y where X, Y are 2x2
    # We decompose each 2x2 multiplication using Strassen (rank 7)
    
    # Define the 7 Strassen terms for a 2x2 multiplication
    # For 2x2 matrices P and Q:
    # S1 = (P00+P11)(Q00+Q11)
    # S2 = (P10+P11)Q00
    # S3 = P00(Q01-Q11)
    # S4 = P11(Q10-Q00)
    # S5 = (P00+P01)Q11
    # S6 = (P10-P00)(Q00+Q01)
    # S7 = (P01-P11)(Q10+Q11)
    #
    # R00 = S1+S4-S5+S7
    # R01 = S3+S5
    # R10 = S2+S4
    # R11 = S1-S2+S3+S6
    
    # For each of the 7 block-level Strassen multiplications,
    # we need to express the 2x2*2x2 product using rank 7.
    
    # Let me define helper: given two 2x2 matrices expressed as linear combos,
    # produce the rank-7 Strassen decomposition.
    
    # Actually, let me think of it differently. 
    # The 4x4 multiplication tensor can be built from 7 copies of the 2x2 multiplication tensor,
    # with appropriate linear combinations.
    
    # Let me define the 2x2 Strassen decomposition explicitly.
    # For 2x2: indices i,j,k in {0,1}
    # U_2x2[i*2+j, r], V_2x2[j*2+k, r], W_2x2[k*2+i, r] for r in 0..6
    
    # Strassen 2x2 decomposition:
    # Term 0: S1 = (P00+P11)(Q00+Q11)
    #   U: (0,0)->1, (1,1)->1  => rows 0,3
    #   V: (0,0)->1, (0,1)->1  => rows 0,1  [Q00, Q01]... wait
    #   
    # Let me be very explicit. For 2x2 matrices P[2,2] and Q[2,2]:
    # P00=P[0,0], P01=P[0,1], P10=P[1,0], P11=P[1,1]
    # Q00=Q[0,0], Q01=Q[0,1], Q10=Q[1,0], Q11=Q[1,1]
    #
    # S1 = (P00+P11)*(Q00+Q11)
    #   U row for P: P00 and P11 have coeff 1 => U[0,0]=1, U[3,0]=1 (row i*2+j: (0,0)=0, (1,1)=3)
    #   V row for Q: Q00 and Q11 have coeff 1 => V[0,0]=1, V[3,0]=1 (row j*2+k: (0,0)=0, (1,1)=3)
    #   W row for R: R00 gets +1 from S1 => W[0*2+0,0]=1 (k*n+i with n=2: k*2+i)
    #     Also R11 gets +1 from S1 => W[1*2+1,0]=1 (k=1,i=1 => row 3)
    #
    # This is getting complex. Let me just build it programmatically.
    
    def build_2x2_strassen():
        """Build rank-7 Strassen decomposition for 2x2 matrix multiplication.
        Returns U, V, W each of shape (4, 7).
        U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # 7 Strassen terms
        # Term r: U has entries for P, V has entries for Q, W has entries for R
        
        # S1 = (P00+P11)(Q00+Q11)
        # U: P00(row 0)=1, P11(row 3)=1
        # V: Q00(row 0)=1, Q11(row 3)=1
        # W: R00(k=0,i=0,row 0)=1, R11(k=1,i=1,row 3)=1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # S2 = (P10+P11)(Q00)
        # U: P10(row 2)=1, P11(row 3)=1
        # V: Q00(row 0)=1
        # W: R10(k=0,i=1,row 1)=1
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[1, 1] = 1
        
        # S3 = P00(Q01-Q11)
        # U: P00(row 0)=1
        # V: Q01(row 1)=1, Q11(row 3)=-1
        # W: R01(k=1,i=0,row 2)=1
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[2, 2] = 1
        
        # S4 = P11(Q10-Q00)
        # U: P11(row 3)=1
        # V: Q10(row 2)=1, Q00(row 0)=-1
        # W: R00(k=0,i=0,row 0)=1, R10(k=0,i=1,row 1)=1
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[1, 3] = 1
        
        # S5 = (P00+P01)(Q11)
        # U: P00(row 0)=1, P01(row 1)=1
        # V: Q11(row 3)=1
        # W: R00(k=0,i=0,row 0)=-1, R01(k=1,i=0,row 2)=1
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # S6 = (P10-P00)(Q00+Q01)
        # U: P10(row 2)=1, P00(row 0)=-1
        # V: Q00(row 0)=1, Q01(row 1)=1
        # W: R11(k=1,i=1,row 3)=1
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # S7 = (P01-P11)(Q10+Q11)
        # U: P01(row 1)=1, P11(row 3)=-1
        # V: Q10(row 2)=1, Q11(row 3)=1
        # W: R00(k=0,i=0,row 0)=1, R11(k=1,i=1,row 3)=-1
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W
    
    U2, V2, W2 = build_2x2_strassen()
    
    # Now build the 4x4 decomposition using block Strassen.
    # The 4x4 matrix A has blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    
    # For the 4x4 Strassen:
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
    
    # Each Mx is a 2x2 matrix product. We decompose each using the 2x2 Strassen.
    
    # For M1 = (A00+A11)(B00+B11):
    # The "P" matrix is (A00+A11), the "Q" matrix is (B00+B11)
    # P00 = A[0,0]+A[2,2], P01 = A[0,1]+A[2,3], P10 = A[1,0]+A[3,2], P11 = A[1,1]+A[3,3]
    # Q00 = B[0,0]+B[2,2], Q01 = B[0,1]+B[2,3], Q10 = B[1,0]+B[3,2], Q11 = B[1,1]+B[3,3]
    
    # In the 4x4 indexing:
    # A[i,j] corresponds to U row i*4+j
    # B[j,k] corresponds to V row j*4+k
    # C[i,k] corresponds to W row k*4+i
    
    # For each of the 7 block-level Strassen terms, and for each of the 7 sub-Strassen terms,
    # we add contributions to U, V, W.
    
    # Let me define the block-level Strassen combinations:
    # For each block-level term t (0..6), we have:
    #   P_block = linear combination of A blocks
    #   Q_block = linear combination of B blocks
    #   R_block = linear combination of C blocks (with coefficients)
    
    # The 7 block-level terms and their P, Q, R block assignments:
    block_terms = [
        # (P_blocks, Q_blocks, R_blocks)
        # Each block is identified by (block_row, block_col) in {0,1}x{0,1}
        # P_blocks: list of (block, coeff)
        # Q_blocks: list of (block, coeff)  
        # R_blocks: list of (block, coeff)
        
        # M1 = (A00+A11)(B00+B11), contributes to C00(+1), C11(+1)
        ([((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)], [((0,0),1), ((1,1),1)]),
        
        # M2 = (A10+A11)B00, contributes to C10(+1)
        ([((1,0),1), ((1,1),1)], [((0,0),1)], [((1,0),1)]),
        
        # M3 = A00(B01-B11), contributes to C01(+1), C11(+1)
        ([((0,0),1)], [((0,1),1), ((1,1),-1)], [((0,1),1), ((1,1),1)]),
        
        # M4 = A11(B10-B00), contributes to C00(+1), C10(+1)
        ([((1,1),1)], [((1,0),1), ((0,0),-1)], [((0,0),1), ((1,0),1)]),
        
        # M5 = (A00+A01)B11, contributes to C00(-1), C01(+1)
        ([((0,0),1), ((0,1),1)], [((1,1),1)], [((0,0),-1), ((0,1),1)]),
        
        # M6 = (A10-A00)(B00+B01), contributes to C11(+1)
        ([((1,0),1), ((0,0),-1)], [((0,0),1), ((0,1),1)], [((1,1),1)]),
        
        # M7 = (A01-A11)(B10+B11), contributes to C00(+1), C11(-1)
        ([((0,1),1), ((1,1),-1)], [((1,0),1), ((1,1),1)], [((0,0),1), ((1,1),-1)]),
    ]
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for t in range(7):
        P_blocks, Q_blocks, R_blocks = block_terms[t]
        
        # For this block-level term, we have a 2x2 "virtual" matrix P (from A blocks)
        # and a 2x2 "virtual" matrix Q (from B blocks).
        # We decompose P*Q using the 2x2 Strassen (7 sub-terms).
        
        for s in range(7):
            col = t * 7 + s
            
            # The s-th sub-Strassen term has:
            # U2[0:4, s] for P's elements (P00, P01, P10, P11)
            # V2[0:4, s] for Q's elements (Q00, Q01, Q10, Q11)
            # W2[0:4, s] for R's elements (R00, R01, R10, R11)
            
            # Map P's 2x2 elements to A's 4x4 elements:
            # P00 = sum of A block contributions at position (0,0)
            # P01 = sum of A block contributions at position (0,1)
            # P10 = sum of A block contributions at position (1,0)
            # P11 = sum of A block contributions at position (1,1)
            
            # P block (br, bc) means A[2*br:2*br+2, 2*bc:2*bc+2]
            # P00 corresponds to A elements at rows 0,1 and cols 0,1 (block (0,0))
            # P01 corresponds to A elements at rows 0,1 and cols 2,3 (block (0,1))
            # P10 corresponds to A elements at rows 2,3 and cols 0,1 (block (1,0))
            # P11 corresponds to A elements at rows 2,3 and cols 2,3 (block (1,1))
            
            # U2 row indices: 0=P00, 1=P01, 2=P10, 3=P11
            # Map to A 4x4 row indices (i*4+j):
            # P00: A[0,0], A[0,1], A[1,0], A[1,1] => rows 0,1,4,5
            # P01: A[0,2], A[0,3], A[1,2], A[1,3] => rows 2,3,6,7
            # P10: A[2,0], A[2,1], A[3,0], A[3,1] => rows 8,9,12,13
            # P11: A[2,2], A[2,3], A[3,2], A[3,3] => rows 10,11,14,15
            
            # For each P element (pe in 0..3), U2[pe, s] gives the coefficient
            # We need to distribute this to the appropriate A rows.
            
            # P element pe corresponds to block position:
            # pe=0: block (0,0), local positions (0,0),(0,1),(1,0),(1,1)
            # pe=1: block (0,1), local positions (0,0),(0,1),(1,0),(1,1)
            # pe=2: block (1,0), local positions (0,0),(0,1),(1,0),(1,1)
            # pe=3: block (1,1), local positions (0,0),(0,1),(1,0),(1,1)
            
            # For P element pe, the A rows are:
            # base_i = 2 * (pe // 2)
            # base_j = 2 * (pe % 2)
            # A rows: (base_i + 0)*4 + (base_j + 0), (base_i + 0)*4 + (base_j + 1),
            #          (base_i + 1)*4 + (base_j + 0), (base_i + 1)*4 + (base_j + 1)
            
            for pe in range(4):
                coeff_p = U2[pe, s]
                if coeff_p == 0:
                    continue
                base_i = 2 * (pe // 2)
                base_j = 2 * (pe % 2)
                a_rows = [(base_i + di) * 4 + (base_j + dj) for di in range(2) for dj in range(2)]
                
                # Now, P element pe gets contributions from P_blocks:
                # P_blocks is a list of ((br, bc), coeff)
                # P element pe corresponds to block (pe//2, pe%2)
                p_block = (pe // 2, pe % 2)
                
                # Sum contributions from all P_blocks that match this block
                total_coeff = 0
                for (pb, pcoeff) in P_blocks:
                    if pb == p_block:
                        total_coeff += pcoeff
                
                if total_coeff == 0:
                    continue
                
                final_coeff = coeff_p * total_coeff
                for ar in a_rows:
                    U4[ar, col] += final_coeff
            
            # Similarly for V (Q blocks -> B elements)
            for qe in range(4):
                coeff_q = V2[qe, s]
                if coeff_q == 0:
                    continue
                base_j = 2 * (qe // 2)
                base_k = 2 * (qe % 2)
                b_rows = [(base_j + dj) * 4 + (base_k + dk) for dj in range(2) for dk in range(2)]
                
                q_block = (qe // 2, qe % 2)
                
                total_coeff = 0
                for (qb, qcoeff) in Q_blocks:
                    if qb == q_block:
                        total_coeff += qcoeff
                
                if total_coeff == 0:
                    continue
                
                final_coeff = coeff_q * total_coeff
                for br in b_rows:
                    V4[br, col] += final_coeff
            
            # Similarly for W (R blocks -> C elements)
            for re in range(4):
                coeff_r = W2[re, s]
                if coeff_r == 0:
                    continue
                base_k = 2 * (re // 2)
                base_i = 2 * (re % 2)
                c_rows = [(base_k + dk) * 4 + (base_i + di) for dk in range(2) for di in range(2)]
                
                r_block = (re // 2, re % 2)
                
                total_coeff = 0
                for (rb, rcoeff) in R_blocks:
                    if rb == r_block:
                        total_coeff += rcoeff
                
                if total_coeff == 0:
                    continue
                
                final_coeff = coeff_r * total_coeff
                for cr in c_rows:
                    W4[cr, col] += final_coeff
    
    # Verify the decomposition
    # Check: sum_r U4[i*4+j, r] * V4[j*4+k, r] * W4[k*4+i, r] == 1 for all i,j,k
    max_err = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.sum(U4[i*4+j, :] * V4[j*4+k, :] * W4[k*4+i, :])
                err = abs(val - 1)
                if err > max_err:
                    max_err = err
    
    if max_err < 1e-10:
        # Convert to lists for the interface
        U_list = U4.tolist()
        V_list = V4.tolist()
        W_list = W4.tolist()
        
        # Round to integers if possible
        for mat in [U_list, V_list, W_list]:
            for i in range(len(mat)):
                for j in range(len(mat[i])):
                    if abs(mat[i][j] - round(mat[i][j])) < 1e-10:
                        mat[i][j] = round(mat[i][j])
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank_fb = 64
        U_fb = [[0] * rank_fb for _ in range(16)]
        V_fb = [[0] * rank_fb for _ in range(16)]
        W_fb = [[0] * rank_fb for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j][column] = 1
                    V_fb[j * 4 + k][column] = 1
                    W_fb[k * 4 + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

# EVOLVE-BLOCK-END
