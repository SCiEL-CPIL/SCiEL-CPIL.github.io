"""Matrix multiplication 2x4 x 4x7 rank-minimal decomposition.

Strategy: Block decomposition using Strassen's algorithm.
Split the 4-dimensional middle index into two groups of 2.
This gives 4 sub-problems of type 2x2 x 2x7 (two for each half of A)
and 2 sub-problems of type 2x2 x 2x3 (cross terms).

Actually, the correct decomposition:
- A is 2x4, B is 4x7
- Split columns of A into A1 (2x2) and A2 (2x2)  
- Split rows of B into B1 (2x7) and B2 (2x7)
- C = A1@B1 + A2@B2

Each A_i @ B_i is a 2x2 @ 2x7 multiplication.
For 2x2 @ 2x7: we can use Strassen on the 2x2 part, giving rank 7*7=49? No.

Let me think more carefully. The tensor for 2x2 @ 2x7 has:
- U shape: (2*2, r) = (4, r)
- V shape: (2*7, r) = (14, r)  
- W shape: (7*2, r) = (14, r)
- Schoolbook rank: 2*2*7 = 28

Using Strassen on the 2x2 block structure:
The 2x2 @ 2x7 can be viewed as: for each of the 7 columns of B,
do a 2x2 @ 2x1 multiplication. Each 2x2@2x1 uses rank 7 (Strassen).
But these share structure across columns.

Better approach: The 2x2 @ 2x7 tensor has a Kronecker-like structure.
It's the Kronecker product of the 2x2@2x1 tensor with... no, not exactly.

Let me just directly construct the rank-52 decomposition:
- 4 sub-problems of type 2x2 x 2x2 (from splitting both A and B into 2x2 blocks)
  Each uses Strassen rank 7, total: 4 * 7 = 28
- 2 sub-problems of type 2x2 x 2x3 (cross terms)
  Each uses schoolbook rank 12, total: 2 * 12 = 24
- Grand total: 28 + 24 = 52

Wait, that's not quite right either. Let me think about the actual block structure.

A (2x4) = [A_top_left (2x2) | A_top_right (2x2)]
           [A_bot_left (2x2) | A_bot_right (2x2)]

No wait, A is 2x4, so:
A = [a00 a01 a02 a03]
    [a10 a11 a12 a13]

B (4x7) = [b00 b01 ... b06]
           [b10 b11 ... b16]
           [b20 b21 ... b26]
           [b30 b31 ... b36]

C (2x7) = A @ B

C[i,k] = sum_j A[i,j] * B[j,k] for j=0..3

Split j into j1 in {0,1} and j2 in {0,1}, j = j1*2 + j2:
C[i,k] = sum_{j1=0,1} sum_{j2=0,1} A[i, j1*2+j2] * B[j1*2+j2, k]

This is like a 2x2 matrix multiplication where:
- The "left matrix" has entries A[i, j1*2+j2] indexed by (j1,j2) for fixed i
- The "right matrix" has entries B[j1*2+j2, k] indexed by (j1,j2) for fixed k

For fixed i and k, we're computing a dot product of two vectors of length 4,
which can be seen as a 2x2 matrix product of "matrices" M_A[j1,j2] = A[i, j1*2+j2]
and M_B[j1,j2] = B[j1*2+j2, k], giving sum M_A[j1,j2]*M_B[j1,j2].

This is actually just a sum of 4 products, not a matrix multiplication in the usual sense.
The Strassen trick applies to matrix multiplication C = AB where C[i,k] = sum_j A[i,j]*B[j,k],
and the "matrix" structure is in the (i,k) output indices.

For our problem, the output is C[i,k] with i in {0,1} and k in {0,...,6}.
The middle index j in {0,1,2,3} is summed over.

To apply Strassen, we need to view this as a matrix multiplication.
One way: group the j indices.

Let's split j into two groups: j_low = {0,1} and j_high = {2,3}.
Then C[i,k] = sum_{j in {0,1}} A[i,j]*B[j,k] + sum_{j in {2,3}} A[i,j]*B[j,k]

Let A_low[i,j] = A[i,j] for j in {0,1} (a 2x2 matrix)
Let B_low[j,k] = B[j,k] for j in {0,1} (a 2x7 matrix)
Let A_high[i,j] = A[i,j] for j in {2,3} (a 2x2 matrix)  
Let B_high[j,k] = B[j,k] for j in {2,3} (a 2x7 matrix)

C = A_low @ B_low + A_high @ B_high

Each term A_low @ B_low is a 2x2 @ 2x7 multiplication.

For 2x2 @ 2x7, the tensor has:
- Input A: 2x2 = 4 elements
- Input B: 2x7 = 14 elements
- Output C: 2x7 = 14 elements
- Schoolbook rank: 2*2*7 = 28

Can we do 2x2 @ 2x7 in less than 28? 
The 2x2 @ 2x7 tensor is actually the tensor product of the 2x2@2x1 tensor with the identity on 7 dimensions? No.

Actually, 2x2 @ 2x7 can be seen as 7 independent 2x2 @ 2x1 multiplications (one per column of B).
Each 2x2 @ 2x1 uses Strassen rank 7. But they're not independent - they share the A matrix.

The correct way: 2x2 @ 2x7 is a bilinear map from (R^2 x R^2) x (R^2 x R^7) to R^2 x R^7.
The rank is at most 2*2*7 = 28 (schoolbook).

Using the structure: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1}, k in {0,...,6}.

This is equivalent to: for each k, compute a 2x2 @ 2x1 multiplication.
The 7 columns of B are independent, so the rank is 7 * (rank of 2x2@2x1).
Rank of 2x2@2x1 = 7 (Strassen). So total rank = 7*7 = 49? That's worse than 28.

No wait, that's not right. The 2x2 @ 2x7 tensor has rank at most 28 by schoolbook.
The Strassen decomposition of 2x2@2x2 has rank 7. 
For 2x2 @ 2x7, we can write it as a sum of 7 rank-1 tensors where each uses one Strassen multiplication
applied to all 7 columns simultaneously.

Specifically, Strassen for 2x2@2x2 gives 7 multiplications M1..M7 of 1x1 "matrices" (scalars).
For 2x2@2x7, each "scalar" multiplication becomes a 1x7 vector multiplication (i.e., scalar times vector).
So we still have 7 "multiplications", but each produces a 1x7 vector.
The rank is still 7 in terms of the number of scalar multiplications, but the tensor rank is different.

Let me just directly verify: the tensor for 2x2@2x7 has dimensions (4, 14, 14).
Schoolbook: 28 rank-1 terms.
Can we do it in 21? (7 Strassen mults * 3... no)

I think the rank of 2x2@2x7 is actually 28 (same as schoolbook) because the 7 columns are independent
and each requires 4 multiplications (2x2@2x1 has rank 4 by schoolbook, 7 by Strassen but that's for 2x2@2x2).

Hmm, I'm getting confused. Let me just go with the known result:
- 2x4 x 4x7 can be done in rank 52 using a block decomposition.

The decomposition:
Split the 4 middle indices into two groups of 2.
This gives two 2x2 @ 2x7 sub-problems.
Each 2x2 @ 2x7 uses schoolbook rank 28.
Total: 2 * 28 = 56 (same as original schoolbook).

To get rank 52, we need to use Strassen on some parts.

Alternative: Split differently.
View 2x4 x 4x7 as a combination of smaller multiplications.

Let me try: split the output k into k1 in {0,...,3} and k2 in {4,...,6}.
Then C[i,k1] = sum_j A[i,j]*B[j,k1] and C[i,k2] = sum_j A[i,j]*B[j,k2].
These are 2x4 @ 4x4 and 2x4 @ 4x3.

2x4 @ 4x4: split into 2x2@2x4 + 2x2@2x4 = 2*(2x2@2x4)
2x2@2x4: schoolbook rank 16, or use Strassen on the 2x2 part: rank 7*4=28? No.

I think I'm overcomplicating this. Let me just use the known rank-52 construction.

The rank-52 decomposition for 2x4 x 4x7:
- Use Strassen (rank 7) for four 2x2x2 sub-problems: 4*7 = 28
- Use schoolbook (rank 12) for two 2x2x3 sub-problems: 2*12 = 24
- Total: 52

The four 2x2x2 sub-problems come from splitting both the middle index and the output index.
The two 2x2x3 sub-problems come from the remaining part.

Let me construct this explicitly.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14

    # Build the target tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def tensor_from_factors(U, V, W):
        return np.einsum('ad,bd,cd->abc', U, V, W)

    def check_decomposition(U, V, W, tol=1e-8):
        T_rec = tensor_from_factors(U, V, W)
        return np.allclose(T_rec, T, atol=tol, rtol=tol)

    # Schoolbook decomposition as fallback
    rank_sb = n * m * p  # 56
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    # Construct rank-52 decomposition using block structure
    # Split middle index j into j_low in {0,1} and j_high in {2,3}
    # Split output index k into k_low in {0,1,2,3} and k_high in {4,5,6}
    
    # Sub-problem 1: A_low (2x2) @ B_low_low (2x4) -> C_low_low (2x4)
    # Sub-problem 2: A_high (2x2) @ B_high_low (2x4) -> C_low_low (2x4)  
    # Sub-problem 3: A_low (2x2) @ B_low_high (2x3) -> C_low_high (2x3)
    # Sub-problem 4: A_high (2x2) @ B_high_high (2x3) -> C_low_high (2x3)
    
    # For sub-problems 1 and 2 (2x2 @ 2x4):
    # Further split the 4 columns into two groups of 2
    # 2x2 @ 2x2 uses Strassen rank 7
    # So 2x2 @ 2x4 = 2 * (2x2 @ 2x2) = 2 * 7 = 14 rank terms each
    # Total for sub-problems 1 and 2: 2 * 14 = 28
    
    # For sub-problems 3 and 4 (2x2 @ 2x3):
    # Schoolbook rank = 2*2*3 = 12 each
    # Total for sub-problems 3 and 4: 2 * 12 = 24
    
    # Grand total: 28 + 24 = 52
    
    rank_52 = 52
    U_52 = np.zeros((nm, rank_52))
    V_52 = np.zeros((mp, rank_52))
    W_52 = np.zeros((pn, rank_52))
    
    d = 0  # column index
    
    # Strassen coefficients for 2x2 @ 2x2
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M1 + M7 - M6
    # C10 = M5 + M2 - M1 - M4
    # C11 = M6 + M3 + M4 + M1
    
    def add_strassen_2x2_2x2(i_base, j_base, k_base, U_mat, V_mat, W_mat, d_start):
        """Add Strassen rank-7 decomposition for a 2x2 @ 2x2 sub-problem.
        
        i_base: base index for i (0 or 1, but here we use the same i for both rows)
        j_base: base index for j (0 or 2)
        k_base: base index for k (0,1,2,3,4,5,6)
        U_mat: 2x2 matrix of coefficients for U
        V_mat: 2x2 matrix of coefficients for V  
        W_mat: 4x1 matrix of coefficients for W (C00, C01, C10, C11)
        d_start: starting column index
        """
        d = d_start
        # M1 = (A00+A11)*(B00+B11)
        U_mat[i_base + 0, j_base + 0, d] = 1
        U_mat[i_base + 1, j_base + 1, d] = 1
        V_mat[j_base + 0, k_base + 0, d] = 1
        V_mat[j_base + 1, k_base + 1, d] = 1
        W_mat[k_base + 0, i_base + 0, d] = 1
        W_mat[k_base + 1, i_base + 0, d] = 1
        W_mat[k_base + 0, i_base + 1, d] = -1
        W_mat[k_base + 1, i_base + 1, d] = -1
        d += 1
        
        # M2 = (A10+A11)*B00
        U_mat[i_base + 1, j_base + 0, d] = 1
        U_mat[i_base + 1, j_base + 1, d] = 1
        V_mat[j_base + 0, k_base + 0, d] = 1
        W_mat[k_base + 0, i_base + 1, d] = 1
        d += 1
        
        # M3 = A00*(B01-B11)
        U_mat[i_base + 0, j_base + 0, d] = 1
        V_mat[j_base + 0, k_base + 1, d] = 1
        V_mat[j_base + 1, k_base + 1, d] = -1
        W_mat[k_base + 1, i_base + 0, d] = 1
        d += 1
        
        # M4 = A11*(B10-B00)
        U_mat[i_base + 1, j_base + 1, d] = 1
        V_mat[j_base + 1, k_base + 0, d] = 1
        V_mat[j_base + 0, k_base + 0, d] = -1
        W_mat[k_base + 0, i_base + 1, d] = 1
        W_mat[k_base + 1, i_base + 1, d] = 1
        d += 1
        
        # M5 = (A00+A10)*B11
        U_mat[i_base + 0, j_base + 0, d] = 1
        U_mat[i_base + 1, j_base + 0, d] = 1
        V_mat[j_base + 1, k_base + 1, d] = 1
        W_mat[k_base + 0, i_base + 0, d] = -1
        W_mat[k_base + 1, i_base + 1, d] = 1
        d += 1
        
        # M6 = (A01-A10)*(B00+B10)
        U_mat[i_base + 0, j_base + 1, d] = 1
        U_mat[i_base + 1, j_base + 0, d] = -1
        V_mat[j_base + 0, k_base + 0, d] = 1
        V_mat[j_base + 1, k_base + 0, d] = 1
        W_mat[k_base + 1, i_base + 0, d] = 1
        W_mat[k_base + 1, i_base + 1, d] = 1
        d += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U_mat[i_base + 0, j_base + 1, d] = 1
        U_mat[i_base + 1, j_base + 1, d] = -1
        V_mat[j_base + 0, k_base + 1, d] = 1
        V_mat[j_base + 1, k_base + 1, d] = 1
        W_mat[k_base + 0, i_base + 0, d] = -1
        W_mat[k_base + 1, i_base + 0, d] = 1
        d += 1
        
        return d
    
    # We need to be more careful with the indexing.
    # Let me restart with a cleaner approach.
    
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # For a rank-1 term with factors U[:,d], V[:,d], W[:,d]:
    # Contribution to T[i*m+j, j*p+k, k*n+i] is U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d]
    
    # For a 2x2 @ 2x2 sub-problem with i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}:
    # We want: sum_d U[i*m+j, d] * V[j*p+k, d] * W[k*n+i, d] = 1 if (i,j,k) in the sub-problem
    
    # Let me define the Strassen decomposition more carefully.
    # For 2x2 @ 2x2: C[i,k] = sum_j A[i,j] * B[j,k] for i,j,k in {0,1}
    # In our tensor notation: T[i*m+j, j*p+k, k*n+i] = 1 for these indices
    
    # Strassen:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[1,0])*B[1,1]
    # M6 = (A[0,1]-A[1,0])*(B[0,0]+B[1,0])
    # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
    
    # C[0,0] = M1 + M4 - M5 - M7
    # C[0,1] = M3 + M1 + M7 - M6
    # C[1,0] = M5 + M2 - M1 - M4
    # C[1,1] = M6 + M3 + M4 + M1
    
    # For our tensor, the "output" index is k*n+i, so:
    # T[i*m+j, j*p+k, k*n+i] should equal 1 for the correct (i,j,k)
    
    # Let's define helper to add a 2x2@2x2 Strassen block
    def add_strassen_block(i0, j0, k0, U_arr, V_arr, W_arr, d_start):
        """Add rank-7 Strassen decomposition for 2x2@2x2 sub-problem.
        i0: first i index
        j0: first j index  
        k0: first k index
        Returns next d index.
        """
        d = d_start
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        U_arr[i0*m + j0, d] += 1
        U_arr[(i0+1)*m + (j0+1), d] += 1
        V_arr[j0*p + k0, d] += 1
        V_arr[(j0+1)*p + (k0+1), d] += 1
        W_arr[k0*n + i0, d] += 1
        W_arr[(k0+1)*n + (i0+1), d] += 1
        W_arr[(k0+1)*n + i0, d] -= 1
        W_arr[k0*n + (i0+1), d] -= 1
        d += 1
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        U_arr[(i0+1)*m + j0, d] += 1
        U_arr[(i0+1)*m + (j0+1), d] += 1
        V_arr[j0*p + k0, d] += 1
        W_arr[k0*n + (i0+1), d] += 1
        d += 1
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        U_arr[i0*m + j0, d] += 1
        V_arr[j0*p + (k0+1), d] += 1
        V_arr[(j0+1)*p + (k0+1), d] -= 1
        W_arr[(k0+1)*n + i0, d] += 1
        d += 1
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        U_arr[(i0+1)*m + (j0+1), d] += 1
        V_arr[(j0+1)*p + k0, d] += 1
        V_arr[j0*p + k0, d] -= 1
        W_arr[k0*n + (i0+1), d] += 1
        W_arr[(k0+1)*n + (i0+1), d] += 1
        d += 1
        
        # M5 = (A[0,0]+A[1,0])*B[1,1]
        U_arr[i0*m + j0, d] += 1
        U_arr[(i0+1)*m + j0, d] += 1
        V_arr[(j0+1)*p + (k0+1), d] += 1
        W_arr[k0*n + i0, d] -= 1
        W_arr[(k0+1)*n + (i0+1), d] += 1
        d += 1
        
        # M6 = (A[0,1]-A[1,0])*(B[0,0]+B[1,0])
        U_arr[i0*m + (j0+1), d] += 1
        U_arr[(i0+1)*m + j0, d] -= 1
        V_arr[j0*p + k0, d] += 1
        V_arr[(j0+1)*p + k0, d] += 1
        W_arr[(k0+1)*n + i0, d] += 1
        W_arr[(k0+1)*n + (i0+1), d] += 1
        d += 1
        
        # M7 = (A[0,1]-A[1,1])*(B[0,1]+B[1,1])
        U_arr[i0*m + (j0+1), d] += 1
        U_arr[(i0+1)*m + (j0+1), d] -= 1
        V_arr[j0*p + (k0+1), d] += 1
        V_arr[(j0+1)*p + (k0+1), d] += 1
        W_arr[k0*n + i0, d] -= 1
        W_arr[(k0+1)*n + i0, d] += 1
        d += 1
        
        return d
    
    def add_schoolbook_2x2_2x3(i0, j0, k0, U_arr, V_arr, W_arr, d_start):
        """Add schoolbook rank-12 decomposition for 2x2@2x3 sub-problem.
        i0: first i index
        j0: first j index
        k0: first k index
        Returns next d index.
        """
        d = d_start
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    U_arr[(i0+di)*m + (j0+dj), d] += 1
                    V_arr[(j0+dj)*p + (k0+dk), d] += 1
                    W_arr[(k0+dk)*n + (i0+di), d] += 1
                    d += 1
        return d
    
    # Now construct the rank-52 decomposition:
    # Split j into j_low={0,1} and j_high={2,3}
    # Split k into k_low={0,1,2,3} and k_high={4,5,6}
    
    # Sub-problem 1: i in {0,1}, j in {0,1}, k in {0,1} -> 2x2@2x2, Strassen rank 7
    d = add_strassen_block(0, 0, 0, U_52, V_52, W_52, d)
    
    # Sub-problem 2: i in {0,1}, j in {0,1}, k in {2,3} -> 2x2@2x2, Strassen rank 7
    d = add_strassen_block(0, 0, 2, U_52, V_52, W_52, d)
    
    # Sub-problem 3: i in {0,1}, j in {2,3}, k in {0,1} -> 2x2@2x2, Strassen rank 7
    d = add_strassen_block(0, 2, 0, U_52, V_52, W_52, d)
    
    # Sub-problem 4: i in {0,1}, j in {2,3}, k in {2,3} -> 2x2@2x2, Strassen rank 7
    d = add_strassen_block(0, 2, 2, U_52, V_52, W_52, d)
    
    # Sub-problem 5: i in {0,1}, j in {0,1}, k in {4,5,6} -> 2x2@2x3, schoolbook rank 12
    d = add_schoolbook_2x2_2x3(0, 0, 4, U_52, V_52, W_52, d)
    
    # Sub-problem 6: i in {0,1}, j in {2,3}, k in {4,5,6} -> 2x2@2x3, schoolbook rank 12
    d = add_schoolbook_2x2_2x3(0, 2, 4, U_52, V_52, W_52, d)
    
    # d should be 52
    assert d == rank_52, f"Expected d={rank_52}, got d={d}"
    
    # Verify the decomposition
    if check_decomposition(U_52, V_52, W_52):
        best_U, best_V, best_W = U_52, V_52, W_52
        best_rank = rank_52
    else:
        # If verification fails, use schoolbook
        pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
