"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 120

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We want to find rank-r decomposition:
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # where U is (n*m, r), V is (m*p, r), W is (p*n, r)

    # Strategy: Use Strassen-like decomposition by partitioning
    # Partition m=5 into 4+1, p=6 into 4+2 (or 4+1+1)
    # Actually, let's try partitioning m into 4+1 and p into 4+2
    
    # For a 4x4x4 matrix multiplication, Strassen gives rank 7
    # For the full problem, we can partition:
    # A is 4x5, B is 5x6
    # Partition A into A1 (4x4) and A2 (4x1)
    # Partition B into B1 (4x6) and B2 (1x6)
    # C = A1*B1 + A2*B2
    
    # A1*B1 is 4x4 times 4x6. We can further partition B1 into B1a (4x4) and B1b (4x2)
    # A1*B1a is 4x4*4x4 -> Strassen rank 7
    # A1*B1b is 4x4*4x2 -> can use Strassen-like for rectangular
    # A2*B2 is 4x1*1x6 -> rank 4 (outer product)
    
    # Let's be more careful. The tensor for 4x5*5x6:
    # T[i*m+j, j*p+k, k*n+i] = 1 for i in 0..3, j in 0..4, k in 0..5
    
    # Partition j into j1 (0..3) and j2 (4)
    # Partition k into k1 (0..3), k2 (4..5)
    
    # Block 1: i in 0..3, j in 0..3, k in 0..3 -> 4x4x4 tensor, Strassen rank 7
    # Block 2: i in 0..3, j in 0..3, k in 4..5 -> 4x4x2 tensor
    # Block 3: i in 0..3, j in 4, k in 0..3 -> 4x1x4 tensor  
    # Block 4: i in 0..3, j in 4, k in 4..5 -> 4x1x2 tensor
    
    # For 4x4x4: Strassen rank 7
    # For 4x4x2: We can view this as multiplying 4x4 by 4x2. 
    #   Partition the 4x2 into two 4x1 columns. Each 4x4*4x1 can be done with rank 4 (each column is a separate scalar multiplication of 4x4 by vector... actually no)
    #   Actually, 4x4 * 4x2: result is 4x2. 
    #   T2[i*4+j, j*2+k, k*4+i] = 1 for j in 0..3, k in 0..1
    #   This has 4*4*2 = 32 nonzero entries. 
    #   We can use a decomposition. Let's think: for each k (2 values), we have a 4x4*4x1 multiplication.
    #   4x4*4x1: T[i*4+j, j*1+0, 0*4+i] = T[i*4+j, j, i] = 1. This is just extracting column i of the 4x4 matrix multiplied... 
    #   Actually this is the "diagonal" tensor. Rank of this: it's the tensor for multiplying 4x4 by a 4-vector (treating 4x1 as vector).
    #   This is equivalent to: for each i, sum_j A[i,j] * B[j] gives C[i]. 
    #   This can be done with rank 4: for each i, U[i*4+j, i] = A[i,j], V[j, i] = B[j], W[i, i] = C[i]. 
    #   Wait, that's not right. Let me re-index.
    #   For 4x4 * 4x1: A is 4x4, B is 4x1, C is 4x1.
    #   C[i] = sum_j A[i,j] * B[j]
    #   Tensor: T[i*4+j, j, i] = 1 (using row-major for A, B is just j, C is just i)
    #   Actually with the convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    #   For 4x4*4x1: n=4, m=4, p=1. U[i*4+j, r], V[j*1+0, r], W[0*4+i, r] = W[i, r]
    #   T[i*4+j, j, i] = 1
    #   Rank: This is the "matrix-vector" multiplication tensor. It has rank 4 (one for each output i, but actually we need to sum over j).
    #   Actually, rank is 4: for each i, we can't separate because we sum over j. 
    #   Let's use the standard: this tensor has rank 4. 
    #   U[i*4+j, i] = 1 if... no. 
    #   Let's use: for each j, U[i*4+j, j] = 1, V[j, j] = 1, W[i, j] = 1 if i==j? No.
    #   C[i] = sum_j A[i,j] * B[j]. 
    #   This is a bilinear form. The tensor T[i*4+j, j, i] = 1.
    #   This can be decomposed as: for each j, a term where U[i*4+j, j] = A[i,j], V[j, j] = B[j], W[i, j] = delta(i,j)? No.
    #   Actually: sum_j A[i,j] * B[j] * delta_{i,j}? No, that's not right.
    #   Let me just use the fact that 4x4*4x1 has rank 4. 
    #   Decomposition: for j=0..3: U[i*4+j, j] = 1, V[j, j] = 1, W[i, j] = 1 if i == j? No.
    #   Hmm, let me think differently. 
    #   C[i] = sum_j A[i,j] * B[j]. 
    #   This is 4 separate dot products. Each dot product is 4x4 * 4x1 -> scalar. 
    #   Each 4x4*4x1 -> scalar has rank 4 (it's just a vector of 4 elements times a vector of 4 elements, but here it's a matrix row times a column).
    #   Actually, a single dot product of two 4-vectors has rank 4: sum_j u_j * v_j = sum_j (u_j * 1) * (1 * v_j), so rank 4.
    #   But here we have 4 such dot products (one for each i). They share the B vector.
    #   So total rank for 4x4*4x1 is 4*4 = 16? No, that's schoolbook.
    #   Actually, it can be done better. The tensor is T[i*4+j, j, i] = 1.
    #   This is the "transpose" of a 4x4 matrix times a vector. 
    #   I think the rank is 4. Let's use: for each i, we compute C[i] = sum_j A[i,j] * B[j].
    #   We can write this as: C[i] = sum_j (A[i,j] * B[j]).
    #   This is 16 scalar multiplications in schoolbook. 
    #   But we can use: for each j, define S_j = sum_i A[i,j] * B[j]? No, that doesn't work.
    #   Let me just use schoolbook for the small blocks and Strassen for the 4x4x4 block.
    
    # Let's use a simpler approach: 
    # Total rank = rank(4x4x4) + rank(4x4x2) + rank(4x1x4) + rank(4x1x2)
    # rank(4x4x4) = 7 (Strassen)
    # rank(4x4x2): schoolbook is 4*4*2 = 32. Can we do better?
    #   4x4 * 4x2: We can partition the 4x2 into two 4x1. Each 4x4*4x1 has rank... 
    #   Let's just use schoolbook for now: 32
    # rank(4x1x4): 4*1*4 = 16 schoolbook. 
    #   4x1 * 1x4: This is an outer product of a 4-vector and a 4-vector. Rank 4.
    #   T[i*1+0, 0*4+k, k*4+i] = T[i, k, k*4+i] = 1 for i in 0..3, k in 0..3
    #   C[i,k] = A[i] * B[k]. This is an outer product, rank 4.
    # rank(4x1x2): 4*1*2 = 8 schoolbook. 
    #   4x1 * 1x2: outer product of 4-vector and 2-vector. Rank 4.
    #   T[i, k, k*4+i] = 1 for i in 0..3, k in 0..1
    #   C[i,k] = A[i] * B[k]. Rank 4.
    
    # So total rank with this partition: 7 + 32 + 4 + 4 = 47
    
    # But wait, I need to verify the Strassen decomposition for 4x4x4.
    # Strassen's algorithm for 2x2x2 has rank 7. For 4x4x4, we can use Strassen recursively:
    # 4x4 = 2*2x2x2, so rank 7^2 = 49? No, that's not right.
    # Actually, Strassen's algorithm gives rank 7 for 2x2x2. For 4x4x4, using Strassen recursively gives 7^2 = 49.
    # But the optimal rank for 4x4x4 is unknown, but it's at most 49.
    # However, there might be better decompositions.
    
    # Let me reconsider. The task is to find a rank-minimal exact decomposition.
    # The schoolbook rank is 120. 
    # Using Strassen for the 4x4x4 block: if we use the standard Strassen decomposition, 
    # the 4x4x4 block (which is actually two 2x2x2 blocks combined) would have rank 7*7 = 49? 
    # No, Strassen's algorithm for 2x2 matrices uses 7 multiplications. For 4x4 matrices, 
    # we can partition into four 2x2 blocks and apply Strassen, giving 7*7 = 49 multiplications.
    # But that's for the full 4x4*4x4 multiplication. 
    # In our case, the 4x4x4 block is exactly a 4x4*4x4 multiplication (with n=4, m=4, p=4).
    # So its rank is at most 49 using Strassen.
    
    # Let's compute the total rank:
    # Block 1 (4x4x4): rank 49 (Strassen)
    # Block 2 (4x4x2): schoolbook 32
    # Block 3 (4x1x4): rank 4 (outer product)
    # Block 4 (4x1x2): rank 4 (outer product)
    # Total: 49 + 32 + 4 + 4 = 89
    
    # That's better than 120. Let's implement this.
    
    # Actually, let me reconsider the block decomposition more carefully.
    # The tensor T[i*m+j, j*p+k, k*n+i] = 1
    # With n=4, m=5, p=6.
    # i in 0..3, j in 0..4, k in 0..5
    
    # Partition j: j1 = j (0..3), j2 = 4
    # Partition k: k1 = k (0..3), k2 = k-4 (4..5, so k2 in 0..1)
    
    # Block (j1, k1): i in 0..3, j1 in 0..3, k1 in 0..3
    #   This is a 4x4x4 tensor. T[i*5+j1, j1*6+k1, k1*4+i] = 1
    #   But this is NOT a standard 4x4x4 matrix multiplication tensor because of the strides.
    #   The standard 4x4x4 tensor has T[i*4+j, j*4+k, k*4+i] = 1.
    #   Here we have T[i*5+j, j*6+k, k*4+i] = 1.
    #   These are different tensors! The Strassen decomposition doesn't directly apply.
    
    # So my approach is flawed. The blocks are not standard matrix multiplication tensors.
    
    # Let me think of a different approach.
    # We need to find a rank-r decomposition of the tensor T of shape (20, 30, 24).
    # T[a, b, c] = 1 if there exist i, j, k such that a = i*5+j, b = j*6+k, c = k*4+i, and T[a,b,c] = 1.
    # Otherwise T[a,b,c] = 0.
    
    # The number of non-zero entries is 4*5*6 = 120.
    
    # One approach: use the fact that this tensor is a "matrix multiplication" tensor and 
    # apply known results. But I don't have access to a database of minimal ranks.
    
    # Let me try a different partition that gives standard matrix multiplication tensors.
    # If I partition n=4 into 2+2, m=5 into 2+2+1, p=6 into 3+3, then:
    # Block (2x2, 2x2, 2x2): standard 2x2x2 tensor, Strassen rank 7
    # But the other blocks won't be standard either.
    
    # Actually, the key insight is that the tensor T is a "slice" of a larger tensor, 
    # and the non-zero entries form a specific pattern.
    
    # Let me try a direct approach: use the schoolbook decomposition but try to find 
    # linear dependencies among the 120 basis tensors to reduce the rank.
    
    # The 120 basis tensors are: for each (i,j,k), the tensor E_{i,j,k} which has 
    # T[i*5+j, j*6+k, k*4+i] = 1 and 0 elsewhere.
    # We want to find the minimum number of these (or linear combinations) that span the same tensor.
    
    # Actually, the rank of the tensor is the minimum r such that T can be written as 
    # sum_{r} u_r * v_r * w_r where u_r, v_r, w_r are vectors.
    
    # This is equivalent to finding the minimum number of rank-1 tensors that sum to T.
    
    # One way to approach this: use the fact that T is a "matrix multiplication" tensor 
    # and apply the "Strassen" idea in a different way.
    
    # Let me try: partition the middle index j (0..4) into two parts: j in 0..2 and j in 3..4.
    # And partition k (0..5) into k in 0..2 and k in 3..5.
    
    # This is getting complicated. Let me just use the schoolbook decomposition for now 
    # and see if I can find any improvements within the time limit.
    
    # Actually, let me try a simpler improvement: use the fact that for the 4x1*1x6 block 
    # (j=4, k in 0..5), the tensor is T[i*5+4, 4*6+k, k*4+i] = T[i*5+4, 24+k, k*4+i] = 1.
    # This is an outer product: C[i,k] = A[i] * B[k].
    # The rank of this sub-tensor is 4 (one for each i).
    # Similarly, for the 4x5*5x1 block (k=5, j in 0..4), the tensor is 
    # T[i*5+j, j*6+5, 5*4+i] = T[i*5+j, j*6+5, 20+i] = 1.
    # This is C[i] = sum_j A[i,j] * B[j]. This has rank 4 (one for each i, but with sum over j).
    # Actually, this is a 4x5*5x1 multiplication, which has rank 4 (each output i is a dot product).
    # Wait, no. The rank of the 4x5*5x1 tensor is 4 (it's the "matrix-vector" product).
    
    # Let me just implement the schoolbook decomposition and return it, since finding a 
    # better decomposition is complex and I'm running out of time.
    
    # Actually, let me try one more idea: use the "Strassen" decomposition for the 2x2x2 blocks.
    # Partition n=4 into 2+2, m=5 into 2+2+1, p=6 into 3+3.
    # This gives 2*3*3 = 18 blocks, each of which is a small matrix multiplication.
    # But this is getting too complex to implement correctly in the time available.
    
    # Let me just return the schoolbook decomposition.
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
    
    # Check if we're within time limit
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
