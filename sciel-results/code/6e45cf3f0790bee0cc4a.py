"""Matrix multiplication 3x4x8 tensor decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Try to construct a rank-32 decomposition using Winograd-style convolution
    # For M_{3,4,8}, we can use the structure:
    # Split j dimension (4) into two 2s, and k dimension (8) into four 2s
    # Use Winograd for 2x2x2 blocks: rank 8 per block, but with sharing
    
    # Actually, let's use a direct algebraic construction.
    # For M_{1,2,2}, rank is 4 (schoolbook) or 4 (Winograd 2x2).
    # For M_{2,2,2}, rank is 7 (Winograd) vs 8 (schoolbook).
    # 
    # Strategy: Write A (3x4) and B (4x8) as:
    # A = [A0 A1] where A0, A1 are 3x2
    # B = [B0; B1] where B0, B1 are 2x8
    # C = A0*B0 + A1*B1
    # 
    # For each 3x2 times 2x8 product, use Winograd-style decomposition.
    # M_{3,2,8}: can we do better than 48?
    # M_{1,2,8}: rank 8 (each row: 1x2x8, Winograd 2x2 blocks -> 4 blocks * 4 = 16? No)
    # 1x2x8: A is 1x2, B is 2x8. C = a0*B0 + a1*B1 where B0,B1 are 1x8.
    # This is just 16 independent products, rank 16.
    # Winograd doesn't help for 1x2x8 since there's no convolution structure in a single row.
    # 
    # Let me think differently. Use the known rank for M_{3,4,4} = 16 (if it exists).
    # Actually, the rank of M_{3,4,4} is at most 48 (schoolbook).
    # Known: r(M_{2,2,2}) = 7, r(M_{3,3,3}) <= 22 (Landsberg), r(M_{2,3,3}) = 9?
    # 
    # Let me just try to use a simple rank reduction via linear dependence.
    # The multiplication tensor T_{i,j,k} has structure T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
    # where the (i,j) input and (j,k) output... wait, let me re-examine the indices.
    # 
    # U is (n*m, rank) = (12, rank), V is (m*p, rank) = (32, rank), W is (p*n, rank) = (24, rank)
    # The decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # 
    # Schoolbook: rank = 96, with U[i*m+j, i*m*j+k] = 1, V[j*p+k, i*m*j+k] = 1, W[k*n+i, i*m*j+k] = 1
    # 
    # To reduce rank, we need to find linear combinations.
    # For M_{1,2,2}: T = e0⊗e0⊗e0 + e0⊗e1⊗e1 (in appropriate basis)
    # Winograd: T = (1/2)(e0+e1)⊗(e0-e1)⊗... no.
    # Winograd for 2x2: T = sum of 4 rank-1 terms still, or can we do better?
    # M_{1,2,2} has rank 4? No: T = e0⊗e0⊗e0 + e0⊗e1⊗e1. That's 2 terms!
    # Wait: (AB)_{0,k} = A_{0,0}B_{0,k} + A_{0,1}B_{1,k}
    # T_{i,j,k} where i=0 only, j∈{0,1}, k∈{0,1}
    # T_{0,0,0} = 1, T_{0,1,1} = 1, others 0.
    # This is 2 rank-1 tensors: e_0 ⊗ e_0 ⊗ e_0 and e_0 ⊗ e_1 ⊗ e_1.
    # So r(M_{1,2,2}) = 2.
    # 
    # For M_{1,2,8}: T_{0,j,k} = 1 if... (AB)_{0,k} = sum_j A_{0,j} B_{j,k}
    # T_{0,j,k} has entries: for each j,k, it's 1 in the sense of the tensor.
    # Actually the multiplication tensor for M_{n,m,p} has T_{i,j,k} = δ_{j,k}? No.
    # Let me re-read: U[i*m+j], V[j*p+k], W[k*n+i].
    # The product C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k with the understanding that
    # T_{i,j,k} contributes A_{i,j} * B_{j,k} to C_{i,k}.
    # The tensor entries: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k.
    # So it's a 3-way tensor with T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
    # 
    # Number of non-zero entries = n*m*p = 96.
    # 
    # For M_{1,2,2}: n=1,m=2,p=2. T has entries at (i*m+j, j*p+k, k*n+i) = (j, 2j+k, 2k) for i=0.
    # j=0,k=0: (0,0,0); j=0,k=1: (0,1,2); j=1,k=0: (1,2,0); j=1,k=1: (1,3,2)
    # 4 non-zero entries. Rank of this tensor: can we decompose into 2?
    # e_0⊗e_0⊗e_0 + e_0⊗e_1⊗e_2 + e_1⊗e_2⊗e_0 + e_1⊗e_3⊗e_2
    # Can we write as sum of 2 rank-1? Probably not easily. Let's say rank 4.
    # 
    # OK, this is getting complex. Let me just implement a rank-32 construction
    # based on Winograd for the inner product, or return schoolbook if uncertain.
    # 
    # Actually, for the 120s time limit, let me just return the schoolbook decomposition
    # but with a clean implementation. The score will be 1.0 (same as seed).
    # But the task asks for a stronger algorithm...
    
    # Let me try a rank-48 decomposition by exploiting that p=8 is even.
    # M_{3,4,8} = M_{3,4,4} ⊕ M_{3,4,4} in some sense? No, because k ranges 0-7.
    # C_{i,k} for k=0..3 depends on all j, and k=4..7 depends on all j.
    # The k=0..3 part is a 3x4x4 multiplication, and k=4..7 is another 3x4x4.
    # So rank(M_{3,4,8}) <= 2 * rank(M_{3,4,4}).
    # If rank(M_{3,4,4}) = 16, then rank(M_{3,4,8}) <= 32.
    # 
    # rank(M_{3,4,4}): M_{1,4,4} has rank 16 (each of 16 products is independent? No.)
    # (AB)_{0,k} = sum_j A_{0,j} B_{j,k}. The tensor for M_{1,4,4} has T_{j,4j+k,k} = 1 for j,k in 0..3.
    # 16 non-zero entries. The rank is 16 in general (no algebraic structure to exploit for 1x4x4).
    # Actually, 1x4x4 is just a linear map from R^4 x R^4 -> R^4, which is the bilinear map
    # (a,b) -> ab (polynomial multiplication truncated to degree 3). The rank of this bilinear map
    # is the tensor rank of the multiplication tensor of R[x]/(x^4).
    # For R[x]/(x^m), the tensor rank is 2m-1? No, for 1xm, it's m (each output coordinate is independent).
    # Wait: (ab)_k = sum_j a_j b_{k-j}. For k=0: a_0 b_0. For k=1: a_0 b_1 + a_1 b_0. Etc.
    # The tensor T_{j, 4j+k, k} ... this is getting confused with indices.
    # 
    # Let me just go with schoolbook rank 96. It's safe and correct.
    
    rank = n * m * p  # 96
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
# EVOLVE-BLOCK-END
