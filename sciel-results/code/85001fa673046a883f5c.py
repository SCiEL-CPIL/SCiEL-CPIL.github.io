"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _build_strassen_block(n, m, p):
    """
    Build a rank-(n*m*p - 1) decomposition for the 2x2x2 block using Strassen,
    combined with trivial multiplication for the remaining parts.
    
    For n=2, m=3, p=2:
    - The matrix is 2x3 times 3x2 -> 2x2 result
    - We can split the 3 columns into 2+1: use Strassen on the 2x2 part (rank 7)
      and trivial on the 2x1 part (rank 2), total rank 9 per block.
    - But wait, the task is 2x3 * 3x10. Let me think about this differently.
    
    Actually, for 2x3 * 3x10:
    - We can decompose the 3x10 into 5 blocks of 3x2
    - For each 3x2 block, we multiply 2x3 * 3x2 -> 2x2
    - Using Strassen on the 2x2 part: split the 3 columns as 2+1
      - 2x2 part: Strassen rank 7
      - 2x1 part: trivial rank 2
      - Total per 3x2 block: 9
    - 5 blocks * 9 = 45 total rank
    
    Wait, let me reconsider. The standard approach:
    For C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    For n=2, m=3, p=10:
    We can use the fact that for the 2x2 sub-problem (taking 2 columns of B),
    Strassen gives rank 7 instead of 8.
    
    Let's split B into 5 blocks of 3x2. For each block:
    C[:, 2b:2b+2] = A * B[:, 2b:2b+2]
    
    For A (2x3) and B_block (3x2):
    Split A into A1 (2x2, first 2 cols) and A2 (2x1, last col)
    Split B_block into B1 (3x1, first col) and B2 (3x1, second col)
    
    C[:,0] = A1[:,0] * B1 + A1[:,1] * B1 + A2 * B1
    C[:,1] = A1[:,0] * B2 + A1[:,1] * B2 + A2 * B2
    
    Hmm, this is getting complex. Let me use a cleaner approach.
    
    For 2x2 matrix multiplication (2x2 * 2x2 -> 2x2), Strassen uses 7 multiplications:
    M1 = (A11 + A22) * (B11 + B22)
    M2 = (A21 + A22) * B11
    M3 = A11 * (B12 - B22)
    M4 = A22 * (B21 - B11)
    M5 = (A11 + A12) * B22
    M6 = (A21 - A11) * (B11 + B12)
    M7 = (A12 - A22) * (B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    For our case 2x3 * 3x2:
    Let A = [[a11, a12, a13], [a21, a22, a23]]
    Let B = [[b11, b12], [b21, b22], [b31, b32]]
    
    C11 = a11*b11 + a12*b21 + a13*b31
    C12 = a11*b12 + a12*b22 + a13*b32
    C21 = a21*b11 + a22*b21 + a23*b31
    C22 = a21*b12 + a22*b22 + a23*b32
    
    We can write this as:
    C11 = (a11+a12)*b11 + a12*b21 + a13*b31  -- not quite
    
    Let me try a different decomposition. Split A into 2x2 + 2x1:
    A = [A_left (2x2), A_right (2x1)]
    B = [B_top (2x2), B_bottom (2x2)] -- wait, B is 3x2
    
    Actually, let me just use the known result: for 2x3 * 3x2, the rank is 9.
    One construction:
    
    Let's define 9 bilinear forms:
    f1 = (a11+a12) * (b11+b21)  -- hmm, this doesn't work directly
    
    Let me look at this more carefully. The tensor for 2x3*3x2 has 4*3*2 = 24 entries.
    The rank is known to be 9 (since 2x2x2 has rank 7, and we have one extra dimension).
    
    Actually, I think the cleanest way is:
    For each pair of output columns (2 columns), we have a 2x3 * 3x2 problem.
    The rank of this is at most 9 (Strassen on 2x2 part + 1 trivial).
    
    Let me construct it explicitly. For 2x3 * 3x2:
    A = [[a, b, c], [d, e, f]]
    B = [[g, h], [i, j], [k, l]]
    
    C11 = a*g + b*i + c*k
    C12 = a*h + b*j + c*l
    C21 = d*g + e*i + f*k
    C22 = d*h + e*j + f*l
    
    We can write:
    Let A' = [[a, b], [d, e]] (2x2), a'' = [c, f]^T (2x1)
    Let B' = [[g, h], [i, j]] (2x2), B'' = [k, l]^T (2x1)
    
    C = A' * B' + a'' * B''
    
    For A'*B', use Strassen (rank 7).
    For a''*B'', this is outer product of two 2-vectors, rank 2.
    
    Total: 7 + 2 = 9.
    
    Now let's write the Strassen terms for A'*B':
    M1 = (a+e)*(g+i)
    M2 = (d+e)*g
    M3 = a*(j-i)
    M4 = e*(h-g)
    M5 = (a+b)*j
    M6 = (d-a)*(g+h)
    M7 = (b-e)*(i+h)
    
    C11' = M1 + M4 - M5 + M7
    C12' = M3 + M5
    C21' = M2 + M4
    C22' = M1 - M2 + M3 + M6
    
    And for a''*B'':
    N1 = c*k
    N2 = c*l
    N3 = f*k
    N4 = f*l
    
    C11 = C11' + N1 = M1 + M4 - M5 + M7 + N1
    C12 = C12' + N2 = M3 + M5 + N2
    C21 = C21' + N3 = M2 + M4 + N3
    C22 = C22' + N4 = M1 - M2 + M3 + M6 + N4
    
    So the 9 bilinear forms are:
    M1 = (a+e)*(g+i)
    M2 = (d+e)*g
    M3 = a*(j-i)
    M4 = e*(h-g)
    M5 = (a+b)*j
    M6 = (d-a)*(g+h)
    M7 = (b-e)*(i+h)
    N1 = c*k
    N2 = c*l
    N3 = f*k
    N4 = f*l
    
    Now for the full 2x3 * 3x10 problem:
    Split B into 5 blocks of 3x2.
    For block b (0-indexed), B_block = B[:, 2b:2b+2]
    
    The U matrix has 2*3=6 rows, V has 3*10=30 rows, W has 10*2=20 rows.
    Rank = 5 * 9 = 45.
    
    For each block b, we have 9 terms. Let's index them:
    Term 0: M1 = (a+e)*(g+i) -> U rows for a,e; V rows for g,i; W rows for C11,C12 (via M1)
    ...
    
    Let me be more careful with the indexing.
    
    A[i,j] for i in 0,1 and j in 0,1,2
    B[j,k] for j in 0,1,2 and k in 0,...,9
    C[i,k] for i in 0,1 and k in 0,...,9
    
    For block b, let k0 = 2*b, k1 = 2*b+1
    A_left: A[0,0]=a, A[0,1]=b, A[1,0]=d, A[1,1]=e
    A_right: A[0,2]=c, A[1,2]=f
    B_top: B[0,k0]=g, B[0,k1]=h, B[1,k0]=i, B[1,k1]=j
    B_bottom: B[2,k0]=k, B[2,k1]=l
    
    M1 = (a+e)*(g+i):
      U: row A[0,0] (idx 0) += 1, row A[1,1] (idx 1) += 1
      V: row B[0,k0] (idx k0) += 1, row B[1,k0] (idx 3+k0) += 1  -- wait, B is 3x10, so B[j,k] is at row j*10+k
      V: row B[0,k0] = 0*10+k0 = k0, row B[1,k0] = 1*10+k0 = 10+k0
      W: C[0,k0] gets +1, C[1,k0] gets +1? No wait.
      
      C11' contribution from M1: +1 to C[0,k0]
      C22' contribution from M1: +1 to C[1,k1]
      
      So W: row C[0,k0] = k0*2+0 = 2*k0, row C[1,k1] = k1*2+1 = 2*(2*b+1)+1 = 4*b+3
      
      Hmm wait, W[k*n+i, r] where n=2. So W index for C[i,k] is k*n+i = 2*k+i.
      
      C[0,k0]: W index = 2*k0+0 = 4*b
      C[1,k1]: W index = 2*k1+1 = 2*(2*b+1)+1 = 4*b+3
      
      Let me redo this more carefully.
    """
    n, m, p = 2, 3, 10
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 9
    total_rank = num_blocks * rank_per_block
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 45
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 45
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 45
    
    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        col_base = b * rank_per_block
        
        # A indices: A[i,j] -> U row i*m+j = i*3+j
        # B indices: B[j,k] -> V row j*p+k = j*10+k
        # C indices: C[i,k] -> W row k*n+i = k*2+i
        
        # M1 = (a+e)*(g+i)
        # a = A[0,0], e = A[1,1], g = B[0,k0], i = B[1,k0]
        # C[0,k0] += 1, C[1,k1] += 1
        col = col_base + 0
        U[0*3+0][col] += 1   # a
        U[1*3+1][col] += 1   # e
        V[0*10+k0][col] += 1 # g
        V[1*10+k0][col] += 1 # i
        W[k0*2+0][col] += 1  # C[0,k0]
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M2 = (d+e)*g
        # d = A[1,0], e = A[1,1], g = B[0,k0]
        # C[1,k0] += 1, C[1,k1] -= 1
        col = col_base + 1
        U[1*3+0][col] += 1   # d
        U[1*3+1][col] += 1   # e
        V[0*10+k0][col] += 1 # g
        W[k0*2+1][col] += 1  # C[1,k0]
        W[k1*2+1][col] -= 1  # C[1,k1]
        
        # M3 = a*(j-i)
        # a = A[0,0], j = B[1,k1], i = B[1,k0]
        # C[0,k1] += 1, C[1,k1] += 1
        col = col_base + 2
        U[0*3+0][col] += 1   # a
        V[1*10+k1][col] += 1 # j
        V[1*10+k0][col] -= 1 # i
        W[k1*2+0][col] += 1  # C[0,k1]
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M4 = e*(h-g)
        # e = A[1,1], h = B[0,k1], g = B[0,k0]
        # C[0,k0] += 1, C[1,k0] += 1
        col = col_base + 3
        U[1*3+1][col] += 1   # e
        V[0*10+k1][col] += 1 # h
        V[0*10+k0][col] -= 1 # g
        W[k0*2+0][col] += 1  # C[0,k0]
        W[k0*2+1][col] += 1  # C[1,k0]
        
        # M5 = (a+b)*j
        # a = A[0,0], b = A[0,1], j = B[1,k1]
        # C[0,k0] -= 1, C[0,k1] += 1
        col = col_base + 4
        U[0*3+0][col] += 1   # a
        U[0*3+1][col] += 1   # b
        V[1*10+k1][col] += 1 # j
        W[k0*2+0][col] -= 1  # C[0,k0]
        W[k1*2+0][col] += 1  # C[0,k1]
        
        # M6 = (d-a)*(g+h)
        # d = A[1,0], a = A[0,0], g = B[0,k0], h = B[0,k1]
        # C[1,k1] += 1
        col = col_base + 5
        U[1*3+0][col] += 1   # d
        U[0*3+0][col] -= 1   # a
        V[0*10+k0][col] += 1 # g
        V[0*10+k1][col] += 1 # h
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M7 = (b-e)*(i+h)
        # b = A[0,1], e = A[1,1], i = B[1,k0], h = B[0,k1]
        # C[0,k0] += 1
        col = col_base + 6
        U[0*3+1][col] += 1   # b
        U[1*3+1][col] -= 1   # e
        V[1*10+k0][col] += 1 # i
        V[0*10+k1][col] += 1 # h
        W[k0*2+0][col] += 1  # C[0,k0]
        
        # N1 = c*k
        # c = A[0,2], k = B[2,k0]
        # C[0,k0] += 1
        col = col_base + 7
        U[0*3+2][col] += 1   # c
        V[2*10+k0][col] += 1 # k
        W[k0*2+0][col] += 1  # C[0,k0]
        
        # N2 = c*l
        # c = A[0,2], l = B[2,k1]
        # C[0,k1] += 1
        col = col_base + 8
        U[0*3+2][col] += 1   # c
        V[2*10+k1][col] += 1 # l
        W[k1*2+0][col] += 1  # C[0,k1]
        
        # N3 = f*k
        # f = A[1,2], k = B[2,k0]
        # C[1,k0] += 1
        col = col_base + 9 - 1  # col_base + 8 already used, wait
        # Actually N3 is col_base + 8? No, let me recount.
        # I have 9 terms: M1-M7 (7 terms) + N1-N4 (4 terms) = 11? No, that's wrong.
        # Strassen is 7, outer product is 4, total 11? But I said rank 9.
        
    # I made an error. Let me recount.
    # Strassen: 7 multiplications
    # a'' * B'': outer product of 2x1 and 1x2, which is 2*2=4 terms? 
    # No, a'' is 2x1, B'' is 2x1 (column vectors). Their product is not defined as outer product in the usual sense for matrix mult.
    # Actually, a'' * B'' where a'' is 2x1 and B'' is 1x2 would be 2x2. But here B'' is 2x1 (the bottom row of B_block).
    # 
    # Wait, I think I confused myself. Let me restart the decomposition.
    
    # A is 2x3, B_block is 3x2.
    # A = [A1 A2] where A1 is 2x2 (cols 0,1) and A2 is 2x1 (col 2)
    # B_block = [B1; B2] where B1 is 2x2 (rows 0,1) and B2 is 1x2 (row 2)
    # 
    # C = A1*B1 + A2*B2
    # A2*B2 is 2x1 * 1x2 = 2x2, which is an outer product.
    # A2 = [c; f], B2 = [k, l]
    # A2*B2 = [[c*k, c*l], [f*k, f*l]]
    # This requires 4 multiplications: c*k, c*l, f*k, f*l
    # 
    # So total = 7 (Strassen) + 4 (outer product) = 11? 
    # But that's worse than schoolbook (12). Hmm.
    
    # Actually, the outer product of two vectors of length 2 has rank 4 (since it's a rank-1 matrix, but we need to decompose it into bilinear forms for the tensor).
    # Wait, no. The tensor for outer product of u (length 2) and v (length 2) to get a 2x2 matrix:
    # M[i,j] = u[i]*v[j]
    # This is just 4 independent products, so rank 4.
    # 
    # So total per block = 7 + 4 = 11. That's not better than schoolbook 12.
    # 
    # Hmm, maybe I need a different approach. Let me think about whether there's a rank-9 decomposition for 2x3*3x2.
    
    # Actually, I recall that the rank of the 2x3*3x2 matrix multiplication tensor is indeed 9.
    # Let me try a different construction.
    
    # Alternative: Use the fact that 2x3*3x2 can be embedded in 3x3*3x3 (rank 21) or use other known results.
    
    # Actually, let me just verify: is rank(2x3*3x2) = 9?
    # The trivial rank is 2*3*2 = 12.
    # I believe the rank is 9 based on known results for small matrix multiplication tensors.
    
    # Let me try to construct it differently.
    # C[0,0] = a*g + b*i + c*k
    # C[0,1] = a*h + b*j + c*l
    # C[1,0] = d*g + e*i + f*k
    # C[1,1] = d*h + e*j + f*l
    
    # 9 bilinear forms. Let me try:
    # f1 = (a+d)*(g+i) -- no, this mixes rows which isn't standard.
    
    # Actually, I think the issue is that I should just use the schoolbook method but with a smarter grouping.
    # Or maybe the rank is actually 12 and I was wrong about 9.
    
    # Let me just go with the schoolbook approach for now and see if I can find improvements.
    # Actually, the knowledge says rank 55 was achieved (from 60), which is 5 blocks * 11.
    # So 11 per block is the known result. Let me implement that.
    
    return None  # Placeholder, will redo


def construct():
    """
    Construct a rank-55 decomposition for 2x3 * 3x10 matrix multiplication.
    Uses Strassen (7) + outer product (4) = 11 per 3x2 block, 5 blocks = 55.
    """
    n, m, p = 2, 3, 10
    num_blocks = p // 2  # 5 blocks
    rank_per_block = 11
    total_rank = num_blocks * rank_per_block  # 55
    
    U = [[0] * total_rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 55
    
    for b in range(num_blocks):
        k0 = 2 * b
        k1 = 2 * b + 1
        col_base = b * rank_per_block
        
        # A[i,j] -> U row i*3+j
        # B[j,k] -> V row j*10+k
        # C[i,k] -> W row k*2+i
        
        # Strassen terms for A1 (2x2) * B1 (2x2):
        # A1 = [[a, b], [d, e]], B1 = [[g, h], [i, j]]
        # a=A[0,0], b=A[0,1], d=A[1,0], e=A[1,1]
        # g=B[0,k0], h=B[0,k1], i=B[1,k0], j=B[1,k1]
        
        # M1 = (a+e)*(g+i)
        col = col_base + 0
        U[0*3+0][col] += 1   # a
        U[1*3+1][col] += 1   # e
        V[0*10+k0][col] += 1 # g
        V[1*10+k0][col] += 1 # i
        W[k0*2+0][col] += 1  # C[0,k0]
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M2 = (d+e)*g
        col = col_base + 1
        U[1*3+0][col] += 1   # d
        U[1*3+1][col] += 1   # e
        V[0*10+k0][col] += 1 # g
        W[k0*2+1][col] += 1  # C[1,k0]
        W[k1*2+1][col] -= 1  # C[1,k1]
        
        # M3 = a*(j-i)
        col = col_base + 2
        U[0*3+0][col] += 1   # a
        V[1*10+k1][col] += 1 # j
        V[1*10+k0][col] -= 1 # i
        W[k1*2+0][col] += 1  # C[0,k1]
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M4 = e*(h-g)
        col = col_base + 3
        U[1*3+1][col] += 1   # e
        V[0*10+k1][col] += 1 # h
        V[0*10+k0][col] -= 1 # g
        W[k0*2+0][col] += 1  # C[0,k0]
        W[k0*2+1][col] += 1  # C[1,k0]
        
        # M5 = (a+b)*j
        col = col_base + 4
        U[0*3+0][col] += 1   # a
        U[0*3+1][col] += 1   # b
        V[1*10+k1][col] += 1 # j
        W[k0*2+0][col] -= 1  # C[0,k0]
        W[k1*2+0][col] += 1  # C[0,k1]
        
        # M6 = (d-a)*(g+h)
        col = col_base + 5
        U[1*3+0][col] += 1   # d
        U[0*3+0][col] -= 1   # a
        V[0*10+k0][col] += 1 # g
        V[0*10+k1][col] += 1 # h
        W[k1*2+1][col] += 1  # C[1,k1]
        
        # M7 = (b-e)*(i+h)
        col = col_base + 6
        U[0*3+1][col] += 1   # b
        U[1*3+1][col] -= 1   # e
        V[1*10+k0][col] += 1 # i
        V[0*10+k1][col] += 1 # h
        W[k0*2+0][col] += 1  # C[0,k0]
        
        # Outer product terms for A2 (2x1) * B2 (1x2):
        # A2 = [c; f], B2 = [k, l]
        # c=A[0,2], f=A[1,2], k=B[2,k0], l=B[2,k1]
        
        # N1 = c*k
        col = col_base + 7
        U[0*3+2][col] += 1   # c
        V[2*10+k0][col] += 1 # k
        W[k0*2+0][col] += 1  # C[0,k0]
        
        # N2 = c*l
        col = col_base + 8
        U[0*3+2][col] += 1   # c
        V[2*10+k1][col] += 1 # l
        W[k1*2+0][col] += 1  # C[0,k1]
        
        # N3 = f*k
        col = col_base + 9
        U[1*3+2][col] += 1   # f
        V[2*10+k0][col] += 1 # k
        W[k0*2+1][col] += 1  # C[1,k0]
        
        # N4 = f*l
        col = col_base + 10
        U[1*3+2][col] += 1   # f
        V[2*10+k1][col] += 1 # l
        W[k1*2+1][col] += 1  # C[1,k1]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
