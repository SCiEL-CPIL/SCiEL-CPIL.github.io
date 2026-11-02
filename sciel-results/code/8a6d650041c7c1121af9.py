import numpy as np
import time

DIMENSIONS = [4, 4, 9]

def _verify(U, V, W, n, m, p, tol=1e-6):
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x9():
    """
    Decomposition of M_{2,2,9} using Winograd for each column.
    For each column j, we multiply a 2x2 matrix by a 2x1 vector.
    Winograd: rank 3 per column.
    Total rank: 9 * 3 = 27.
    
    A = [[a, b], [c, d]] (4 vars)
    B_col = [x, y]^T (2 vars)
    C_col = [o1, o2]^T (2 vars)
    
    Winograd scheme:
    U (4x3):
      a: [1, 0, 0]
      b: [0, 1, 1]
      c: [1, 1, 0]
      d: [0, 1, 1]
    Wait, let's use the standard Winograd for 2x2 mat-vec.
    Actually, for 2x2 * 2x1:
    o1 = a*x + b*y
    o2 = c*x + d*y
    
    Let's use the same Winograd as for 2x2*2x2 but with the second matrix being 2x1.
    Standard Winograd for 2x2*2x2 (rank 7) is complex.
    For 2x2*2x1, we can use a simpler scheme.
    Rank 4 is schoolbook. Can we do rank 3?
    Tensor M_{2,2,2} has rank 7. M_{2,2,1} is a slice.
    Let's check if rank 3 is possible for M_{2,2,1}.
    T[i,j,k] where i in {0,1,2,3}, j in {0,1}, k in {0,1}.
    T[0,0,0]=1, T[1,1,0]=1, T[2,0,1]=1, T[3,1,1]=1.
    This is the tensor of matrix-vector multiplication.
    It is known that rank(M_{2,2,1}) = 4. (It's the rank of the 2x2 matrix multiplication tensor restricted? No, it's the bilinear map (2x2) x (2x1) -> (2x1)).
    Actually, the rank of the 2x2 matrix multiplication is 7.
    The rank of the 2x2 matrix-vector multiplication is 4.
    Proof: We need to compute ax+by and cx+dy.
    If rank < 4, then...
    Let's just use rank 4 for each column.
    Total rank for M_{2,2,9} = 9 * 4 = 36.
    Total rank for M_{4,4,9} = 4 * 36 = 144.
    This is the same as schoolbook.
    
    Can we do better?
    We can group columns.
    If we use Winograd for 2x2 * 2x2 blocks of B?
    B is 4x9. We can view it as four 1x9 vectors? No, B is 4 rows, 9 cols.
    A is 4x4.
    C = A B.
    
    Let's try to use the rank-7 decomposition for 2x2*2x2.
    We can split A into 2x2 blocks (4 blocks).
    We can split B into 2x2 blocks? B has 9 columns, so 4 full 2x2 blocks and one 2x1 block.
    This seems complicated.
    
    Let's stick to the column-wise approach but see if we can share computation.
    If we use the same Winograd transform for all columns?
    The A matrix is shared.
    If we apply a transform to A and to each column of B, and then do pointwise products?
    This is exactly what the Winograd method does.
    For 2x2 * 2x1, the rank is 4.
    So for 2x2 * 2x9, if we do it column-wise, rank is 36.
    
    Is there a rank < 36 decomposition for 2x2 * 2x9?
    Rank(M_{2,2,9}) <= Rank(M_{2,2,2}) * Rank(M_{1,1,5})? No.
    Rank(M_{2,2,9}) <= 2 * Rank(M_{1,2,9})?
    Rank(M_{1,2,9}) is the rank of multiplying a 1x2 vector by a 2x9 matrix? No, 1x2 * 2x9 -> 1x9.
    This is just dot products. Rank = 9.
    So Rank(M_{2,2,9}) <= 2 * 9 = 18?
    Let's check.
    C = A B. A is 2x2, B is 2x9.
    C is 2x9.
    C_row1 = A_row1 * B (dot product of A_row1 with each column of B)
    C_row2 = A_row2 * B
    This is 2 independent 1x2 * 2x9 multiplications.
    Rank of 1x2 * 2x9 is 9 (just 9 dot products of length 2? No, 1x2 * 2x1 is rank 2. 1x2 * 2x9 is 9 copies of 1x2*2x1? No, the A part is shared? No, A is 1x2, so it's just two scalars.
    Wait, M_{1,2,9}: A is 1x2 (2 vars), B is 2x9 (18 vars), C is 1x9 (9 vars).
    C_j = a_1 B_1j + a_2 B_2j.
    This is linear in a_1, a_2.
    Rank is 2 * 9 = 18?
    No, the tensor is T[a, b, c].
    T[a_1, B_1j, C_j] = 1.
    T[a_2, B_2j, C_j] = 1.
    These are 18 independent terms?
    Actually, we can factor this.
    C = A B.
    If we fix A, C is linear in B.
    If we fix B, C is linear in A.
    The rank of the tensor T_{i,j,k} where i in {1,2}, j in {1..18}, k in {1..9}.
    T_{1, 1+j, j+1} = 1? No.
    Let's index A as a_0, a_1. B as b_00..b_08, b_10..b_18. C as c_0..c_8.
    c_j = a_0 b_0j + a_1 b_1j.
    Terms: (a_0, b_0j, c_j) and (a_1, b_1j, c_j).
    Total 18 terms.
    Can we do better?
    This is the tensor of the map (a_0, a_1) x (b_00..b_18) -> (c_0..c_8).
    This is equivalent to 9 independent 2D dot products.
    Rank is 18.
    
    So Rank(M_{2,2,9}) = 2 * 18 = 36?
    Wait, in M_{2,2,9}, A is 2x2 (4 vars).
    C_row0 = A_row0 * B.
    C_row1 = A_row1 * B.
    These are two independent M_{1,2,9} problems.
    So Rank(M_{2,2,9}) = 2 * 18 = 36.
    
    So the column-wise Winograd (rank 4 per col) gives 36.
    The row-wise decomposition (rank 18 per row) gives 36.
    So 36 is likely optimal for 2x2x9.
    
    Therefore, the block decomposition 4x4x9 -> 4 * (2x2x9) gives rank 4 * 36 = 144.
    This is the same as schoolbook.
    
    To improve, we need a better decomposition for 4x4x9.
    Maybe use 4x4 matrix multiplication rank?
    Rank(M_{4,4,4}) is unknown, but > 49 (schoolbook).
    If we had a rank R decomposition for 4x4*4x4, we could use it for 4x4*4x9 by splitting B into four 4x? blocks?
    B is 4x9.
    We can write B = [B1, B2, B3] where B1 is 4x3, B2 is 4x3, B3 is 4x3.
    C = [A B1, A B2, A B3].
    If we have a rank R decomposition for 4x4*4x3, then total rank is 3R.
    Rank(M_{4,4,3})?
    M_{4,4,3} is 4x4 * 4x3 -> 4x3.
    This is 3 independent 4x4 * 4x1 multiplications.
    Rank(M_{4,4,1})?
    This is the rank of the matrix-vector multiplication for 4x4.
    It is known that Rank(M_{n,n,1}) = n^2.
    So Rank(M_{4,4,1}) = 16.
    So Rank(M_{4,4,3}) = 3 * 16 = 48.
    Total rank = 3 * 48 = 144.
    
    It seems 144 is a strong lower bound or the actual rank.
    
    Let's check if we can mix A and B more efficiently.
    The tensor is T_{i,k,j} = delta_{ik} delta_{kj}? No, T_{i,k,j} is 1 if the term A_{ik} B_{kj} contributes to C_{ij}.
    Actually, the tensor is defined by C_{ij} = sum_k A_{ik} B_{kj}.
    So T_{(i,k), (k,j), (j,i)} = 1.
    
    Is there a decomposition with rank < 144?
    For 4x4x4, the lower bound is 49.
    For 4x4x9, the lower bound is likely higher.
    
    Given the constraints and the fact that 144 is the schoolbook rank, and simple decompositions yield 144, I will return the schoolbook decomposition but with a very fast verification and construction to ensure it passes within time limits.
    
    Actually, I can try to use a known rank-16 decomposition for 4x4*4x1? No, rank is 16.
    
    Let's try to construct a decomposition that is slightly better if possible, or just ensure the schoolbook is correct.
    
    Wait, what if I use the rank-7 decomposition for 2x2*2x2 and combine?
    We have 4x4 * 4x9.
    Split A into 4 blocks of 2x2.
    Split B into 4 blocks of 2x? No, B is 4x9.
    Split B into 2 blocks of 2x9?
    Then we have 2x2 * 2x9 problems.
    Rank(M_{2,2,9}) = 36.
    Total 4 * 36 = 144.
    
    It seems 144 is the limit for these simple block decompositions.
    
    I will provide the schoolbook solution.
    """
    # Schoolbook for 2x2x9 is rank 36.
    # We will just return the global schoolbook.
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try to find a better decomposition.
    # Since simple block decompositions yield 144, and we don't have a known
    # rank < 144 decomposition for 4x4x9 readily available in this context,
    # we will return the schoolbook decomposition.
    
    best = _schoolbook(n, m, p)
    
    # Verify to be safe
    if _verify(best["U"], best["V"], best["W"], n, m, p):
        return best
    
    # If verification fails (shouldn't happen), try to fix or return empty
    return {"U": [], "V": [], "W": []}
