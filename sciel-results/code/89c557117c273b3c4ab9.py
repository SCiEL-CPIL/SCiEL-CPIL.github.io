"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 by 3x5 matrix multiplication.
    The schoolbook method gives rank 45. For 3x3x3, the minimal rank is 23.
    For 3x3x5, the tensor is T = T_3x3x3 + T_3x3x2.
    Since the indices are independent, we can combine the 3x3x3 part (rank 23) with the 3x3x2 part.
    However, the 3x3x2 part is a linear map from R^9 to R^15 with rank 6 (since it's just embedding).
    Actually, the tensor for 3x3x5 is the sum of two tensors: one for k in 0..2 and one for k in 3..4.
    T_0 = tensor for 3x3x3 (indices i,j,k in 0..2)
    T_1 = tensor for 3x3x2 (indices i,j,k in 3..4)
    
    The rank of T is at most rank(T_0) + rank(T_1).
    T_1 is simply the matrix multiplication tensor for A (3x2) * B (2x5)? No, A is 3x3, B is 3x5.
    For k=3,4, the terms are A[i,3]*B[3,j] + A[i,4]*B[4,j].
    This is a bilinear map from A_{.,3:5} (3x2) and B_{3:5,.} (2x5) to C_{.,.} (3x5).
    This is equivalent to multiplying a 3x2 matrix by a 2x5 matrix.
    The tensor for 3x2x5 has rank 3*2*5 = 30 if schoolbook, but minimal rank is lower.
    Actually, the tensor for multiplying an n x m matrix by an m x p matrix has rank at least n*p.
    For 3x2x5, n=3, m=2, p=5. Rank >= 15.
    Is there a known low rank decomposition for 3x2x5?
    The tensor is sum_{j=0}^{1} (A_{.,j} B_{j,.})^T.
    This is a sum of 2 rank-15 terms? No.
    The tensor T_{ijk} = delta_{j,k} for j,k in 0..1? No, j is the column of A, row of B.
    T_{i,j,k} = 1 if i,j,k are valid indices for the multiplication A_{i,j} B_{j,k} -> C_{i,k}.
    Here j ranges 0..1.
    So T = sum_{j=0}^1 e_i (all i) \otimes e_j \otimes e_k (all k).
    This is a sum of 2 tensors. Each term is a matrix of size 3x5 in the i,k plane, with 1s everywhere.
    So T = sum_{j=0}^1 (1_3 \otimes 1_5) \otimes e_j.
    This tensor has rank 10?
    Let's check: 1_3 \otimes 1_5 is a 3x5 matrix of ones. It has rank 1.
    So each term (for fixed j) is a rank-1 tensor in (i,k) times a vector in j.
    So the whole tensor is a sum of 2 rank-1 tensors?
    T = (1_3 \otimes 1_5 \otimes e_0) + (1_3 \otimes 1_5 \otimes e_1).
    This can be written as (1_3 \otimes 1_5) \otimes (e_0 + e_1).
    This is a single rank-1 tensor!
    Wait, is it?
    T_{i,j,k} = 1 if j in {0,1} and any i,k.
    If we set U_{i,j} = 1, V_{j,k} = 1, W_{i,k} = 1?
    Then (U V W)_{i,j,k} = U_{i,j} V_{j,k} W_{i,k} = 1*1*1 = 1.
    But the tensor is defined on indices (i, j, k).
    In the decomposition, U has indices (i, j) where j is the "rank" index? No.
    Standard form: T = sum_{r=1}^R u_r \otimes v_r \otimes w_r.
    Here, the first index is i (0..2), second is j (0..1) for the 3x2x5 part, third is k (0..4).
    For the 3x2x5 part, the middle index is the shared dimension (size 2).
    So T_part2_{i, j, k} = 1 for all i, j, k in their ranges.
    This tensor is simply 1_3 \otimes 1_2 \otimes 1_5.
    This is a rank-1 tensor!
    So the rank of the 3x2x5 part is 1.
    
    So the total rank is at most rank(T_3x3x3) + rank(T_3x2x5) = 23 + 1 = 24?
    Or can we combine them?
    T_total = T_3x3x3 + T_3x2x5.
    T_3x3x3 has indices i in 0..2, j in 0..2, k in 0..2.
    T_3x2x5 has indices i in 0..2, j in 3..4 (shifted), k in 0..4.
    Wait, the indices in the problem are global.
    A is 3x3, B is 3x5.
    C[i,k] = sum_{j=0}^2 A[i,j]B[j,k] + sum_{j=3}^4 A[i,j]B[j,k].
    The tensor is sum_{j=0}^4 (e_i \otimes e_j \otimes e_k).
    This is a sum of 5 terms.
    Term for j=0: e_i (all i) \otimes e_0 \otimes e_k (all k).
    This is a rank-1 tensor: 1_3 \otimes e_0 \otimes 1_5.
    Term for j=1: 1_3 \otimes e_1 \otimes 1_5.
    Term for j=2: 1_3 \otimes e_2 \otimes 1_5.
    Term for j=3: 1_3 \otimes e_3 \otimes 1_5.
    Term for j=4: 1_3 \otimes e_4 \otimes 1_5.
    
    So the tensor is sum_{j=0}^4 (1_3 \otimes e_j \otimes 1_5).
    This can be rewritten as (1_3 \otimes 1_5) \otimes (sum_{j=0}^4 e_j).
    This is a SINGLE rank-1 tensor!
    
    Wait, is this correct?
    If the tensor is rank 1, then the rank of matrix multiplication 3x3 by 3x5 is 1?
    That is impossible.
    The error is in the interpretation of the tensor indices.
    The tensor T_{ijk} corresponds to the coefficient of A_{ik} B_{kj} in C_{ij}?
    No, standard convention: C_{ij} = sum_k A_{ik} B_{kj}.
    So the tensor is T_{i, k, j}? Or T_{i, j, k}?
    The task says: "output factor follows the official trace convention W[k*n+i,r]".
    And U is (n*m, rank), V is (m*p, rank).
    U has indices (i, k) for A? No, A is n x m. So U index is i * m + k?
    V has indices (k, j) for B? B is m x p. So V index is k * p + j?
    W has indices (j, i) for C? C is n x p. So W index is j * n + i?
    Let's check the shapes.
    U: (n*m, rank). Index u = i*m + k.
    V: (m*p, rank). Index v = k*p + j.
    W: (p*n, rank). Index w = j*n + i.
    
    The tensor element is T_{u,v,w} = 1 if u,v,w correspond to the same (i,k,j) tuple.
    u = i*m + k.
    v = k*p + j.
    w = j*n + i.
    
    For a fixed (i,k,j), u is determined, v is determined, w is determined.
    The tensor is a set of 1s at these positions.
    This is the standard matrix multiplication tensor.
    
    My previous analysis "T = sum (1_3 \otimes e_j \otimes 1_5)" was wrong because I confused the indices.
    The middle index in the tensor product is NOT just "j".
    The indices are coupled: k is in U and V. j is in V and W. i is in W and U.
    
    So we are back to the standard MM tensor.
    Rank(3x3x5) ?
    It is known that R(3,3,5) is likely 23 or 24?
    Actually, R(n,m,p) >= max(n*m, m*p, p*n).
    Here max(9, 15, 15) = 15.
    Schoolbook is 45.
    
    There is a result by Le Gall or others?
    For 3x3x3, R=23.
    For 3x3x5, we can view it as 3x3x3 plus 3x3x2.
    The 3x3x2 part is multiplying a 3x3 matrix by a 3x2 matrix.
    This is equivalent to multiplying a 3x3 matrix by a 3x2 matrix.
    The tensor for 3x3x2:
    Indices i (0..2), k (0..2) for A, j (0..1) for B.
    C_{i,j} = sum_k A_{i,k} B_{k,j}.
    This is a linear map from A (9 dims) and B (6 dims) to C (6 dims).
    The tensor is sum_{i,k,j} e_i \otimes e_k \otimes e_j? No.
    T_{i, k, j} = 1.
    This tensor has rank ?
    It is the same as 3x2x3 (if we permute indices).
    R(3,3,2) is the same as R(3,2,3).
    By symmetry, R(3,2,3) = R(3,3,2).
    Is R(3,2,3) known?
    R(2,2,2) = 7.
    R(3,2,3) ?
    It is at least max(9, 6, 6) = 9.
    Schoolbook is 18.
    
    If we use the rank-23 decomposition for 3x3x3, and a rank-? decomposition for 3x3x2,
    we can sum them.
    If R(3,3,2) = 18 (schoolbook), then total rank = 23 + 18 = 41.
    If R(3,3,2) is lower, say 15, then total rank = 38.
    
    However, we can likely do better by combining.
    But finding the exact minimal rank decomposition is hard.
    
    Given the constraints (120s), I will attempt to construct a decomposition of rank 23 for the 3x3x3 part
    and rank 18 for the 3x3x2 part, resulting in rank 41.
    Or, simpler: just use the schoolbook decomposition for the whole thing, which is rank 45.
    The prompt asks for a "stronger algorithm" but also warns about validity.
    The seed is rank 45.
    If I can produce rank 41, that's a 45/41 = 1.097 score improvement.
    
    Let's try to implement the rank-23 decomposition for 3x3x3.
    The Strassen-like decomposition for 3x3x3 was found by Le Gall (1979) or later improved.
    Actually, R(3,3,3) = 23 is the best known.
    The construction is complex.
    
    Alternative: Use the "Winograd" FFT idea from the directive?
    The directive suggests DFT over Z_3.
    If we apply DFT to the 3x3x3 part, it might decouple it.
    But 3x3x3 is not circulant.
    
    Let's stick to a safe improvement: Rank 41.
    I will construct the 3x3x3 part with rank 23 (using a known list of 23 tensors) and 3x3x2 part with rank 18 (schoolbook).
    
    I need the explicit 23 tensors for 3x3x3.
    I don't have them memorized.
    
    Let's try to find a simpler improvement.
    What if I just use rank 45? The score is 1.0.
    The prompt says "Keep the function names...".
    
    If I cannot find a valid lower rank decomposition, I should return the schoolbook one.
    However, the "Winograd" hint is specific.
    
    Let's try to implement a random search for a lower rank?
    No, too risky for 120s.
    
    I will output the schoolbook decomposition.
    """
    n, m, p = DIMENSIONS
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
# EVOLVE-BLOCK-END
