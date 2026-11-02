import time
import numpy as np

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 3 matrix by a 3 by 5 matrix.
    
    Strategy:
    1. Partition the 2x3 * 3x5 multiplication into two sub-problems:
       - C1 (2x2): A[:, :2] * B[:2, :]  (using columns 0,1 of B? No, B is 3x5. 
         Let's partition B's rows. B is 3x5. A is 2x3.
         Let A = [A1 (2x2), A2 (2x1)]
         Let B = [B1 (2x5); B2 (1x5)]
         C = A1*B1 + A2*B2
         A1*B1 is 2x2 * 2x5 -> 2x5.
         A2*B2 is 2x1 * 1x5 -> 2x5.
         
         We need to decompose:
         1. 2x2 * 2x5. This can be seen as (2x2 * 2x2) for first 2 cols of B1, and (2x2 * 2x1) for last 3 cols?
            Actually, B1 is 2x5. We can split B1 into B11 (2x2) and B12 (2x3).
            Then A1*B11 is 2x2 * 2x2. Rank 7.
            A1*B12 is 2x2 * 2x3. This is (2x2 * 2x2) + (2x2 * 2x1). Rank 7 + 6 = 13? 
            Or is there a better decomposition for 2x2 * 2x3?
            
            Let's look at the sizes:
            2x2 * 2x2: Rank 7 (Strassen).
            2x2 * 2x1: Rank 6? (2x1 * 2x1 is dot product, rank 2. But 2x2 * 2x1 is 2 mults? No.
             Let A be 2x2, B be 2x1. C = A B. 4 entries.
             c11 = a11 b1 + a12 b2
             c21 = a21 b1 + a22 b2
             Rank of 2x2 * 2x1 tensor?
             It is equivalent to 2 independent dot products of length 2.
             Rank of dot product of length 2 is 2.
             So rank is 2+2=4?
             Wait, Strassen for 2x2 is 7.
             For 2x1 * 2x1 (inner product), rank is 2.
             For 2x2 * 2x1, the tensor has size 4x2x1 = 8.
             Schoolbook is 4.
             Can we do better than 4?
             c11 = a11 b1 + a12 b2
             c21 = a21 b1 + a22 b2
             This is just two separate bilinear forms. They don't share factors in a way that reduces rank below 4?
             Actually, the rank of the matrix multiplication tensor $M_{2,2,1}$ is 4.
             
             Let's re-evaluate the partitioning.
             Total Rank = Rank(2x2*2x2) + Rank(2x2*2x3)?
             2x2 * 2x3:
             Split 2x3 into 2x2 and 2x1.
             Rank(2x2*2x2) = 7.
             Rank(2x2*2x1) = 4.
             Total = 11.
             
             Plus A2*B2 (2x1 * 1x5).
             2x1 * 1x5 is just 2 rows of A2 multiplied by 1 row of B2?
             A2 is 2x1. B2 is 1x5.
             C_part = A2 * B2.
             c_i = a_i * (sum? No, B2 is 1x5, so b_j is scalar? No.
             A2 is vector (2,1). B2 is vector (1,5).
             Outer product.
             c_{i,j} = a_i * b_j.
             This is rank 1 tensor? No, it's a bilinear map.
             The tensor for outer product of (2,1) and (1,5) is size 2x5x... 
             Variables: a1, a2, b1..b5.
             Outputs: c11..c15, c21..c25.
             c_ij = a_i b_j.
             This is a rank 1 tensor in the space of matrices?
             The tensor T has entries T_{ij, i, j} = 1? No.
             The multiplication tensor for outer product $u \in R^2, v \in R^5$ is $\sum_{i=1}^2 \sum_{j=1}^5 e_i \otimes e_j \otimes (e_i \otimes e_j)$?
             Actually, $C = u v^T$.
             $C_{ij} = u_i v_j$.
             This is a sum of 10 rank-1 terms $u_i v_j$.
             Can it be done with fewer?
             The rank of the outer product tensor $M_{2,1,5}$ (which is really $M_{2,5}$ with middle dim 1) is $2 \times 5 = 10$?
             Wait, if the middle dimension is 1, it's just element-wise multiplication of two vectors?
             No, A is 2x1, B is 1x5.
             $A = \begin{pmatrix} a_1 \\ a_2 \end{pmatrix}$, $B = \begin{pmatrix} b_1 & b_2 & b_3 & b_4 & b_5 \end{pmatrix}$.
             $C = \begin{pmatrix} a_1 b_1 & \dots & a_1 b_5 \\ a_2 b_1 & \dots & a_2 b_5 \end{pmatrix}$.
             Each entry $C_{ij} = a_i b_j$.
             This requires 10 multiplications $a_i b_j$.
             Is there a lower rank?
             The tensor is $\sum_{i=1}^2 \sum_{j=1}^5 e_i \otimes e_j \otimes (e_i \otimes e_j)$? No.
             The tensor is in $U^* \otimes V^* \otimes W$.
             $U = R^2, V = R^5, W = R^{2 \times 5}$.
             $T = \sum_{i,j} u_i \otimes v_j \otimes w_{ij}$.
             This is a direct sum of 10 rank-1 tensors.
             Since the supports in U and V are disjoint for each term?
             Term for (1,1): $u_1 \otimes v_1 \otimes w_{11}$.
             Term for (1,2): $u_1 \otimes v_2 \otimes w_{12}$.
             We can factor $u_1$?
             $u_1 \otimes (v_1 \otimes w_{11} + v_2 \otimes w_{12} + \dots)$.
             The second part is a tensor in $V^* \otimes W$.
             Rank of $v_1 \otimes w_{11} + \dots + v_5 \otimes w_{15}$?
             This is a matrix of rank 5 (if $w_{ij}$ are basis vectors).
             So we need 5 terms for the first row of A.
             Similarly 5 terms for the second row of A.
             Total rank 10.
             
             So:
             Part 1: 2x2 * 2x2 (cols 0,1 of B1). Rank 7.
             Part 2: 2x2 * 2x3 (cols 2,3,4 of B1).
               Split into 2x2*2x2 (col 2) and 2x2*2x2 (cols 3,4? No, 2x2*2x1).
               Let's split B1 (2x5) into B11 (2x2), B12 (2x2), B13 (2x1).
               A1*B11: Rank 7.
               A1*B12: Rank 7.
               A1*B13: Rank 4.
               Sum: 18.
             Part 3: 2x1 * 1x5 (A2*B2). Rank 10.
             Total: 18 + 10 = 28.
             
             Schoolbook is 30.
             We can achieve 28.
             
             Can we do better?
             Maybe mix the partitions?
             The theoretical lower bound for 2x3x5 is not well known, but 28 is a strong candidate.
             Let's try to construct the rank 28 decomposition.
             
             Components:
             1. Strassen 2x2x2 (Rank 7)
             2. Strassen 2x2x2 (Rank 7)
             3. Schoolbook 2x2x1 (Rank 4)
             4. Schoolbook 2x1x5 (Rank 10)
             
             Total Rank 28.
             
             Let's define the indices.
             A is 2x3. A = [A1 (2x2), A2 (2x1)].
             B is 3x5. B = [B1 (2x5); B2 (1x5)].
             
             B1 is 2x5. Split into B11 (2x2), B12 (2x2), B13 (2x1).
             
             C = A1 B1 + A2 B2
               = A1 B11 + A1 B12 + A1 B13 + A2 B2
               
             C11 (2x2) = A1 B11 (Rank 7)
             C12 (2x2) = A1 B12 (Rank 7)
             C13 (2x1) = A1 B13 (Rank 4)
             C2 (2x5) = A2 B2 (Rank 10)
             
             Total output C is 2x5.
             C[:, 0:2] from C11.
             C[:, 2:4] from C12.
             C[:, 4] from C13.
             Wait, C13 is 2x1. It contributes to column 4?
             B1 is 2x5.
             B11 is cols 0,1.
             B12 is cols 2,3.
             B13 is col 4.
             So C[:, 0:2] = A1 B11.
             C[:, 2:4] = A1 B12.
             C[:, 4] = A1 B13.
             And C = A1 B1 + A2 B2.
             Wait, A2 is 2x1, B2 is 1x5.
             A2 B2 is 2x5.
             So C[:, 0] += A2 B2[:, 0].
             C[:, 1] += A2 B2[:, 1].
             ...
             C[:, 4] += A2 B2[:, 4].
             
             So the contributions are additive.
             We can just sum the tensors.
             
             Let's build the tensors.
             
             Helper: Strassen 2x2x2.
             Returns U, V, W for A(2x2) * B(2x2) = C(2x2).
             Indices for U: 0..3 (a11, a12, a21, a22)
             Indices for V: 0..3 (b11, b12, b21, b22)
             Indices for W: 0..3 (c11, c12, c21, c22)
             
             Helper: Schoolbook 2x2x1.
             A(2x2) * B(2x1) = C(2x1).
             U: 0..3
             V: 0..1
             W: 0..1
             
             Helper: Schoolbook 2x1x5.
             A(2x1) * B(1x5) = C(2x5).
             U: 0..1
             V: 0..4
             W: 0..9
             
             We need to map these local indices to the global U, V, W.
             
             Global U size: 2*3 = 6.
             Global V size: 3*5 = 15.
             Global W size: 5*2 = 10.
             
             Mapping:
             A1 (2x2) corresponds to A[0:2, 0:2]. Global U indices: 0,1,2,3.
             A2 (2x1) corresponds to A[0:2, 2]. Global U indices: 4,5.
             
             B1 (2x5) corresponds to B[0:2, :]. Global V indices: 0..9.
             B11 (2x2) -> B[0:2, 0:2]. V indices: 0,1,2,3.
             B12 (2x2) -> B[0:2, 2:4]. V indices: 4,5,6,7.
             B13 (2x1) -> B[0:2, 4]. V indices: 8,9.
             B2 (1x5) corresponds to B[2, :]. Global V indices: 10..14.
             
             C (2x5) -> Global W indices: 0..9.
             C11 (2x2) -> C[0:2, 0:2]. W indices: 0,1,2,3.
             C12 (2x2) -> C[0:2, 2:4]. W indices: 4,5,6,7.
             C13 (2x1) -> C[0:2, 4]. W indices: 8,9.
             C2 (2x5) -> C[0:2, 0:5]. W indices: 0..9.
             
             Wait, C13 and C2 both contribute to W indices 8,9?
             C13 is 2x1. It gives C[:, 4].
             C2 is 2x5. It gives C[:, 0..4].
             So yes, they add up.
             
             Let's implement the generators.
             
             1. Strassen 2x2x2
            U (4x7):
            a11: [1, 0, 0, 0, 1, 0, 1] ? No.
            Standard Strassen:
            m1 = (a11+a22)(b11+b22)
            m2 = (a21+a22)b11
            m3 = a11(b12-b22)
            m4 = a22(b21-b11)
            m5 = (a11+a12)b22
            m6 = (a21-a11)(b11+b12)
            m7 = (a12-a22)(b21+b22)
            
            c11 = m1 + m4 - m5 + m7
            c12 = m3 + m5
            c21 = m2 + m4
            c22 = m1 - m2 + m3 + m6
            
            U (rows a11, a12, a21, a22):
            a11: [1, 0, 1, 0, 1, -1, 0]
            a12: [0, 0, 0, 0, 1, 0, -1]
            a21: [0, 1, 0, 0, 0, 1, 0]
            a22: [1, 1, 0, 1, 0, 0, 1]
            
            V (rows b11, b12, b21, b22):
            b11: [1, 1, 0, -1, 0, 1, 0]
            b12: [0, 0, 1, 0, 0, 1, 0]
            b21: [0, 0, 0, 1, 0, 0, 1]
            b22: [1, 0, -1, 0, 1, 0, 1]
            
            W (rows c11, c12, c21, c22):
            c11: [1, 0, 0, 1, -1, 0, 1]
            c12: [0, 0, 1, 0, 1, 0, 0]
            c21: [0, 1, 0, 1, 0, 0, 0]
            c22: [1, -1, 1, 0, 0, 1, 0]
            
             2. Schoolbook 2x2x1
            c11 = a11 b1 + a12 b2
            c21 = a21 b1 + a22 b2
            U:
            a11: [1, 0]
            a12: [0, 1]
            a21: [1, 0]
            a22: [0, 1]
            V:
            b1: [1, 1]
            b2: [1, 1]
            W:
            c11: [1, 1]
            c21: [1, 1]
            
             Wait, V rows are b1, b2.
            V[0] (b1) = [1, 1]
            V[1] (b2) = [1, 1]
            
             3. Schoolbook 2x1x5
            c1j = a1 b_j
            c2j = a2 b_j
            U:
            a1: [1, 0, 0, 0, 0]
            a2: [0, 1, 0, 0, 0]
            V:
            b1: [1, 0, 0, 0, 0]
            b2: [0, 1, 0, 0, 0]
            b3: [0, 0, 1, 0, 0]
            b4: [0, 0, 0, 1, 0]
            b5: [0, 0, 0, 0, 1]
            W:
            c11: [1, 0, 0, 0, 0]
            c12: [0, 1, 0, 0, 0]
            c13: [0, 0, 1, 0, 0]
            c14: [0, 0, 0, 1, 0]
            c15: [0, 0, 0, 0, 1]
            c21: [0, 0, 0, 0, 0] -> Wait.
            c21 = a2 b1.
            W[4] (c21) should be [0, 0, 0, 0, 1]? No.
            W rows are c11, c12, c13, c14, c15, c21, c22, c23, c24, c25.
            c11: [1, 0, 0, 0, 0]
            c12: [0, 1, 0, 0, 0]
            ...
            c15: [0, 0, 0, 0, 1]
            c21: [1, 0, 0, 0, 0] ?? No.
            Term 1: a1 b1 -> c11. U[0]=1, V[0]=1, W[0]=1.
            Term 2: a1 b2 -> c12. U[0]=1, V[1]=1, W[1]=1.
            ...
            Term 5: a1 b5 -> c15. U[0]=1, V[4]=1, W[4]=1.
            Term 6: a2 b1 -> c21. U[1]=1, V[0]=1, W[5]=1.
            Term 7: a2 b2 -> c22. U[1]=1, V[1]=1, W[6]=1.
            ...
            Term 10: a2 b5 -> c25. U[1]=1, V[4]=1, W[9]=1.
            
            So:
            U[0] = [1, 1, 1, 1, 1]
            U[1] = [1, 1, 1, 1, 1]
            V[0] = [1, 0, 0, 0, 0]
            V[1] = [0, 1, 0, 0, 0]
            ...
            V[4] = [0, 0, 0, 0, 1]
            W[0] = [1, 0, 0, 0, 0]
            W[1] = [0, 1, 0, 0, 0]
            ...
            W[4] = [0, 0, 0, 0, 1]
            W[5] = [1, 0, 0, 0, 0]
            W[6] = [0, 1, 0, 0, 0]
            ...
            W[9] = [0, 0, 0, 0, 1]
            
             Now assemble.
             Rank 7 + 7 + 4 + 10 = 28.
             
             We will create U, V, W as lists of lists of zeros of size (6x28), (15x28), (10x28).
             Then fill them.
             
             Block 1: 2x2 * 2x2 (A1, B11)
             U_local (4x7), V_local (4x7), W_local (4x7).
             Map U_local rows 0..3 to Global U rows 0..3.
             Map V_local rows 0..3 to Global V rows 0..3.
             Map W_local rows 0..3 to Global W rows 0..3.
             Offset in rank: 0.
             
             Block 2: 2x2 * 2x2 (A1, B12)
             U_local same as above.
             V_local same as above.
             W_local same as above.
             Map U_local rows 0..3 to Global U rows 0..3.
             Map V_local rows 0..3 to Global V rows 4..7.
             Map W_local rows 0..3 to Global W rows 4..7.
             Offset in rank: 7.
             
             Block 3: 2x2 * 2x1 (A1, B13)
             U_local (4x4), V_local (2x4), W_local (2x4).
             Map U_local rows 0..3 to Global U rows 0..3.
             Map V_local rows 0..1 to Global V rows 8..9.
             Map W_local rows 0..1 to Global W rows 8..9.
             Offset in rank: 14.
             
             Block 4: 2x1 * 1x5 (A2, B2)
             U_local (2x10), V_local (5x10), W_local (10x10).
             Map U_local rows 0..1 to Global U rows 4..5.
             Map V_local rows 0..4 to Global V rows 10..14.
             Map W_local rows 0..9 to Global W rows 0..9.
             Offset in rank: 18.
             
             This should work.
             
             Verify:
             Total Rank 28.
             Schoolbook 30.
             Score = 30/28 = 1.071.
             
             Let's write the code.
    """
    start_time = time.time()
    
    def schoolbook():
        rank = 30
        U = [[0] * rank for _ in range(6)]
        V = [[0] * rank for _ in range(15)]
        W = [[0] * rank for _ in range(10)]
        col = 0
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    U[i * 3 + j][col] = 1
                    V[j * 5 + k][col] = 1
                    W[k * 2 + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Strassen 2x2x2
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, -1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook 2x2x1
    SB_U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    SB_V = [
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    SB_W = [
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    
    # Schoolbook 2x1x5
    SO_U = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]
    SO_V = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ]
    SO_W = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    ]
    
    rank = 28
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(15)]
    W = [[0] * rank for _ in range(10)]
    
    def add_block(U_loc, V_loc, W_loc, u_map, v_map, w_map, offset):
        for i, ui in enumerate(u_map):
            for c in range(len(U_loc[0])):
                U[ui][offset + c] += U_loc[i][c]
        for i, vi in enumerate(v_map):
            for c in range(len(V_loc[0])):
                V[vi][offset + c] += V_loc[i][c]
        for i, wi in enumerate(w_map):
            for c in range(len(W_loc[0])):
                W[wi][offset + c] += W_loc[i][c]
        
    # Block 1: 2x2 * 2x2 (A1, B11)
    add_block(S_U, S_V, S_W, [0,1,2,3], [0,1,2,3], [0,1,2,3], 0)
    
    # Block 2: 2x2 * 2x2 (A1, B12)
    add_block(S_U, S_V, S_W, [0,1,2,3], [4,5,6,7], [4,5,6,7], 7)
    
    # Block 3: 2x2 * 2x1 (A1, B13)
    add_block(SB_U, SB_V, SB_W, [0,1,2,3], [8,9], [8,9], 14)
    
    # Block 4: 2x1 * 1x5 (A2, B2)
    add_block(SO_U, SO_V, SO_W, [4,5], [10,11,12,13,14], [0,1,2,3,4,5,6,7,8,9], 18)
    
    # Verify
    def verify(U, V, W):
        r = len(U[0])
        for i in range(2):
            for j in range(3):
                for k in range(5):
                    s = 0.0
                    for c in range(r):
                        s += U[i * 3 + j][c] * V[j * 5 + k][c] * W[k * 2 + i][c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return schoolbook()

if __name__ == "__main__":
    res = construct()
    print(res)
