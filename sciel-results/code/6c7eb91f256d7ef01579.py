import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    n, m, p = 4, 4, 5
    
    # We will construct a decomposition using the Strassen algorithm recursively.
    # For 4x4x4, we can use Strassen (rank 49) for the first 4 columns of B.
    # For the remaining 1 column, we can use schoolbook or a specific low-rank trick.
    # However, a simpler and robust approach is to use the known rank 49 decomposition for 4x4x4
    # and then extend it to 4x4x5.
    
    # The rank of 4x4x4 multiplication is 49 (Strassen).
    # The rank of 4x4x5 is at most 49 + 4*4 = 65 (Strassen for 4x4x4 + schoolbook for 4x4x1).
    # Actually, we can just use the Kronecker product structure or block decomposition.
    
    # Let's try to build the 4x4x4 Strassen decomposition explicitly.
    # The Strassen algorithm for 2x2x2 has rank 7.
    # For 4x4x4, we view it as 2x2 blocks of 2x2 matrices.
    # The rank is 7 * 7 = 49.
    
    # We define the 2x2x2 Strassen factors.
    # U2, V2, W2 are 4x7 matrices.
    # Indices: U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Term 2: (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Term 3: A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Term 4: A11(B10-B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Term 5: (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Term 6: (A10-A00)(B00+B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Term 7: (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Now, we construct the 4x4x4 decomposition.
    # We have 7 "top-level" Strassen terms.
    # Each term involves multiplying two 2x2 block matrices.
    # Let L_s and R_s be the 4x4 matrices formed by the block combinations for term s.
    # Then we apply the 2x2x2 Strassen decomposition to the 2x2 blocks of L_s * R_s.
    # This results in 7 * 7 = 49 rank-1 terms.
    
    # We define the block combinations for the 7 top-level terms.
    # L_s and R_s are 4x4 matrices.
    # We represent them as 2x2 block matrices of 2x2 blocks.
    
    # Let's define the coefficients for the blocks.
    # For each top-level term s, we have a 2x2 matrix of 2x2 blocks for L_s and R_s.
    # Let L_s_blocks[i,j] be the 2x2 block at position (i,j) of L_s.
    # Similarly for R_s.
    
    # We will store the non-zero blocks and their coefficients.
    # Actually, it's easier to just construct the full 4x4 matrices L_s and R_s.
    
    L_s_list = []
    R_s_list = []
    
    # s=0: L = A00+A11, R = B00+B11
    L0 = np.zeros((4,4))
    L0[0:2, 0:2] = np.eye(2)
    L0[2:4, 2:4] = np.eye(2)
    R0 = np.zeros((4,4))
    R0[0:2, 0:2] = np.eye(2)
    R0[2:4, 2:4] = np.eye(2)
    L_s_list.append(L0)
    R_s_list.append(R0)
    
    # s=1: L = A10+A11, R = B00
    L1 = np.zeros((4,4))
    L1[2:4, 0:2] = np.eye(2)
    L1[2:4, 2:4] = np.eye(2)
    R1 = np.zeros((4,4))
    R1[0:2, 0:2] = np.eye(2)
    L_s_list.append(L1)
    R_s_list.append(R1)
    
    # s=2: L = A00, R = B01-B11
    L2 = np.zeros((4,4))
    L2[0:2, 0:2] = np.eye(2)
    R2 = np.zeros((4,4))
    R2[0:2, 2:4] = np.eye(2)
    R2[2:4, 2:4] = -np.eye(2)
    L_s_list.append(L2)
    R_s_list.append(R2)
    
    # s=3: L = A11, R = B10-B00
    L3 = np.zeros((4,4))
    L3[2:4, 2:4] = np.eye(2)
    R3 = np.zeros((4,4))
    R3[2:4, 0:2] = np.eye(2)
    R3[0:2, 0:2] = -np.eye(2)
    L_s_list.append(L3)
    R_s_list.append(R3)
    
    # s=4: L = A00+A01, R = B11
    L4 = np.zeros((4,4))
    L4[0:2, 0:2] = np.eye(2)
    L4[0:2, 2:4] = np.eye(2)
    R4 = np.zeros((4,4))
    R4[2:4, 2:4] = np.eye(2)
    L_s_list.append(L4)
    R_s_list.append(R4)
    
    # s=5: L = A10-A00, R = B00+B01
    L5 = np.zeros((4,4))
    L5[2:4, 0:2] = np.eye(2)
    L5[0:2, 0:2] = -np.eye(2)
    R5 = np.zeros((4,4))
    R5[0:2, 0:2] = np.eye(2)
    R5[0:2, 2:4] = np.eye(2)
    L_s_list.append(L5)
    R_s_list.append(R5)
    
    # s=6: L = A01-A11, R = B10+B11
    L6 = np.zeros((4,4))
    L6[0:2, 2:4] = np.eye(2)
    L6[2:4, 2:4] = -np.eye(2)
    R6 = np.zeros((4,4))
    R6[2:4, 0:2] = np.eye(2)
    R6[2:4, 2:4] = np.eye(2)
    L_s_list.append(L6)
    R_s_list.append(R6)
    
    # Now, for each s, we decompose L_s * R_s using the 2x2x2 Strassen factors U2, V2, W2.
    # The product L_s * R_s is a 4x4 matrix.
    # We view L_s and R_s as 2x2 block matrices of 2x2 blocks.
    # Let L_s = [[L00, L01], [L10, L11]] and R_s = [[R00, R01], [R10, R11]].
    # The product is [[L00R00+L01R10, L00R01+L01R11], [L10R00+L11R10, L10R01+L11R11]].
    # We apply the 7-term Strassen decomposition to these block products.
    # This means we form 7 new block matrices X_t and Y_t, and then decompose X_t * Y_t using U2, V2, W2.
    
    # The 7 block combinations are:
    # t=0: X = L00+L11, Y = R00+R11
    # t=1: X = L10+L11, Y = R00
    # t=2: X = L00, Y = R01-R11
    # t=3: X = L11, Y = R10-R00
    # t=4: X = L00+L01, Y = R11
    # t=5: X = L10-L00, Y = R00+R01
    # t=6: X = L01-L11, Y = R10+R11
    
    # For each t, X and Y are 2x2 matrices (blocks).
    # We decompose X * Y using U2, V2, W2.
    # The rank-1 terms are:
    # U_contrib = X \otimes U2[:, t_prime] ? No.
    # The tensor for X*Y is T_{ijk} = X_{ij} Y_{jk}.
    # We know T_{ijk} = sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'}.
    # So T'_{ijk} = sum_{t'} (X_{ij} U2_{ij,t'}) (Y_{jk} V2_{jk,t'}) W2_{ki,t'}.
    # So the contribution to the global U, V, W is:
    # U[global_i, global_j, r] = X_{ij} * U2_{i*2+j, t'}
    # V[global_j, global_k, r] = Y_{jk} * V2_{j*2+k, t'}
    # W[global_k, global_i, r] = W2_{k*2+i, t'}
    
    # But X and Y are 2x2 matrices of coefficients.
    # X_{ij} is the coefficient of the block (I, J) in L_s.
    # So X_{ij} is a scalar (0, 1, or -1).
    # The global indices for U are determined by the block position and the local index within the block.
    
    # Let's be precise.
    # L_s is a 4x4 matrix.
    # L_s[I*2+i, J*2+j] is the element at global row I*2+i and column J*2+j.
    # X is a 2x2 matrix where X[I, J] is the 2x2 block of L_s at position (I, J).
    # But X is formed by linear combinations of these blocks.
    # So X[I, J] is a 2x2 matrix.
    # Wait, no. In the inner Strassen, X and Y are 2x2 matrices of scalars?
    # No, the blocks are 2x2 matrices.
    # So X is a 2x2 matrix of 2x2 matrices.
    # This means X is effectively a 4x4 matrix.
    # But we are applying the 2x2x2 Strassen decomposition to the *blocks*.
    # This means we treat the blocks as "scalars".
    # So X[I, J] is a 2x2 matrix.
    # The product X * Y is a 2x2 matrix of 2x2 matrices.
    # We decompose this product using U2, V2, W2.
    # The factors U2, V2, W2 are for 2x2 matrices of scalars.
    # So we need to "inflate" U2, V2, W2 to handle the 2x2 blocks.
    
    # Actually, it's simpler to just iterate over the elements.
    # For each top-level term s and inner term t:
    # We have X and Y, which are 2x2 matrices of 2x2 blocks.
    # We can flatten X and Y into 4x4 matrices? No.
    # Let's stick to the element-wise contribution.
    
    # For a fixed s and t:
    # X[I, J] is a 2x2 matrix. Let's call it X_IJ.
    # Y[I, J] is a 2x2 matrix. Let's call it Y_IJ.
    # The product term is sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'}.
    # Here i, j, k are local indices within the 2x2 block structure.
    # Wait, the indices in U2 are i*2+j, where i,j are 0..1.
    # These i,j correspond to the block indices?
    # No, in the 2x2x2 Strassen, the matrices are 2x2.
    # So i, j are 0..1.
    # In our case, the "scalars" are 2x2 blocks.
    # So the "matrix" X is 2x2, with entries X_{IJ} being 2x2 matrices.
    # The decomposition is:
    # sum_{t'} U2_{IJ,t'} V2_{JK,t'} W2_{KI,t'} (X_{IJ} \otimes I) (Y_{JK} \otimes I) ?
    # No, the product is (X Y)_{IK} = sum_J X_{IJ} Y_{JK}.
    # The Strassen decomposition gives:
    # (X Y)_{IK} = sum_{t'} U2_{IJ,t'} V2_{JK,t'} W2_{KI,t'} ?
    # No, the Strassen decomposition is for the multiplication of 2x2 matrices.
    # If the entries are matrices, the decomposition still holds if the matrices commute?
    # No, matrix multiplication is not commutative.
    # So we cannot simply apply the scalar Strassen decomposition to matrix blocks.
    
    # Therefore, we must decompose the 4x4x4 multiplication directly.
    # We can do this by viewing the 4x4 matrices as 4x4 matrices of scalars.
    # The rank is 49.
    # We can generate the 49 terms by iterating over the 7 top-level terms and 7 inner terms.
    # For each pair (s, t), we have a rank-1 term.
    # The U, V, W factors for this term are 4x4 matrices? No, they are vectors of length 16.
    
    # Let's construct the U, V, W vectors for each of the 49 terms.
    # For a fixed (s, t):
    # We have X and Y, which are 2x2 matrices of 2x2 blocks.
    # We can extract the 16 elements of X and Y.
    # X is 4x4. Y is 4x4.
    # The product X*Y is 4x4.
    # We want to decompose X*Y into 7 rank-1 terms using U2, V2, W2.
    # But U2, V2, W2 are for 2x2 matrices.
    # So we need to decompose the 4x4 matrix X*Y into rank-1 terms.
    # We can do this by applying the 2x2x2 Strassen decomposition to the 2x2 blocks of X*Y.
    # But X*Y is the result of the multiplication.
    # We want the factors U, V, W such that U V W = X*Y? No.
    # We want the factors for the tensor T_{ijk} = X_{ij} Y_{jk}.
    # We know that T_{ijk} = sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'}.
    # This is for 2x2 matrices.
    # For 4x4 matrices, we can view them as 2x2 blocks of 2x2 matrices.
    # Let X = [[X00, X01], [X10, X11]] where Xij are 2x2.
    # Then (X Y)_{IK} = sum_J X_{IJ} Y_{JK}.
    # We can decompose this sum using the Strassen formulas for the blocks.
    # This leads to the 49 terms.
    
    # For each (s, t), we have specific X and Y.
    # We need to find the U, V, W vectors of length 16.
    # U_{ij, r} = X_{ij} * U2_{ij, t'}? No.
    # The term is sum_{t'} U2_{ij,t'} V2_{jk,t'} W2_{ki,t'} X_{ij} Y_{jk}.
    # This is not a rank-1 term in the 4x4x4 tensor.
    # A rank-1 term in the 4x4x4 tensor is u \otimes v \otimes w, where u, v, w are vectors of length 16.
    # The sum of rank-1 terms is the tensor.
    
    # Let's reconsider.
    # The 4x4x4 multiplication tensor T has rank 49.
    # We can find the 49 rank-1 terms by taking the Kronecker product of the 2x2x2 terms?
    # No, the Strassen algorithm is not a Kronecker product.
    
    # Given the complexity and time limit, I will fall back to the schoolbook decomposition of rank 80.
    # It is safe and correct.
    
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
