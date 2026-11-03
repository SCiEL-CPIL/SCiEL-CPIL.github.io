import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block decomposition: 6 = 3 * 2
    # R_6 <= R_3 * R_2
    # R_3 = 27 (Schoolbook)
    # R_2 = 7 (Strassen)
    # Total Rank = 189
    
    # --- 3x3 Schoolbook Decomposition ---
    # U3, V3, W3 of shape (9, 27)
    U3 = np.zeros((9, 27))
    V3 = np.zeros((9, 27))
    W3 = np.zeros((9, 27))
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                # Term for A[I,K] * B[K,J] -> C[I,J]
                # U3 index: I*3 + K
                U3[I * 3 + K, col] = 1
                # V3 index: K*3 + J
                V3[K * 3 + J, col] = 1
                # W3 index: K*3 + I (Trace convention W[k*n+i])
                # Here k=K, i=I for the 3x3 block?
                # Wait. C[I,J] = sum K A[I,K] B[K,J].
                # Output C[I,J].
                # W index is k*n + i.
                # In the tensor definition, W[k, i] multiplies the product term.
                # The product term generates C[I,J].
                # So W should have 1 at [J, I]?
                # Let's check trace convention.
                # C[i, j] = sum_k A[i, k] B[k, j].
                # Tensor T_{i, j, k} = A_{i, k} B_{k, j}.
                # W is usually W[k, i] in the prompt? "W[k*n+i,r]"
                # If W[k*n+i, r] = 1, then the term contributes to C[i, k]?
                # No.
                # Let's verify with the seed program.
                # Seed:
                # U[i*m+j][column] = 1  -> A[i,j]
                # V[j*p+k][column] = 1  -> B[j,k]
                # W[k*n+i][column] = 1  -> C[i,k]?
                # If U is A[i,j] and V is B[j,k], the product is A[i,j]B[j,k].
                # Summing over j gives C[i,k].
                # So the output element is C[i,k].
                # The index in W is k*n + i.
                # This corresponds to row k, column i?
                # If C is n x p, C[i, k] is row i, col k.
                # Index k*n + i is row k, col i.
                # This implies C is transposed? Or the convention is C[k, i]?
                # "Output factor follows the official trace convention W[k*n+i,r]"
                # Usually trace convention for matrix mult tensor T_{ijk} (A_ij B_jk -> C_ik)
                # If W is indexed by k*n+i, it accesses C[i,k]? No, k*n+i is row k, col i.
                # So W corresponds to C[k, i].
                # Let's check seed again.
                # Loop: i, j, k.
                # U[i*m+j] = A[i,j]
                # V[j*p+k] = B[j,k]
                # W[k*n+i] = C[k, i] ??
                # Product A[i,j] B[j,k] sums to C[i,k].
                # If W[k*n+i] is 1, it adds to C[k, i].
                # This means the seed computes C^T?
                # Or my interpretation of W index is wrong.
                # k*n+i: if n=6, k=0, i=1 -> 1. Row 0, Col 1.
                # C[i,k] = C[1,0].
                # So W[k*n+i] puts the value into C[k, i].
                # But the term is A[i,j]B[j,k] which is C[i,k].
                # So Seed computes C[k, i] = A[i,j]B[j,k]?
                # This is C^T[i, k] = C[k, i].
                # So the output is C^T?
                # "Output factor follows the official trace convention"
                # If the evaluator expects C[i, j] = sum A[i, k] B[k, j],
                # and we provide W[k*n+i], we are putting C[i, k] into slot k*n+i (which is C[k, i] in row-major).
                # So we are effectively computing C^T.
                # BUT, the evaluator checks if the decomposition is exact.
                # If the evaluator computes U * V * W^T or similar?
                # Standard check:
                # C = U * (V * W.T)? No.
                # Usually: C[i, j] = sum_r U[i, r] V[j, r] W[k, r]?
                # No.
                # Decomposition: T = sum_r U_r V_r W_r^T.
                # C[i, j] = sum_r U[i, r] V[j, r] W[i_out, r]?
                
                # Let's assume the seed is correct for the evaluator.
                # Seed uses W[k*n+i] for term A[i,j]B[j,k].
                # This term contributes to C[i,k].
                # So W[k*n+i] must correspond to the slot for C[i,k].
                # This implies the evaluator expects the result in a matrix where index k*n+i holds C[i,k].
                # i.e. Result[k, i] = C[i, k].
                # i.e. Result = C^T.
                # Or the evaluator indexes C as C[k, i] for the term A[i, j] B[j, k].
                # This is consistent if the "output" is defined as C^T.
                
                # I will follow the seed's indexing exactly.
                # Seed:
                # U[i*m+j] = 1
                # V[j*p+k] = 1
                # W[k*n+i] = 1
                
                # So for 3x3 block:
                # U3[I*3+K, col] = 1 (A[I,K])
                # V3[K*3+J, col] = 1 (B[K,J])
                # W3[? , col] = 1
                # Term is A[I,K] B[K,J] -> C[I,J].
                # Following seed logic: W index should be [J*3 + I]?
                # Seed: k corresponds to 2nd index of V, i corresponds to 1st index of U.
                # V[j*p+k]. Here j=K, k=J. So V index K*3+J.
                # U[i*m+j]. Here i=I, j=K. So U index I*3+K.
                # W[k*n+i]. Here k=J (from V), i=I (from U).
                # So W index J*3 + I.
                
                W3[J * 3 + I, col] = 1
                col += 1

    # --- 2x2 Strassen Decomposition ---
    # U2, V2, W2 of shape (4, 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Strassen terms
    # m1 = (a+d)(e+h) -> C00, C11
    # m2 = (c+d)e    -> C10, C11
    # m3 = a(f-h)    -> C01, C11
    # m4 = d(g-c)    -> C00, C10
    # m5 = (a+b)h    -> C01, C00
    # m6 = (c-a)(e+f) -> C11
    # m7 = (b-d)(g+h) -> C00
    
    # Indices:
    # U: a=0, b=1, c=2, d=3
    # V: e=0, f=1, g=2, h=3
    # W: C00=0, C01=1, C10=2, C11=3 (Following seed convention W[k*n+i] -> C[i,k]? 
    # Wait. Seed: W[k*n+i] for A[i,j]B[j,k] -> C[i,k].
    # So W index for C[i,k] is k*n+i.
    # For 2x2:
    # C[0,0] -> index 0*2+0 = 0.
    # C[0,1] -> index 1*2+0 = 2.
    # C[1,0] -> index 0*2+1 = 1.
    # C[1,1] -> index 1*2+1 = 3.
    
    # Let's re-verify seed mapping for 2x2.
    # Term A[0,0]B[0,0] -> C[0,0]. W[0,0].
    # Term A[0,1]B[1,0] -> C[0,0]. W[0,0].
    # Term A[0,0]B[0,1] -> C[0,1]. W[1,0] (k=1, i=0).
    # Term A[0,1]B[1,1] -> C[0,1]. W[1,0].
    # Term A[1,0]B[0,0] -> C[1,0]. W[0,1] (k=0, i=1).
    # Term A[1,1]B[1,0] -> C[1,0]. W[0,1].
    # Term A[1,0]B[0,1] -> C[1,1]. W[1,1] (k=1, i=1).
    # Term A[1,1]B[1,1] -> C[1,1]. W[1,1].
    
    # So W indices:
    # C[0,0] -> 0
    # C[0,1] -> 2
    # C[1,0] -> 1
    # C[1,1] -> 3
    
    # Strassen m1 = (a+d)(e+h)
    # Contributes to C[0,0] and C[1,1].
    # C[0,0] coeff 1. Index 0.
    # C[1,1] coeff 1. Index 3.
    # U2[:, 0] = [1, 0, 0, 1]
    # V2[:, 0] = [1, 0, 0, 1]
    # W2[0, 0] = 1, W2[3, 0] = 1
    
    # m2 = (c+d)e
    # Contributes to C[1,0] and C[1,1].
    # C[1,0] coeff 1. Index 1.
    # C[1,1] coeff -1. Index 3.
    # U2[:, 1] = [0, 0, 1, 1]
    # V2[:, 1] = [1, 0, 0, 0]
    # W2[1, 1] = 1, W2[3, 1] = -1
    
    # m3 = a(f-h)
    # Contributes to C[0,1] and C[1,1].
    # C[0,1] coeff 1. Index 2.
    # C[1,1] coeff 1. Index 3.
    # U2[:, 2] = [1, 0, 0, 0]
    # V2[:, 2] = [0, 1, 0, -1]
    # W2[2, 2] = 1, W2[3, 2] = 1
    
    # m4 = d(g-c)
    # Contributes to C[0,0] and C[1,0].
    # C[0,0] coeff 1. Index 0.
    # C[1,0] coeff 1. Index 1.
    # U2[:, 3] = [0, 0, 0, 1]
    # V2[:, 3] = [0, 0, 1, -1]
    # W2[0, 3] = 1, W2[1, 3] = 1
    
    # m5 = (a+b)h
    # Contributes to C[0,1] and C[0,0].
    # C[0,1] coeff 1. Index 2.
    # C[0,0] coeff -1. Index 0.
    # U2[:, 4] = [1, 1, 0, 0]
    # V2[:, 4] = [0, 0, 0, 1]
    # W2[2, 4] = 1, W2[0, 4] = -1
    
    # m6 = (c-a)(e+f)
    # Contributes to C[1,1].
    # C[1,1] coeff 1. Index 3.
    # U2[:, 5] = [1, 0, -1, 0]
    # V2[:, 5] = [1, 1, 0, 0]
    # W2[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    # Contributes to C[0,0].
    # C[0,0] coeff 1. Index 0.
    # U2[:, 6] = [0, 1, 0, -1]
    # V2[:, 6] = [0, 0, 1, 1]
    # W2[0, 6] = 1

    # Combine
    U = np.kron(U3, U2)
    V = np.kron(V3, V2)
    W = np.kron(W3, W2)
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
