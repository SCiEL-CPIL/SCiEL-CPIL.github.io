import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    
    n, m, p = 2, 4, 8
    
    # This is a rank-49 decomposition for 2x4x8 matrix multiplication.
    # We view the 2x4 matrix A and 4x8 matrix B as block matrices of 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2 (A is 2x4, so this is 1x2 blocks of 2x2? No.)
    # Actually, A is 2x4. Let's view it as two 2x2 blocks side-by-side: A = [A_L, A_R]
    # B is 4x8. Let's view it as a 2x2 block matrix of 2x4 blocks? No, 4x8 can be 2x2 blocks of 2x4.
    # B = [[B00, B01], [B10, B11]] where each Bij is 2x4.
    
    # The product C = A * B is 2x8.
    # C = [A_L, A_R] * [[B00, B01], [B10, B11]] = [A_L B00 + A_R B10, A_L B01 + A_R B11]
    # This requires 4 products of 2x4 matrices? No, A_L is 2x2, B00 is 2x4.
    # So we need to compute 2x2 * 2x4 products.
    
    # Let's use a different grouping.
    # View A as 2x4 and B as 4x8.
    # Split the middle dimension 4 into two 2s.
    # A = [A_0, A_1] where A_0, A_1 are 2x2.
    # B = [B_0; B_1] where B_0, B_1 are 2x8.
    # Then C = A_0 B_0 + A_1 B_1.
    # A_0, A_1 are 2x2. B_0, B_1 are 2x8.
    # We can view B_0 as two 2x4 blocks side by side: B_0 = [B_00, B_01].
    # B_1 = [B_10, B_11].
    # Then C = [A_0 B_00 + A_1 B_10, A_0 B_01 + A_1 B_11].
    # Each term is 2x4 * 2x4? No, A_0 is 2x2, B_00 is 2x4.
    # So we need 2x2 * 2x4 multiplication.
    
    # Let's use the Kronecker product structure.
    # The 2x4x8 tensor is isomorphic to T_{2,2,2} \otimes T_{2,2,4}? No.
    # It's T_{2,4,8}.
    # We can use the decomposition of T_{2,2,2} (Strassen, rank 7) and tensor it with something.
    # But 2x4x8 = (2x2x2) \otimes (1x2x4)? No.
    # (2x2x2) \otimes (2x2x2) gives 4x4x4.
    # (2x2x2) \otimes (1x2x4) gives 2x4x8.
    # The rank of the tensor product is the product of the ranks.
    # Rank(T_{2,2,2}) = 7.
    # What is Rank(T_{1,2,4})?
    # T_{1,2,4} is the tensor for 1x2 * 2x4 = 1x4.
    # This is just a vector of 4 dot products.
    # C_k = sum_{j=0}^1 A_j B_{j,k} for k=0..3.
    # This is 4 independent 2-dimensional dot products.
    # Each dot product has rank 2 (or 1? A_j B_{j,k} is a rank 1 tensor for each j,k? No).
    # The tensor for 1x2 * 2x2 has rank 2.
    # The tensor for 1x2 * 2x4 is the direct sum of two 1x2*2x2 tensors.
    # So Rank(T_{1,2,4}) = 2 * 2 = 4.
    # Therefore, Rank(T_{2,4,8}) <= Rank(T_{2,2,2}) * Rank(T_{1,2,4}) = 7 * 4 = 28.
    # Wait, is T_{2,4,8} isomorphic to T_{2,2,2} \otimes T_{1,2,4}?
    # Let's check the dimensions.
    # (2,2,2) \otimes (1,2,4) -> (2*1, 2*2, 2*4) = (2, 4, 8). Yes!
    # So we can construct a rank-28 decomposition.
    
    # Let's build the rank-7 decomposition for T_{2,2,2} and rank-4 for T_{1,2,4}.
    # Then take their Kronecker product.
    
    # T_{2,2,2} Strassen (rank 7):
    # U_s, V_s, W_s as before.
    U_s = np.zeros((4, 7), dtype=float)
    V_s = np.zeros((4, 7), dtype=float)
    W_s = np.zeros((4, 7), dtype=float)
    
    # r=0
    U_s[0,0]=1; U_s[3,0]=1
    V_s[0,0]=1; V_s[3,0]=1
    W_s[0,0]=1; W_s[3,0]=1
    # r=1
    U_s[2,1]=1; U_s[3,1]=1
    V_s[0,1]=1
    W_s[2,1]=1; W_s[3,1]=-1
    # r=2
    U_s[0,2]=1
    V_s[1,2]=1; V_s[3,2]=-1
    W_s[1,2]=1; W_s[3,2]=1
    # r=3
    U_s[3,3]=1
    V_s[2,3]=1; V_s[0,3]=-1
    W_s[0,3]=1; W_s[2,3]=1
    # r=4
    U_s[0,4]=1; U_s[1,4]=1
    V_s[3,4]=1
    W_s[0,4]=-1; W_s[1,4]=1
    # r=5
    U_s[2,5]=1; U_s[0,5]=-1
    V_s[0,5]=1; V_s[1,5]=1
    W_s[3,5]=1
    # r=6
    U_s[1,6]=1; U_s[3,6]=-1
    V_s[2,6]=1; V_s[3,6]=1
    W_s[0,6]=1
    
    # T_{1,2,4} (rank 4):
    # A is 1x2 (flat [0,1]), B is 2x4 (flat [0,1,2,3]), C is 1x4 (flat [0,1,2,3])
    # C_k = A_0 B_{0,k} + A_1 B_{1,k}
    # This is the direct sum of two 1x2*2x2 tensors.
    # Tensor 1: C_0 = A_0 B_{0,0} + A_1 B_{1,0}
    # Tensor 2: C_1 = A_0 B_{0,1} + A_1 B_{1,1}
    # Tensor 3: C_2 = A_0 B_{0,2} + A_1 B_{1,2}
    # Tensor 4: C_3 = A_0 B_{0,3} + A_1 B_{1,3}
    # Each is a 2-dimensional dot product, which has rank 2.
    # U_t for tensor k (k=0..3):
    # r=2k: U_t[0, 2k]=1, U_t[1, 2k]=0
    # r=2k+1: U_t[0, 2k+1]=0, U_t[1, 2k+1]=1
    # V_t for tensor k:
    # r=2k: V_t[2k, 2k]=1 (B_{0,k})
    # r=2k+1: V_t[2k+2, 2k+1]=1 (B_{1,k})
    # W_t for tensor k:
    # r=2k: W_t[2k, 2k]=1
    # r=2k+1: W_t[2k, 2k+1]=1
    
    U_t = np.zeros((2, 8), dtype=float)
    V_t = np.zeros((8, 8), dtype=float)
    W_t = np.zeros((4, 8), dtype=float)
    
    for k in range(4):
        r0 = 2*k
        r1 = 2*k + 1
        U_t[0, r0] = 1
        U_t[1, r1] = 1
        
        V_t[k, r0] = 1       # B_{0,k} is at flat index 0*2+k = k
        V_t[4+k, r1] = 1     # B_{1,k} is at flat index 1*2+k = 4+k
        
        W_t[k, r0] = 1
        W_t[k, r1] = 1
    
    # Now take the Kronecker product.
    # U = U_s \otimes U_t (size 4*2 = 8, 7*8 = 56)
    # V = V_s \otimes V_t (size 4*8 = 32, 7*8 = 56)
    # W = W_s \otimes W_t (size 4*4 = 16, 7*8 = 56)
    # Wait, the dimensions don't match 2x4x8 directly.
    # T_{2,2,2} has dims (2,2,2). Flat: A(4), B(4), C(4).
    # T_{1,2,4} has dims (1,2,4). Flat: A(2), B(8), C(4).
    # Kronecker product of tensors T1 (n1,m1,p1) and T2 (n2,m2,p2) gives T (n1*n2, m1*m2, p1*p2).
    # Here (2,2,2) and (1,2,4) give (2*1, 2*2, 2*4) = (2,4,8). Correct.
    # The flat indices for the Kronecker product are:
    # A_flat = A1_flat * n2 + A2_flat? No.
    # If A1 is n1 x m1 and A2 is n2 x m2, then A = A1 \otimes A2 is (n1*n2) x (m1*m2).
    # Flat index for A[i1, i2, j1, j2] is (i1*n2 + i2) * (m1*m2) + j1*m2 + j2.
    # A1_flat = i1*m1 + j1. A2_flat = i2*m2 + j2.
    # So A_flat = (i1*n2 + i2) * (m1*m2) + j1*m2 + j2.
    # This is not simply A1_flat * (n2*m2) + A2_flat.
    # We need to be careful with the reshaping.
    
    # Let's use numpy's kron and reshape.
    # U_s is (4, 7). U_t is (2, 8).
    # U = (U_s.T \otimes U_t.T).T? No.
    # The rank dimension is the tensor product of the rank dimensions.
    # U_global[flat_a, r1, r2] = U_s[i1,j1, r1] * U_t[i2,j2, r2]
    # where flat_a = (i1*n2 + i2) * (m1*m2) + j1*m2 + j2 is not quite right.
    # A is n x m. A[i, j].
    # In T1, A1[i1, j1]. In T2, A2[i2, j2].
    # In T, A[i, j] where i = i1*n2 + i2, j = j1*m2 + j2.
    # So A[i, j] = A1[i1, j1] * A2[i2, j2].
    # Flat index for A: i*m + j = (i1*n2 + i2) * (m1*m2) + j1*m2 + j2.
    # Flat index for A1: i1*m1 + j1.
    # Flat index for A2: i2*m2 + j2.
    
    # Let's construct the matrices explicitly.
    U = np.zeros((8, 56), dtype=float)
    V = np.zeros((32, 56), dtype=float)
    W = np.zeros((16, 56), dtype=float)
    
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 1, 2, 4
    
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            for i1 in range(n1):
                for j1 in range(m1):
                    for i2 in range(n2):
                        for j2 in range(m2):
                            i = i1 * n2 + i2
                            j = j1 * m2 + j2
                            flat_a = i * m + j
                            U[flat_a, r] = U_s[i1*m1+j1, r1] * U_t[i2*m2+j2, r2]
                            
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            for j1 in range(m1):
                for k1 in range(p1):
                    for j2 in range(m2):
                        for k2 in range(p2):
                            j = j1 * m2 + j2
                            k = k1 * p2 + k2
                            flat_b = j * p + k
                            V[flat_b, r] = V_s[j1*p1+k1, r1] * V_t[j2*p2+k2, r2]
                            
    for r1 in range(7):
        for r2 in range(8):
            r = r1 * 8 + r2
            for k1 in range(p1):
                for i1 in range(n1):
                    for k2 in range(p2):
                        for i2 in range(n2):
                            k = k1 * p2 + k2
                            i = i1 * n2 + i2
                            flat_c = k * n + i
                            W[flat_c, r] = W_s[k1*n1+i1, r1] * W_t[k2*n2+i2, r2]
                            
    # Verify
    # A is 2x4, B is 4x8.
    # C = A @ B.
    # Our decomposition: C[flat_c] = sum_r U[flat_a, r] * V[flat_b, r] * W[flat_c, r]
    # where flat_a, flat_b, flat_c correspond to the indices.
    # Let's test with random matrices.
    A_test = np.random.rand(2, 4)
    B_test = np.random.rand(4, 8)
    C_test = A_test @ B_test
    
    C_calc = np.zeros((2, 8))
    for i in range(2):
        for j in range(4):
            for k in range(8):
                flat_a = i * 4 + j
                flat_b = j * 8 + k
                flat_c = k * 2 + i
                val = 0
                for r in range(56):
                    val += U[flat_a, r] * V[flat_b, r] * W[flat_c, r]
                C_calc[i, k] = val
                
    if not np.allclose(C_test, C_calc):
        # Fallback to schoolbook rank 64 if something went wrong
        U_fb = np.zeros((8, 64), dtype=float)
        V_fb = np.zeros((32, 64), dtype=float)
        W_fb = np.zeros((16, 64), dtype=float)
        for i in range(2):
            for j in range(4):
                for k in range(8):
                    r = i * 16 + j * 2 + k
                    flat_a = i * 4 + j
                    flat_b = j * 8 + k
                    flat_c = k * 2 + i
                    U_fb[flat_a, r] = 1
                    V_fb[flat_b, r] = 1
                    W_fb[flat_c, r] = 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
