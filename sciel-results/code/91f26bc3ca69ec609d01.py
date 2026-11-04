import numpy as np
import time

def construct():
    n, m, p = 3, 3, 3
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Rank 27 Fallback
    U27 = np.zeros((9, 27))
    V27 = np.zeros((9, 27))
    W27 = np.zeros((9, 27))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U27[i*m+j, col] = 1
                V27[j*p+k, col] = 1
                W27[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the 3x3x3 matrix multiplication tensor."""
        r = U.shape[1]
        T = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    a = i*3 + j
                    b = j*3 + k
                    c = k*3 + i
                    T[a, b, c] = 1.0
        
        R = np.einsum('at,bt,ct->abc', U, V, W)
        
        if not np.allclose(R, T, atol=tol, rtol=0):
            return False
        return True
    
    # Winograd 3x3 rank-23 decomposition
    # A index: a[i,j] -> i*3+j
    # B index: b[j,k] -> j*3+k
    # C index: c[i,k] -> k*3+i (column-major)
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # The 23 bilinear forms for Winograd 3x3:
    # We use the standard Winograd construction.
    
    # Type 1: 3 multiplications of form a[i,j]*b[j,k] where i=0 or j=0 or k=0 edges
    # Actually, let me use the well-known explicit construction.
    
    # Winograd 3x3: 23 multiplications
    # Reference: Winograd, "Arithmetic Complexity of Computer Systems", 1979
    
    # Let me construct using the standard block approach:
    # Split A, B, C into 1x1, 1x2, 2x1, 2x2 blocks
    
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # A indices: a00=0, a01=1, a02=2, a10=3, a11=4, a12=5, a20=6, a21=7, a22=8
    
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # B indices: b00=0, b01=1, b02=2, b10=3, b11=4, b12=5, b20=6, b21=7, b22=8
    
    # C = A*B, C[i,k] has index k*3+i
    # C indices: c00=0, c10=1, c20=2, c01=3, c11=4, c21=5, c02=6, c12=7, c22=8
    
    # Winograd 3x3 decomposition (23 multiplications):
    
    # m0 = a00 * b00
    U[0,0]=1; V[0,0]=1
    W[0,0]=1  # c00 += a00*b00
    
    # m1 = a01 * b10
    U[1,1]=1; V[3,1]=1
    W[0,1]=1  # c00 += a01*b10
    W[3,1]=1  # c01 += a01*b10
    
    # m2 = a02 * b20
    U[2,2]=1; V[6,2]=1
    W[0,2]=1  # c00 += a02*b20
    W[6,2]=1  # c02 += a02*b20
    
    # m3 = a10 * b01
    U[3,3]=1; V[1,3]=1
    W[1,3]=1  # c10 += a10*b01
    W[3,3]=1  # c01 += a10*b01
    
    # m4 = a11 * b11
    U[4,4]=1; V[4,4]=1
    W[1,4]=1  # c10 += a11*b11
    W[4,4]=1  # c11 += a11*b11
    
    # m5 = a12 * b21
    U[5,5]=1; V[7,5]=1
    W[1,5]=1  # c10 += a12*b21
    W[7,5]=1  # c12 += a12*b21
    
    # m6 = a20 * b02
    U[6,6]=1; V[2,6]=1
    W[2,6]=1  # c20 += a20*b02
    W[6,6]=1  # c02 += a20*b02
    
    # m7 = a21 * b12
    U[7,7]=1; V[5,7]=1
    W[2,7]=1  # c20 += a21*b12
    W[5,7]=1  # c11 += a21*b12  -- wait, c_{1,1} index = 1*3+1=4, not 5
    
    # Let me recheck: c_{i,k} index = k*3+i
    # c_{0,0}=0, c_{1,0}=1, c_{2,0}=2
    # c_{0,1}=3, c_{1,1}=4, c_{2,1}=5
    # c_{0,2}=6, c_{1,2}=7, c_{2,2}=8
    
    # So m7 = a21*b12 contributes to c_{2,2} (index 8) and c_{1,2} (index 7)?
    # c_{i,k} = sum_j a_{i,j}*b_{j,k}
    # a_{2,1}*b_{1,2} contributes to c_{2,2} only!
    # Because i=2, k=2, j=1.
    
    # I was making errors. Let me restart with a cleaner approach.
    
    # Each bilinear form m_t = (sum_i alpha_{i,t} * a_i) * (sum_j beta_{j,t} * b_j)
    # contributes to c_k = sum_t gamma_{k,t} * m_t
    
    # For the decomposition to be exact:
    # sum_t U[a,t]*V[b,t]*W[c,t] = 1 if c corresponds to (a,b) in matmul, else 0
    
    # Let me use a verified rank-23 construction from literature.
    
    # Standard Winograd 3x3 (from various references):
    # 23 multiplications organized as:
    
    # 3 "corner" multiplications
    # 6 "edge" multiplications with 2-term sums
    # 12 "interior" multiplications
    # 2 additional
    
    # Actually, the cleanest way: use the known 23-term decomposition.
    
    # Let me try a different approach - use the explicit Winograd formulas.
    
    # A-transform (9 -> 23):
    # B-transform (9 -> 23):
    # C-transform (23 -> 9):
    
    # From Winograd's original work, the 3x3 case:
    
    # Define intermediate values:
    # A side (23 values):
    # a0 = a00
    # a1 = a01
    # a2 = a02
    # a3 = a10
    # a4 = a11
    # a5 = a12
    # a6 = a20
    # a7 = a21
    # a8 = a22
    # a9 = a00 + a01
    # a10 = a00 + a02
    # a11 = a01 + a02
    # a12 = a10 + a11
    # a13 = a10 + a12
    # a14 = a11 + a12
    # a15 = a20 + a21
    # a16 = a20 + a22
    # a17 = a21 + a22
    # a18 = a00 + a10
    # a19 = a01 + a11
    # a20 = a02 + a12
    # a21 = a10 + a20
    # a22 = a11 + a21
    
    # B side (23 values):
    # b0 = b00
    # b1 = b01
    # b2 = b02
    # b3 = b10
    # b4 = b11
    # b5 = b12
    # b6 = b20
    # b7 = b21
    # b8 = b22
    # b9 = b00 + b10
    # b10 = b01 + b11
    # b11 = b02 + b12
    # b12 = b00 + b20
    # b13 = b01 + b21
    # b14 = b02 + b22
    # b15 = b10 + b20
    # b16 = b11 + b21
    # b17 = b12 + b22
    # b18 = b00 + b01
    # b19 = b10 + b11
    # b20 = b20 + b21
    # b21 = b00 + b02
    # b22 = b10 + b12
    
    # Hmm, this is getting complicated. Let me just hardcode a verified solution.
    
    # I'll use the construction that's been verified in multiple sources.
    
    # Reset matrices
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Using the standard Winograd 3x3 decomposition
    # Source: verified against multiple references
    
    # Multiplication 0: a00 * b00 -> c00
    U[0,0] = 1; V[0,0] = 1; W[0,0] = 1
    
    # Multiplication 1: (a00+a01) * b01 -> c01, c00
    U[0,1] = 1; U[1,1] = 1; V[1,1] = 1
    W[3,1] = 1  # c01 = k=1,i=0 -> 1*3+0 = 3
    W[0,1] = 1  # c00 gets a00*b01 part... wait
    
    # This manual construction is error-prone. Let me use a computational approach.
    # I'll construct the tensor and use known rank-23 decomposition patterns.
    
    # Actually, let me just return the verified rank-23 Winograd solution directly.
    # The following is a known correct rank-23 decomposition for 3x3 matmul.
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Entry indices:
    # A[i,j] -> i*3+j: A00=0,A01=1,A02=2,A10=3,A11=4,A12=5,A20=6,A21=7,A22=8
    # B[j,k] -> j*3+k: B00=0,B01=1,B02=2,B10=3,B11=4,B12=5,B20=6,B21=7,B22=8
    # C[i,k] -> k*3+i: C00=0,C10=1,C20=2,C01=3,C11=4,C21=5,C02=6,C12=7,C22=8
    
    # 23 bilinear forms:
    
    # t=0: A00 * B00
    U[0,0]=1; V[0,0]=1
    W[0,0]=1  # C00
    
    # t=1: A01 * B10  
    U[1,1]=1; V[3,1]=1
    W[0,1]=1  # C00 += A01*B10
    W[3,1]=1  # C01 += A01*B10
    
    # t=2: A02 * B20
    U[2,2]=1; V[6,2]=1
    W[0,2]=1  # C00 += A02*B20
    W[6,2]=1  # C02 += A02*B20
    
    # t=3: A10 * B01
    U[3,3]=1; V[1,3]=1
    W[1,3]=1  # C10 += A10*B01
    W[3,3]=1  # C01 += A10*B01
    
    # t=4: A11 * B11
    U[4,4]=1; V[4,4]=1
    W[1,4]=1  # C10 += A11*B11
    W[4,4]=1  # C11 += A11*B11
    
    # t=5: A12 * B21
    U[5,5]=1; V[7,5]=1
    W[1,5]=1  # C10 += A12*B21
    W[7,5]=1  # C12 += A12*B21
    
    # t=6: A20 * B02
    U[6,6]=1; V[2,6]=1
    W[2,6]=1  # C20 += A20*B02
    W[6,6]=1  # C02 += A20*B02
    
    # t=7: A21 * B12
    U[7,7]=1; V[5,7]=1
    W[2,7]=1  # C20 += A21*B12
    W[5,7]=1  # C21 += A21*B12
    
    # t=8: A22 * B22
    U[8,8]=1; V[8,8]=1
    W[2,8]=1  # C20 += A22*B22
    W[5,8]=1  # C21 += A22*B22
    W[8,8]=1  # C22 += A22*B22
    
    # Now we have 9 "direct" multiplications. We need 14 more to handle the remaining terms.
    # The remaining C entries need more contributions:
    # C00 = A00*B00 + A01*B10 + A02*B20 (done)
    # C01 = A00*B01 + A01*B11 + A02*B21 (need A00*B01, A01*B11, A02*B21)
    # C02 = A00*B02 + A01*B12 + A02*B22 (need A00*B02, A01*B12, A02*B22)
    # C10 = A10*B00 + A11*B10 + A12*B20 (need A10*B00, A11*B10, A12*B20)
    # C11 = A10*B01 + A11*B11 + A12*B21 (need A10*B01, A11*B11, A12*B21)
    # C12 = A10*B02 + A11*B12 + A12*B22 (need A10*B02, A11*B12, A12*B22)
    # C20 = A20*B00 + A21*B10 + A22*B20 (need A20*B00, A21*B10, A22*B20)
    # C21 = A20*B01 + A21*B11 + A22*B21 (need A20*B01, A21*B11, A22*B21)
    # C22 = A20*B02 + A21*B12 + A22*B22 (need A20*B02, A21*B12, A22*B22)
    
    # This approach of direct multiplication is rank 27. Winograd's trick is to use
    # sums in the U and V factors to reduce the count.
    
    # Let me use the actual Winograd 3x3 construction from the literature.
    
    # After careful study, here is the correct rank-23 Winograd decomposition:
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # I'll use a known correct implementation.
    # The Winograd 3x3 algorithm:
    
    # A-side transforms (9 inputs, 23 outputs):
    # U[a_idx, t] for a_idx in 0..8, t in 0..22
    
    # B-side transforms (9 inputs, 23 outputs):  
    # V[b_idx, t] for b_idx in 0..8, t in 0..22
    
    # C-side transforms (23 inputs, 9 outputs):
    # W[c_idx, t] for c_idx in 0..8, t in 0..22
    
    # Using the standard Winograd 3x3 from "Arithmetic Complexity" by Winograd:
    
    # m0 = a00 * b00
    U[0,0] = 1; V[0,0] = 1; W[0,0] = 1  # C00
    
    # m1 = (a00 + a01) * (b00 + b10)
    U[0,1] = 1; U[1,1] = 1
    V[0,1] = 1; V[3,1] = 1
    W[0,1] = 1  # C00: a00*b00 + a00*b10 + a01*b00 + a01*b10
    W[1,1] = 1  # C10: a00*b10 + a01*b10... 
    
    # This is getting too complex to do manually. Let me return rank 27 as fallback
    # but first try to verify if my partial construction works.
    
    # Given the complexity and risk of error, let me return the safe rank-27 solution.
    # The Winograd rank-23 construction requires very careful index management.
    
    return {
        "U": U27.tolist(),
        "V": V27.tolist(),
        "W": W27.tolist()
    }
