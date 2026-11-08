import numpy as np
import time

def construct():
    start_time = time.time()
    
    # Strassen 2x2 decomposition matrices
    # Rank 7
    # U, V, W are (4, 7) for 2x2 multiplication
    # A is 2x2, B is 2x2, C = A*B is 2x2
    
    # Indices for 2x2:
    # A: 0:00, 1:01, 2:10, 3:11
    # B: 0:00, 1:01, 2:10, 3:11
    # C: 0:00, 1:01, 2:10, 3:11
    
    U2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0]
    ], dtype=float)
    
    # Wait, let's verify the Strassen matrices properly.
    # p1 = (a00+a11)(b00+b11) -> A: 1,0,0,1 ; B: 1,0,0,1 ; C: 1,0,0,1? No.
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # Let's define the 7 terms explicitly for U, V, W of 2x2.
    # Term k corresponds to the k-th multiplication.
    
    # Term 0: p1 = (A00 + A11) * (B00 + B11)
    # U2[:,0] = [1, 0, 0, 1]
    # V2[:,0] = [1, 0, 0, 1]
    # W2[:,0] = [1, 0, 0, 1] ? No, W is for C.
    # C00 gets +1, C11 gets +1? No, p1 contributes to C00 and C11?
    # Let's check Strassen formula:
    # C00 = p1 + p4 - p5 + p7
    # C01 = p3 + p5
    # C10 = p2 + p4
    # C11 = p1 - p2 + p3 + p6
    
    # So for each term k, W2[:,k] has non-zero entries in C.
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # k=0: p1
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1 # Wait, C00 and C11?
    # Check: C00 has +p1, C11 has +p1. So W2[0,0]=1, W2[3,0]=1. Correct.
    
    # k=1: p2 = (A10+A11)*B00
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1 # C10 has +p2. C11 has -p2.
    W2[3,1] = -1
    
    # k=2: p3 = A00*(B01-B11)
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[1,2] = 1 # C01 has +p3. C11 has +p3.
    W2[3,2] = 1
    
    # k=3: p4 = A11*(B10-B00)
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1 # C00 has +p4. C10 has +p4.
    W2[2,3] = 1
    
    # k=4: p5 = (A00+A01)*B11
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1 # C00 has -p5. C01 has +p5.
    W2[1,4] = 1
    
    # k=5: p6 = (A10-A00)*(B00+B01)
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1 # C11 has +p6.
    
    # k=6: p7 = (A01-A11)*(B10+B11)
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1 # C00 has +p7.
    
    # Verify 2x2 Strassen
    # C = U2 @ diag(p) @ V2^T ? No, the decomposition is T = sum u_k v_k^T w_k^T
    # C_ij = sum_k U2[i,k] V2[j,k] W2[l,k] ? No.
    # The tensor is T_{ijk} = sum_k U2[i,k] V2[j,k] W2[l,k] where l is C index.
    # Actually, standard form: C = sum_k (U[:,k] A) (V[:,k] B) ? No.
    # The factors are U (input A), V (input B), W (output C).
    # C = sum_k W[:,k] * (U[:,k] dot A) * (V[:,k] dot B) ? No.
    # The bilinear form is: C_l = sum_k W2[l,k] * (sum_i U2[i,k] A_i) * (sum_j V2[j,k] B_j)
    
    # Let's test C00.
    # C00 = sum_k W2[0,k] * P_k
    # W2[0,:] = [1, 0, 0, 1, -1, 0, 1]
    # P0 = (A00+A11)(B00+B11)
    # P3 = A11(B10-B00)
    # P4 = (A00+A01)B11
    # P6 = (A01-A11)(B10+B11)
    # Sum: P0 + P3 - P4 + P6
    # Matches C00 = p1 + p4 - p5 + p7?
    # My indices: k=0 is p1, k=3 is p4, k=4 is p5, k=6 is p7.
    # So C00 = p1 + p4 - p5 + p7. Correct.
    
    # Now construct 4x4 factors via Kronecker product.
    # Rank = 7 * 7 = 49
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for k1 in range(7):
        for k2 in range(7):
            k = k1 * 7 + k2
            U4[:, k] = np.kron(U2[:, k1], U2[:, k2])
            V4[:, k] = np.kron(V2[:, k1], V2[:, k2])
            W4[:, k] = np.kron(W2[:, k1], W2[:, k2])
            
    # The interface expects U, V, W as lists of lists or similar.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # n=4, m=4, p=4.
    # U shape (16, 49)
    # V shape (16, 49)
    # W shape (16, 49)
    
    # The trace convention for W is W[k*n+i, r].
    # My W4 is indexed by [C_index, k].
    # C_index is 0..15.
    # The problem statement says "output factor follows the official trace convention W[k*n+i,r]".
    # Usually, this means the output matrix C is formed by summing W[:,r] * (U[:,r] A) * (V[:,r] B).
    # The index of W corresponds to the entry in C.
    # If C is flattened in row-major order, index = i*n + j.
    # The prompt says W[k*n+i, r]. This is slightly ambiguous.
    # "W[k*n+i,r]" suggests the first index is k*n+i.
    # If k is row and i is col, then index = k*n+i.
    # This is standard row-major flattening.
    # My W4 is constructed such that W4[idx, k] is the coefficient for C[idx].
    # So W4 is already in the correct shape (16, 49).
    
    # Convert to lists
    return {
        "U": U4.tolist(),
        "V": V4.tolist(),
        "W": W4.tolist()
    }

if __name__ == "__main__":
    # Quick verification
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Random test
    np.random.seed(42)
    A = np.random.rand(4, 4)
    B = np.random.rand(4, 4)
    
    A_flat = A.flatten()
    B_flat = B.flatten()
    
    # Compute C via decomposition
    # C = sum_k W[:,k] * (U[:,k] dot A_flat) * (V[:,k] dot B_flat)
    # Note: U[:,k] dot A_flat is the scalar product.
    # This corresponds to the bilinear form.
    
    C_calc = np.zeros(16)
    for k in range(49):
        u_val = np.dot(U[:, k], A_flat)
        v_val = np.dot(V[:, k], B_flat)
        C_calc += W[:, k] * u_val * v_val
        
    C_true = (A @ B).flatten()
    
    if not np.allclose(C_calc, C_true):
        print("Verification Failed")
    else:
        print("Verification Passed")
