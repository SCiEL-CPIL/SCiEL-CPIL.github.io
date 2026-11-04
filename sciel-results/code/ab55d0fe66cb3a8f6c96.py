"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen's rank-7 algorithm for 2x2 matrix multiplication
    # U[i*m+j, s], V[j*p+k, s], W[k*n+i, s]
    # For 2x2: U is 4x7, V is 4x7, W is 4x7
    # Indices: i,j in {0,1} for U; j,k in {0,1} for V; k,i in {0,1} for W
    
    # Strassen's 7 products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # U matrices (4x7): row index = i*2+j
    # V matrices (4x7): row index = j*2+k
    # W matrices (4x7): row index = k*2+i
    
    # For each Strassen term s, we have:
    # U_s: which entries of A are used (with coefficients)
    # V_s: which entries of B are used (with coefficients)
    # W_s: how the product contributes to C (with coefficients)
    
    # Let's define the 7 terms carefully
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A*B = [[C00, C01], [C10, C11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # s=0: M1
    U2[0, 0] = 1  # A00
    U2[3, 0] = 1  # A11
    V2[0, 0] = 1  # B00
    V2[3, 0] = 1  # B11
    W2[0, 0] = 1  # C00 += 1
    W2[3, 0] = 1  # C11 += 1
    
    # s=1: M2
    U2[2, 1] = 1  # A10
    U2[3, 1] = 1  # A11
    V2[0, 1] = 1  # B00
    W2[2, 1] = 1  # C10 += 1
    W2[3, 1] = -1 # C11 -= 1
    
    # s=2: M3
    U2[0, 2] = 1  # A00
    V2[1, 2] = 1  # B01
    V2[3, 2] = -1 # B11
    W2[1, 2] = 1  # C01 += 1
    W2[3, 2] = 1  # C11 += 1
    
    # s=3: M4
    U2[3, 3] = 1  # A11
    V2[2, 3] = 1  # B10
    V2[0, 3] = -1 # B00
    W2[0, 3] = -1 # C00 -= 1
    W2[2, 3] = 1  # C10 += 1
    
    # s=4: M5
    U2[0, 4] = 1  # A00
    U2[1, 4] = 1  # A01
    V2[3, 4] = 1  # B11
    W2[0, 4] = 1  # C00 += 1
    W2[1, 4] = 1  # C01 += 1
    
    # s=5: M6
    U2[2, 5] = 1  # A10
    U2[0, 5] = -1 # A00
    V2[0, 5] = 1  # B00
    V2[1, 5] = 1  # B01
    W2[3, 5] = 1  # C11 += 1
    
    # s=6: M7
    U2[1, 6] = 1  # A01
    U2[3, 6] = -1 # A11
    V2[2, 6] = 1  # B10
    V2[3, 6] = 1  # B11
    W2[0, 6] = -1 # C00 -= 1
    W2[3, 6] = 1  # C11 += 1
    
    # Now use Kronecker product to get 4x4 decomposition of rank 49
    # 4x4 = 2x2 block matrices, each block is 2x2
    # A_block[i2,j2] is 2x2, indexed by (i2*2+i1, j2*2+j1) for i1,j1 in {0,1}
    # But wait - the problem uses flat indexing: U[i*m+j, c] where m=4
    # So A is flattened as A[0..15] where A[i*4+j]
    
    # For 4x4 multiplication, we can view A as a 2x2 block matrix where each
    # block is 2x2. The Kronecker product approach:
    # If we have a rank-r decomposition for 2x2, then the Kronecker product
    # gives rank r^2 for 4x4.
    
    # U_4 has shape (16, 49), V_4 has shape (16, 49), W_4 has shape (16, 49)
    # For each pair (s1, s2) where s1,s2 in 0..6:
    #   col = s1*7 + s2
    #   U_4[:, col] = U2[:, s1] ⊗ U2[:, s2] (Kronecker product, reshaped)
    #   V_4[:, col] = V2[:, s1] ⊗ V2[:, s2]
    #   W_4[:, col] = W2[:, s1] ⊗ W2[:, s2]
    
    rank = 49
    U4 = np.zeros((16, rank))
    V4 = np.zeros((16, rank))
    W4 = np.zeros((16, rank))
    
    for s1 in range(7):
        for s2 in range(7):
            col = s1 * 7 + s2
            # Kronecker product: (a⊗b)[i*4+j] = a[i]*b[j] for a,b in R^4
            U4[:, col] = np.kron(U2[:, s1], U2[:, s2])
            V4[:, col] = np.kron(V2[:, s1], V2[:, s2])
            W4[:, col] = np.kron(W2[:, s1], W2[:, s2])
    
    # Verify the decomposition is correct
    # The matrix multiplication tensor T satisfies:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = A[i,j] * B[j,k] -> C[i,k]
    # With U,V,W: sum_s U[i,j,s] * V[j,k,s] * W[k,i,s] should give the
    # correct contribution to C[i,k]
    
    # Let's verify by checking a few entries
    # Build the full tensor and verify
    def verify_decomposition(U, V, W, n, m, p):
        """Check that sum_s U[i*m+j,s]*V[j*p+k,s]*W[k*n+i,s] 
        correctly reconstructs matrix multiplication."""
        # For each (i,k), check that sum_j A[i,j]*B[j,k] is recovered
        # The identity is: for all i,j,k:
        # sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] 
        # should equal the coefficient that maps A[i,j]*B[j,k] to C[k,i]... 
        # Actually the convention is W[k*n+i], so output index is (k,i) meaning
        # C[i,k] gets contribution. Let me re-read: "W[k*n+i,r]"
        # So the output is indexed by k*n+i, meaning the (i,k)-th entry of C.
        
        # The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
        # In the factorization: C[i,k] = sum_s (sum_j U[i*m+j,s] A[i,j]) * 
        #                              (sum_? V[?,s] B[?]) * W[k*n+i,s]
        # Actually the standard form is:
        # sum_s U_s(A) * V_s(B) * W_s where U_s(A) = sum_{i,j} U[i*m+j,s]*A[i,j]
        # V_s(B) = sum_{j,k} V[j*p+k,s]*B[j,k]
        # and C[i,k] gets contribution sum_s U_s(A)*V_s(B)*W[k*n+i,s]
        
        # So the identity to verify: for all i,j,k,
        # sum_s U[i*m+j,s] * V[j*p+k,s] * W[k*n+i,s] 
        # should be 1 if this is the "delta" term, i.e., it should
        # reconstruct the multiplication.
        
        # Actually, the correct check is:
        # sum_j A[i,j]*B[j,k] = sum_s (sum_{j'} U[i*m+j',s]*A[i,j']) * 
        #                          (sum_{j''} V[j''*p+k,s]*B[j'',k]) * W[k*n+i,s]
        # For this to work for all A,B, we need:
        # sum_s U[i*m+j,s] * V[j'*p+k,s] * W[k*n+i,s] = delta_{j,j'}
        # i.e., it should be 1 when j=j' and 0 otherwise, for each (i,k).
        
        # Wait, that's not quite right either. Let me think more carefully.
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_s [sum_j A[i,j] U[i*m+j,s]] [sum_j B[j,k] V[j*p+k,s]] W[k*n+i,s]
        # For this to equal sum_j A[i,j] B[j,k] for all A,B, we need:
        # sum_s U[i*m+j1,s] V[j2*p+k,s] W[k*n+i,s] = delta_{j1,j2}
        
        # So verify: for all i, k, j1, j2:
        # sum_s U[i*m+j1, s] * V[j2*p+k, s] * W[k*n+i, s] == (1 if j1==j2 else 0)
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        rank_val = U_arr.shape[1]
        
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        expected = 1.0 if j1 == j2 else 0.0
                        # sum_s U[i*m+j1, s] * V[j2*p+k, s] * W[k*n+i, s]
                        val = np.sum(U_arr[i*m+j1, :] * V_arr[j2*p+k, :] * W_arr[k*n+i, :])
                        if abs(val - expected) > 1e-9:
                            return False
        return True
    
    # Convert to lists with Fraction for exactness
    U_list = [[int(U4[i, s]) for s in range(rank)] for i in range(16)]
    V_list = [[int(V4[i, s]) for s in range(rank)] for i in range(16)]
    W_list = [[int(W4[i, s]) for s in range(rank)] for i in range(16)]
    
    # Verify
    if not verify_decomposition(U_list, V_list, W_list, n, m, p):
        # Fallback to schoolbook if verification fails
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
    
    # Check time
    if time.time() - start_time > DEADLINE:
        return {"U": U_list, "V": V_list, "W": W_list}
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
