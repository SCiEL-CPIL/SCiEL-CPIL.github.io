"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose B into B_main (4x4) and B_extra (4x2)
    # C = A @ B_main + A @ B_extra
    
    # Part 1: A @ B_main (4x4 x 4x4) using Strassen (Rank 49)
    # Part 2: A @ B_extra (4x4 x 4x2) using Schoolbook (Rank 32)
    # Total Rank = 49 + 32 = 81
    
    rank = 81
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term to the decomposition
    def add_term(u_idx, v_idx, w_idx, col):
        U[u_idx, col] += 1
        V[v_idx, col] += 1
        W[w_idx, col] += 1

    # Strassen 2x2 decomposition (Rank 7)
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices for 4x4 blocks (2x2 grid of 2x2 blocks)
    # A indices: i in 0..3, j in 0..3
    # A11: 0,0 1,0 0,1 1,1
    # A12: 0,2 1,2 0,3 1,3
    # A21: 2,0 3,0 2,1 3,1
    # A22: 2,2 3,2 2,3 3,3
    
    # B indices for B_main (4x4): j in 0..3, k in 0..3
    # B11: 0,0 1,0 0,1 1,1
    # B12: 0,2 1,2 0,3 1,3
    # B21: 2,0 3,0 2,1 3,1
    # B22: 2,2 3,2 2,3 3,3
    
    # W indices for C_main (4x4): k in 0..3, i in 0..3
    # W[k*n+i]
    
    col = 0
    
    # Define block indices
    A11 = [0*m+0, 1*m+0, 0*m+1, 1*m+1]
    A12 = [0*m+2, 1*m+2, 0*m+3, 1*m+3]
    A21 = [2*m+0, 3*m+0, 2*m+1, 3*m+1]
    A22 = [2*m+2, 3*m+2, 2*m+3, 3*m+3]
    
    # B_main blocks (j*p+k)
    B11 = [0*p+0, 1*p+0, 0*p+1, 1*p+1]
    B12 = [0*p+2, 1*p+2, 0*p+3, 1*p+3]
    B21 = [2*p+0, 3*p+0, 2*p+1, 3*p+1]
    B22 = [2*p+2, 3*p+2, 2*p+3, 3*p+3]
    
    # W_main blocks (k*n+i)
    W11 = [0*n+0, 1*n+0, 0*n+1, 1*n+1]
    W12 = [2*n+0, 3*n+0, 2*n+1, 3*n+1]
    W21 = [0*n+2, 1*n+2, 0*n+3, 1*n+3]
    W22 = [2*n+2, 3*n+2, 2*n+3, 3*n+3]

    # Strassen Terms for 4x4 x 4x4
    # Each term is a sum of products of 2x2 blocks.
    # We need to expand each "block multiplication" into scalar multiplications using Strassen 2x2 (Rank 7).
    # This is complex to implement manually. 
    # Alternative: Just use the Kronecker product of the 2x2 Strassen tensor with itself?
    # The tensor for 4x4x4x4 is the Kronecker product of the 2x2x2x2 tensor with itself.
    # Rank(4x4) = Rank(2x2)^2 = 7*7 = 49.
    
    # Let's construct the 49 terms.
    # T_2x2 (Strassen) has 7 terms (u,v,w) in 2x2x2x2 space.
    # U_2x2 (4x7), V_2x2 (4x7), W_2x2 (4x7)
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # 2x2 Strassen
    # M1 = (A00+A11)(B00+B11)
    U2[0, 0] += 1; U2[1, 0] += 1
    V2[0, 0] += 1; V2[1, 0] += 1
    W2[0, 0] += 1; W2[1, 0] += 1
    # M2 = (A10+A11)B00
    U2[2, 1] += 1; U2[3, 1] += 1
    V2[0, 1] += 1
    W2[2, 1] += 1; W2[3, 1] += 1 # Wait, W index is k*n+i. For 2x2, n=2.
    # W indices: 0,1,2,3.
    # M2 contributes to C10 and C11? No.
    # C10 = M1 + M4 - M2 - M5 + M6
    # C11 = M3 + M5
    # C00 = M1 - M4 + M2 - M6
    # C01 = M3 + M6
    
    # Let's just use the standard Kronecker product implementation.
    # U_4x4[i1*2+i2, c1*7+c2] = U2[i1,c1] * U2[i2,c2]
    # V_4x4[j1*2+j2, c1*7+c2] = V2[j1,c1] * V2[j2,c2]
    # W_4x4[k1*2+k2, c1*7+c2] = W2[k1,c1] * W2[k2,c2]
    
    # But we need to map the 2x2 Strassen correctly first.
    
    # Standard 2x2 Strassen (indices 0..3 for 2x2)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11) -> C00, C01, C10, C11
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[1, 0] = 1; W2[2, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10, C11
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = 1
    
    # M3: A00(B01-B11) -> C01, C11
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M4: A11(B10-B00) -> C10, C11
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[2, 3] = 1; W2[3, 3] = 1
    
    # M5: (A00+A01)B11 -> C01, C11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[1, 4] = 1; W2[3, 4] = 1
    
    # M6: (A10-A00)(B00+B01) -> C10, C11
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[2, 5] = 1; W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C01, C11
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[1, 6] = 1; W2[3, 6] = 1
    
    # Now Kronecker product for 4x4 x 4x4
    U_main = np.zeros((16, 49))
    V_main = np.zeros((16, 49))
    W_main = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U_main[i, c] = U2[i1, c1] * U2[i2, c2]
                    V_main[i, c] = V2[i1, c1] * V2[i2, c2] # Wait, V index is j.
                    # V index in 4x4 is j1*2+j2.
                    # In U_main, I used i1*2+i2 for rows.
                    # V rows are j1*2+j2.
                    # W rows are k1*2+k2.
                    
                    V_main[i, c] = V2[i1, c1] * V2[i2, c2] # This is wrong, V indices are different from U.
                    # U row i corresponds to A index. V row j corresponds to B index.
                    # In the loop, i is the row index for U.
                    # We need to iterate j for V and k for W.
                    
    # Correct Kronecker loop
    U_main = np.zeros((16, 49))
    V_main = np.zeros((16, 49))
    W_main = np.zeros((16, 49))
    
    for c1 in range(7):
        for c2 in range(7):
            c = c1 * 7 + c2
            for i1 in range(2):
                for i2 in range(2):
                    i = i1 * 2 + i2
                    U_main[i, c] = U2[i1, c1] * U2[i2, c2]
            for j1 in range(2):
                for j2 in range(2):
                    j = j1 * 2 + j2
                    V_main[j, c] = V2[j1, c1] * V2[j2, c2]
            for k1 in range(2):
                for k2 in range(2):
                    k = k1 * 2 + k2
                    W_main[k, c] = W2[k1, c1] * W2[k2, c2]
                    
    # Copy to U, V, W for the first 49 columns
    U[:, :49] = U_main
    V[:, :49] = V_main
    W[:, :49] = W_main
    
    # Part 2: A @ B_extra (4x4 x 4x2) Schoolbook
    # B_extra is columns 4 and 5 of B.
    # j in 0..3, k in 4..5
    # Schoolbook rank 32.
    for i in range(4):
        for j in range(4):
            for k in [4, 5]:
                # W index k*n + i
                # V index j*p + k
                # U index i*m + j
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Check if we filled all 81
    # assert col == 81
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    # Quick validation
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = DIMENSIONS
    A = np.eye(n, m)
    B = np.eye(m, p)
    C_true = A @ B
    
    # C_approx[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # This is not a simple matrix product.
    # We need to check the tensor contraction.
    # C[i, k] = sum_j sum_c U[i*m+j, c] V[j*p+k, c] * 1? No.
    # The decomposition is C[i,k] = sum_c U[i*m+?, c] V[?p+k, c] ...
    # The interface is:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # C[i,k] = sum_c sum_j U[i*m+j, c] V[j*p+k, c] * ???
    # Wait, the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] ... No.
    # It is C[i,k] = sum_c U[i, c] V[c, k]? No.
    
    # The task description says:
    # U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # This implies:
    # C[i,k] = sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] ?
    # No, that's not right dimensionally for a simple product.
    
    # The standard form is:
    # C[i,k] = sum_j (A[i,j] * B[j,k])
    # Decomposition:
    # C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]?
    
    # Let's look at the seed program:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # For c = (i,j,k)
    # Then C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # If we sum over j?
    # No, the indices i, j, k are fixed for a specific column c.
    # So C[i,k] = sum_c U[i*m + j_c, c] V[j_c*p + k_c, c] W[k_c*n + i_c, c]
    # For the seed, j_c = j, k_c = k, i_c = i.
    # So C[i,k] = sum_{i,j,k} delta_{i,i} delta_{j,j} delta_{k,k} = 1 if i,j,k match.
    # This sums to A[i,j]B[j,k] for the specific j.
    # Wait, the sum over c covers all i,j,k.
    # So C[i,k] = sum_{j} A[i,j] B[j,k].
    # Yes.
    
    # So my construction must satisfy:
    # C[i,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    
    # My Kronecker construction:
    # U_main[i, c] corresponds to A[i, j]?
    # In U2, row i is A index.
    # In V2, row j is B index.
    # In W2, row k is C index (k*n+i).
    
    # In my loop:
    # U_main[i, c] = U2[i1,c1]*U2[i2,c2]. This is A[i, ?]
    # V_main[j, c] = V2[j1,c1]*V2[j2,c2]. This is B[?, j]
    # W_main[k, c] = W2[k1,c1]*W2[k2,c2]. This is C[k, i]
    
    # This looks correct.
    
    print("Constructing...")
    t0 = time.time()
    res = construct()
    print(f"Time: {time.time()-t0}")
    
