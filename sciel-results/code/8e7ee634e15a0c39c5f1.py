"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    
    # We use a tensor product decomposition:
    # The multiplication tensor M_{n,m,p} can be viewed as a product structure
    # when p is even: group the p output columns into pairs.
    # 
    # For each pair of output columns (2t, 2t+1), we have a sub-tensor that
    # multiplies A (n x m) by a 2-column block B_t (m x 2).
    # This sub-tensor has the structure of a 2x2x2 matrix multiplication.
    #
    # We use Strassen's 7-term decomposition for each 2x2x2 block.
    # Total rank = 7 * (p/2) = 7 * 8 = 56.
    
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56
    
    # Strassen's 7 products for 2x2 matrix multiplication:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B where C_{ik} = sum_j A_{ij} B_{jk}
    #
    # Strassen factors (u, v, w) such that:
    # sum_r u_r[i] v_r[j] w_r[k] = T[i,j,k]
    # where T[i,j,k] is 1 if C[i,k] involves A[i,j]*B[j,k]
    # 
    # The 7 rank-1 tensors for 2x2x2 matrix multiplication:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each factor is defined by vectors u (len 2), v (len 2), w (len 2)
    # u corresponds to A's rows/cols structure? No, u[i,j] is the coefficient for A[i,j].
    # In the tensor decomposition T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i] ... 
    # Wait, the standard decomposition is:
    # C_{i,k} = sum_r (sum_j U_{i,j}^{(r)} A_{i,j}) * (sum_k V_{j,k}^{(r)} B_{j,k}) * W_{k,i}^{(r)} ?
    # No, the bilinear form is:
    # C_{i,k} = sum_r U_{i}^{(r)} * (sum_j A_{i,j} V_{j}^{(r)}) * W_{k}^{(r)}? No.
    
    # Correct Tensor Decomposition Definition:
    # The tensor T for matrix mult has entries T[i, j, k] = 1.
    # A rank-1 term is u ⊗ v ⊗ w.
    # The contribution to C[i,k] from term r is:
    # sum_j (u_r[i] * v_r[j]) * (w_r[k] * ...)?
    
    # Let's stick to the explicit factor matrices provided in the task description format:
    # U has shape (n*m, rank). Row index is i*m + j.
    # V has shape (m*p, rank). Row index is j*p + k.
    # W has shape (p*n, rank). Row index is k*n + i.
    
    # The reconstruction formula is:
    # C[i, k] = sum_r sum_j U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] ?
    # No, that would be a trilinear form on A, B, C.
    # The standard interpretation for these factor matrices in tensor decomposition of matrix mult is:
    # C = sum_r (U_r ⊗ V_r) W_r^T ? No.
    
    # Let's look at the schoolbook case again.
    # For each (i,j,k), U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1.
    # This implies the term is: A[i,j] * B[j,k] * 1 (for C[i,k]).
    # So, C[i,k] = sum_{i,j,k} A[i,j] B[j,k] (U[i*m+j] V[j*p+k] W[k*n+i])
    # This works if we sum over all i,j,k.
    
    # So for Strassen, we need to find U, V, W such that:
    # sum_r sum_j U[i*m+j, r] A[i,j] V[j*p+k, r] B[j,k] W[k*n+i, r] = C[i,k]
    # Note: In the schoolbook, U, V, W are 0/1.
    # In Strassen, U, V, W can be integers.
    
    # The local 2x2x2 Strassen factors (u, v, w) where u is 2x2 matrix for A, v is 2x2 for B, w is 2x2 for C?
    # No, the factors are vectors in the decomposition space.
    # Let's define the 7 factors for a single 2x2x2 block (indices 0,1 for i,j,k):
    # Factor r contributes:
    # C_local[i,k] += sum_j U_local[i,j,r] * A[i,j] * V_local[j,k,r] * B[j,k] * W_local[k,i,r]
    # Actually, usually it's simpler:
    # C = sum_r (A U_r) (V_r B) W_r ? No.
    
    # Let's use the explicit definition from the "unbalanced" strategy hint:
    # The tensor is a sum of rank-1 tensors.
    # T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r is in R^{n*m}, v_r in R^{m*p}, w_r in R^{p*n}.
    # The condition is that for any A, B:
    # (A ⊗ B) : T = C
    # i.e., sum_{i,j,k} A[i,j] B[j,k] T[i,j,k] = C[i,k] ... wait, C is a matrix, not a scalar.
    # The bilinear map is (A, B) -> C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is defined such that C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k]?
    # No, T[i,j,k] is just 1 for all i,j,k in schoolbook?
    # If T[i,j,k] = 1, then sum_j 1 * A[i,j] B[j,k] = C[i,k]. Yes.
    # So the tensor entries T[i,j,k] are all 1.
    # We want to decompose T[i,j,k] = 1 into sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # No, the indices don't match.
    # T is a 3-tensor with indices i (n), j (m), k (p).
    # U is indexed by (i,j). V by (j,k). W by (k,i).
    # This is a different structure.
    
    # Let's re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    # This corresponds to the "matrix multiplication tensor" in the sense of the bilinear map.
    # The standard decomposition is:
    # C_{i,k} = sum_r (sum_j U_{i,j}^{(r)} A_{i,j}) * (sum_k V_{j,k}^{(r)} B_{j,k}) ... no.
    
    # Let's look at the schoolbook mapping again:
    # U[i*m + j, col] = 1
    # V[j*p + k, col] = 1
    # W[k*n + i, col] = 1
    # This creates a term that picks out A[i,j] and B[j,k] and puts it into C[i,k].
    # So the term is A[i,j] * B[j,k] * 1 (for C[i,k]).
    # Summing over all i,j,k gives C[i,k] = sum_j A[i,j] B[j,k].
    
    # So, for Strassen, we need to group these terms.
    # For a 2x2x2 block (i,j,k in {0,1}), we have 8 schoolbook terms.
    # We want to replace them with 7 terms.
    # The 7 Strassen terms for 2x2x2 are:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B01-B11) -> C01, C11
    # M4 = A11(B10-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C01, C11
    # M6 = (A10-A00)(B00+B01) -> C10, C11
    # M7 = (A01-A11)(B10+B11) -> C00, C01
    
    # Let's define the factors U, V, W for each Strassen product r in a 2x2x2 block.
    # U_r is a 4-vector (for A00, A01, A10, A11).
    # V_r is a 4-vector (for B00, B01, B10, B11).
    # W_r is a 4-vector (for C00, C01, C10, C11).
    
    # M1:
    # A part: A00+A11 -> U = [1, 0, 0, 1]
    # B part: B00+B11 -> V = [1, 0, 0, 1]
    # C part: C00+C11 -> W = [1, 0, 0, 1] (Indices: C00=0, C01=1, C10=2, C11=3)
    # Check: C00 += 1*1*1 = 1. C11 += 1*1*1 = 1.
    
    # M2:
    # A part: A10+A11 -> U = [0, 0, 1, 1]
    # B part: B00 -> V = [1, 0, 0, 0]
    # C part: C10+C11 -> W = [0, 0, 1, 1]
    # Check: C10 += 1*1*1 = 1. C11 += 1*1*1 = 1.
    # Wait, C11 needs to accumulate from multiple terms.
    # Let's verify the sum for C11:
    # M1: 1
    # M2: 1
    # M3: 1
    # M6: 1
    # Sum = 4? But C11 = A00B01 + A01B11 + A10B01 + A11B11.
    # Let's check the coefficients.
    
    # Actually, the standard Strassen factors are:
    # m1 = (a11 + a22)(b11 + b22)
    # m2 = (a21 + a22)b11
    # m3 = a11(b12 - b22)
    # m4 = a22(b21 - b11)
    # m5 = (a11 + a12)b22
    # m6 = (a21 - a11)(b11 + b12)
    # m7 = (a12 - a22)(b21 + b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Let's map indices:
    # A: 00, 01, 10, 11 (indices 0,1,2,3)
    # B: 00, 01, 10, 11 (indices 0,1,2,3)
    # C: 00, 01, 10, 11 (indices 0,1,2,3)
    
    # Factor 1 (m1):
    # U1 = [1, 0, 0, 1]  (A00+A11)
    # V1 = [1, 0, 0, 1]  (B00+B11)
    # W1 = [1, 0, 0, 1]  (C00+C11)
    
    # Factor 2 (m2):
    # U2 = [0, 0, 1, 1]  (A10+A11)
    # V2 = [1, 0, 0, 0]  (B00)
    # W2 = [0, 0, 1, 1]  (C10+C11)
    
    # Factor 3 (m3):
    # U3 = [1, 0, 0, 0]  (A00)
    # V3 = [0, 1, 0, -1] (B01-B11)
    # W3 = [0, 1, 0, 1]  (C01+C11)
    
    # Factor 4 (m4):
    # U4 = [0, 0, 0, 1]  (A11)
    # V4 = [0, 0, 1, -1] (B10-B00)
    # W4 = [1, 0, 1, 0]  (C00+C10)
    
    # Factor 5 (m5):
    # U5 = [1, 1, 0, 0]  (A00+A01)
    # V5 = [0, 0, 0, 1]  (B11)
    # W5 = [0, 1, 0, 1]  (C01+C11)
    
    # Factor 6 (m6):
    # U6 = [0, 0, 1, -1] (A10-A00)
    # V6 = [1, 1, 0, 0]  (B00+B01)
    # W6 = [0, 0, 1, 1]  (C10+C11)
    
    # Factor 7 (m7):
    # U7 = [0, 1, 0, -1] (A01-A11)
    # V7 = [0, 0, 1, 1]  (B10+B11)
    # W7 = [1, 0, 0, 1]  (C00+C11)
    
    # Let's verify C00:
    # m1: 1
    # m4: 1
    # m5: 0
    # m7: 1
    # Sum: 1 + 1 - 0 + 1 = 3?
    # Formula: C00 = m1 + m4 - m5 + m7.
    # Coeffs: 1, 1, -1, 1.
    # My W vectors:
    # W1[0]=1, W4[0]=1, W5[0]=0, W7[0]=1.
    # So C00 gets 1*m1 + 1*m4 + 0*m5 + 1*m7.
    # But the formula says -m5.
    # So W5[0] should be -1?
    # Let's check m5 contribution to C00.
    # m5 = (A00+A01)B11.
    # C00 = A00B00 + A01B10.
    # m5 does not contribute to C00 directly in the sum?
    # Wait, C00 = m1 + m4 - m5 + m7.
    # So yes, W5[0] must be -1.
    
    # Let's re-verify all W vectors based on the C formulas:
    # C00 = m1 + m4 - m5 + m7  -> W[0] for each factor:
    #   W1[0]=1, W4[0]=1, W5[0]=-1, W7[0]=1. Others 0.
    # C01 = m3 + m5             -> W[1] for each factor:
    #   W3[1]=1, W5[1]=1. Others 0.
    # C10 = m2 + m4             -> W[2] for each factor:
    #   W2[2]=1, W4[2]=1. Others 0.
    # C11 = m1 - m2 + m3 + m6   -> W[3] for each factor:
    #   W1[3]=1, W2[3]=-1, W3[3]=1, W6[3]=1. Others 0.
    
    # Now let's fix the W vectors:
    # W1 = [1, 0, 0, 1]
    # W2 = [0, 0, 1, -1]
    # W3 = [0, 1, 0, 1]
    # W4 = [1, 0, 1, 0]
    # W5 = [-1, 1, 0, 1]
    # W6 = [0, 0, 1, 1]
    # W7 = [1, 0, 0, 1]
    
    # Now U and V vectors:
    # U1 = [1, 0, 0, 1]
    # V1 = [1, 0, 0, 1]
    
    # U2 = [0, 0, 1, 1]
    # V2 = [1, 0, 0, 0]
    
    # U3 = [1, 0, 0, 0]
    # V3 = [0, 1, 0, -1]
    
    # U4 = [0, 0, 0, 1]
    # V4 = [0, 0, 1, -1]
    
    # U5 = [1, 1, 0, 0]
    # V5 = [0, 0, 0, 1]
    
    # U6 = [0, 0, 1, -1]
    # V6 = [1, 1, 0, 0]
    
    # U7 = [0, 1, 0, -1]
    # V7 = [0, 0, 1, 1]
    
    # These are the local factors for a 2x2x2 block.
    # We now map these to the global 2x2x16 problem.
    # For each block t (0 to 7), the local indices 0,1 map to global columns 2t, 2t+1.
    
    strassen_factors = [
        # (U, V, W)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 1, 1]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1]),
    ]
    
    col = 0
    for s in range(7):
        u_s, v_s, w_s = strassen_factors[s]
        for t in range(num_blocks):
            # Fill U
            # U row index: i*m + j. i in 0..1, j in 0..1.
            # u_s is length 4: [A00, A01, A10, A11]
            # Map: u_s[0]->(0,0), u_s[1]->(0,1), u_s[2]->(1,0), u_s[3]->(1,1)
            U[0][col] = u_s[0]
            U[1][col] = u_s[1]
            U[2][col] = u_s[2]
            U[3][col] = u_s[3]
            
            # Fill V
            # V row index: j*p + k. j in 0..1, k in 0..15.
            # v_s is length 4: [B00, B01, B10, B11] for local block.
            # Local B00 -> Global B[j=0, k=2t]
            # Local B01 -> Global B[j=0, k=2t+1]
            # Local B10 -> Global B[j=1, k=2t]
            # Local B11 -> Global B[j=1, k=2t+1]
            
            V[0 * p + (2 * t)][col] = v_s[0]
            V[0 * p + (2 * t + 1)][col] = v_s[1]
            V[1 * p + (2 * t)][col] = v_s[2]
            V[1 * p + (2 * t + 1)][col] = v_s[3]
            
            # Fill W
            # W row index: k*n + i. k in 0..15, i in 0..1.
            # w_s is length 4: [C00, C01, C10, C11] for local block.
            # Local C00 -> Global C[k=2t, i=0]
            # Local C01 -> Global C[k=2t+1, i=0]
            # Local C10 -> Global C[k=2t, i=1]
            # Local C11 -> Global C[k=2t+1, i=1]
            
            W[(2 * t) * n + 0][col] = w_s[0]
            W[(2 * t + 1) * n + 0][col] = w_s[1]
            W[(2 * t) * n + 1][col] = w_s[2]
            W[(2 * t + 1) * n + 1][col] = w_s[3]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
