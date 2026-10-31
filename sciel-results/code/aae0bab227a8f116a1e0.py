"""Matrix multiplication 4x4x4 - 49-rank Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a valid 49-rank exact tensor decomposition for 4x4 matrix multiplication.
    
    This uses the Kronecker product of the 7-rank Strassen algorithm for 2x2 matrices.
    The tensor T[i,j,k] is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    
    U shape: (16, 49)
    V shape: (16, 49)
    W shape: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    
    # Initialize factor matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2 decomposition coefficients
    # Indices for A (U): 0:00, 1:01, 2:10, 3:11
    # Indices for B (V): 0:00, 1:01, 2:10, 3:11
    # Indices for C (W): 0:00, 1:01, 2:10, 3:11 (Note: W index is k*n+i, so row=col, col=row in standard C=AB? 
    # Let's check the trace convention: W[k*n+i, r]. 
    # Standard matmul: C[i,j] = sum_k A[i,k]B[k,j].
    # Tensor T[a,b,c] = 1 if a=i*m+k, b=k*p+j, c=j*n+i.
    # So c corresponds to the pair (j,i) which is the transpose of C's indices (i,j) if we view C as row i col j.
    # Actually, usually C is stored row-major. C_flat = i*n + j.
    # Here W index is k*n + i? No, W[k*n+i]. 
    # Let's trace:
    # Term r contributes to T[i,j,k] if U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] != 0.
    # If we set U,V,W based on Strassen, we need to ensure the indices match.
    #
    # Let's stick to the verified logic from Inspiration 1 which scored 1.3061.
    # It mapped:
    # U[i*4+j] corresponds to A[i,j]
    # V[k*4+l] corresponds to B[k,l]
    # W[j*4+i] corresponds to C[j,i] ? 
    # Wait, the task says W[k*n+i]. 
    # In Inspiration 1: W[j*4+i]. 
    # If n=4, W index is j*4+i. 
    # The task says W[k*n+i]. 
    # There is a mismatch in variable names in my head vs the task.
    # Task: T[i,j,k] -> U[i*m+j], V[j*p+k], W[k*n+i].
    # Inspiration 1: W[j*4+i]. 
    # If we identify the third index of T as 'k' in task notation, it maps to W row.
    # In standard matmul C = A B, C[i,j] = sum_k A[i,k]B[k,j].
    # The tensor is T[i, k, j] = A[i,k] B[k,j] C[j,i]? No.
    # The tensor for matmul is T_{i,j,k} = 1 if we are computing C_{i,j} using A_{i,k} and B_{k,j}.
    # Usually T is indexed by (row_A, col_A/row_B, col_B).
    # Let's assume the standard bilinear form:
    # C_{i,j} = sum_k A_{i,k} B_{k,j}
    # This is a 3-tensor T_{i,k,j}.
    # The task defines indices as i, j, k for U, V, W respectively?
    # U[i*m+j] -> i is row of A, j is col of A.
    # V[j*p+k] -> j is row of B, k is col of B.
    # W[k*n+i] -> k is row of C? i is col of C?
    # If C = A B, then C_{row, col} = sum_mid A_{row, mid} B_{mid, col}.
    # So A index: (row, mid). B index: (mid, col). C index: (col, row)?? 
    # If W is indexed by k*n+i, and k is the "mid" index from B's row and A's col...
    # Let's look at the indices in the task:
    # U[i*m+j]: i (row A), j (col A)
    # V[j*p+k]: j (row B), k (col B)
    # W[k*n+i]: k (row C?), i (col C?)
    # For consistency in matmul C = A B:
    # A is n x m. B is m x p. C is n x p.
    # C_{i, l} = sum_j A_{i,j} B_{j,l}.
    # Here m is the shared dimension.
    # U index: i (row A), j (col A / row B).
    # V index: j (row B), k (col B / col C).
    # W index: k (col C), i (row C).
    # So W[k*n+i] corresponds to C_{i,k} but flattened as k*n+i? 
    # If C is n x p (4x4), and we flatten row-major, C_{i,k} is i*4+k.
    # But the task says W[k*n+i]. This is column-major flattening of C, or C is transposed.
    # Let's trust the Inspiration 1 implementation which worked.
    # It used W[j*4+i] where j was the "col B" index and i was "row A" index.
    # In my loop variables from Insp 1:
    # i, j for U (A)
    # k, l for V (B) -- wait, Insp 1 used k,l for V.
    # j, i for W (C) -- Insp 1 used j,i for W.
    # Let's re-verify Insp 1 variable usage.
    # U[i*4+j]
    # V[k*4+l]
    # W[j*4+i]
    # This implies the third index of the tensor is linked to (j,i).
    # If the task requires W[k*n+i], and in my derivation k is the col of B (which is col of C) and i is row of A (which is row of C), then W[k*n+i] matches W[col_C * n + row_C].
    # So the mapping is consistent if we treat the W index as `col_C * 4 + row_C`.
    #
    # Let's hardcode the Strassen 2x2 matrices.
    
    # Strassen 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = A B
    
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # We define U2, V2, W2 for the 2x2 case.
    # Rows are 0:00, 1:01, 2:10, 3:11
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M0
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1 # C00, C11
    
    # M1
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1 # C10, C11
    
    # M2
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1 # C01, C11
    
    # M3
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1 # C00, C10
    
    # M4
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1 # C00, C01
    
    # M5
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1 # C11
    
    # M6
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1 # C00

    # Precompute non-zero entries for efficiency
    # List of (row, col, val)
    U2_nz = []
    V2_nz = []
    W2_nz = []
    for r in range(4):
        for c in range(7):
            if U2[r][c] != 0: U2_nz.append((r, c, U2[r][c]))
            if V2[r][c] != 0: V2_nz.append((r, c, V2[r][c]))
            if W2[r][c] != 0: W2_nz.append((r, c, W2[r][c]))
    
    # Construct 4x4 decomposition
    # Rank index r4 = s * 7 + t, where s is block Strassen index, t is inner Strassen index.
    # A[i,j] where i = I*2 + ii, j = J*2 + jj.
    # B[k,l] where k = K*2 + kk, l = L*2 + ll.
    # C[j,i] (based on W index j*4+i) where j = J*2 + jj, i = I*2 + ii.
    # Note: W index in task is k*n+i. In our mapping, the "col" of C is the 2nd index of the pair (col, row).
    # So W index = col_C * 4 + row_C.
    # col_C corresponds to the column index of the result matrix.
    # In C = A B, C_{row, col}. 
    # A is 4x4, B is 4x4.
    # A[i, j] is row i, col j.
    # B[j, k] is row j, col k.
    # C[i, k] is row i, col k.
    # Task U: i*m+j -> i (row A), j (col A).
    # Task V: j*p+k -> j (row B), k (col B).
    # Task W: k*n+i -> k (col C), i (row C).
    # So W index is col_C * 4 + row_C.
    
    # In Strassen 2x2:
    # Block A[I,J] has inner A[ii,jj].
    # Block B[K,L] has inner B[kk,ll].
    # Block C[I,J] has inner C[ii,jj].
    # Wait, C = A B.
    # C[I,J] = sum_K A[I,K] B[K,J].
    # The Strassen algorithm computes C_block[I,J] from A_block[I,K] and B_block[K,J].
    # The "outer" Strassen (s) operates on the blocks.
    # The "inner" Strassen (t) operates on the elements within the blocks.
    
    # So for a given term (s, t):
    # U[i*4+j, r4] = U2_block[I*2+J, s] * U2_inner[ii*2+jj, t]
    # V[k*4+l, r4] = V2_block[K*2+L, s] * V2_inner[kk*2+ll, t]
    # W[j*4+i, r4] = W2_block[J*2+I, s] * W2_inner[jj*2+ii, t] 
    # Why J*2+I for W block?
    # W index is col_C * 4 + row_C.
    # Block index for C is (I, J) where I is row block, J is col block.
    # So col_C block is J, row_C block is I.
    # So W block index is J*2 + I.
    # Inner W index: col_C inner is jj, row_C inner is ii.
    # So W inner index is jj*2 + ii.
    
    for s in range(7):
        for t in range(7):
            r4 = s * 7 + t
            
            # We can iterate over non-zeros to be faster, but 16x49 is small.
            # Let's just do the direct calculation for clarity and correctness.
            
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    J = j // 2
                    ii = i % 2
                    jj = j % 2
                    
                    # U
                    coeff_block = U2[I * 2 + J][s]
                    coeff_inner = U2[ii * 2 + jj][t]
                    val = coeff_block * coeff_inner
                    if val != 0:
                        U[i * 4 + j][r4] = val

            for k in range(4):
                for l in range(4):
                    K = k // 2
                    L = l // 2
                    kk = k % 2
                    ll = l % 2
                    
                    # V
                    coeff_block = V2[K * 2 + L][s]
                    coeff_inner = V2[kk * 2 + ll][t]
                    val = coeff_block * coeff_inner
                    if val != 0:
                        V[k * 4 + l][r4] = val

            for j in range(4):
                for i in range(4):
                    J = j // 2
                    I = i // 2
                    jj = j % 2
                    ii = i % 2
                    
                    # W
                    # Block index: J (col block) * 2 + I (row block)
                    coeff_block = W2[J * 2 + I][s]
                    # Inner index: jj (col inner) * 2 + ii (row inner)
                    coeff_inner = W2[jj * 2 + ii][t]
                    val = coeff_block * coeff_inner
                    if val != 0:
                        W[j * 4 + i][r4] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
