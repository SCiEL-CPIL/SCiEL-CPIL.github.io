"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # --- Block A1 (rows i=0,1) * B1 (cols j=0,1) using Strassen ---
    # B1 has 8 columns, split into 4 groups of 2 columns each (r=0..3)
    # For each group r, we multiply A1 (2x2) with B1[:, 2r:2r+2] (2x2)
    # Strassen for 2x2 * 2x2 uses 7 multiplications
    
    col = 0
    for r in range(4):
        k0 = 2 * r
        k1 = 2 * r + 1
        # B1 submatrix indices: B[0, k0], B[0, k1], B[1, k0], B[1, k1]
        # A1 submatrix: A[0,0], A[0,1], A[1,0], A[1,1]
        # C submatrix: C[k0, 0], C[k0, 1], C[k1, 0], C[k1, 1]
        
        # M1 = (A00 + A11)(B00 + B11)
        U[0][col] = 1  # A00
        U[3][col] = 1  # A11
        V[0][col] = 1  # B[0, k0]
        V[2][col] = 1  # B[1, k0] -- Wait, V index is j*p+k. B1 is j=0,1.
        # V index for B[j,k] is j*8 + k
        # B[0, k0] -> V[0*8+k0] = V[k0]
        # B[1, k0] -> V[1*8+k0] = V[8+k0]
        # Let's re-evaluate V indices carefully.
        # V has shape (m*p, rank) = (3*8, rank) = (24, rank).
        # V[j*p + k, t] corresponds to B[j, k].
        # B1 is rows j=0,1.
        # So V[0*8 + k0] = V[k0] is B[0, k0].
        # V[1*8 + k0] = V[8+k0] is B[1, k0].
        # V[0*8 + k1] = V[k1] is B[0, k1].
        # V[1*8 + k1] = V[8+k1] is B[1, k1].
        
        # Correction to previous lines:
        V[k0][col] = 1
        V[8 + k0][col] = 1
        # W index for C[k, i] is k*n + i.
        # W[k0*n + 0] = W[2*k0] is C[k0, 0]
        # W[k0*n + 1] = W[2*k0+1] is C[k0, 1]
        # W[k1*n + 0] = W[2*k1] is C[k1, 0]
        # W[k1*n + 1] = W[2*k1+1] is C[k1, 1]
        
        W[2*k0][col] = 1
        W[2*k0+1][col] = 1
        W[2*k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M2 = (A10 + A11) B00
        U[2][col] = 1  # A10
        U[3][col] = 1  # A11
        V[k0][col] = 1  # B[0, k0]
        W[2*k0+1][col] = 1  # C[k0, 1]
        W[2*k1+1][col] = 1  # C[k1, 1]
        col += 1
        
        # M3 = A00 (B10 - B11)
        U[0][col] = 1  # A00
        V[8+k0][col] = 1  # B[1, k0]
        V[8+k1][col] = -1  # B[1, k1]
        W[2*k0+1][col] = 1  # C[k0, 1]
        W[2*k1+1][col] = 1  # C[k1, 1]
        col += 1
        
        # M4 = A11 (B00 - B01)
        U[3][col] = 1  # A11
        V[k0][col] = 1  # B[0, k0]
        V[k1][col] = -1  # B[0, k1]
        W[2*k0][col] = 1  # C[k0, 0]
        W[2*k1][col] = 1  # C[k1, 0]
        col += 1
        
        # M5 = (A00 + A01) B11
        U[0][col] = 1  # A00
        U[1][col] = 1  # A01
        V[8+k1][col] = 1  # B[1, k1]
        W[2*k0][col] = 1  # C[k0, 0]
        W[2*k1][col] = 1  # C[k1, 0]
        col += 1
        
        # M6 = (A10 - A00) (B00 + B01)
        U[2][col] = 1  # A10
        U[0][col] = -1  # A00
        V[k0][col] = 1  # B[0, k0]
        V[k1][col] = 1  # B[0, k1]
        W[2*k1][col] = 1  # C[k1, 0]
        col += 1
        
        # M7 = (A01 - A11) (B10 + B11)
        U[1][col] = 1  # A01
        U[3][col] = -1  # A11
        V[8+k0][col] = 1  # B[1, k0]
        V[8+k1][col] = 1  # B[1, k1]
        W[2*k0+1][col] = 1  # C[k0, 1]
        col += 1
    
    # --- Block A2 (row i=2) * B2 (col j=2) ---
    # A2 is 1x3, B2 is 3x8.
    # We can do this with 3 multiplications? No, 1x3 * 3x8.
    # C[k, 2] = sum_j A[2, j] B[j, k].
    # This is a dot product of a 1x3 vector and 3x8 matrix.
    # We can decompose this into 3 rank-1 terms?
    # C[k, 2] = A[2,0]B[0,k] + A[2,1]B[1,k] + A[2,2]B[2,k].
    # Each term is rank 1.
    # So we need 3 more columns.
    
    # Term 1: A[2,0] * B[0,k]
    U[5][col] = 1  # A[2,0]
    V[0*8 + 0][col] = 1  # B[0, 0] -- Wait, B[0,k] varies with k.
    # We need to sum over k.
    # Actually, for each k, C[k,2] gets a contribution.
    # This is not a single rank-1 tensor for the whole 8x1 block.
    # It is 8 rank-1 tensors if done naively.
    # But we can do better?
    # The subproblem is 1x3 * 3x8.
    # Rank of 1x3 * 3x8 is 3?
    # C[k, 2] = sum_j A[2,j] B[j,k].
    # We can write this as:
    # Let U_vec be A[2,0], A[2,1], A[2,2].
    # Let V_mat be B[0,:], B[1,:], B[2,:].
    # C[:, 2] = U_vec dot V_mat.
    # This is a matrix product of 1x3 and 3x8.
    # The standard decomposition uses 3*8 = 24 terms? No.
    # Rank of (1,3,8) tensor is 3.
    # C[k,2] = sum_j A[2,j] B[j,k].
    # We can define 3 rank-1 components?
    # Let t=0,1,2.
    # U[2,0,t] = 1? No, U is indexed by (i,j).
    # U[2*m + j, t] = A[2,j].
    # V[j*p + k, t] = B[j,k].
    # W[k*n + 2, t] = C[k,2].
    # If we set t=j, then U[2*m+j, j]=1, V[j*p+k, j]=B[j,k]? No, V entries must be constant.
    # We need V[j*p+k, t] to be 1 or 0.
    # If we choose t=k, then we have 8 terms.
    # If we choose t=j, we have 3 terms.
    # Let's try t=j (j=0,1,2).
    # U[2*m+j, j] = 1.
    # V[j*p+k, j] = 1? No, V must be 1 for all k?
    # If V[j*p+k, j] = 1, then sum_t U[..,t] V[..,t] W[..,t] = sum_j A[2,j] * 1 * C[k,2]_t?
    # No, W[k*n+2, j] must be C[k,2]. But C[k,2] is the output, not input.
    # The tensor is T[i,j,k] = delta_{i,j,k}? No, T[i,j,k] = 1 if i,j,k are valid indices?
    # The multiplication tensor T has T[i,j,k] = 1.
    # We want sum_t U[i,j,t] V[j,k,t] W[k,i,t] = 1.
    # For the A2*B2 block: i=2, j=0,1,2, k=0..7.
    # We want sum_t U[2,j,t] V[j,k,t] W[k,2,t] = 1 for all j,k?
    # No, the product is C[k,i] = sum_j A[i,j] B[j,k].
    # The tensor decomposition is sum_t U[i,j,t] V[j,k,t] W[k,i,t].
    # For i=2, we want sum_j A[2,j] B[j,k] to be represented.
    # This is not a constant 1. The tensor T[i,j,k] is 1 for all i,j,k.
    # So for i=2, j, k, we want sum_t U[2,j,t] V[j,k,t] W[k,2,t] = 1.
    # This is a 3x8 sub-tensor of all 1s.
    # The rank of a 3x8 all-ones matrix (as a bilinear form) is 3.
    # We can use 3 rank-1 terms.
    # Let t=0,1,2.
    # We need U[2,j,t], V[j,k,t], W[k,2,t] such that sum_t U[2,j,t] V[j,k,t] W[k,2,t] = 1.
    # Let U[2,j,0]=1, U[2,j,1]=0, U[2,j,2]=0 for all j? No.
    # Let's use the identity matrix trick.
    # Let V[j,k,t] = delta_{j,t}. (V is 3x8x3).
    # Then sum_t U[2,j,t] delta_{j,t} W[k,2,t] = U[2,j,j] W[k,2,j].
    # We want this to be 1 for all j,k.
    # This requires U[2,j,j] W[k,2,j] = 1 for all j,k.
    # This is impossible for all k unless W is constant and U is constant.
    # Let U[2,j,j] = 1. Then W[k,2,j] = 1 for all k,j.
    # Then sum_t 1 * delta_{j,t} * 1 = delta_{j,j} = 1.
    # This works!
    # So for the A2*B2 block:
    # For t=0,1,2:
    # U[2*m+t, t] = 1  (i.e., U[6+t, t])
    # V[t*8+k, t] = 1 for all k? No, V[j,k,t] = delta_{j,t}.
    # So V[t*8+k, t] = 1 for all k.
    # W[k*n+2, t] = 1 for all k.
    
    for t in range(3):
        U[6 + t][col] = 1
        for k in range(8):
            V[t * 8 + k][col] = 1
            W[k * 2 + 2][col] = 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
