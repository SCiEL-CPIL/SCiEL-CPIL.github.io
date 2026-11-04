"""Matrix multiplication 2,4,8 - Strassen-based rank 56 decomposition."""
import time

DIMENSIONS = [2, 4, 8]
_DEADLINE = 120.0

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """
    Rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) such that
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = product tensor.
    Here n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    
    Strassen's 7 products for 2x2:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1+M4-M5+M7
    C12 = M3+M5
    C21 = M2+M4
    C22 = M1-M2+M3+M6
    
    Tensor decomposition: for each product Mc, we have a rank-1 term.
    U row index: i*2+j (i=row of A, j=col of A)
    V row index: j*2+k (j=row of B, k=col of B)
    W row index: k*2+i (k=row of C, i=col of C)
    """
    # 7 Strassen products
    # Each product is a bilinear form sum of terms (a_ia * b_ib)
    # We need to express each Mc as a sum of rank-1 terms in the tensor.
    # Actually, each Mc IS already a single product of two linear forms,
    # so each Mc contributes exactly ONE rank-1 term to the decomposition.
    
    # Mc = (sum_i u_c[i] * a_i) * (sum_j v_c[j] * b_j)
    # where a_i are the 4 entries of A (indexed i*2+j), b_j are 4 entries of B (j*2+k)
    
    # Product 1: (A11+A22)(B11+B22)
    # A11 is a[0], A22 is a[3]
    # B11 is b[0], B22 is b[3]
    # M1 = (a0+a3)(b0+b3) = a0*b0 + a0*b3 + a3*b0 + a3*b3
    # This is NOT a single rank-1 term in the 2x2x2 tensor sense.
    # Wait - in the tensor decomposition, each rank-1 term is:
    # U[*,r] * V[*,r] * W[*,r]
    # This means: for each r, the contribution to C is:
    # C_ki += W[k*2+i, r] * (sum over ij of U[i*2+j,r]*V[j*2+k,r]*A[i,j]*B[j,k])
    # 
    # Actually the correct interpretation:
    # The multiplication tensor T has entries T[i,j,k] where
    # C = sum_{i,j} A[i,j]*B[j,k]
    # The rank-1 decomposition means:
    # T = sum_r U[*,r] outer V[*,r] outer W[*,r]
    # where T is viewed as a 3-way tensor with indices (i*2+j, j*2+k, k*2+i)
    # 
    # More precisely: for the 2x2x2 case:
    # A is 2x2 (indices i=0,1; j=0,1), B is 2x2 (indices j=0,1; k=0,1), C is 2x2 (indices k=0,1; i=0,1)
    # U has shape (4, r) indexed by (i,j)
    # V has shape (4, r) indexed by (j,k)
    # W has shape (4, r) indexed by (k,i)
    # 
    # The tensor T[i,j,k] (where i is row of A/C, j is col of A / row of B, k is col of B / row of C)
    # satisfies: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid (i,j,k), and the multiplication is:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # 
    # Rank-1 term r contributes:
    # C[i,k] += W[k*2+i, r] * (sum_j U[i*2+j, r] * A[i,j] * V[j*2+k, r] * B[j,k])
    # 
    # For this to equal the standard multiplication, we need:
    # sum_r W[k*2+i, r] * U[i*2+j, r] * V[j*2+k, r] = delta_{j,j} (i.e., 1)
    # for all i, j, k.
    
    # Let me just directly construct the 7 rank-1 terms from Strassen.
    
    # M1 = (a00+a11)(b00+b11)
    # This means: U contribution: a00 and a11 get coefficient 1
    #              V contribution: b00 and b11 get coefficient 1
    #              W: C11 += M1, so W[0,0] += 1 (C[0,0])
    # But wait, M1 = (a00+a11)(b00+b11) expands to:
    # a00*b00 + a00*b11 + a11*b00 + a11*b11
    # In the tensor: 
    #   term (i=0,j=0,k=0): coeff from U[0,0]*V[0,0]*W[0,0]
    #   term (i=0,j=0,k=1): coeff from U[0,0]*V[1,0]*W[1,0]  -- wait this doesn't work simply
    
    # I think the cleaner way: each Strassen product Mc corresponds to ONE rank-1 slice.
    # Mc = (linear form in A) * (linear form in B)
    # The rank-1 term has:
    #   U[*, r_c] = coefficients of the linear form in A (indexed by i,j)
    #   V[*, r_c] = coefficients of the linear form in B (indexed by j,k)  
    #   W[*, r_c] = how Mc contributes to C (indexed by k,i)
    
    # M1 = (a00+a11)(b00+b11)
    # Linear form in A: a00 + a11 -> U[0,0]=1, U[3,0]=1 (i.e., U[i*2+j])
    # Linear form in B: b00 + b11 -> V[0,0]=1, V[3,0]=1 (i.e., V[j*2+k])
    # Contribution to C: C00 += M1 -> W[0,0]=1 (k=0,i=0)
    
    # But wait - this gives C00 += (a00+a11)(b00+b11) which is correct for the Strassen formula.
    # However, the FULL Strassen formula uses 7 such products and combines them.
    # Each product is a separate rank-1 term. The W matrix encodes the combination.
    
    # M1: A-form: (1,0,0,1) on (a00,a01,a10,a11)
    #     B-form: (1,0,0,1) on (b00,b01,b10,b11)
    #     C-contribution: +1 to C[0,0], +0 to C[0,1], +0 to C[1,0], +0 to C[1,1]
    #     W indices: k*2+i, so C[0,0]->idx 0, C[0,1]->idx 1, C[1,0]->idx 2, C[1,1]->idx 3
    
    products = []
    # M1 = (a00+a11)(b00+b11) -> C00 += M1
    products.append((
        [1,0,0,1],   # U: a00,a01,a10,a11
        [1,0,0,1],   # V: b00,b01,b10,b11
        [1,0,0,0]    # W: C00,C01,C10,C11
    ))
    # M2 = (a10+a11)(b00) -> C10 += M2
    products.append((
        [0,0,1,1],
        [1,0,0,0],
        [0,0,1,0]
    ))
    # M3 = a00*(b10-b11) -> C01 += M3, C10 += M3
    products.append((
        [1,0,0,0],
        [0,0,1,-1],
        [0,1,1,0]
    ))
    # M4 = a11*(b01-b00) -> C00 += M4, C10 += M4
    products.append((
        [0,0,0,1],
        [0,1,-1,0],
        [1,0,1,0]
    ))
    # M5 = (a00+a01)*b11 -> C00 -= M5, C01 += M5
    products.append((
        [1,1,0,0],
        [0,0,0,1],
        [-1,1,0,0]
    ))
    # M6 = (a10-a00)*(b00+b01) -> C11 += M6
    products.append((
        [-1,0,1,0],
        [1,1,0,0],
        [0,0,0,1]
    ))
    # M7 = (a01-a11)*(b10+b11) -> C00 += M7
    products.append((
        [0,1,-1,0],
        [0,0,1,1],
        [1,0,0,0]
    ))
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    for c in range(7):
        for idx in range(4):
            U[idx][c] = products[c][0][idx]
            V[idx][c] = products[c][1][idx]
            W[idx][c] = products[c][2][idx]
    return U, V, W

def _strassen_2x2x8():
    """
    Decompose 2x2x8 multiplication using Strassen.
    A is 2x2, B is 2x8, C is 2x8.
    Split B into B_left (2x4) and B_right (2x4).
    C = A * B = [A*B_left, A*B_right]
    Each A*B_left and A*B_right is a 2x2x4 multiplication.
    
    For 2x2x4: split B into 2 blocks of 2x2.
    Each 2x2x2 uses rank 7. So 2x2x4 uses rank 14.
    2x2x8 uses rank 28.
    
    Actually, let me think more carefully.
    2x2x4: A(2x2) * B(2x4) = C(2x4)
    Split B into B1(2x2) and B2(2x2).
    C1 = A*B1 (2x2x2, rank 7)
    C2 = A*B2 (2x2x2, rank 7)
    Total rank: 14.
    
    2x2x8: split B into 4 blocks of 2x2.
    Total rank: 4 * 7 = 28.
    """
    U_222, V_222, W_222 = _strassen_2x2x2()
    
    # 2x2x8: A is 2x2, B is 2x8, C is 2x8
    # U: shape (4, 28) - indexed by (i,j) of A, i in 0..1, j in 0..1
    # V: shape (16, 28) - indexed by (j,k) of B, j in 0..1, k in 0..7
    # W: shape (16, 28) - indexed by (k,i) of C, k in 0..7, i in 0..1
    
    rank = 28
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    for block in range(4):
        base = block * 7
        for r in range(7):
            for i in range(4):
                U[i][base+r] = U_222[i][r]
                # V index: j*8 + k, where j in 0..1, k in 0..1 (within block)
                # Actually V is indexed by (j, k) where j is row of B (0..1), k is col (0..7)
                # For block, columns are 2*block, 2*block+1
                for j in range(2):
                    for k in range(2):
                        v_idx = j * 8 + (2*block + k)
                        # In the 2x2x2, V is indexed by (j,k) with j,k in 0..1
                        v_local_idx = j * 2 + k
                        V[v_idx][base+r] = V_222[v_local_idx][r]
                # W index: k*2 + i, where k in 0..7, i in 0..1
                for k in range(2):
                    for i in range(2):
                        w_idx = (2*block + k) * 2 + i
                        w_local_idx = k * 2 + i
                        W[w_idx][base+r] = W_222[w_local_idx][r]
    return U, V, W

def construct():
    """
    Rank-56 decomposition for 2x4x8 matrix multiplication.
    
    A is 2x4, B is 4x8, C is 2x8.
    Split A into A1 (2x2, cols 0-1) and A2 (2x2, cols 2-3).
    Split B into B1 (rows 0-1, 2x8) and B2 (rows 2-3, 2x8).
    C = A1*B1 + A2*B2.
    
    Each term is a 2x2x8 multiplication (rank 28).
    Total rank: 56.
    """
    start_time = time.time()
    
    U_228, V_228, W_228 = _strassen_2x2x8()
    
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0]*rank for _ in range(n*m)]  # 8 x 56
    V = [[0]*rank for _ in range(m*p)]  # 32 x 56
    W = [[0]*rank for _ in range(p*n)]  # 16 x 56
    
    # C = A1*B1 + A2*B2
    # A1 uses columns 0,1 of A; A2 uses columns 2,3
    # B1 uses rows 0,1 of B; B2 uses rows 2,3
    
    # For A1*B1 (first 28 columns of U,V,W):
    for r in range(28):
        # U: A has rows 0,1 and cols 0,1 for A1
        # U index: i*4+j
        for i in range(2):
            for j in range(2):
                U[i*4 + j][r] = U_228[i*2 + j][r]
        
        # V: B has rows 0,1 and cols 0..7 for B1
        # V index: j*8+k
        for j in range(2):
            for k in range(8):
                V[j*8 + k][r] = V_228[j*8 + k][r]
        
        # W: C has rows 0..7 and cols 0,1
        # W index: k*2+i
        for k in range(8):
            for i in range(2):
                W[k*2 + i][r] = W_228[k*2 + i][r]
    
    # For A2*B2 (next 28 columns, offset by 28):
    for r in range(28):
        rr = r + 28
        # U: A has rows 0,1 and cols 2,3 for A2
        # U index: i*4+j where j = 2,3
        for i in range(2):
            for j in range(2):
                U[i*4 + (2+j)][rr] = U_228[i*2 + j][r]
        
        # V: B has rows 2,3 and cols 0..7 for B2
        # V index: j*8+k where j = 2,3
        for j in range(2):
            for k in range(8):
                V[(2+j)*8 + k][rr] = V_228[j*8 + k][r]
        
        # W: same C
        for k in range(8):
            for i in range(2):
                W[k*2 + i][rr] = W_228[k*2 + i][r]
    
    elapsed = time.time() - start_time
    if elapsed > _DEADLINE - 5:
        # Should not happen since this is O(1), but guard anyway
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
