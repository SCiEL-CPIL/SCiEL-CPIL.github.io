"""Matrix multiplication 2x4 x 4x4 rank-minimal decomposition."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Use Strassen's algorithm for 2x2 block multiplication.
    # A is 2x4 = [A1 A2] where A1, A2 are 2x2
    # B is 4x4 = [[B11, B12], [B21, B22]] where each is 2x2
    # C = A*B is 2x4 = [C1 C2] where C1, C2 are 2x2
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22
    #
    # For each 2x2 product, use Strassen with 7 multiplications.
    # We need 4 independent 2x2x2 tensor decompositions (one for each block product).
    #
    # Block structure:
    # A has rows 0,1 (block row 0) and 2,3 (block row 1)
    # B has cols 0,1 (block col 0) and 2,3 (block col 1), and rows 0,1 (block row 0) and 2,3 (block row 1)
    #
    # C1 = A1*B11 + A2*B21  (2x2 = 2x2 * 2x2 + 2x2 * 2x2)
    # C2 = A1*B12 + A2*B22  (2x2 = 2x2 * 2x2 + 2x2 * 2x2)
    #
    # Each 2x2x2 matrix multiplication can be done with Strassen rank 7.
    # We have 4 such products: A1*B11, A2*B21, A1*B12, A2*B22
    # Total rank = 4 * 7 = 28
    #
    # Strassen's 7 bilinear forms for 2x2:
    # M1 = (a+b)(e+f)
    # M2 = (g+h)e
    # M3 = a(d-f)
    # M4 = c(g-h)
    # M5 = (a+d)(h+f)
    # M6 = (b+c)(d-g)
    # M7 = (a+b)(e+f)  -- wait, let me use the standard form
    #
    # Standard Strassen for X (2x2) * Y (2x2) = Z (2x2):
    # X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y11-y10)
    # M4 = x11*(y00-y01)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y10)
    # M7 = (x01-x11)*(y10+y11)
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # Now I need to map these to the global index scheme.
    # A is 2x4: A[i][j] for i in 0..1, j in 0..3
    # B is 4x4: B[j][k] for j in 0..3, k in 0..3
    # C is 2x4: C[i][k] for i in 0..1, k in 0..3
    #
    # U has shape (n*m, rank) = (8, rank) -- indexed by [i*m+j] = [i*4+j]
    # V has shape (m*p, rank) = (16, rank) -- indexed by [j*p+k] = [j*4+k]
    # W has shape (p*n, rank) = (8, rank) -- indexed by [k*n+i] = [k*2+i]
    #
    # The tensor element T[i,j,k] = delta (1 if C[i,k] depends on A[i,j]*B[j,k])
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # For block product A1*B11: A1 = A[0:2, 0:2], B11 = B[0:2, 0:2]
    # This contributes to C[0:2, 0:2]
    # For A1*B11, the Strassen bilinear forms involve:
    #   x00=A[0,0], x01=A[0,1], x10=A[1,0], x11=A[1,1]
    #   y00=B[0,0], y01=B[0,1], y10=B[1,0], y11=B[1,1]
    #   z00=C[0,0], z01=C[0,1], z10=C[1,0], z11=C[1,1]
    #
    # For A2*B21: A2 = A[0:2, 2:4], B21 = B[2:4, 0:2]
    #   x00=A[0,2], x01=A[0,3], x10=A[1,2], x11=A[1,3]
    #   y00=B[2,0], y01=B[2,1], y10=B[3,0], y11=B[3,1]
    #   z00=C[0,0], z01=C[0,1], z10=C[1,0], z11=C[1,1]
    #
    # For A1*B12: A1 = A[0:2, 0:2], B12 = B[0:2, 2:4]
    #   x00=A[0,0], x01=A[0,1], x10=A[1,0], x11=A[1,1]
    #   y00=B[0,2], y01=B[0,3], y10=B[1,2], y11=B[1,3]
    #   z00=C[0,2], z01=C[0,3], z10=C[1,2], z11=C[1,3]
    #
    # For A2*B22: A2 = A[0:2, 2:4], B22 = B[2:4, 2:4]
    #   x00=A[0,2], x01=A[0,3], x10=A[1,2], x11=A[1,3]
    #   y00=B[2,2], y01=B[2,3], y10=B[3,2], y11=B[3,3]
    #   z00=C[0,2], z01=C[0,3], z10=C[1,2], z11=C[1,3]
    #
    # U index for A[i,j]: i*4+j
    # V index for B[j,k]: j*4+k
    # W index for C[k,i]: k*2+i  (trace convention W[k*n+i,r])
    #
    # Let me define the Strassen bilinear forms for a given block product.
    # Each bilinear form r has:
    #   U factors: which (i,j) pairs of A are involved and with what coefficient
    #   V factors: which (j,k) pairs of B are involved and with what coefficient
    #   W factors: which (k,i) pairs of C are involved and with what coefficient
    #
    # For a 2x2x2 Strassen decomposition, the 7 bilinear forms are:
    # r=0: M1 = (x00+x11)(y00+y11)
    #   U: x00 coeff 1, x11 coeff 1
    #   V: y00 coeff 1, y11 coeff 1
    #   W: z00 coeff 1, z11 coeff 1  (M1 appears in Z00 and Z11)
    #   Actually W encodes the output combination. Let me think more carefully.
    #
    # The decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where T[i,j,k] = 1 if C[i,k] = A[i,j]*B[j,k]
    #
    # For Strassen, the 7 intermediate products are:
    # M1 = (x00+x11)(y00+y11)
    # M2 = (x10+x11)y00
    # M3 = x00(y11-y10)
    # M4 = x11(y00-y01)
    # M5 = (x00+x01)y11
    # M6 = (x10-x00)(y00+y10)
    # M7 = (x01-x11)(y10+y11)
    #
    # And:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # So for each bilinear form M_r, we have:
    # U_r: coefficients on the x variables
    # V_r: coefficients on the y variables
    # W_r: coefficients on the z variables (output)
    #
    # M1: U={x00:1, x11:1}, V={y00:1, y11:1}, W={z00:1, z11:1}
    # M2: U={x10:1, x11:1}, V={y00:1}, W={z10:1, z11:-1}
    # M3: U={x00:1}, V={y11:1, y10:-1}, W={z01:1, z11:1}
    # M4: U={x11:1}, V={y00:1, y01:-1}, W={z00:1, z10:1}
    # M5: U={x00:1, x01:1}, V={y11:1}, W={z00:-1, z01:1}
    # M6: U={x10:1, x00:-1}, V={y00:1, y10:1}, W={z11:1}
    # M7: U={x01:1, x11:-1}, V={y10:1, y11:1}, W={z00:1}
    #
    # Now I need to map x,y,z to global indices for each of the 4 block products.
    #
    # Block product 1: A1*B11 -> C[0:2,0:2]
    #   x00=A[0,0]=U_idx 0, x01=A[0,1]=U_idx 1, x10=A[1,0]=U_idx 4, x11=A[1,1]=U_idx 5
    #   y00=B[0,0]=V_idx 0, y01=B[0,1]=V_idx 1, y10=B[1,0]=V_idx 4, y11=B[1,1]=V_idx 5
    #   z00=C[0,0]=W_idx 0*2+0=0, z01=C[0,1]=W_idx 1*2+0=2, z10=C[1,0]=W_idx 0*2+1=1, z11=C[1,1]=W_idx 1*2+1=3
    #
    # Block product 2: A2*B21 -> C[0:2,0:2]
    #   x00=A[0,2]=U_idx 2, x01=A[0,3]=U_idx 3, x10=A[1,2]=U_idx 6, x11=A[1,3]=U_idx 7
    #   y00=B[2,0]=V_idx 8, y01=B[2,1]=V_idx 9, y10=B[3,0]=V_idx 12, y11=B[3,1]=V_idx 13
    #   z00=C[0,0]=W_idx 0, z01=C[0,1]=W_idx 2, z10=C[1,0]=W_idx 1, z11=C[1,1]=W_idx 3
    #
    # Block product 3: A1*B12 -> C[0:2,2:4]
    #   x00=A[0,0]=U_idx 0, x01=A[0,1]=U_idx 1, x10=A[1,0]=U_idx 4, x11=A[1,1]=U_idx 5
    #   y00=B[0,2]=V_idx 2, y01=B[0,3]=V_idx 3, y10=B[1,2]=V_idx 6, y11=B[1,3]=V_idx 7
    #   z00=C[0,2]=W_idx 2*2+0=4, z01=C[0,3]=W_idx 3*2+0=6, z10=C[1,2]=W_idx 2*2+1=5, z11=C[1,3]=W_idx 3*2+1=7
    #
    # Block product 4: A2*B22 -> C[0:2,2:4]
    #   x00=A[0,2]=U_idx 2, x01=A[0,3]=U_idx 3, x10=A[1,2]=U_idx 6, x11=A[1,3]=U_idx 7
    #   y00=B[2,2]=V_idx 10, y01=B[2,3]=V_idx 11, y10=B[3,2]=V_idx 14, y11=B[3,3]=V_idx 15
    #   z00=C[0,2]=W_idx 4, z01=C[0,3]=W_idx 6, z10=C[1,2]=W_idx 5, z11=C[1,3]=W_idx 7

    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Define the 4 block products with their index mappings
    # Each block product: (u_indices for x00,x01,x10,x11), (v_indices for y00,y01,y10,y11), (w_indices for z00,z01,z10,z11)
    blocks = [
        # Block 1: A1*B11
        ([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3]),
        # Block 2: A2*B21
        ([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3]),
        # Block 3: A1*B12
        ([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7]),
        # Block 4: A2*B22
        ([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7]),
    ]

    # Strassen bilinear forms: (u_coeffs for x00,x01,x10,x11), (v_coeffs for y00,y01,y10,y11), (w_coeffs for z00,z01,z10,z11)
    strassen_forms = [
        # M1
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        # M2
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        # M3
        ((1, 0, 0, 0), (0, 0, -1, 1), (0, 1, 0, 1)),
        # M4
        ((0, 0, 0, 1), (1, -1, 0, 0), (1, 0, 1, 0)),
        # M5
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        # M6
        ((-1, 0, 1, 0), (1, 0, 1, 0), (0, 0, 0, 1)),
        # M7
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]

    for block_idx, (u_map, v_map, w_map) in enumerate(blocks):
        for form_idx, (u_coeff, v_coeff, w_coeff) in enumerate(strassen_forms):
            col = block_idx * 7 + form_idx
            for xi in range(4):
                if u_coeff[xi] != 0:
                    U[u_map[xi]][col] = u_coeff[xi]
            for yi in range(4):
                if v_coeff[yi] != 0:
                    V[v_map[yi]][col] = v_coeff[yi]
            for zi in range(4):
                if w_coeff[zi] != 0:
                    W[w_map[zi]][col] = w_coeff[zi]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
