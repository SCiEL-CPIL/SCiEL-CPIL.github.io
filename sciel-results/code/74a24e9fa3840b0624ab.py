import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Build the standard 4-Strassen-block decomposition (rank 28)
    # A is 2x4, B is 4x4, C is 2x4
    # A11 = A[0:2,0:2], A12 = A[0:2,2:4]
    # B11 = B[0:2,0:2], B12 = B[0:2,2:4], B21 = B[2:4,0:2], B22 = B[2:4,2:4]
    # C[:,0:2] = A11*B11 + A12*B21
    # C[:,2:4] = A11*B12 + A12*B22
    
    # Flatten indices:
    # A[i,j] -> a_idx = i*m + j = i*4 + j
    # B[j,k] -> b_idx = j*p + k = j*4 + k
    # C[i,k] -> c_idx = k*n + i = k*2 + i (per W[k*n+i, r] convention)
    
    # Strassen for 2x2 * 2x2:
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    # Z11 = M1+M4-M5+M7
    # Z12 = M3+M5
    # Z21 = M2+M4
    # Z22 = M1-M2+M3+M6
    
    # For each 2x2 sub-product, we have 7 multiplications.
    # 4 sub-products -> 28 multiplications.
    
    # Let's build U, V, W matrices of shape (8,28), (16,28), (8,28)
    
    U = np.zeros((n*m, 28))
    V = np.zeros((m*p, 28))
    W = np.zeros((p*n, 28))
    
    # Define Strassen coefficients for a single 2x2*2x2 product
    # For product X*Y where X = [[x00,x01],[x10,x11]], Y = [[y00,y01],[y10,y11]]
    # M1: X part: x00 + x11; Y part: y00 + y11
    # M2: X part: x10 + x11; Y part: y00
    # M3: X part: x00; Y part: y01 - y11
    # M4: X part: x11; Y part: y10 - y00
    # M5: X part: x00 + x01; Y part: y11
    # M6: X part: x10 - x00; Y part: y00 + y01
    # M7: X part: x01 - x11; Y part: y10 + y11
    
    # Z contributions:
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_x = [
        {0: 1, 3: 1},      # M1: x00 + x11
        {2: 1, 3: 1},      # M2: x10 + x11
        {0: 1},            # M3: x00
        {3: 1},            # M4: x11
        {0: 1, 1: 1},      # M5: x00 + x01
        {2: 1, 0: -1},     # M6: x10 - x00
        {1: 1, 3: -1}      # M7: x01 - x11
    ]
    
    strassen_y = [
        {0: 1, 3: 1},      # M1: y00 + y11
        {0: 1},            # M2: y00
        {1: 1, 3: -1},     # M3: y01 - y11
        {2: 1, 0: -1},     # M4: y10 - y00
        {3: 1},            # M5: y11
        {0: 1, 1: 1},      # M6: y00 + y01
        {2: 1, 3: 1}       # M7: y10 + y11
    ]
    
    strassen_z = [
        {0: 1, 3: 1, 2: -1, 1: 1},   # Z00: M1+M4-M5+M7
        {1: 1, 2: 1},                 # Z01: M3+M5
        {1: 1, 3: 1},                 # Z10: M2+M4
        {0: 1, 1: -1, 2: 1, 5: 1}    # Z11: M1-M2+M3+M6
    ]
    
    # Four sub-products:
    # P1 = A11 * B11 -> contributes to C[:, 0:2]
    # P2 = A12 * B21 -> contributes to C[:, 0:2]
    # P3 = A11 * B12 -> contributes to C[:, 2:4]
    # P4 = A12 * B22 -> contributes to C[:, 2:4]
    
    # For each sub-product, the X matrix is a 2x2 block of A, Y is a 2x2 block of B
    # X entries map to A indices: X[0,0]=A[i0,j0], X[0,1]=A[i0,j0+1], X[1,0]=A[i0+1,j0], X[1,1]=A[i0+1,j0+1]
    # Y entries map to B indices: Y[0,0]=B[i0,j0], etc.
    # Z entries map to C indices: Z[0,0]=C[i0,j0], Z[0,1]=C[i0,j0+1], Z[1,0]=C[i0+1,j0], Z[1,1]=C[i0+1,j0+1]
    
    # Sub-product definitions: (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    sub_products = [
        (0, 0, 0, 0, 0, 0),  # P1: A11 * B11 -> C[:,0:2]
        (0, 2, 2, 0, 0, 0),  # P2: A12 * B21 -> C[:,0:2]
        (0, 0, 0, 2, 0, 2),  # P3: A11 * B12 -> C[:,2:4]
        (0, 2, 2, 2, 0, 2),  # P4: A12 * B22 -> C[:,2:4]
    ]
    
    for sp_idx, (a_r0, a_c0, b_r0, b_c0, c_r0, c_c0) in enumerate(sub_products):
        for s in range(7):
            col = sp_idx * 7 + s
            
            # Fill U: A entries
            for x_idx, val in strassen_x[s].items():
                # x_idx: 0=x00, 1=x01, 2=x10, 3=x11
                xi = x_idx // 2
                xj = x_idx % 2
                a_row = a_r0 + xi
                a_col = a_c0 + xj
                a_idx = a_row * m + a_col
                U[a_idx, col] += val
            
            # Fill V: B entries
            for y_idx, val in strassen_y[s].items():
                # y_idx: 0=y00, 1=y01, 2=y10, 3=y11
                yi = y_idx // 2
                yj = y_idx % 2
                b_row = b_r0 + yi
                b_col = b_c0 + yj
                b_idx = b_row * p + b_col
                V[b_idx, col] += val
            
            # Fill W: C entries
            for z_idx, val in strassen_z[s].items():
                # z_idx: 0=Z00, 1=Z01, 2=Z10, 3=Z11
                zi = z_idx // 2
                zj = z_idx % 2
                c_row = c_r0 + zi
                c_col = c_c0 + zj
                c_idx = c_col * n + c_row  # W[k*n+i, r] convention
                W[c_idx, col] += val
    
    # Now attempt to reduce rank by finding linear dependencies.
    # The rank-1 tensors are U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # But for the decomposition to be valid, we need:
    # sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] = T (the multiplication tensor)
    
    # A key observation: if we can find coefficients c_r such that
    # sum_r c_r * U[:,r] = 0 AND sum_r c_r * V[:,r] = 0
    # then we can eliminate those terms (they contribute zero to the tensor).
    # But that's unlikely.
    
    # More realistically: check if any column of U⊗V is linearly dependent.
    # If U[:,r1] ⊗ V[:,r1] = sum_{r≠r1} alpha_r * U[:,r] ⊗ V[:,r]
    # then we can potentially merge, but only if W also satisfies the relation.
    
    # Let's check the rank of the matrix whose columns are vec(U[:,r] ⊗ V[:,r])
    # This is an 8*16 = 128 x 28 matrix.
    
    UV = np.zeros((n*m * m*p, 28))
    for r in range(28):
        UV[:, r] = np.kron(U[:, r], V[:, r])
    
    # Check rank of UV
    try:
        rank_uv = np.linalg.matrix_rank(UV, tol=1e-10)
    except:
        rank_uv = 28
    
    # If rank_uv < 28, there are linear dependencies among the U⊗V columns.
    # In that case, we can potentially reduce the rank.
    
    # For now, just return the rank-28 decomposition.
    # The peeling strategy would require checking if the dependencies
    # are compatible with W, which is complex.
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
