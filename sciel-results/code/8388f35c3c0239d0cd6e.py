import time
import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    
    # Build a rank-28 tensor decomposition using Strassen's 7-multiplication
    # algorithm applied to each of the four 2x2 * 2x2 sub-block products.
    # A is n x m = 2x4, B is m x p = 4x4, C is n x p = 2x4
    # C = A * B
    # Partition: A = [A11 A12], B = [[B11 B12],[B21 B22]]
    # C[:,0:2] = A11*B11 + A12*B21
    # C[:,2:4] = A11*B12 + A12*B22
    
    # Each 2x2*2x2 product uses 7 multiplications (Strassen)
    # Total: 4 * 7 = 28 multiplications
    
    # Index conventions:
    # A[i,j] -> a_idx = i*m + j = i*4 + j  (rows 0-1, cols 0-3)
    # B[j,k] -> b_idx = j*p + k = j*4 + k  (rows 0-3, cols 0-3)
    # C[i,k] -> c_idx = k*n + i = k*2 + i  (W[k*n+i, r] convention)
    
    # Strassen for X*Y where X,Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Result: Z00=M1+M4-M5+M7, Z01=M3+M5, Z10=M2+M4, Z11=M1-M2+M3+M6
    
    # For each sub-product, define which A entries, B entries, and C entries
    # participate with which coefficients in each of the 7 Strassen multiplications.
    
    # strassen_x[s]: {x_local_idx: coeff} for multiplication s (s=0..6)
    # x_local_idx: 0=X00, 1=X01, 2=X10, 3=X11
    # strassen_y[s]: {y_local_idx: coeff}
    # strassen_z[s]: {z_local_idx: coeff}
    # z_local_idx: 0=Z00, 1=Z01, 2=Z10, 3=Z11
    
    sx = [
        {0: 1, 3: 1},    # M1: X00 + X11
        {2: 1, 3: 1},    # M2: X10 + X11
        {0: 1},          # M3: X00
        {3: 1},          # M4: X11
        {0: 1, 1: 1},    # M5: X00 + X01
        {2: 1, 0: -1},   # M6: X10 - X00
        {1: 1, 3: -1},   # M7: X01 - X11
    ]
    sy = [
        {0: 1, 3: 1},    # M1: Y00 + Y11
        {0: 1},          # M2: Y00
        {1: 1, 3: -1},   # M3: Y01 - Y11
        {2: 1, 0: -1},   # M4: Y10 - Y00
        {3: 1},          # M5: Y11
        {0: 1, 1: 1},    # M6: Y00 + Y01
        {2: 1, 3: 1},    # M7: Y10 + Y11
    ]
    sz = [
        {0: 1, 3: 1},    # M1 -> Z00, Z11
        {2: 1, 3: -1},   # M2 -> Z10, Z11
        {1: 1, 3: 1},    # M3 -> Z01, Z11
        {0: 1, 2: 1},    # M4 -> Z00, Z10
        {0: -1, 1: 1},   # M5 -> Z00, Z01
        {3: 1},          # M6 -> Z11
        {0: 1},          # M7 -> Z00
    ]
    
    # Four sub-products: (a_row0, a_col0, b_row0, b_col0, c_col0)
    # c_row is always 0 for the 2-row output (n=2), and the 2x2 block spans both rows
    # Actually C is 2x4, so c_row spans 0-1, c_col determines which half
    sub_products = [
        (0, 0, 0, 0, 0),   # A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
        (0, 2, 2, 0, 0),   # A[0:2,2:4] * B[2:4,0:2] -> C[0:2,0:2]
        (0, 0, 0, 2, 2),   # A[0:2,0:2] * B[0:2,2:4] -> C[0:2,2:4]
        (0, 2, 2, 2, 2),   # A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    ]
    
    rank = 28
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (16, 28)
    W = np.zeros((p * n, rank))  # (8, 28)
    
    for sp_idx, (ar0, ac0, br0, bc0, cc0) in enumerate(sub_products):
        for s in range(7):
            col = sp_idx * 7 + s
            
            for x_idx, val in sx[s].items():
                xi = x_idx // 2
                xj = x_idx % 2
                a_row = ar0 + xi
                a_col = ac0 + xj
                a_idx = a_row * m + a_col
                U[a_idx, col] += val
            
            for y_idx, val in sy[s].items():
                yi = y_idx // 2
                yj = y_idx % 2
                b_row = br0 + yi
                b_col = bc0 + yj
                b_idx = b_row * p + b_col
                V[b_idx, col] += val
            
            for z_idx, val in sz[s].items():
                zi = z_idx // 2
                zj = z_idx % 2
                c_row = zi
                c_col = cc0 + zj
                c_idx = c_col * n + c_row
                W[c_idx, col] += val
            
            if time.time() - start > 130:
                break
        if time.time() - start > 130:
            break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    result = construct()
    print(f"U shape: {len(result['U'])}x{len(result['U'][0])}")
    print(f"V shape: {len(result['V'])}x{len(result['V'][0])}")
    print(f"W shape: {len(result['W'])}x{len(result['W'][0])}")
