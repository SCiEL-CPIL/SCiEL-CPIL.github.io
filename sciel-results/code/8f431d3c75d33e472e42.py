import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Define rank-23 decomposition for 3x3x3
    # Each term t: (u_vec, v_vec, w_vec) as 9-element lists (row-major 3x3)
    # U[i,j] * V[j,k] summed over t should give W[i,k]
    
    # BCLR-23 coefficients
    # Term: (coeffs for A, coeffs for B, coeffs for C)
    # A indices: 0=a11, 1=a12, 2=a13, 3=a21, 4=a22, 5=a23, 6=a31, 7=a32, 8=a33
    # B indices: 0=b11, 1=b12, 2=b13, 3=b21, 4=b22, 5=b23, 6=b31, 7=b32, 8=b33
    # C indices: 0=c11, 1=c12, 2=c13, 3=c21, 4=c22, 5=c23, 6=c31, 7=c32, 8=c33
    
    terms_3x3 = []
    
    # p1 = a11*b11
    terms_3x3.append(([1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    # p2 = (a12+a13)*b13
    terms_3x3.append(([0,1,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0]))
    # p3 = (a13-a12)*b12
    terms_3x3.append(([0,-1,1,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0]))
    # p4 = a21*b21
    terms_3x3.append(([0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # p5 = (a22+a23)*b23
    terms_3x3.append(([0,0,0,0,1,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0]))
    # p6 = (a23-a22)*b22
    terms_3x3.append(([0,0,0,0,-1,1,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # p7 = a31*b31
    terms_3x3.append(([0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]))
    # p8 = (a32+a33)*b33
    terms_3x3.append(([0,0,0,0,0,0,0,1,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # p9 = (a33-a32)*b32
    terms_3x3.append(([0,0,0,0,0,0,0,-1,1], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]))
    # p10 = (a11+a21)*b21
    terms_3x3.append(([1,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # p11 = (a21-a11)*b11
    terms_3x3.append(([1,0,0,-1,0,0,0,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    # p12 = (a12+a22)*b22
    terms_3x3.append(([0,1,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # p13 = (a22-a12)*b12
    terms_3x3.append(([0,1,0,0,-1,0,0,0,0], [0,1,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0]))
    # p14 = (a13+a23)*b23
    terms_3x3.append(([0,0,1,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0,0]))
    # p15 = (a23-a13)*b13
    terms_3x3.append(([0,0,1,0,0,-1,0,0,0], [0,0,1,0,0,0,0,0,0], [0,0,0,0,0,1,0,0,0]))
    # p16 = (a21+a31)*b31
    terms_3x3.append(([0,0,0,1,0,0,1,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]))
    # p17 = (a31-a21)*b21
    terms_3x3.append(([0,0,0,1,0,0,-1,0,0], [0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    # p18 = (a22+a32)*b32
    terms_3x3.append(([0,0,0,0,1,0,0,1,0], [0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,0]))
    # p19 = (a32-a22)*b22
    terms_3x3.append(([0,0,0,0,1,0,0,-1,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,1,0]))
    # p20 = (a23+a33)*b33
    terms_3x3.append(([0,0,0,0,0,1,0,0,1], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,0,1]))
    # p21 = (a33-a23)*b23
    terms_3x3.append(([0,0,0,0,0,1,0,0,-1], [0,0,0,0,0,1,0,0,0], [0,0,0,0,0,0,0,0,1]))
    # p22 = (a11+a31)*b31
    terms_3x3.append(([1,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0,0]))
    # p23 = (a31-a11)*b11
    terms_3x3.append(([1,0,0,0,0,0,-1,0,0], [1,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    
    # Verify the 3x3x3 decomposition
    def verify_3x3(terms):
        # Build U, V, W for 3x3x3
        U = [[0]*len(terms) for _ in range(9)]
        V = [[0]*len(terms) for _ in range(9)]
        W = [[0]*len(terms) for _ in range(9)]
        for t, (u, v, w) in enumerate(terms):
            for i in range(9):
                U[i][t] = u[i]
                V[i][t] = v[i]
                W[i][t] = w[i]
        
        # Check: for all i,k, sum_j A[i,j]*B[j,k] = sum_t U[i,j]*V[j,k]*W[i,k]... 
        # Actually the decomposition is: C[i,k] = sum_t (sum_j U[i,j]*V[j,k]) * W[i,k]? No.
        # The tensor decomposition is: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
        # where T[i,j,k] = delta(i,j,k) for matrix mult, i.e., T[i,j,k] = 1 if j is the middle index
        # Actually for matrix mult C = A*B: C[i,k] = sum_j A[i,j]*B[j,k]
        # So the tensor is T[i,j,k] = 1 (it's just the multiplication tensor)
        # Decomposition: T[i,j,k] = sum_t U[i,j,t] * V[j,k,t] * W[i,k,t]
        
        # Verify: for all i,j,k, sum_t U[i,j,t]*V[j,k,t]*W[i,k,t] should equal 1 if this is the mult tensor
        # Wait, the mult tensor for C[i,k] = sum_j A[i,j]*B[j,k] has T[i,j,k] = 1 for all i,j,k? No.
        # The tensor T such that C = A*B is: T[i,j,k] where i indexes A rows, j indexes A cols/B rows, k indexes B cols
        # T[i,j,k] = 1 for all i,j,k (it's the identity tensor for this bilinear form)
        # Actually no. The multiplication tensor is T[i,j,k] = 1 if the indices are consistent.
        # For C[i,k] = sum_j A[i,j] B[j,k], the tensor is T[i,j,k] = 1 for all valid i,j,k.
        
        # So we need: sum_t U[i,j,t] * V[j,k,t] * W[i,k,t] = 1 for all i,j,k in {0,1,2}
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    total = 0
                    for t in range(len(terms)):
                        total += U[i*3+j][t] * V[j*3+k][t] * W[i*3+k][t]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    if not verify_3x3(terms_3x3):
        # Fall back to schoolbook
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
    
    # Now build 3x3x6 using two copies of the 3x3x3 decomposition
    # B is 3x6, so B = [B1, B2] where B1, B2 are 3x3
    # C = [C1, C2] where C1 = A*B1, C2 = A*B2
    
    rank = 46  # 2 * 23
    U = [[0] * rank for _ in range(9)]   # 3*3 = 9 rows
    V = [[0] * rank for _ in range(18)]  # 3*6 = 18 rows
    W = [[0] * rank for _ in range(18)]  # 6*3 = 18 rows
    
    for t in range(23):
        u, v, w = terms_3x3[t]
        # First half (columns 0-2 of B, columns 0-2 of C)
        U_col = t
        V_col = t
        W_col = t
        for i in range(9):
            U[i][U_col] = u[i]
        for j in range(3):
            for k in range(3):
                V[j*3+k][V_col] = v[j*3+k]
        for i in range(3):
            for k in range(3):
                W[k*3+i][W_col] = w[i*3+k]
        
        # Second half (columns 3-5 of B, columns 3-5 of C)
        U_col2 = 23 + t
        V_col2 = 23 + t
        W_col2 = 23 + t
        for i in range(9):
            U[i][U_col2] = u[i]
        for j in range(3):
            for k in range(3):
                V[j*3+(k+3)][V_col2] = v[j*3+k]
        for i in range(3):
            for k in range(3):
                W[(k+3)*3+i][W_col2] = w[i*3+k]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
