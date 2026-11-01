"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-7 exact tensor decomposition for 2x2 matrix multiplication.
    Uses Strassen's algorithm. This is the theoretical minimum rank for 2x2
    matrix multiplication over the reals.
    """
    n, m, p = DIMENSIONS  # 2, 2, 2
    rank = 7

    # Strassen's 7 products define linear combinations of input matrix elements.
    # Let A = [[a00, a01], [a10, a11]] and B = [[b00, b01], [b10, b11]].
    # The products are:
    # m1 = (a00 + a11) * (b00 + b11)
    # m2 = (a10 + a11) * b00
    # m3 = a00 * (b01 - b11)
    # m4 = a11 * (b10 - b00)
    # m5 = (a00 + a01) * b11
    # m6 = (a10 - a00) * (b00 + b01)
    # m7 = (a01 - a11) * (b10 + b11)
    
    # We construct U, V, W such that:
    # C[i, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] is not quite right.
    # Standard tensor decomposition: T = sum_c U_c \otimes V_c \otimes W_c
    # Where T_{i,j,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ? 
    # The contract says:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank).
    # Output C is n x p.
    # C[i, k] = sum_{j, c} A[i, j] * (sum_{c'} U[i*m+j, c'] * V[j*p+k, c'] ... no.
    # The multiplication is A * B = C.
    # C[i, k] = sum_j A[i, j] B[j, k].
    # The decomposition approximates this bilinear map.
    # Usually: C[i, k] = sum_c (sum_j U[i*m+j, c] A[i, j]) * (sum_j' V[j'*p+k, c] B[j', k])
    # Wait, the standard form is:
    # C = sum_c (U_c \otimes W_c) * (V_c^T \otimes I)? No.
    # Let's stick to the specific interface:
    # U[i*m+j, c] is coeff of A[i,j] in c-th product.
    # V[j*p+k, c] is coeff of B[j,k] in c-th product.
    # W[k*n+i, c] is coeff of product c in C[i,k]? 
    # The contract says W shape (p*n, rank). Index is k*n+i.
    # So C[i, k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * (sum_j V[j*p+k, c] B[j,k])
    # This is a rank-r decomposition of the multiplication tensor.
    
    # Let's define the linear forms for the 7 Strassen products.
    # Product c is L_A(c) * L_B(c).
    # L_A(c) = sum_{i,j} u_{i,j,c} A[i,j]
    # L_B(c) = sum_{j,k} v_{j,k,c} B[j,k]
    # Then C[i, k] = sum_c w_{i,k,c} * L_A(c) * L_B(c)
    
    # Strassen Forms:
    # c=0: L_A = A[0,0] + A[1,1], L_B = B[0,0] + B[1,1]
    # c=1: L_A = A[1,0] + A[1,1], L_B = B[0,0]
    # c=2: L_A = A[0,0], L_B = B[0,1] - B[1,1]
    # c=3: L_A = A[1,1], L_B = B[1,0] - B[0,0]
    # c=4: L_A = A[0,0] + A[0,1], L_B = B[1,1]
    # c=5: L_A = A[1,0] - A[0,0], L_B = B[0,0] + B[0,1]
    # c=6: L_A = A[0,1] - A[1,1], L_B = B[1,0] + B[1,1]
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def set_u(i, j, c, val):
        U[i*m+j][c] = val
        
    def set_v(j, k, c, val):
        V[j*p+k][c] = val
        
    def set_w(i, k, c, val):
        W[k*n+i][c] = val

    # Strassen coefficients
    # c=0: (A00+A11)(B00+B11)
    set_u(0,0,0,1); set_u(1,1,0,1)
    set_v(0,0,0,1); set_v(1,1,0,1)
    # c=1: (A10+A11)B00
    set_u(1,0,1,1); set_u(1,1,1,1)
    set_v(0,0,1,1)
    # c=2: A00(B01-B11)
    set_u(0,0,2,1)
    set_v(0,1,2,1); set_v(1,1,2,-1)
    # c=3: A11(B10-B00)
    set_u(1,1,3,1)
    set_v(1,0,3,1); set_v(0,0,3,-1)
    # c=4: (A00+A01)B11
    set_u(0,0,4,1); set_u(0,1,4,1)
    set_v(1,1,4,1)
    # c=5: (A10-A00)(B00+B01)
    set_u(1,0,5,1); set_u(0,0,5,-1)
    set_v(0,0,5,1); set_v(0,1,5,1)
    # c=6: (A01-A11)(B10+B11)
    set_u(0,1,6,1); set_u(1,1,6,-1)
    set_v(1,0,6,1); set_v(1,1,6,1)
    
    # Reconstruction
    # C00 = m0 + m3 - m4 + m6
    set_w(0, 0, 0, 1)
    set_w(0, 0, 3, 1)
    set_w(0, 0, 4, -1)
    set_w(0, 0, 6, 1)
    
    # C10 = m1 + m3
    set_w(1, 0, 1, 1)
    set_w(1, 0, 3, 1)
    
    # C01 = m2 + m4
    set_w(0, 1, 2, 1)
    set_w(0, 1, 4, 1)
    
    # C11 = m0 - m1 + m2 + m5
    set_w(1, 1, 0, 1)
    set_w(1, 1, 1, -1)
    set_w(1, 1, 2, 1)
    set_w(1, 1, 5, 1)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
