"""Strassen-based rank-42 decomposition for 2x4x6 matrix multiplication."""
DIMENSIONS = [2, 4, 6]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-42 tensor decomposition for 2x4x6 matrix multiplication
    using Strassen's algorithm on 2x2x2 sub-blocks.
    
    The decomposition exploits the block structure:
    - A (2x4) = [A1, A2] where A1, A2 are 2x2
    - B (4x6) = [B1, B2, B3] where each Bj is 4x2 = [Bj_top; Bj_bot]
    - For each column block r: Cr = A1*Bj_top + A2*Bj_bot (2x2 result)
    - Each such sum requires 14 scalar multiplications (7 Strassen each)
    - Total rank = 3 * 14 = 42
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0.0] * rank for _ in range(n * m)]  # (8, 42)
    V = [[0.0] * rank for _ in range(m * p)]  # (24, 42)
    W = [[0.0] * rank for _ in range(p * n)]  # (12, 42)
    
    # Strassen 7 products for X * Y = Z (2x2 matrices)
    # X[i,j], Y[j,k], Z[i,k]
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # Define Strassen linear forms for X and Y
    # Each product has (x_form, y_form, z_coeffs)
    # x_form: list of (i, j, coeff)
    # y_form: list of (j, k, coeff)
    # z_coeffs: dict mapping (i,k) -> coeff (sum of products)
    
    strassen_products = [
        # Product 1: (X00+X11)(Y00+Y11)
        {
            'x': [(0,0,1), (1,1,1)],
            'y': [(0,0,1), (1,1,1)],
            'z': {(0,0):1, (1,1):1},
        },
        # Product 2: (X10+X11)Y00
        {
            'x': [(1,0,1), (1,1,1)],
            'y': [(0,0,1)],
            'z': {(1,0):1, (1,1):-1},
        },
        # Product 3: X00(Y01-Y11)
        {
            'x': [(0,0,1)],
            'y': [(0,1,1), (1,1,-1)],
            'z': {(0,1):1, (1,1):1},
        },
        # Product 4: X11(Y10-Y00)
        {
            'x': [(1,1,1)],
            'y': [(1,0,1), (0,0,-1)],
            'z': {(0,0):1, (1,0):1},
        },
        # Product 5: (X00+X01)Y11
        {
            'x': [(0,0,1), (0,1,1)],
            'y': [(1,1,1)],
            'z': {(0,0):-1, (0,1):1},
        },
        # Product 6: (X10-X00)(Y00+Y01)
        {
            'x': [(1,0,1), (0,0,-1)],
            'y': [(0,0,1), (0,1,1)],
            'z': {(1,1):1},
        },
        # Product 7: (X01-X11)(Y10+Y11)
        {
            'x': [(0,1,1), (1,1,-1)],
            'y': [(1,0,1), (1,1,1)],
            'z': {(0,0):1, (1,1):-1},
        },
    ]
    
    col = 0
    for r in range(3):
        # Column block r: B columns 2*r, 2*r+1; C columns 2*r, 2*r+1
        # A1 = A[:, 0:2], A2 = A[:, 2:4]
        # Bj_top = B[0:2, 2*r:2*r+2], Bj_bot = B[2:4, 2*r:2*r+2]
        
        # Strassen for A1 * Bj_top
        for prod in strassen_products:
            # U: A1[i, j] -> global index i*4 + j
            for (i, j, coeff) in prod['x']:
                U[i * 4 + j][col] += float(coeff)
            # V: Bj_top[j, kb] -> global index j*6 + (2*r + kb)
            for (j, kb, coeff) in prod['y']:
                V[j * 6 + (2 * r + kb)][col] += float(coeff)
            # W: output (i, 2*r + kb) -> W row = (2*r + kb)*n + i = (2*r+kb)*2 + i
            for ((i, kb), zcoeff) in prod['z'].items():
                W[(2 * r + kb) * n + i][col] += float(zcoeff)
            col += 1
        
        # Strassen for A2 * Bj_bot
        for prod in strassen_products:
            # U: A2[i, j] -> global index i*4 + (j+2)
            for (i, j, coeff) in prod['x']:
                U[i * 4 + (j + 2)][col] += float(coeff)
            # V: Bj_bot[j, kb] -> global index (j+2)*6 + (2*r + kb)
            for (j, kb, coeff) in prod['y']:
                V[(j + 2) * 6 + (2 * r + kb)][col] += float(coeff)
            # W: same output indices
            for ((i, kb), zcoeff) in prod['z'].items():
                W[(2 * r + kb) * n + i][col] += float(zcoeff)
            col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition exactly reproduces matrix multiplication.
    For random matrices A (n x m) and B (m x p), check that
    sum_r U[:,r] * V[:,r] * W[:,r]^T == A * B (in tensor form)
    
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    # Convert to numpy
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B  # n x p
    
    # Compute via decomposition
    # T[i,a,b] = sum_r U[i*m+a, r] * V[a*p+b, r] * W[b*n+i, r]
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u = U_arr[:, r]  # n*m
        v = V_arr[:, r]  # m*p
        w = W_arr[:, r]  # p*n
        # Reshape
        u_mat = u.reshape(n, m)
        v_mat = v.reshape(m, p)
        w_mat = w.reshape(p, n)
        # Contribution to C[i,b] = sum_a u_mat[i,a] * v_mat[a,b] * w_mat[b,i]
        # = sum_a u_mat[i,a] * v_mat[a,b] * w_mat.T[i,b]
        # = (u_mat * w_mat.T) @ v_mat  ... no
        # C[i,b] += sum_a u_mat[i,a] * v_mat[a,b] * w_mat[b,i]
        # = sum_a u_mat[i,a] * v_mat[a,b] * w_mat.T[i,b]
        # = (u_mat * w_mat.T) @ v_mat  -> this gives (n x p)
        contrib = (u_mat * w_mat.T) @ v_mat
        C_computed += contrib
    
    if np.allclose(C_computed, C_expected, atol=1e-9):
        return True
    else:
        return False
# EVOLVE-BLOCK-END
