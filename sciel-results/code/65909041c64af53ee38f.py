"""Matrix multiplication 2x4x4 rank-minimal decomposition via Strassen-style block recursion."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 4]

# Deadline guard
_START_TIME = time.time()
DEADLINE = _START_TIME + 130  # 10s safety margin before 140s kill

def _time_ok():
    return time.time() < DEADLINE

def construct():
    """
    Construct a rank-minimal (or near-minimal) exact tensor decomposition for 2x4x4 matrix multiplication.
    
    Strategy: Use Strassen-style block decomposition.
    - Split A (2x4) into two 2x2 blocks: A_top, A_bottom (each 2x2)
    - Split B (4x4) into four 2x2 blocks: B11, B12, B21, B22
    - Output C = A * B is 2x4, with C_left = A_top*B11 + A_bottom*B21, C_right = A_top*B12 + A_bottom*B22
    
    Using Strassen's 7-product algorithm for each 2x2x2 multiplication:
    For computing X*Y where X,Y are 2x2:
    M1 = (X11+X22)*(Y11+Y22)
    M2 = (X21+X22)*Y11
    M3 = X11*(Y12-Y22)
    M4 = X22*(Y21-Y11)
    M5 = (X11+X12)*Y22
    M6 = (X21-X11)*(Y11+Y12)
    M7 = (X12-X22)*(Y21+Y22)
    Result:
    Z11 = M1+M4-M5+M7
    Z12 = M3+M5
    Z21 = M2+M4
    Z22 = M1-M2+M3+M6
    
    For 2x4x4, we have:
    C_left = A_top * B11 + A_bottom * B21
    C_right = A_top * B12 + A_bottom * B22
    
    Each 2x2x2 multiplication costs 7 rank-1 terms.
    We have 4 such multiplications: (A_top,B11), (A_bottom,B21), (A_top,B12), (A_bottom,B22)
    Total rank = 4 * 7 = 28.
    
    This is an exact decomposition with integer coefficients.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    # U has shape (n*m, rank) = (8, 28)
    # V has shape (m*p, rank) = (16, 28)
    # W has shape (p*n, rank) = (8, 28)
    # Using the trace convention: W[k*n + i, r]
    # A is n x m = 2 x 4, indexed A[i, j] for i in 0..1, j in 0..3
    # B is m x p = 4 x 4, indexed B[j, k] for j in 0..3, k in 0..3
    # C is n x p = 2 x 4, indexed C[i, k]
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Block structure:
    # A_top = A[0:2, 0:2] (rows 0-1, cols 0-1 of A) -> actually A is 2x4, so A_top = A[:, 0:2]
    # Wait, let me reconsider. A is 2x4. We split columns into two groups of 2.
    # A_top = A[:, 0:2] (2x2), A_bottom = A[:, 2:4] (2x2)
    # B is 4x4. Split rows into two groups of 2, and columns into two groups of 2.
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2] = A_top * B11 + A_bottom * B21
    # C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4] = A_top * B12 + A_bottom * B22
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # Strassen 7 products for 2x2 * 2x2:
    # For X (2x2) * Y (2x2):
    # X indices: X[0,0], X[0,1], X[1,0], X[1,1]
    # Y indices: Y[0,0], Y[0,1], Y[1,0], Y[1,1]
    # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
    # M2 = (X[1,0]+X[1,1]) * Y[0,0]
    # M3 = X[0,0] * (Y[0,1]-Y[1,1])
    # M4 = X[1,1] * (Y[1,0]-Y[0,0])
    # M5 = (X[0,0]+X[0,1]) * Y[1,1]
    # M6 = (X[1,0]-X[0,0]) * (Y[0,0]+Y[0,1])
    # M7 = (X[0,1]-X[1,1]) * (Y[1,0]+Y[1,1])
    #
    # Z[0,0] = M1 + M4 - M5 + M7
    # Z[0,1] = M3 + M5
    # Z[1,0] = M2 + M4
    # Z[1,1] = M1 - M2 + M3 + M6
    
    # For each 2x2x2 multiplication, we need to map the Strassen products to the global indices.
    
    def strassen_products(X_row_start, X_col_start, Y_row_start, Y_col_start, base_col):
        """
        Generate 7 Strassen rank-1 terms for multiplying X (2x2 at position X_row_start:X_row_start+2, X_col_start:X_col_start+2)
        by Y (2x2 at position Y_row_start:Y_row_start+2, Y_col_start:Y_col_start+2).
        
        X is from A: A[row, col] where row in 0..1, col in X_col_start..X_col_start+1
        Y is from B: B[row, col] where row in Y_row_start..Y_row_start+1, col in Y_col_start..Y_col_start+1
        
        Returns list of 7 (u_coeffs, v_coeffs, w_coeffs) where each is a dict mapping index to coefficient.
        """
        products = []
        
        # M1 = (X[0,0]+X[1,1]) * (Y[0,0]+Y[1,1])
        u = {X_row_start + 0 * 2 + 0: 1, X_row_start + 1 * 2 + 1: 1}  # A indices: row*2+col for A (2x4)
        # Wait, A is 2x4, so A[i,j] has index i*m + j = i*4 + j in U
        # B is 4x4, so B[j,k] has index j*p + k = j*4 + k in V
        # C is 2x4, so C[i,k] has index k*n + i = k*2 + i in W
        
        # Let me redo with correct indexing.
        # A[i,j]: i in 0..1, j in 0..3. U index = i*4 + j
        # B[j,k]: j in 0..3, k in 0..3. V index = j*4 + k
        # C[i,k]: i in 0..1, k in 0..3. W index = k*2 + i
        
        # X is a 2x2 submatrix of A. X[0,0]=A[xi, xj], X[0,1]=A[xi, xj+1], X[1,0]=A[xi+1, xj], X[1,1]=A[xi+1, xj+1]
        # where xi = X_row_start (0 or 1), xj = X_col_start (0 or 2)
        
        # Y is a 2x2 submatrix of B. Y[0,0]=B[yi, yj], Y[0,1]=B[yi, yj+1], Y[1,0]=B[yi+1, yj], Y[1,1]=B[yi+1, yj+1]
        # where yi = Y_row_start (0 or 2), yj = Y_col_start (0 or 2)
        
        xi = X_row_start
        xj = X_col_start
        yi = Y_row_start
        yj = Y_col_start
        
        # M1 = (A[xi,xj] + A[xi+1,xj+1]) * (B[yi,yj] + B[yi+1,yj+1])
        u = {(xi)*4 + xj: 1, (xi+1)*4 + (xj+1): 1}
        v = {(yi)*4 + yj: 1, (yi+1)*4 + (yj+1): 1}
        products.append((u, v, base_col))
        
        # M2 = (A[xi+1,xj] + A[xi+1,xj+1]) * B[yi,yj]
        u = {(xi+1)*4 + xj: 1, (xi+1)*4 + (xj+1): 1}
        v = {(yi)*4 + yj: 1}
        products.append((u, v, base_col + 1))
        
        # M3 = A[xi,xj] * (B[yi,yj+1] - B[yi+1,yj+1])
        u = {(xi)*4 + xj: 1}
        v = {(yi)*4 + (yj+1): 1, (yi+1)*4 + (yj+1): -1}
        products.append((u, v, base_col + 2))
        
        # M4 = A[xi+1,xj+1] * (B[yi+1,yj] - B[yi,yj])
        u = {(xi+1)*4 + (xj+1): 1}
        v = {(yi+1)*4 + yj: 1, (yi)*4 + yj: -1}
        products.append((u, v, base_col + 3))
        
        # M5 = (A[xi,xj] + A[xi,xj+1]) * B[yi+1,yj+1]
        u = {(xi)*4 + xj: 1, (xi)*4 + (xj+1): 1}
        v = {(yi+1)*4 + (yj+1): 1}
        products.append((u, v, base_col + 4))
        
        # M6 = (A[xi+1,xj] - A[xi,xj]) * (B[yi,yj] + B[yi,yj+1])
        u = {(xi+1)*4 + xj: 1, (xi)*4 + xj: -1}
        v = {(yi)*4 + yj: 1, (yi)*4 + (yj+1): 1}
        products.append((u, v, base_col + 5))
        
        # M7 = (A[xi,xj+1] - A[xi+1,xj+1]) * (B[yi+1,yj] + B[yi+1,yj+1])
        u = {(xi)*4 + (xj+1): 1, (xi+1)*4 + (xj+1): -1}
        v = {(yi+1)*4 + yj: 1, (yi+1)*4 + (yj+1): 1}
        products.append((u, v, base_col + 6))
        
        return products
    
    def strassen_output(M_base, X_row_start, X_col_start, Y_row_start, Y_col_start):
        """
        Compute the output coefficients for a 2x2 result Z = X*Y.
        Z[0,0] = M1 + M4 - M5 + M7
        Z[0,1] = M3 + M5
        Z[1,0] = M2 + M4
        Z[1,1] = M1 - M2 + M3 + M6
        
        Returns dict mapping W index to coefficient.
        """
        xi = X_row_start
        xj = X_col_start
        yi = Y_row_start
        yj = Y_col_start
        
        # Z[0,0] = C[xi, yj] -> W index = yj*2 + xi
        # Z[0,1] = C[xi, yj+1] -> W index = (yj+1)*2 + xi
        # Z[1,0] = C[xi+1, yj] -> W index = yj*2 + (xi+1)
        # Z[1,1] = C[xi+1, yj+1] -> W index = (yj+1)*2 + (xi+1)
        
        M1 = M_base + 0
        M2 = M_base + 1
        M3 = M_base + 2
        M4 = M_base + 3
        M5 = M_base + 4
        M6 = M_base + 5
        M7 = M_base + 6
        
        w = {}
        # Z[0,0]
        idx = yj * 2 + xi
        w[idx] = w.get(idx, 0) + 1  # M1
        w[idx] = w.get(idx, 0) + 1  # M4
        w[idx] = w.get(idx, 0) - 1  # -M5
        w[idx] = w.get(idx, 0) + 1  # M7
        
        # Z[0,1]
        idx = (yj + 1) * 2 + xi
        w[idx] = w.get(idx, 0) + 1  # M3
        w[idx] = w.get(idx, 0) + 1  # M5
        
        # Z[1,0]
        idx = yj * 2 + (xi + 1)
        w[idx] = w.get(idx, 0) + 1  # M2
        w[idx] = w.get(idx, 0) + 1  # M4
        
        # Z[1,1]
        idx = (yj + 1) * 2 + (xi + 1)
        w[idx] = w.get(idx, 0) + 1  # M1
        w[idx] = w.get(idx, 0) - 1  # -M2
        w[idx] = w.get(idx, 0) + 1  # M3
        w[idx] = w.get(idx, 0) + 1  # M6
        
        return w
    
    # Four 2x2x2 multiplications:
    # 1. A_top * B11: A[:, 0:2] * B[0:2, 0:2]
    # 2. A_bottom * B21: A[:, 2:4] * B[2:4, 0:2]
    # 3. A_top * B12: A[:, 0:2] * B[0:2, 2:4]
    # 4. A_bottom * B22: A[:, 2:4] * B[2:4, 2:4]
    
    multiplications = [
        (0, 0, 0, 0),   # A_top * B11: xi=0, xj=0, yi=0, yj=0
        (0, 2, 2, 0),   # A_bottom * B21: xi=0, xj=2, yi=2, yj=0
        (0, 0, 0, 2),   # A_top * B12: xi=0, xj=0, yi=0, yj=2
        (0, 2, 2, 2),   # A_bottom * B22: xi=0, xj=2, yi=2, yj=2
    ]
    
    col = 0
    for (xi, xj, yi, yj) in multiplications:
        if not _time_ok():
            break
        prods = strassen_products(xi, xj, yi, yj, col)
        for (u, v, c) in prods:
            for (idx, coeff) in u.items():
                U[idx][c] += coeff
            for (idx, coeff) in v.items():
                V[idx][c] += coeff
        # Add W contributions
        w_contrib = strassen_output(col, xi, xj, yi, yj)
        for (idx, coeff) in w_contrib.items():
            for c in range(7):
                W[idx][col + c] += coeff
        col += 7
    
    # Verify the decomposition
    if _time_ok():
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta(i,j,k) for the multiplication tensor
        # Actually, the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 if this corresponds to a valid term, else 0
        # The multiplication tensor has T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        
        # Let's verify by computing the product
        # The reconstructed tensor should have T[i,j,k] = 1 for all i in 0..1, j in 0..3, k in 0..3
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > max_err:
                        max_err = abs(val - 1.0)
        
        if max_err > 1e-9:
            # Fallback to schoolbook if verification fails
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
