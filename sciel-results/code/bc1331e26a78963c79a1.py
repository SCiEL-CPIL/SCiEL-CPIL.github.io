"""Matrix multiplication 2x4x4 with Strassen-based rank-28 decomposition."""
import time

DIMENSIONS = [2, 4, 4]

def _strassen_2x2(U, V, W, col, x_rows, y_rows, z_cols, x_cols, y_cols, z_rows):
    """
    Apply Strassen's 7 multiplications for a (2x2)*(2x2) product.
    
    U: list of lists, shape (n*m, rank) - A entries
    V: list of lists, shape (m*p, rank) - B entries  
    W: list of lists, shape (p*n, rank) - C entries
    
    x_rows: indices into A's rows for the 2x2 block (2 entries)
    x_cols: indices into A's cols for the 2x2 block (2 entries)
    y_rows: indices into B's rows for the 2x2 block (2 entries)
    y_cols: indices into B's cols for the 2x2 block (2 entries)
    z_rows: indices into C's rows for the 2x2 block (2 entries)
    z_cols: indices into C's cols for the 2x2 block (2 entries)
    
    Returns: next column index (col + 7)
    """
    n, m, p = DIMENSIONS
    rank = 28
    
    # A[i,j] is at U index i*m + j
    # B[j,k] is at V index j*p + k
    # C[i,k] is at W index k*n + i (trace convention W[k*n+i])
    
    x00 = x_rows[0] * m + x_cols[0]
    x01 = x_rows[0] * m + x_cols[1]
    x10 = x_rows[1] * m + x_cols[0]
    x11 = x_rows[1] * m + x_cols[1]
    
    y00 = y_rows[0] * p + y_cols[0]
    y01 = y_rows[0] * p + y_cols[1]
    y10 = y_rows[1] * p + y_cols[0]
    y11 = y_rows[1] * p + y_cols[1]
    
    z00 = z_cols[0] * n + z_rows[0]
    z01 = z_cols[1] * n + z_rows[0]
    z10 = z_cols[0] * n + z_rows[1]
    z11 = z_cols[1] * n + z_rows[1]
    
    # M1 = (x00+x11)*(y00+y11)
    U[x00][col] += 1
    U[x11][col] += 1
    V[y00][col] += 1
    V[y11][col] += 1
    W[z00][col] += 1
    W[z11][col] += 1
    col += 1
    
    # M2 = (x10+x11)*y00
    U[x10][col] += 1
    U[x11][col] += 1
    V[y00][col] += 1
    W[z10][col] += 1
    W[z11][col] += 1
    col += 1
    
    # M3 = x00*(y01-y11)
    U[x00][col] += 1
    V[y01][col] += 1
    V[y11][col] -= 1
    W[z01][col] += 1
    W[z11][col] += 1
    col += 1
    
    # M4 = x11*(y10-y00)
    U[x11][col] += 1
    V[y10][col] += 1
    V[y00][col] -= 1
    W[z00][col] += 1
    W[z10][col] += 1
    col += 1
    
    # M5 = (x00+x01)*y11
    U[x00][col] += 1
    U[x01][col] += 1
    V[y11][col] += 1
    W[z00][col] -= 1
    W[z11][col] += 1
    col += 1
    
    # M6 = (x10-x00)*(y00+y01)
    U[x10][col] += 1
    U[x00][col] -= 1
    V[y00][col] += 1
    V[y01][col] += 1
    W[z10][col] += 1
    W[z11][col] -= 1
    col += 1
    
    # M7 = (x01-x11)*(y10+y11)
    U[x01][col] += 1
    U[x11][col] -= 1
    V[y10][col] += 1
    V[y11][col] += 1
    W[z00][col] += 1
    W[z11][col] += 1
    col += 1
    
    return col

def construct():
    """
    Construct a rank-28 tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    C = A * B where A is 2x4, B is 4x4, C is 2x4.
    
    Split A into A1=A[:,0:2], A2=A[:,2:4] (each 2x2).
    Split B into 4 blocks: B11=B[0:2,0:2], B12=B[0:2,2:4], B21=B[2:4,0:2], B22=B[2:4,2:4].
    
    C[:,0:2] = A1*B11 + A2*B21
    C[:,2:4] = A1*B12 + A2*B22
    
    Use 7 Strassen multiplications for each (2x2)*(2x2) sub-product: 4*7 = 28 total.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 28)
    V = [[0] * rank for _ in range(m * p)]  # shape (16, 28)
    W = [[0] * rank for _ in range(p * n)]  # shape (8, 28)
    
    # A rows: 0, 1; A cols: 0, 1, 2, 3
    # B rows: 0, 1, 2, 3; B cols: 0, 1, 2, 3
    # C rows: 0, 1; C cols: 0, 1, 2, 3
    
    col = 0
    
    # Product 1: A1 * B11 -> contributes to C[:, 0:2]
    # A1 rows = [0,1], A1 cols = [0,1]
    # B11 rows = [0,1], B11 cols = [0,1]
    # C block rows = [0,1], C block cols = [0,1]
    col = _strassen_2x2(U, V, W, col, 
                         x_rows=[0,1], x_cols=[0,1],
                         y_rows=[0,1], y_cols=[0,1],
                         z_rows=[0,1], z_cols=[0,1])
    
    # Product 2: A2 * B21 -> contributes to C[:, 0:2]
    # A2 rows = [0,1], A2 cols = [2,3]
    # B21 rows = [2,3], B21 cols = [0,1]
    # C block rows = [0,1], C block cols = [0,1]
    col = _strassen_2x2(U, V, W, col,
                         x_rows=[0,1], x_cols=[2,3],
                         y_rows=[2,3], y_cols=[0,1],
                         z_rows=[0,1], z_cols=[0,1])
    
    # Product 3: A1 * B12 -> contributes to C[:, 2:4]
    # A1 rows = [0,1], A1 cols = [0,1]
    # B12 rows = [0,1], B12 cols = [2,3]
    # C block rows = [0,1], C block cols = [2,3]
    col = _strassen_2x2(U, V, W, col,
                         x_rows=[0,1], x_cols=[0,1],
                         y_rows=[0,1], y_cols=[2,3],
                         z_rows=[0,1], z_cols=[2,3])
    
    # Product 4: A2 * B22 -> contributes to C[:, 2:4]
    # A2 rows = [0,1], A2 cols = [2,3]
    # B22 rows = [2,3], B22 cols = [2,3]
    # C block rows = [0,1], C block cols = [2,3]
    col = _strassen_2x2(U, V, W, col,
                         x_rows=[0,1], x_cols=[2,3],
                         y_rows=[2,3], y_cols=[2,3],
                         z_rows=[0,1], z_cols=[2,3])
    
    # Verify: col should be 28
    assert col == rank, f"Expected col={rank}, got col={col}"
    
    return {"U": U, "V": V, "W": W}
