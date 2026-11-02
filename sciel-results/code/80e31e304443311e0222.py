import numpy as np
import time

def construct():
    n, m, p = 2, 2, 12
    
    rank = 20
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Verify the rank-5 decomposition for 2x2x3
    # A is 2x2, B is 2x3, C = A*B is 2x3
    # U indices: U[0]=a[0][0], U[1]=a[0][1], U[2]=a[1][0], U[3]=a[1][1]
    # V indices: V[0]=b[0][0], V[1]=b[0][1], V[2]=b[0][2], V[3]=b[1][0], V[4]=b[1][1], V[5]=b[1][2]
    # W indices: W[0]=c[0][0], W[1]=c[0][1], W[2]=c[0][2], W[3]=c[1][0], W[4]=c[1][1], W[5]=c[1][2]
    
    # Let me find a valid rank-5 decomposition by solving the system
    
    # The tensor T has entries T[a_idx, b_idx, c_idx] = 1 if c = a*b (appropriate contraction)
    # For our purposes, we need: sum_r U[a_idx, r] * V[b_idx, r] * W[c_idx, r] = delta_{c, (a*b)}
    # where a*b means the matrix product.
    
    # Actually, let me just construct it directly.
    # For 2x2x3, I'll use the following known decomposition:
    
    # M1 = a00 * b00 -> contributes to c00
    # M2 = a00 * b01 -> contributes to c01  
    # M3 = a00 * b02 -> contributes to c02
    # M4 = a01 * b10 -> contributes to c00
    # M5 = a01 * b11 -> contributes to c01
    # M6 = a01 * b12 -> contributes to c02
    # M7 = a10 * b00 -> contributes to c10
    # M8 = a10 * b01 -> contributes to c11
    # M9 = a10 * b02 -> contributes to c12
    # M10 = a11 * b10 -> contributes to c10
    # M11 = a11 * b11 -> contributes to c11
    # M12 = a11 * b12 -> contributes to c12
    
    # That's rank 12 (naive). We need rank 5.
    
    # Let me try a different approach: use linear algebra to find a rank-5 decomposition.
    # The tensor T can be viewed as a matrix of size (n*m)*(m*p) x (p*n) = 4*6 x 6 = 24 x 6
    # We need to find rank-1 decompositions.
    
    # Actually, for 2x2x3, let me try to construct it manually.
    
    # c[i][j] = sum_k a[i][k] * b[k][j]
    # c[0][j] = a[0][0]*b[0][j] + a[0][1]*b[1][j]
    # c[1][j] = a[1][0]*b[0][j] + a[1][1]*b[1][j]
    
    # Define linear combinations:
    # S1 = a[0][0] + a[0][1] + a[1][0] + a[1][1]
    # This gets complicated. Let me just use a numerical search.
    
    # Actually, I recall that the rank of 2x2x3 is indeed 5.
    # Here's one construction:
    
    # M1 = a00 * (b00 + b01 + b02)
    # M2 = a01 * (b10 + b11 + b12)
    # M3 = (a00 + a01 + a10 + a11) * (b00 + b10)
    # M4 = (a00 - a01 + a10 - a11) * (b01 + b11)
    # M5 = (a00 + a01 - a10 - a11) * (b02 + b12)
    
    # Let me verify if this works...
    # c[0][0] should = a00*b00 + a01*b10
    # From M1: a00*b00
    # From M2: a01*b10
    # From M3: (a00+a01+a10+a11)*b00 + ... (contributes a00*b00, a01*b00, a10*b00, a11*b00)
    # This doesn't seem to work directly.
    
    # Let me just do a numerical search for the rank-5 decomposition.
    # I'll set up the equations and solve.
    
    # The tensor entries: T[i, j, k] where i is (a_row, a_col) flattened, j is (b_row, b_col) flattened, k is (c_row, c_col) flattened
    # T = 1 if the entry is "active" in the multiplication
    
    # For our problem:
    # A is 2x2: entries (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # B is 2x3: entries (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> indices 0,1,2,3,4,5
    # C is 2x3: entries (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) -> indices 0,1,2,3,4,5
    
    # The multiplication: c[i][j] = sum_k a[i][k] * b[k][j]
    # So T[a_idx, b_idx, c_idx] = 1 if a_idx = (i,k), b_idx = (k,j), c_idx = (i,j)
    
    # Let me build the tensor and find its rank-5 decomposition numerically.
    
    # Actually, let me just hardcode a known valid decomposition.
    # After more thought, here's a rank-5 decomposition for 2x2x3:
    
    # Using the following 5 terms:
    # M1 = a00 * b00
    # M2 = a00 * (b01 + b02)
    # M3 = a01 * (b10 + b11 + b12)
    # M4 = (a10 + a11) * (b00 + b01 + b02)
    # M5 = (a00 + a01 + a10 + a11) * (b00 + b10)
    
    # This is getting too complex. Let me just use a safe rank-12 decomposition (4 blocks x 12) = rank 48.
    # Wait, that's worse than 42.
    
    # Actually, let me just use the schoolbook approach: for each of the 12 columns of B,
    # do a 2x2x1 multiplication which has rank 2 (since c[i] = a[i][0]*b[0] + a[i][1]*b[1]).
    # So rank = 12 * 2 = 24.
    
    rank = 24
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # For each column j of B (j = 0..11):
    # c[0][j] = a[0][0]*b[0][j] + a[0][1]*b[1][j]
    # c[1][j] = a[1][0]*b[0][j] + a[1][1]*b[1][j]
    
    # Term 1: a[0][0] * b[0][j] -> c[0][j]
    # Term 2: a[0][1] * b[1][j] -> c[0][j]
    # Term 3: a[1][0] * b[0][j] -> c[1][j]
    # Term 4: a[1][1] * b[1][j] -> c[1][j]
    
    for j in range(p):
        base_col = j * 2
        
        # c[0][j] = a[0][0]*b[0][j] + a[0][1]*b[1][j]
        col = base_col + 0
        U[0][col] = 1.0  # a[0][0]
        V[0 * 12 + j][col] = 1.0  # b[0][j]
        W[j * 2 + 0][col] = 1.0  # c[0][j]
        
        col = base_col + 1
        U[1][col] = 1.0  # a[0][1]
        V[1 * 12 + j][col] = 1.0  # b[1][j]
        W[j * 2 + 0][col] = 1.0  # c[0][j]
        
        # c[1][j] = a[1][0]*b[0][j] + a[1][1]*b[1][j]
        # Need 2 more terms
        # But we only allocated 2 per column. Let me fix this.
    
    # I need 4 terms per column, so rank = 4 * 12 = 48.
    # That's worse than 42. Let me just return the rank-42 Strassen result.
    
    rank = 42
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for t in range(6):
        base_col = t * 7
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # M1 = (a00+a11)(b[k0][0]+b[k1][1]) -> wait, B is 2x2 here
        # b00 = B[0][k0], b01 = B[0][k1], b10 = B[1][k0], b11 = B[1][k1]
        
        # M1 = (a00+a11)*(b00+b11)
        col = base_col
        U[0][col] += 1
        U[3][col] += 1
        V[0 * 12 + k0][col] += 1  # b00
        V[1 * 12 + k1][col] += 1  # b11
        W[k0 * 2 + 0][col] += 1   # c[0][k0]
        W[k1 * 2 + 1][col] += 1   # c[1][k1]
        
        # M2 = (a10+a11)*b00
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[0 * 12 + k0][col] += 1  # b00
        W[k0 * 2 + 1][col] += 1   # c[1][k0]
        W[k1 * 2 + 1][col] += -1  # c[1][k1]
        
        # M3 = a00*(b01-b11)
        col = base_col + 2
        U[0][col] += 1
        V[0 * 12 + k1][col] += 1  # b01
        V[1 * 12 + k1][col] += -1 # b11
        W[k0 * 2 + 0][col] += 1   # c[0][k0]
        W[k1 * 2 + 0][col] += 1   # c[0][k1]
        
        # M4 = a11*(b10-b00)
        col = base_col + 3
        U[3][col] += 1
        V[1 * 12 + k0][col] += 1  # b10
        V[0 * 12 + k0][col] += -1 # b00
        W[k0 * 2 + 0][col] += 1   # c[0][k0]
        W[k0 * 2 + 1][col] += 1   # c[1][k0]
        
        # M5 = (a00+a01)*b11
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[1 * 12 + k1][col] += 1  # b11
        W[k0 * 2 + 0][col] += -1  # c[0][k0]
        W[k0 * 2 + 1][col] += 1   # c[1][k0]
        
        # M6 = (a10-a00)*(b00+b01)
        col = base_col + 5
        U[2][col] += 1
        U[0][col] += -1
        V[0 * 12 + k0][col] += 1  # b00
        V[0 * 12 + k1][col] += 1  # b01
        W[k1 * 2 + 1][col] += 1   # c[1][k1]
        
        # M7 = (a01-a11)*(b10+b11)
        col = base_col + 6
        U[1][col] += 1
        U[3][col] += -1
        V[1 * 12 + k0][col] += 1  # b10
        V[1 * 12 + k1][col] += 1  # b11
        W[k0 * 2 + 1][col] += 1   # c[1][k0]
    
    return {"U": U, "V": V, "W": W}

