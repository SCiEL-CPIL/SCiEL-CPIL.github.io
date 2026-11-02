"""Matrix multiplication 2x2 x 2x13 with Strassen-based rank-46 decomposition."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # Strassen's 7 products for 2x2 x 2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a-b)(f+h)
    # M4 = (d)(g-h)
    # M5 = (a+c)(h)
    # M6 = (b-c)(e+g)
    # M7 = (b+d)(f-h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # For 2x2 A times 2x2 B_block (two columns of B):
    # We need U (4 x rank), V (4 x rank), W (4 x rank) for the 2x2x2 subproblem
    # Then embed into the full problem.
    
    # Full problem: A is 2x2 (indices 0,1 for rows, 0,1 for cols)
    # B is 2x13 (indices 0,1 for rows, 0..12 for cols)
    # C is 2x13 (indices 0,1 for rows, 0..12 for cols)
    
    # U has shape (n*m, rank) = (4, rank): indexed by [i*m+j] = [i*2+j]
    # V has shape (m*p, rank) = (26, rank): indexed by [j*p+k] = [j*13+k]
    # W has shape (p*n, rank) = (26, rank): indexed by [k*n+i] = [k*2+i]
    
    # Group columns: 0,1 | 2,3 | 4,5 | 6,7 | 8,9 | 10,11 | 12
    # 6 pairs + 1 single
    
    rank = 6 * 7 + 1 * 4  # 42 + 4 = 46
    
    U = [[0] * rank for _ in range(4)]   # 4 rows (2x2 A)
    V = [[0] * rank for _ in range(26)]  # 26 rows (2x13 B)
    W = [[0] * rank for _ in range(26)]  # 26 rows (2x13 C)
    
    col = 0
    
    # Strassen decomposition for 2x2 x 2x2 block
    # A indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1] -> U index: i*2+j
    #   a: U[0], b: U[1], c: U[2], d: U[3]
    # B block (2 columns, say cols k0,k1): 
    #   e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
    #   V index: j*13+k -> e: V[0*13+k0]=V[k0], f: V[0*13+k1]=V[k1]
    #   g: V[1*13+k0]=V[13+k0], h: V[1*13+k1]=V[13+k1]
    # C block (2 columns):
    #   C[0,k0]=M1+M4-M5+M7, C[0,k1]=M3+M5
    #   C[1,k0]=M2+M4, C[1,k1]=M1-M2+M3+M6
    #   W index: k*2+i -> C[0,k0]: W[0*2+0]=W[0], C[0,k1]: W[1*2+0]=W[2]
    #   C[1,k0]: W[0*2+1]=W[1], C[1,k1]: W[1*2+1]=W[3]
    
    def strassen_block(k0, k1, col_start):
        """Add Strassen decomposition for columns k0,k1. Returns next col index."""
        nonlocal col
        c = col_start
        
        # M1 = (a+d)(e+h): U[a]+U[d] times V[e]+V[h]
        # a=U[0], d=U[3], e=V[k0], h=V[13+k1]
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[13 + k1][c] = 1
        # C[0,k0] += M1, C[1,k1] += M1
        W[k0 * 2 + 0][c] = 1   # C[0,k0]
        W[k1 * 2 + 1][c] = 1   # C[1,k1]
        col = c + 1
        
        # M2 = (c+d)(e): U[c]+U[d] times V[e]
        # c=U[2], d=U[3], e=V[k0]
        U[2][col] = 1
        U[3][col] = 1
        V[k0][col] = 1
        # C[1,k0] += M2, C[1,k1] -= M2
        W[k0 * 2 + 1][col] = 1   # C[1,k0]
        W[k1 * 2 + 1][col] = -1  # C[1,k1]
        col += 1
        
        # M3 = (a-b)(f+h): U[a]-U[b] times V[f]+V[h]
        # a=U[0], b=U[1], f=V[k1], h=V[13+k1]
        U[0][col] = 1
        U[1][col] = -1
        V[k1][col] = 1
        V[13 + k1][col] = 1
        # C[0,k1] += M3, C[1,k1] += M3
        W[k1 * 2 + 0][col] = 1   # C[0,k1]
        W[k1 * 2 + 1][col] = 1   # C[1,k1]
        col += 1
        
        # M4 = (d)(g-h): U[d] times V[g]-V[h]
        # d=U[3], g=V[13+k0], h=V[13+k1]
        U[3][col] = 1
        V[13 + k0][col] = 1
        V[13 + k1][col] = -1
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0 * 2 + 0][col] = 1   # C[0,k0]
        W[k0 * 2 + 1][col] = 1   # C[1,k0]
        col += 1
        
        # M5 = (a+c)(h): U[a]+U[c] times V[h]
        # a=U[0], c=U[2], h=V[13+k1]
        U[0][col] = 1
        U[2][col] = 1
        V[13 + k1][col] = 1
        # C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * 2 + 0][col] = -1  # C[0,k0]
        W[k1 * 2 + 0][col] = 1   # C[0,k1]
        col += 1
        
        # M6 = (b-c)(e+g): U[b]-U[c] times V[e]+V[g]
        # b=U[1], c=U[2], e=V[k0], g=V[13+k0]
        U[1][col] = 1
        U[2][col] = -1
        V[k0][col] = 1
        V[13 + k0][col] = 1
        # C[1,k1] += M6
        W[k1 * 2 + 1][col] = 1   # C[1,k1]
        col += 1
        
        # M7 = (b+d)(f-h): U[b]+U[d] times V[f]-V[h]
        # b=U[1], d=U[3], f=V[k1], h=V[13+k1]
        U[1][col] = 1
        U[3][col] = 1
        V[k1][col] = 1
        V[13 + k1][col] = -1
        # C[0,k0] += M7
        W[k0 * 2 + 0][col] = 1   # C[0,k0]
        col += 1
        
        return col
    
    # Process 6 pairs of columns
    for block in range(6):
        k0 = 2 * block
        k1 = 2 * block + 1
        col = strassen_block(k0, k1, col)
    
    # Process single column 12 with schoolbook (4 terms)
    # C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    # C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    
    # Term 1: A[0,0] * B[0,12] -> C[0,12]
    U[0][col] = 1
    V[0 * 13 + 12][col] = 1  # V[12]
    W[12 * 2 + 0][col] = 1   # W[24]
    col += 1
    
    # Term 2: A[0,1] * B[1,12] -> C[0,12]
    U[1][col] = 1
    V[1 * 13 + 12][col] = 1  # V[25]
    W[12 * 2 + 0][col] = 1   # W[24]
    col += 1
    
    # Term 3: A[1,0] * B[0,12] -> C[1,12]
    U[2][col] = 1
    V[0 * 13 + 12][col] = 1  # V[12]
    W[12 * 2 + 1][col] = 1   # W[25]
    col += 1
    
    # Term 4: A[1,1] * B[1,12] -> C[1,12]
    U[3][col] = 1
    V[1 * 13 + 12][col] = 1  # V[25]
    W[12 * 2 + 1][col] = 1   # W[25]
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
