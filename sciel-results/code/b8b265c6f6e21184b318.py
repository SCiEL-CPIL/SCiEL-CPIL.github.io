"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    terms = []
    
    def add_2x2_schoolbook(a_row, a_col, b_row, b_col, c_row, c_col):
        """
        Add 4 schoolbook terms for C[c_row:c_row+2, c_col:c_col+2] += 
        A[a_row:a_row+2, a_col:a_col+2] * B[b_row:b_row+2, b_col:b_col+2]
        
        A is n×m = 4×4, indexed A[i,j] -> U index i*m+j = i*4+j
        B is m×p = 4×5, indexed B[j,k] -> V index j*p+k = j*5+k
        C is n×p = 4×5, indexed C[i,k] -> W index k*n+i = k*4+i
        """
        for di in range(2):
            for dj in range(2):
                u = [0]*16
                v = [0]*20
                w = [0]*20
                # A[a_row+di, a_col+dj]
                u[(a_row+di)*4 + (a_col+dj)] = 1
                # B[b_row+dj, b_col+di]
                v[(b_row+dj)*5 + (b_col+di)] = 1
                # C[c_row+di, c_col+dj] -> W index = (c_col+dj)*4 + (c_row+di)
                w[(c_col+dj)*4 + (c_row+di)] = 1
                terms.append((u, v, w))
    
    # 4x4x4 part: 8 2x2x2 schoolbook decompositions
    add_2x2_schoolbook(0, 0, 0, 0, 0, 0)
    add_2x2_schoolbook(0, 2, 2, 0, 0, 0)
    add_2x2_schoolbook(0, 0, 0, 2, 0, 2)
    add_2x2_schoolbook(0, 2, 2, 2, 0, 2)
    add_2x2_schoolbook(2, 0, 0, 0, 2, 0)
    add_2x2_schoolbook(2, 2, 2, 0, 2, 0)
    add_2x2_schoolbook(2, 0, 0, 2, 2, 2)
    add_2x2_schoolbook(2, 2, 2, 2, 2, 2)
    
    # 4x4x1 part: C[i,4] = sum_j A[i,j]*B[j,4]
    for i in range(4):
        for j in range(4):
            u = [0]*16
            v = [0]*20
            w = [0]*20
            u[i*4+j] = 1
            v[j*5+4] = 1
            w[4*4+i] = 1  # k=4, so index = 4*4+i = 16+i
            terms.append((u, v, w))
    
    rank = len(terms)
    assert rank == 48, f"Expected 48 terms, got {rank}"
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    for col, (u, v, w) in enumerate(terms):
        for row in range(16):
            U[row][col] = u[row]
        for row in range(20):
            V[row][col] = v[row]
        for row in range(20):
            W[row][col] = w[row]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
