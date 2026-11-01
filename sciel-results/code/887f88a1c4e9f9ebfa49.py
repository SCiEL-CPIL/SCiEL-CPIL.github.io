import time

def construct():
    n, m, p = 2, 2, 5
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for B columns k_offset, k_offset+1
    def add_strassen_block(k_offset):
        nonlocal col
        # P1 = (a00+a11)(b00+b11) -> c00:1, c11:1
        U[0][col] = 1; U[3][col] = 1
        V[k_offset][col] = 1; V[5+k_offset+1][col] = 1 # b_j, k_offset+j? No.
        # V index is j*p + k.
        # b_0, k_offset -> 0*5 + k_offset
        # b_1, k_offset+1 -> 1*5 + (k_offset+1)
        V[0*5 + k_offset][col] = 1
        V[1*5 + (k_offset+1)][col] = 1
        # W index is k*n + i.
        # c_0, k_offset -> k_offset*2 + 0
        # c_1, k_offset+1 -> (k_offset+1)*2 + 1
        W[k_offset*2 + 0][col] = 1
        W[(k_offset+1)*2 + 1][col] = 1
        col += 1
        
        # P2 = (a10+a11)b00 -> c10:1, c11:-1
        U[2][col] = 1; U[3][col] = 1
        V[0*5 + k_offset][col] = 1
        W[k_offset*2 + 1][col] = 1 # c1, k_offset
        W[(k_offset+1)*2 + 1][col] = -1 # c1, k_offset+1
        col += 1
        
        # P3 = a00(b01-b11) -> c01:1, c11:1
        U[0][col] = 1
        V[0*5 + (k_offset+1)][col] = 1
        V[1*5 + (k_offset+1)][col] = -1
        W[(k_offset+1)*2 + 0][col] = 1 # c0, k_offset+1
        W[(k_offset+1)*2 + 1][col] = 1 # c1, k_offset+1
        col += 1
        
        # P4 = a11(b10-b00) -> c00:1, c10:1
        U[3][col] = 1
        V[1*5 + k_offset][col] = 1
        V[0*5 + k_offset][col] = -1
        W[k_offset*2 + 0][col] = 1 # c0, k_offset
        W[k_offset*2 + 1][col] = 1 # c1, k_offset
        col += 1
        
        # P5 = (a00+a01)b11 -> c00:-1, c01:1
        U[0][col] = 1; U[1][col] = 1
        V[1*5 + (k_offset+1)][col] = 1
        W[k_offset*2 + 0][col] = -1
        W[(k_offset+1)*2 + 0][col] = 1
        col += 1
        
        # P6 = (a10-a00)(b00+b01) -> c11:1
        U[2][col] = 1; U[0][col] = -1
        V[0*5 + k_offset][col] = 1
        V[0*5 + (k_offset+1)][col] = 1
        W[(k_offset+1)*2 + 1][col] = 1
        col += 1
        
        # P7 = (a01-a11)(b10+b11) -> c00:1
        U[1][col] = 1; U[3][col] = -1
        V[1*5 + k_offset][col] = 1
        V[1*5 + (k_offset+1)][col] = 1
        W[k_offset*2 + 0][col] = 1
        col += 1

    add_strassen_block(0)
    add_strassen_block(2)
    
    # Schoolbook for k=4
    # c04 = a00 b04 + a01 b14
    # c14 = a10 b04 + a11 b14
    
    # a00 b04 -> c04
    U[0][col] = 1
    V[0*5 + 4][col] = 1
    W[4*2 + 0][col] = 1
    col += 1
    
    # a01 b14 -> c04
    U[1][col] = 1
    V[1*5 + 4][col] = 1
    W[4*2 + 0][col] = 1
    col += 1
    
    # a10 b04 -> c14
    U[2][col] = 1
    V[0*5 + 4][col] = 1
    W[4*2 + 1][col] = 1
    col += 1
    
    # a11 b14 -> c14
    U[3][col] = 1
    V[1*5 + 4][col] = 1
    W[4*2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
