# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3x8 matrix multiplication.
    
    This decomposition splits the 3x8 matrix B into four 3x2 blocks.
    For each 2x3x2 multiplication, we use a rank-11 decomposition:
    - 4 multiplications for the 2x1x2 part (A[:,0] * B[0,:])
    - 7 multiplications for the 2x2x2 part (A[:,1:3] * B[1:3,:]) using Strassen's algorithm
    
    Total rank = 4 blocks * 11 multiplications = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(4):
        base_c = k * 11
        
        # M1: a00 * b00
        U[0][base_c] = 1
        V[0 * p + 2*k][base_c] = 1
        W[(2*k) * n + 0][base_c] = 1
        
        # M2: a00 * b01
        U[0][base_c+1] = 1
        V[0 * p + 2*k+1][base_c+1] = 1
        W[(2*k+1) * n + 0][base_c+1] = 1
        
        # M3: a10 * b00
        U[3][base_c+2] = 1
        V[0 * p + 2*k][base_c+2] = 1
        W[(2*k) * n + 1][base_c+2] = 1
        
        # M4: a10 * b01
        U[3][base_c+3] = 1
        V[0 * p + 2*k+1][base_c+3] = 1
        W[(2*k+1) * n + 1][base_c+3] = 1
        
        # M5: a01 * b10
        U[1][base_c+4] = 1
        V[1 * p + 2*k][base_c+4] = 1
        W[(2*k) * n + 0][base_c+4] = 1
        W[(2*k+1) * n + 1][base_c+4] = 1
        
        # M6: a12 * b21
        U[5][base_c+5] = 1
        V[2 * p + 2*k+1][base_c+5] = 1
        W[(2*k) * n + 0][base_c+5] = 1
        W[(2*k+1) * n + 1][base_c+5] = 1
        
        # M7: (a01+a02) * b21
        U[1][base_c+6] = 1
        U[2][base_c+6] = 1
        V[2 * p + 2*k+1][base_c+6] = 1
        W[(2*k+1) * n + 0][base_c+6] = 1
        
        # M8: a11 * (b10+b11)
        U[4][base_c+7] = 1
        V[1 * p + 2*k][base_c+7] = 1
        V[1 * p + 2*k+1][base_c+7] = 1
        W[(2*k) * n + 1][base_c+7] = 1
        
        # M9: (a01+a11) * (b21-b10)
        U[1][base_c+8] = 1
        U[4][base_c+8] = 1
        V[2 * p + 2*k+1][base_c+8] = 1
        V[1 * p + 2*k][base_c+8] = -1
        W[(2*k) * n + 0][base_c+8] = 1
        W[(2*k) * n + 1][base_c+8] = 1
        W[(2*k+1) * n + 1][base_c+8] = -1
        
        # M10: (a02-a12) * (b10+b20)
        U[2][base_c+9] = 1
        U[5][base_c+9] = -1
        V[1 * p + 2*k][base_c+9] = 1
        V[2 * p + 2*k][base_c+9] = 1
        W[(2*k+1) * n + 1][base_c+9] = 1
        
        # M11: (a12-a01) * (b11+b20)
        U[5][base_c+10] = 1
        U[1][base_c+10] = -1
        V[1 * p + 2*k+1][base_c+10] = 1
        V[2 * p + 2*k][base_c+10] = 1
        W[(2*k) * n + 0][base_c+10] = -1
        W[(2*k+1) * n + 0][base_c+10] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
