"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    # Transposed problem: M_10,3,2
    # A' is 10x3, B' is 3x2, C' is 10x2
    # U' shape (10*3, rank) = (30, rank)
    # V' shape (3*2, rank) = (6, rank)
    # W' shape (2*10, rank) = (20, rank)
    # We need to return U, V, W for original M_2,3,10
    # U (6, rank), V (30, rank), W (20, rank)
    # The decomposition of M_10,3,2 gives U', V', W'
    # Due to symmetry, we can map them back.
    # However, it's easier to just construct the original directly but using the logic of the transpose.
    
    n, m, p = 2, 3, 10
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)] # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)] # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)] # 20 x 55
    
    idx = 0
    
    # We use the same 55 rank decomposition but constructed via the transposed logic
    # Split m=3 into 2+1
    # Part 1: M_2,2,10 (Strassen on 5 blocks of 2x2x2)
    # Part 2: M_2,1,10 (Schoolbook)
    
    # Part 1: Strassen
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices (2x2 block from cols 0,1)
        a00 = 0
        a01 = 1
        a10 = 3
        a11 = 4
        
        # B indices (2x2 block from rows 0,1, cols k0,k1)
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices (2x2 block from rows k0,k1, cols 0,1)
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1

    # Part 2: Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
            
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
