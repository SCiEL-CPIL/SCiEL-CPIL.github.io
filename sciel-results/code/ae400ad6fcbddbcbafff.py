"""Matrix multiplication 2x4 by 4x4 - rank-minimal tensor decomposition."""

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for multiplying a 2x4 matrix
    by a 4x4 matrix using Strassen's algorithm on four 2x2 blocks.
    
    U: (n*m, rank) = (8, 28)
    V: (m*p, rank) = (16, 28)
    W: (p*n, rank) = (8, 28)
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]   # 8 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 8 x 28
    
    # Strassen's 7 terms for 2x2 matrix multiplication
    # For A = [[a,b],[c,d]] and B = [[e,f],[g,h]], C = A*B:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)*e
    # M3 = a*(f-h)
    # M4 = d*(g-e)
    # M5 = (a+b)*h
    # M6 = (c-a)*(e+f)
    # M7 = (b-d)*(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # Each term: list of (index, coeff) for U, V, W
    # U indices: li*2 + lj (li in {0,1}, lj in {0,1} local)
    # V indices: lj*2 + lk
    # W indices: lk*2 + li
    
    strassen_terms = [
        # M1 = (a+d)(e+h)
        {
            'U': [(0, 1), (3, 1)],        # a at (0,0), d at (1,1)
            'V': [(0, 1), (3, 1)],        # e at (0,0), h at (1,1)
            'W': [(0, 1), (3, 1)]         # C[0,0] and C[1,1]
        },
        # M2 = (c+d)*e
        {
            'U': [(2, 1), (3, 1)],        # c at (1,0), d at (1,1)
            'V': [(0, 1)],                # e at (0,0)
            'W': [(2, 1), (3, -1)]        # C[1,0] += M2, C[1,1] -= M2
        },
        # M3 = a*(f-h)
        {
            'U': [(0, 1)],                # a at (0,0)
            'V': [(1, 1), (3, -1)],       # f at (0,1), h at (1,1)
            'W': [(1, 1), (3, 1)]         # C[0,1] += M3, C[1,1] += M3
        },
        # M4 = d*(g-e)
        {
            'U': [(3, 1)],                # d at (1,1)
            'V': [(2, 1), (0, -1)],       # g at (1,0), e at (0,0)
            'W': [(0, 1), (2, 1)]         # C[0,0] += M4, C[1,0] += M4
        },
        # M5 = (a+b)*h
        {
            'U': [(0, 1), (1, 1)],        # a at (0,0), b at (0,1)
            'V': [(3, 1)],                # h at (1,1)
            'W': [(0, -1), (1, 1)]        # C[0,0] -= M5, C[0,1] += M5
        },
        # M6 = (c-a)*(e+f)
        {
            'U': [(2, 1), (0, -1)],       # c at (1,0), a at (0,0)
            'V': [(0, 1), (1, 1)],        # e at (0,0), f at (0,1)
            'W': [(3, 1)]                 # C[1,1] += M6
        },
        # M7 = (b-d)*(g+h)
        {
            'U': [(1, 1), (3, -1)],       # b at (0,1), d at (1,1)
            'V': [(2, 1), (3, 1)],        # g at (1,0), h at (1,1)
            'W': [(0, 1)]                 # C[0,0] += M7
        }
    ]
    
    # Four 2x2 block multiplications
    # A is 2x4, split into A_left (cols 0,1) and A_right (cols 2,3)
    # B is 4x4, split into four 2x2 blocks
    # C = A_left * B_00 + A_left * B_01 + A_right * B_10 + A_right * B_11
    
    # Block definitions: (j_offset, k_offset)
    # j_offset: column offset in A (0 or 2)
    # k_offset: column offset in B (0 or 2)
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    
    col = 0
    for j_off, k_off in blocks:
        for term in strassen_terms:
            # Add to U
            for (local_idx, coeff) in term['U']:
                li = local_idx // 2
                lj = local_idx % 2
                global_i = li
                global_j = lj + j_off
                U[global_i * m + global_j][col] += coeff
            
            # Add to V
            for (local_idx, coeff) in term['V']:
                lj = local_idx // 2
                lk = local_idx % 2
                global_j = lj + j_off
                global_k = lk + k_off
                V[global_j * p + global_k][col] += coeff
            
            # Add to W
            for (local_idx, coeff) in term['W']:
                lk = local_idx // 2
                li = local_idx % 2
                global_k = lk + k_off
                global_i = li
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
