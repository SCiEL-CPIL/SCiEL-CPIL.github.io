"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen's 7 terms as (u_indices, v_indices, w_indices)
    # Indices are flattened: A[i,k] -> i*2+k, B[j,l] -> j*2+l, C[i,l] -> i*2+l
    # Term 1: (A00+A11)(B00+B11) -> C00, C11
    # A: 0,3; B: 0,3; C: 0,3
    terms = [
        {
            'u': [(0, 1), (3, 1)],
            'v': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        {
            'u': [(2, 1), (3, 1)],
            'v': [(0, 1)],
            'w': [(1, 1), (3, -1)]
        },
        {
            'u': [(0, 1)],
            'v': [(1, 1), (3, -1)],
            'w': [(2, 1), (3, 1)]
        },
        {
            'u': [(3, 1)],
            'v': [(2, 1), (0, -1)],
            'w': [(0, 1), (1, 1)]
        },
        {
            'u': [(0, 1), (1, 1)],
            'v': [(3, 1)],
            'w': [(0, -1), (2, 1)]
        },
        {
            'u': [(2, 1), (0, -1)],
            'v': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        {
            'u': [(1, 1), (3, -1)],
            'v': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    for r, term in enumerate(terms):
        for idx, val in term['u']:
            U[idx][r] = val
        for idx, val in term['v']:
            V[idx][r] = val
        for idx, val in term['w']:
            W[idx][r] = val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
