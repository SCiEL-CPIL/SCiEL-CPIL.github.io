import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    steps = [
        {
            'x': [(0, 1), (3, 1)],
            'y': [(0, 1), (3, 1)],
            'w': [(0, 1), (3, 1)]
        },
        {
            'x': [(2, 1), (3, 1)],
            'y': [(0, 1)],
            'w': [(2, 1), (3, -1)]
        },
        {
            'x': [(0, 1)],
            'y': [(1, 1), (3, -1)],
            'w': [(1, 1), (3, 1)]
        },
        {
            'x': [(3, 1)],
            'y': [(2, 1), (0, -1)],
            'w': [(0, 1), (2, 1)]
        },
        {
            'x': [(0, 1), (1, 1)],
            'y': [(3, 1)],
            'w': [(0, -1), (1, 1)]
        },
        {
            'x': [(2, 1), (0, -1)],
            'y': [(0, 1), (1, 1)],
            'w': [(3, 1)]
        },
        {
            'x': [(1, 1), (3, -1)],
            'y': [(2, 1), (3, 1)],
            'w': [(0, 1)]
        }
    ]
    
    a_offsets = [(0, 0), (0, 2), (2, 0), (2, 2)]
    b_offsets = [(0, 0), (0, 3), (2, 0), (2, 3)]
    c_offsets = [(0, 0), (0, 3), (2, 0), (2, 3)]
    
    for s_idx, step in enumerate(steps):
        base_col_offset = s_idx * 12
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    col = base_col_offset + (i * 4 + j * 2 + k)
                    
                    # U: (I+i, J+j) -> (I+i)*m + (J+j)
                    for b_idx, coeff in step['x']:
                        I, J = a_offsets[b_idx]
                        u_idx = (I + i) * m + (J + j)
                        U[u_idx][col] += coeff
                        
                    # V: (J+j, K+k) -> (J+j)*p + (K+k)
                    for b_idx, coeff in step['y']:
                        J, K = b_offsets[b_idx]
                        v_idx = (J + j) * p + (K + k)
                        V[v_idx][col] += coeff
                        
                    # W: (K+k, I+i) -> (K+k)*n + (I+i)
                    for b_idx, coeff in step['w']:
                        I, K = c_offsets[b_idx]
                        w_idx = (K + k) * n + (I + i)
                        W[w_idx][col] += coeff
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
