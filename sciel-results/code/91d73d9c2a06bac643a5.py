import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    Attempts rank-59 decomposition using verified rank-23 for 3x3x3 block.
    Falls back to schoolbook (rank 63) if verification fails.
    """
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Candidate rank-23 decomposition for 3x3x3 matrix multiplication
    # U index: i*3+j, V index: j*3+k, W index: k*3+i
    terms_333 = [
        ({0: 1}, {0: 1}, {0: 1}),
        ({1: 1}, {4: 1}, {3: 1}),
        ({2: 1}, {8: 1}, {6: 1}),
        ({3: 1}, {1: 1}, {1: 1}),
        ({4: 1}, {5: 1}, {4: 1}),
        ({5: 1}, {9: 1}, {7: 1}),
        ({6: 1}, {2: 1}, {2: 1}),
        ({0: 1, 1: 1}, {4: 1}, {1: 1}),
        ({1: 1, 0: -1}, {0: 1, 4: 1}, {0: 1}),
        ({0: 1, 3: 1}, {1: 1}, {3: 1}),
        ({3: 1, 4: 1}, {0: 1, 4: 1}, {0: 1}),
        ({4: 1, 7: 1}, {5: 1}, {7: 1}),
        ({7: 1, 8: 1}, {8: 1}, {8: 1}),
        ({6: 1, 7: 1}, {2: 1, 5: 1}, {7: 1}),
        ({7: 1, 6: -1}, {2: 1, 8: 1}, {6: 1}),
        ({8: 1, 7: 1}, {2: 1, 5: 1}, {7: 1}),
        ({0: 1, 4: 1}, {0: 1, 4: 1}, {0: 1}),
        ({1: 1, 4: 1}, {1: 1, 4: 1}, {1: 1}),
        ({3: 1, 7: 1}, {0: 1, 5: 1}, {3: 1}),
        ({4: 1, 8: 1}, {4: 1, 8: 1}, {4: 1}),
        ({6: 1, 8: 1}, {2: 1, 8: 1}, {6: 1}),
        ({0: 1, 6: 1}, {0: 1, 2: 1}, {0: 1}),
        ({1: 1, 7: 1}, {1: 1, 5: 1}, {1: 1}),
    ]
    
    def verify_333(terms):
        r = len(terms)
        U3 = np.zeros((9, r))
        V3 = np.zeros((9, r))
        W3 = np.zeros((9, r))
        
        for c, (ud, vd, wd) in enumerate(terms):
            for idx, coeff in ud.items():
                U3[idx, c] = coeff
            for idx, coeff in vd.items():
                V3[idx, c] = coeff
            for idx, coeff in wd.items():
                W3[idx, c] = coeff
        
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U3[i*3+j] * V3[j*3+k] * W3[k*3+i])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
    
    use_rank_23 = verify_333(terms_333)
    
    if use_rank_23:
        total_rank = 23 + 36
        U = [[0]*total_rank for _ in range(9)]
        V = [[0]*total_rank for _ in range(21)]
        W = [[0]*total_rank for _ in range(21)]
        
        for c, (ud, vd, wd) in enumerate(terms_333):
            for idx, coeff in ud.items():
                U[idx][c] = coeff
            for idx, coeff in vd.items():
                V[idx][c] = coeff
            for idx, coeff in wd.items():
                W[idx][c] = coeff
        
        col_offset = 23
        for i in range(3):
            for j in range(3):
                for k in range(4):
                    col = col_offset + i*3 + j*4 + k
                    U[i*3+j][col] = 1
                    V[j*3 + (k+3)][col] = 1
                    W[(k+3)*3 + i][col] = 1
        
        def verify_full(U, V, W, rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank))
                        if abs(val - 1) > 1e-9:
                            return False
            return True
        
        if verify_full(U, V, W, total_rank):
            return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook, rank 63
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

