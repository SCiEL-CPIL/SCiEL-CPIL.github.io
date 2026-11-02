DIMENSIONS = [2, 2, 12]

import time
import numpy as np

def construct():
    """Rank-42 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Uses the tensor product structure M_{2,2,12} with 6 independent 2x2x2 blocks.
    Each block uses Strassen's rank-7 algorithm. Total rank = 6 * 7 = 42.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j for A[i][j]
      V[j*p+k, r]: row = j*p+k for B[j][k]
      W[k*n+i, r]: row = k*n+i for C[i][k] (trace convention)
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    num_pairs = p // 2    # 6
    rank = num_pairs * 7  # 42
    
    U = np.zeros((n * m, rank), dtype=float)   # 4 x 42
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)   # 24 x 42
    
    # Strassen's 7 products for 2x2 * 2x2
    # A elements: (0,0)=A11, (0,1)=A12, (1,0)=A21, (1,1)=A22
    # B elements within block: (0,0)=B11, (0,1)=B12, (1,0)=B21, (1,1)=B22
    
    products = [
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)]),
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)]),
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)]),
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)]),
    ]
    
    output_contribs = [
        [((0,0), 1), ((1,1), 1)],
        [((1,0), 1), ((1,1), -1)],
        [((0,1), 1), ((1,1), 1)],
        [((0,0), 1), ((1,0), 1)],
        [((0,0), -1), ((0,1), 1)],
        [((1,1), 1)],
        [((0,0), 1)],
    ]
    
    col = 0
    for t in range(num_pairs):
        base_col = 2 * t
        for s in range(7):
            for (ai, aj), coeff in products[s][0]:
                U[ai * m + aj, col] = coeff
            
            for (bj, bk_rel), coeff in products[s][1]:
                V[bj * p + base_col + bk_rel, col] = coeff
            
            for (ci, ck_rel), coeff in output_contribs[s]:
                W[(base_col + ck_rel) * n + ci, col] = coeff
            
            col += 1
    
    # Convert to list of lists for the expected return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
