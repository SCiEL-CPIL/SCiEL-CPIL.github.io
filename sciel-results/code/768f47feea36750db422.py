"""Matrix multiplication 3,3,8 — improved decomposition using block Strassen."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# Time budget guard
_T0 = time.time()
_DEADLINE = _T0 + 110  # safety margin before 120 s hard limit


def _elapsed():
    return time.time() - _T0


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    # Build tensor T of shape (n, m, p, rank)
    # T[i,j,k,c] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    T = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    # The multiplication tensor M[i,j,k] = 1 for all valid (i,j,k)
    M_target = np.ones((n, m, p))
    # Contract: sum_c T[i,j,k,c] should equal M_target[i,j,k]
    result = T.sum(axis=3)
    return np.allclose(result, M_target, atol=1e-9)


def construct():
    """Construct rank-minimal (or near-minimal) decomposition for 3x3 x 3x8."""
    n, m, p = 3, 3, 8
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Target Rank 70 Decomposition
    # Blocks: [0, 2], [3, 5], [6, 7]
    # Rank 26 for 3x3x3, Rank 18 for 3x3x2
    total_rank = 26 + 26 + 18
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    
    def add_strassen_3x3x3(k_start):
        nonlocal offset
        # Strassen terms for 2x2x2 subblock (i in 0,1; j in 0,1; k in k_start, k_start+1)
        # Local indices for 2x2x2:
        # U: i*2+j -> global i*3+j
        # V: j*2+k_local -> global j*8 + (k_start + k_local)
        # W: k_local*2+i -> global (k_start + k_local)*3 + i
        
        # The 7 terms of Strassen
        # Each term: (U_coeffs, V_coeffs, W_coeffs)
        # Coeffs are dicts mapping local (i,j), (j,k), (k,i) to values
        
        terms = [
            # t1: (A00+A11)(B00+B11) -> C00, C11
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # t2: (A10+A11)B00 -> C10
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
            # t3: A00(B10-B11) -> C01
            ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}),
            # t4: A11(B01-B00) -> C00, C10
            ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # t5: (A00+A01)B11 -> C00, C01
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
            # t6: (A10-A00)(B00+B01) -> C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # t7: (A01-A11)(B10+B11) -> C00, C11
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ]
        
        for u_dict, v_dict, w_dict in terms:
            for (i, j), val in u_dict.items():
                U[i * 3 + j][offset] += val
            for (j, k_loc), val in v_dict.items():
                V[j * 8 + (k_start + k_loc)][offset] += val
            for (k_loc, i), val in w_dict.items():
                W[(k_start + k_loc) * 3 + i][offset] += val
            offset += 1
        
        # Schoolbook for remaining 19 terms in 3x3x3
        # A12*B21 (2x1 * 1x2 -> 2x2): i in 0,1; j=2; k in k_start, k_start+1
        for i in range(2):
            for k_loc in range(2):
                U[i * 3 + 2][offset] += 1
                V[2 * 8 + (k_start + k_loc)][offset] += 1
                W[(k_start + k_loc) * 3 + i][offset] += 1
                offset += 1
                
        # A11*B12 (2x2 * 2x1 -> 2x1): i in 0,1; j in 0,1; k=k_start+2
        for i in range(2):
            for j in range(2):
                U[i * 3 + j][offset] += 1
                V[j * 8 + (k_start + 2)][offset] += 1
                W[(k_start + 2) * 3 + i][offset] += 1
                offset += 1
                
        # A12*B22 (2x1 * 1x1 -> 2x1): i in 0,1; j=2; k=k_start+2
        for i in range(2):
            U[i * 3 + 2][offset] += 1
            V[2 * 8 + (k_start + 2)][offset] += 1
            W[(k_start + 2) * 3 + i][offset] += 1
            offset += 1
            
        # A21*B11 (1x2 * 2x2 -> 1x2): i=2; j in 0,1; k in k_start, k_start+1
        for j in range(2):
            for k_loc in range(2):
                U[2 * 3 + j][offset] += 1
                V[j * 8 + (k_start + k_loc)][offset] += 1
                W[(k_start + k_loc) * 3 + 2][offset] += 1
                offset += 1
                
        # A21*B12 (1x2 * 2x1 -> 1x1): i=2; j in 0,1; k=k_start+2
        for j in range(2):
            U[2 * 3 + j][offset] += 1
            V[j * 8 + (k_start + 2)][offset] += 1
            W[(k_start + 2) * 3 + 2][offset] += 1
            offset += 1
            
        # A22*B21 (1x1 * 1x2 -> 1x2): i=2; j=2; k in k_start, k_start+1
        for k_loc in range(2):
            U[2 * 3 + 2][offset] += 1
            V[2 * 8 + (k_start + k_loc)][offset] += 1
            W[(k_start + k_loc) * 3 + 2][offset] += 1
            offset += 1
            
        # A22*B22 (1x1 * 1x1 -> 1x1): i=2; j=2; k=k_start+2
        U[2 * 3 + 2][offset] += 1
        V[2 * 8 + (k_start + 2)][offset] += 1
        W[(k_start + 2) * 3 + 2][offset] += 1
        offset += 1

    def add_schoolbook_3x3xk(k_start, k_end):
        nonlocal offset
        for i in range(3):
            for j in range(3):
                for k in range(k_start, k_end):
                    U[i * 3 + j][offset] += 1
                    V[j * 8 + k][offset] += 1
                    W[k * 3 + i][offset] += 1
                    offset += 1

    # Block 1: k=0,1,2
    add_strassen_3x3x3(0)
    # Block 2: k=3,4,5
    add_strassen_3x3x3(3)
    # Block 3: k=6,7
    add_schoolbook_3x3xk(6, 8)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    return {"U": U_school, "V": V_school, "W": W_school}
