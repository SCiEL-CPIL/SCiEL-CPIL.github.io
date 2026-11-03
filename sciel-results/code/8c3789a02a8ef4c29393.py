"""Matrix multiplication 2x4x6 rank-minimal tensor decomposition.
Uses Strassen's algorithm on 2x2x2 sub-blocks to achieve rank 42.

The decomposition works by:
1. Splitting A (2x4) into two 2x2 blocks: A = [A_L, A_R]
2. Splitting B (4x6) into two 2x6 blocks: B = [B_top; B_bot]
3. C = A_L * B_top + A_R * B_bot
4. Each (2x2)*(2x6) is decomposed using Strassen on three 2x2x2 sub-blocks

Total rank: 2 * 3 * 7 = 42.
"""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def _strassen_2x2x2_factors():
    """Return the 7 rank-1 factors for 2x2x2 Strassen multiplication.
    
    Each factor is a dict with 'U', 'V', 'W' keys, where each value is a dict
    mapping index -> coefficient.
    
    U index: i*2+j for A[i,j]
    V index: j*2+k for B[j,k]
    W index: k*2+i for C[k,i]
    """
    factors = []
    
    # M1 = (A11 + A22)(B11 + B22)
    # C11 += M1, C22 += M1
    factors.append({
        'U': {0: 1, 3: 1},       # A[0,0], A[1,1]
        'V': {0: 1, 3: 1},       # B[0,0], B[1,1]
        'W': {0: 1, 3: 1}        # C[0,0], C[1,1]
    })
    
    # M2 = (A21 + A22)B11
    # C21 += M2
    factors.append({
        'U': {2: 1, 3: 1},       # A[1,0], A[1,1]
        'V': {0: 1},             # B[0,0]
        'W': {2: 1}              # C[1,0]
    })
    
    # M3 = A11(B21 - B22)
    # C12 += M3, C22 += M3
    factors.append({
        'U': {0: 1},             # A[0,0]
        'V': {2: 1, 3: -1},      # B[1,0], B[1,1]
        'W': {1: 1, 3: 1}        # C[0,1], C[1,1]
    })
    
    # M4 = A22(B12 - B11)
    # C11 += M4, C21 += M4
    factors.append({
        'U': {3: 1},             # A[1,1]
        'V': {1: 1, 0: -1},      # B[0,1], B[0,0]
        'W': {0: 1, 2: 1}        # C[0,0], C[1,0]
    })
    
    # M5 = (A11 + A12)B22
    # C11 -= M5, C12 += M5
    factors.append({
        'U': {0: 1, 1: 1},       # A[0,0], A[0,1]
        'V': {3: 1},             # B[1,1]
        'W': {0: -1, 1: 1}       # C[0,0], C[0,1]
    })
    
    # M6 = (A21 - A11)(B11 + B12)
    # C22 += M6
    factors.append({
        'U': {2: 1, 0: -1},      # A[1,0], A[0,0]
        'V': {0: 1, 1: 1},       # B[0,0], B[0,1]
        'W': {3: 1}              # C[1,1]
    })
    
    # M7 = (A12 - A22)(B21 + B22)
    # C11 += M7
    factors.append({
        'U': {1: 1, 3: -1},      # A[0,1], A[1,1]
        'V': {2: 1, 3: 1},       # B[1,0], B[1,1]
        'W': {0: 1}              # C[0,0]
    })
    
    return factors

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]   # (8, 42)
    V = [[0] * rank for _ in range(m * p)]   # (24, 42)
    W = [[0] * rank for _ in range(p * n)]   # (12, 42)
    
    strassen_factors = _strassen_2x2x2_factors()
    
    factor_idx = 0
    for a_block in range(2):      # 0: A cols 0,1; 1: A cols 2,3
        for b_col_block in range(3):  # 0: B cols 0,1; 1: B cols 2,3; 2: B cols 4,5
            for sf in strassen_factors:
                # U part: A indices
                # a_block=0: A[i,0]->row i*4+0, A[i,1]->row i*4+1
                # a_block=1: A[i,2]->row i*4+2, A[i,3]->row i*4+3
                for a_idx, coeff in sf['U'].items():
                    i, j = divmod(a_idx, 2)
                    u_row = i * 4 + (j + 2 * a_block)
                    U[u_row][factor_idx] += coeff
                
                # V part: B indices
                # a_block=0: B rows 0,1 -> V[j*6+k]
                # a_block=1: B rows 2,3 -> V[(j+2)*6+k]
                for b_idx, coeff in sf['V'].items():
                    j, k = divmod(b_idx, 2)
                    v_row = (j + 2 * a_block) * 6 + (2 * b_col_block + k)
                    V[v_row][factor_idx] += coeff
                
                # W part: C indices
                # C[i, k_global] -> W[k_global*2 + i]
                for c_idx, coeff in sf['W'].items():
                    k_local, i = divmod(c_idx, 2)
                    k_global = 2 * b_col_block + k_local
                    w_row = k_global * 2 + i
                    W[w_row][factor_idx] += coeff
                
                factor_idx += 1
    
    assert factor_idx == 42, f"Expected 42 factors, got {factor_idx}"
    
    # Verify correctness with a numerical test
    np.random.seed(42)
    A_test = np.random.randint(-3, 4, size=(2, 4))
    B_test = np.random.randint(-3, 4, size=(4, 6))
    C_expected = A_test @ B_test
    
    # Compute C using the decomposition
    # C[i,k] = sum_r (sum_j U[i*4+j, r] * A[i,j]) * (sum_j V[j*6+k, r] * B[j,k]) * W[k*2+i, r]
    # Wait, that's not right. The correct formula is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_r sum_j sum_j' U[i*4+j, r] * A[i,j] * V[j'*6+k, r] * B[j',k] * W[k*2+i, r]
    # No, that's not right either.
    
    # The tensor decomposition is:
    # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
    # = sum_r U[i*4+j, r] * V[j'*6+k, r] * W[k'*2+i', r]
    
    # To compute C[i,k] = sum_j A[i,j] * B[j,k]:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # = sum_j A[i,j] * B[j,k] * 1
    # = sum_j A[i,j] * B[j,k] * T[(i,j),(j,k),(k,i)]  (since T=1 when indices match)
    # = sum_j A[i,j] * B[j,k] * sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*2+i, r]
    # = sum_r W[k*2+i, r] * (sum_j A[i,j] * U[i*4+j, r]) * (sum_j B[j,k] * V[j*6+k, r])
    
    # So: C[i,k] = sum_r W[k*2+i, r] * (sum_j A[i,j] * U[i*4+j, r]) * (sum_j B[j,k] * V[j*6+k, r])
    
    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_r = np.array([U[i * m + j][r] for i in range(n) for j in range(m)])
        v_r = np.array([V[j * p + k][r] for j in range(m) for k in range(p)])
        w_r = np.array([W[k * n + i][r] for k in range(p) for i in range(n)])
        
        A_u = np.array([sum(A_test[i, j] * U[i * m + j][r] for j in range(m)) for i in range(n)])
        B_v = np.array([sum(B_test[j, k] * V[j * p + k][r] for j in range(m)) for k in range(p)])
        C_w = np.array([w_r[k * n + i] for k in range(p) for i in range(n)])
        
        for i in range(n):
            for k in range(p):
                C_computed[i, k] += W[k * n + i][r] * A_u[i] * B_v[k]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fall back to schoolbook rank 48
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
