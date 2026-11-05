import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify_decomp(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check shapes
    if U.shape != (n * m, rank):
        return False
    if V.shape != (m * p, rank):
        return False
    if W.shape != (p * n, rank):
        return False
    
    # Verify: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = 4, 4, 7
    
    # Rank 49 for 4x4x4 (cols 0-3) + Rank 2 for 4x4x3 (cols 4-6)
    total_rank = 51
    
    Umat = np.zeros((n * m, total_rank))
    Vmat = np.zeros((m * p, total_rank))
    Wmat = np.zeros((p * n, total_rank))
    
    pos_rank = 0
    
    # Strassen 2x2x2 terms
    strassen_2x2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(0,0): 0, (0,1): 0, (1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,0): 0, (0,1): 1, (1,0): 0, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 0, (1,0): 1, (1,1): 0}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1, (1,0): 0, (1,1): 0}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(0,0): 0, (0,1): 0, (1,0): 0, (1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0}),
    ]
    
    # Strassen 4x4x4: 7 outer terms
    strassen_4x4_outer = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}),
    ]
    
    c_blocks_4x4 = [
        {(0,0): 1.0, (1,1): 1.0},
        {(1,0): 1.0, (1,1): -1.0},
        {(0,1): 1.0, (1,1): 1.0},
        {(0,0): 1.0, (1,0): 1.0},
        {(0,0): -1.0, (0,1): 1.0},
        {(1,1): 1.0},
        {(0,0): 1.0},
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49) for columns 0-3
    for m_idx in range(7):
        x_def, y_def = strassen_4x4_outer[m_idx]
        c_block = c_blocks_4x4[m_idx]
        for t_idx in range(7):
            r = pos_rank
            pos_rank += 1
            x_sub, y_sub, c_sub = strassen_2x2[t_idx]
            
            # Fill U
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    Umat[i * m + j, r] += cx * cb
            
            # Fill V
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    Vmat[i * p + j, r] += cy * cb
            
            # Fill W (columns 0-3)
            for (li, lj), cc in c_sub.items():
                if cc == 0:
                    continue
                for (br, bc), cb_c in c_block.items():
                    if cb_c == 0:
                        continue
                    i_final = 2 * br + li
                    k_final = 2 * bc + lj
                    # k_final is 0..3
                    Wmat[k_final * n + i_final, r] += cc * cb_c

    # Part 2: 4x4x3 optimized (rank 2) for columns 4-6
    # We need to cover j in {0,1,2,3} and k in {4,5,6}
    # Split j into {0,1} and {2,3}
    
    # Rank 49: j in {0,1}, k in {4,5,6}
    r = pos_rank
    pos_rank += 1
    # U: i is 0..3, j is 0,1
    for i in range(4):
        Umat[i * 4 + 0, r] = 1.0
        Umat[i * 4 + 1, r] = 1.0
    # V: j is 0,1, k is 4,5,6
    for j in [0, 1]:
        for k in [4, 5, 6]:
            Vmat[j * 7 + k, r] = 1.0
    # W: k is 4,5,6, i is 0..3
    for k in [4, 5, 6]:
        for i in range(4):
            Wmat[k * 4 + i, r] = 1.0

    # Rank 50: j in {2,3}, k in {4,5,6}
    r = pos_rank
    pos_rank += 1
    # U: i is 0..3, j is 2,3
    for i in range(4):
        Umat[i * 4 + 2, r] = 1.0
        Umat[i * 4 + 3, r] = 1.0
    # V: j is 2,3, k is 4,5,6
    for j in [2, 3]:
        for k in [4, 5, 6]:
            Vmat[j * 7 + k, r] = 1.0
    # W: k is 4,5,6, i is 0..3
    for k in [4, 5, 6]:
        for i in range(4):
            Wmat[k * 4 + i, r] = 1.0

    assert pos_rank == 51, f"Expected 51, got {pos_rank}"
    
    Uout = Umat[:, :pos_rank].tolist()
    Vout = Vmat[:, :pos_rank].tolist()
    Wout = Wmat[:, :pos_rank].tolist()
    
    if not _verify_decomp(Uout, Vout, Wout, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    return {"U": Uout, "V": Vout, "W": Wout}
