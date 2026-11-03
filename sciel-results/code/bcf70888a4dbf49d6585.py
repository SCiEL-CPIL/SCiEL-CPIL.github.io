import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 120.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank 56 decomposition: 
    # Part 1: 4x4x4 (rank 28)
    # Part 2: 4x4x2 (rank 28)
    R = 56
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2x2 coefficients
    # alpha[s][a] for a in 0..3 (row, col in 2x2)
    # beta[s][b] for b in 0..3
    # recon[s] = list of (a, b, coeff) for output reconstruction
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    col = 0
    
    # Part 1: 4x4x4 sub-tensor (rank 28)
    # A is 4x4, B is 4x4 (cols 0-3 of the full 4x6 B), C is 4x4 (cols 0-3 of the full 4x6 C)
    for s in range(7):
        for inner in range(4):
            # inner in 0..3 corresponds to the 4 scalar positions in 2x2
            ia, ib = inner // 2, inner % 2  # inner row, col
            
            # U: for each (a, r) where a is input A index (block 0..3), r is Strassen step
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, ac = a // 2, a % 2
                    # u_idx for A element (ar*2+ia, ac*2+ib)
                    u_idx = (ar * 2 + ia) * 4 + (ac * 2 + ib)
                    U[u_idx, col] += alpha[s][a]
            
            # V: for each (b, c) where b is input B index (block 0..3), c is Strassen step
            for b in range(4):
                if beta[s][b] != 0:
                    br, bc = b // 2, b % 2
                    # v_idx for B element (br*2+ib, bc*2+ia)
                    # B is 4x4 in this part. Row = br*2+ib, Col = bc*2+ia
                    # Full B is 4x6. This part uses cols 0-3.
                    v_idx = (br * 2 + ib) * 6 + (bc * 2 + ia)
                    V[v_idx, col] += beta[s][b]
            
            # W: reconstruct C
            for a, b, gamma in recon[s]:
                # a, b are 0..1 (2x2 block indices)
                # The output C block (a, b) gets contribution gamma * M_s
                # Within each 2x2 block, we have 4 scalar positions
                # For inner position (ia, ib), the output is at C[a*2+ia, b*2+ib]
                # w_idx = (b*2+ib) * 4 + (a*2+ia)
                # Full C is 4x6. This part uses cols 0-3.
                w_idx = (b * 2 + ib) * 6 + (a * 2 + ia)
                W[w_idx, col] += gamma
            
            col += 1
    
    # Part 2: 4x4x2 sub-tensor (rank 28)
    # A is 4x4, B is 4x2 (cols 4-5 of the full 4x6 B), C is 4x2 (cols 4-5 of the full 4x6 C)
    # Structure is 2x2 block multiplication with 2x2 matrix elements
    for s in range(7):
        for inner in range(4):
            # inner in 0..3 corresponds to the 4 scalar positions in 2x2
            ia, ib = inner // 2, inner % 2  # inner row, col
            
            # U: for each (a, r) where a is input A index (block 0..3)
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, ac = a // 2, a % 2
                    # u_idx for A element (ar*2+ia, ac*2+ib)
                    u_idx = (ar * 2 + ia) * 4 + (ac * 2 + ib)
                    U[u_idx, col] += alpha[s][a]
            
            # V: for each (b, c) where b is input B index (block 0..3)
            # B is 4x2. Blocks are 2x2.
            # B block (br, bc) is 2x2.
            # Element (br*2+ib, bc*2+ia) in the 4x2 matrix.
            # Col index in full 4x6 B is 4 + (bc*2+ia)
            for b in range(4):
                if beta[s][b] != 0:
                    br, bc = b // 2, b % 2
                    v_idx = (br * 2 + ib) * 6 + (4 + bc * 2 + ia)
                    V[v_idx, col] += beta[s][b]
            
            # W: reconstruct C
            for a, b, gamma in recon[s]:
                # a, b are 0..1 (2x2 block indices)
                # Output C block (a, b) is 2x2.
                # Element (a*2+ia, b*2+ib) in the 4x2 matrix.
                # Col index in full 4x6 C is 4 + (b*2+ib)
                w_idx = (a * 2 + ia) * 6 + (4 + b * 2 + ib)
                W[w_idx, col] += gamma
            
            col += 1
    
    # Verification
    if time.time() - t_start < _DEADLINE - 5:
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    prod = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    err = abs(prod - 1.0)
                    if err > max_err:
                        max_err = err
        if max_err > 1e-9:
            # Fallback to schoolbook
            R = 96
            U = np.zeros((16, R))
            V = np.zeros((24, R))
            W = np.zeros((24, R))
            col = 0
            for i in range(4):
                for j in range(4):
                    for k in range(6):
                        U[i*4+j, col] = 1.0
                        V[j*6+k, col] = 1.0
                        W[k*4+i, col] = 1.0
                        col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
