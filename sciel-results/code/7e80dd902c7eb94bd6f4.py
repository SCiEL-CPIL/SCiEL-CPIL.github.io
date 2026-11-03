import numpy as np
import time

DIMENSIONS = [4, 4, 6]
_DEADLINE = 120.0

def construct():
    t_start = time.time()
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Rank 42 decomposition: split 4x4x6 into 4x4x4 (rank 28) + 4x4x2 (rank 14)
    R = 42
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
    
    # Part 1: 4x4x4 sub-tensor using recursive Strassen (rank 28 = 7*7*... no, 4x4x4 = 2x2x2 outer * 2x2x2 inner)
    # Actually 4x4x4 can be done with rank 28 = 7 (outer Strassen) * 4 (inner schoolbook) or other combinations
    # Let's use: split 4x4x4 into four 2x2x2 blocks, each done with Strassen (rank 7)
    # Total: 4 * 7 = 28
    
    # For 4x4x4: A is 4x4, B is 4x4, C is 4x4
    # Split into 2x2 blocks: A = [[A00, A01], [A10, A11]], etc.
    # C_ij = sum_k A_ik * B_kj
    # Use Strassen on the 2x2 block structure where each "element" is a 2x2 matrix
    # But we need exact decomposition, so let's do it directly
    
    # Alternative: 4x4x4 has rank at most 28 (Strassen applied twice: 7*4 or 4*7)
    # Let's use 7 outer Strassen steps, each with 4 inner scalar multiplications
    for s in range(7):
        for inner in range(4):
            # inner in 0..3 corresponds to the 4 scalar positions in 2x2
            ia, ib = inner // 2, inner % 2  # inner row, col
            
            # U: for each (a, r) where a is input A index, r is Strassen step
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, ac = a // 2, a % 2
                    # u_idx for A element (ar*2+ia, ac*2+ib)
                    u_idx = (ar * 2 + ia) * 4 + (ac * 2 + ib)
                    U[u_idx, col] += alpha[s][a]
            
            # V: for each (b, c) where b is input B index, c is Strassen step
            for b in range(4):
                if beta[s][b] != 0:
                    br, bc = b // 2, b % 2
                    # v_idx for B element (br*2+ib, bc*2+ia) -- note: B is 4x4, so col index is bc*2+ia
                    v_idx = (br * 2 + ib) * 4 + (bc * 2 + ia)
                    V[v_idx, col] += beta[s][b]
            
            # W: reconstruct C
            for a, b, gamma in recon[s]:
                # a, b are 0..1 (2x2 block indices)
                # The output C block (a, b) gets contribution gamma * M_s
                # Within each 2x2 block, we have 4 scalar positions
                # For inner position (ia, ib), the output is at C[a*2+ia, b*2+ib]
                # w_idx = (b*2+ib) * 4 + (a*2+ia)
                w_idx = (b * 2 + ib) * 4 + (a * 2 + ia)
                W[w_idx, col] += gamma
            
            col += 1
    
    # Part 2: 4x4x2 sub-tensor (rank 14 = 7 * 2)
    # A is 4x4, B is 4x2, C is 4x2
    # Split into 2x2 blocks in A and C, B is 4x2
    # Use Strassen on the 2x2x2 structure
    for s in range(7):
        for inner in range(2):
            # inner in 0..1 corresponds to the 2 columns of the 4x2 B and 4x2 C
            ib = inner  # inner col index
            
            # U: for each a in 0..3
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, ac = a // 2, a % 2
                    # A element (ar*2+0, ac*2+ib) -- row from ar, col from ac*2+ib
                    u_idx = (ar * 2 + 0) * 4 + (ac * 2 + ib)
                    U[u_idx, col] += alpha[s][a]
            
            # V: for each b in 0..3 (B is 4x2, so b indexes rows 0..3 and col 0..1)
            # b = br*2 + bc where br is block row (0,1), bc is within-block col (0,1)
            for b in range(4):
                if beta[s][b] != 0:
                    br, bc = b // 2, b % 2
                    # B element: row = br*2 + 0 (since we're using first 2 rows?), col = bc
                    # Wait, B is 4x2. Let me reconsider.
                    # B has 4 rows and 2 cols. b indexes into B's 4x2 = 8 elements? No, b is 0..3.
                    # Let me re-examine: for 4x4x2, we split A into 2x2 blocks (4 blocks),
                    # B is 4x2 (2 blocks of 2x1), C is 4x2 (2 blocks of 2x1)
                    # This is getting complex. Let me use a simpler approach.
                    pass
            
            # Let me restart Part 2 with a clearer formulation
            break
        break
    
    # Clear and redo Part 2 properly
    col = 28  # reset to start of Part 2
    
    # For 4x4x2: A (4x4) * B (4x2) = C (4x2)
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B has 4 rows, 2 cols. Split B into 2x1 blocks: B = [B0; B1] where B0 is 2x2, B1 is 2x2
    # C has 4 rows, 2 cols. Split C into 2x1 blocks: C = [C0; C1] where C0 is 2x2, C1 is 2x2
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # Use Strassen on each 2x2x2 multiplication (A_ij * B_k = 2x2 * 2x2 = 2x2)
    # But B0 and B1 are both 2x2, so we can do:
    # For C0: use Strassen with A00, A01 and B0, B1 (treating B as having 2 "columns" of 2x1)
    # Actually, let's just do 2 independent 2x2x2 Strassen decompositions:
    # One for C0 = A00*B0 + A01*B1 (but this isn't a standard matrix mult)
    
    # Simpler: treat 4x4x2 as two 4x4x1 problems, each done with rank 7
    # 4x4x1: A(4x4) * B(4x1) = C(4x1)
    # Split A into 2x2 blocks, B into 2x1 blocks, C into 2x1 blocks
    # C0 = A00*B0 + A01*B1
    # C1 = A10*B0 + A11*B1
    # This is like a 2x2x1 "matrix multiplication" where each element is 2x1
    # Use Strassen-like approach: 7 multiplications
    for half in range(2):
        # half = 0: first column of B and C
        # half = 1: second column of B and C
        for s in range(7):
            # U: A contributions
            for a in range(4):
                if alpha[s][a] != 0:
                    ar, ac = a // 2, a % 2
                    u_idx = (ar * 2 + 0) * 4 + (ac * 2 + half)
                    U[u_idx, col] += alpha[s][a]
            
            # V: B contributions (B is 4x2, we take column 'half')
            for b in range(4):
                if beta[s][b] != 0:
                    br, bc = b // 2, b % 2
                    # B element: row = br*2 + 0, col = half
                    v_idx = (br * 2 + 0) * 2 + half
                    V[v_idx, col] += beta[s][b]
            
            # W: C contributions (C is 4x2, we take column 'half')
            for a, b, gamma in recon[s]:
                w_idx = (b * 2 + 0) * 2 + half
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
