"""Matrix multiplication tensor decomposition for 2x3x8."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin

    # Build target tensor for verification
    M = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                M[u_idx, v_idx, w_idx] = 1.0

    # Strategy: split m=3 into m1=1 and m2=2
    # Part 1: 2x1x8 (schoolbook rank 16)
    # Part 2: 2x2x8 (Strassen: 4 blocks of 2x2x2, each rank 7, total 28)
    # Total rank: 16 + 28 = 44

    # Part 1: j=0 (first column group)
    # U indices: i*3+0 = 0,1
    # V indices: 0*8+k = 0..7
    # W indices: k*2+i
    rank1 = 0
    U1 = np.zeros((n*m, 16))
    V1 = np.zeros((m*p, 16))
    W1 = np.zeros((p*n, 16))
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U1[u_idx, rank1] = 1.0
            V1[v_idx, rank1] = 1.0
            W1[w_idx, rank1] = 1.0
            rank1 += 1

    # Part 2: j=1,2 (last two column groups)
    # Use Strassen for 2x2x2 blocks
    # 4 blocks: k pairs (0,1), (2,3), (4,5), (6,7)
    # Each 2x2x2 block has rank 7
    
    # Strassen rank-7 decomposition for 2x2x2:
    # Local indices: A[i,j] -> i*2+j, B[j,k] -> j*2+k, C[i,k] -> k*2+i
    # The 7 rank-1 tensors (u, v, w) as length-4 vectors:
    
    # I'll use a known rank-7 decomposition. Let me construct it:
    # The 7 terms can be found by solving the system.
    
    # Actually, let me just hardcode a known decomposition.
    # The standard Strassen-based rank-7 decomposition:
    
    # For 2x2x2 matrix multiplication, the rank-7 decomposition is:
    # M_2x2x2[a,b,c] = sum_{r=1}^{7} u_r[a] * v_r[b] * w_r[c]
    
    # Where a = i*2+j (A index), b = j*2+k (B index), c = k*2+i (C index)
    
    # The 7 terms (verified by substitution):
    # Term 1: u=[1,0,0,1], v=[1,0,0,1], w=[1,0,0,1]  -> contributes to (0,0,0) and (3,3,2)
    # But this doesn't work directly. Let me use the actual Strassen decomposition.
    
    # Actually, for 2x2x2, the rank is 7, and the decomposition is:
    # We need 7 vectors (u_r, v_r, w_r) in R^4 such that
    # sum_r u_r[a] * v_r[b] * w_r[c] = M_2x2x2[a,b,c]
    
    # The valid entries are:
    # (0,0,0), (0,1,2), (1,2,1), (1,3,3), (2,0,1), (2,1,3), (3,2,0), (3,3,2)
    
    # A known rank-7 decomposition:
    # u1 = [1, 0, 0, 0], v1 = [1, 0, 0, 0], w1 = [1, 0, 0, 0]
    # u2 = [0, 1, 0, 0], v2 = [0, 1, 0, 0], w2 = [0, 1, 0, 0]
    # u3 = [0, 0, 1, 0], v3 = [0, 0, 1, 0], w3 = [0, 0, 1, 0]
    # u4 = [0, 0, 0, 1], v4 = [0, 0, 0, 1], w4 = [0, 0, 0, 1]
    # u5 = [1, 1, 0, 0], v5 = [0, 0, 1, 1], w5 = [0, 0, 0, 1]
    # u6 = [0, 0, 1, 1], v6 = [1, 1, 0, 0], w6 = [1, 0, 0, 0]
    # u7 = [1, 0, 1, 0], v7 = [1, 0, 1, 0], w7 = [0, 1, 0, 1]
    
    # Let me verify this more carefully. Actually, let me just use schoolbook for now
    # and get rank 16 + 32 = 48, which is safe.
    
    # For 2x2x8, schoolbook gives rank 2*2*8 = 32
    # Total: 16 + 32 = 48
    
    # Actually, let me try to implement Strassen properly.
    # For 2x2x2, the rank-7 decomposition is well-known.
    
    # The 7 terms for 2x2x2 (using Strassen's algorithm):
    # m1 = (a00 + a11)(b00 + b11)
    # m2 = (a10 + a11)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    
    # c00 = m1 - m4 + m6 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m3 + m5 - m6
    
    # In tensor form, each m_r is a product of 3 vectors:
    # m1: u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]  (but with signs)
    # Actually, the tensor decomposition is:
    # M_2x2x2 = sum_{r=1}^{7} u_r ⊗ v_r ⊗ w_r
    
    # Let me write out the 7 terms explicitly:
    # For m1 = (a00+a11)(b00+b11):
    #   u1 = [1, 0, 0, 1]  (a00 and a11)
    #   v1 = [1, 0, 0, 1]  (b00 and b11)
    #   w1 = [1, 0, 0, 1]  (c00 and c11)
    # But this gives unwanted terms. The correct decomposition uses:
    
    # Actually, the rank-7 decomposition of the 2x2x2 matrix multiplication tensor is:
    # [u1, v1, w1] = [[1,0,0,1], [1,0,0,1], [1,0,0,1]]
    # [u2, v2, w2] = [[0,0,1,1], [1,0,0,0], [0,0,1,0]]
    # [u3, v3, w3] = [[1,0,0,0], [0,1,0,-1], [0,1,0,0]]
    # [u4, v4, w4] = [[0,0,0,1], [0,0,1,-1], [0,0,1,0]]
    # [u5, v5, w5] = [[1,1,0,0], [0,0,0,1], [0,1,0,0]]
    # [u6, v6, w6] = [[0,0,1,-1], [1,1,0,0], [0,0,0,1]]
    # [u7, v7, w7] = [[0,1,0,-1], [0,0,1,1], [1,0,0,0]]
    
    # Let me verify term 1: u1⊗v1⊗w1
    # u1 = [1,0,0,1], v1 = [1,0,0,1], w1 = [1,0,0,1]
    # Nonzero entries: (a,b,c) where a∈{0,3}, b∈{0,3}, c∈{0,3}
    # (0,0,0)=1, (0,0,3)=1, (0,3,0)=1, (0,3,3)=1, (3,0,0)=1, (3,0,3)=1, (3,3,0)=1, (3,3,3)=1
    # But M_2x2x2 has (0,0,0)=1 and (3,3,2)=1, not (3,3,3)=1
    # So this is wrong.
    
    # Let me just use schoolbook for safety: rank = 48
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
