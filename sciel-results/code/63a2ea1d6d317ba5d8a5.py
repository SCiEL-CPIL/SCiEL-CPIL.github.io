"""Rewritten program: search for rank-minimal tensor decomposition of 2x2 * 2x15 matrix multiplication."""
import numpy as np
from itertools import product
import time
import random

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    import time
    _start_time = time.time()
    _DEADLINE = 120.0
    n, m, p = DIMENSIONS
    # n=2, m=2, p=15
    # Schoolbook rank: n*m*p = 60
    # Lower bound: max(n*m, m*p, p*n) = max(4, 30, 30) = 30
    
    # Strategy: Use block decomposition
    # Split the 15 columns of B into blocks
    # For each block of size 2, use Strassen (rank 7)
    # For a block of size 1, use schoolbook (rank 4)
    
    # 15 = 7*2 + 1 -> 7 blocks of size 2 (rank 7 each = 49) + 1 block of size 1 (rank 4) = 53
    # 15 = 3*4 + 3 -> 3 blocks of size 4 (rank 14 each = 42) + 1 block of size 3 (rank 11) = 53
    # 15 = 5*3 -> 5 blocks of size 3 (rank 11 each = 55)
    
    # Let's implement the 7*2 + 1 split for rank 53
    
    # Strassen for 2x2 * 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    # C11 = P1 + P4 - P5 + P7
    # C12 = P3 + P5
    # C21 = P2 + P4
    # C22 = P1 - P2 + P3 + P6
    
    # For 2x2 * 2x1 (single column [e; g]):
    # C1 = a*e + b*g
    # C2 = c*e + d*g
    # Rank 4: just schoolbook
    
    # Let's build the decomposition for 15 = 7*2 + 1
    
    # Total rank = 7*7 + 4 = 53
    
    rank = 53
    U = np.zeros((n*m, rank))  # (4, 53)
    V = np.zeros((m*p, rank))  # (30, 53)
    W = np.zeros((p*n, rank))  # (30, 4)
    
    col = 0
    
    # Process 7 blocks of size 2
    for block_idx in range(7):
        # Columns k = 2*block_idx and 2*block_idx+1
        k0 = 2 * block_idx
        k1 = 2 * block_idx + 1
        
        # Strassen decomposition for this block
        # A is 2x2, B_block is 2x2 (columns k0, k1)
        
        # P1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # P2 = (A[1,0] + A[1,1]) * B[0,k0]
        # P3 = A[0,0] * (B[1,k1] - B[0,k0])
        # P4 = A[1,1] * (B[1,k0] - B[0,k1])
        # P5 = (A[0,0] + A[0,1]) * B[1,k1]
        # P6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[1,k1])
        # P7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[0,k1])
        
        # C[0,k0] = P1 + P4 - P5 + P7
        # C[0,k1] = P3 + P5
        # C[1,k0] = P2 + P4
        # C[1,k1] = P1 - P2 + P3 + P6
        
        # For each Strassen term, we need to set up U, V, W entries
        
        # P1: U factors: A[0,0] and A[1,1] with +1
        #      V factors: B[0,k0] and B[1,k1] with +1
        #      W factors: C[0,k0] with +1, C[1,k1] with +1
        
        # Actually, let's think about this more carefully.
        # The tensor decomposition is:
        # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
        
        # For Strassen, each intermediate product P_i corresponds to a rank-1 term
        # P1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # This contributes to:
        #   C[0,k0]: +1 * P1
        #   C[1,k1]: +1 * P1
        
        # So for P1:
        # U[0, r1] = 1 (A[0,0])
        # U[3, r1] = 1 (A[1,1])  [index 3 = 1*2+1]
        # V[0, r1] = 1 (B[0,k0])
        # V[15, r1] = 1 (B[1,k1])  [index 15 = 1*15+0... wait]
        
        # Let me re-index:
        # A index: i*m + j, where i in {0,1}, j in {0,1}
        #   A[0,0] -> 0, A[0,1] -> 1, A[1,0] -> 2, A[1,1] -> 3
        # B index: j*p + k, where j in {0,1}, k in {0..14}
        #   B[0,k] -> k, B[1,k] -> 15+k
        # C index: k*n + i, where k in {0..14}, i in {0,1}
        #   C[0,k] -> k, C[1,k] -> 15+k... wait, k*n + i = 2k + i
        #   C[0,k] -> 2k, C[1,k] -> 2k+1
        
        # P1 = (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # U: A[0,0] (idx 0) and A[1,1] (idx 3), both with +1
        # V: B[0,k0] (idx k0) and B[1,k1] (idx 15+k1), both with +1
        # W: C[0,k0] (idx 2*k0) with +1, C[1,k1] (idx 2*k1+1) with +1
        
        # But wait, this is a product of sums, not a sum of products.
        # (A[0,0] + A[1,1]) * (B[0,k0] + B[1,k1])
        # = A[0,0]*B[0,k0] + A[0,0]*B[1,k1] + A[1,1]*B[0,k0] + A[1,1]*B[1,k1]
        
        # Each of these is a rank-1 term in the tensor.
        # But Strassen combines them into one multiplication.
        
        # For the tensor decomposition, we need:
        # U[a,r] * V[b,r] * W[c,r] summed over r gives the tensor value.
        
        # For the term A[0,0]*B[0,k0] contributing to C[0,k0]:
        # U[0, r] = 1, V[k0, r] = 1, W[2*k0, r] = 1
        
        # For Strassen, we have 7 such combined terms per 2x2x2 block.
        # Each combined term uses one rank slot.
        
        # Let me just implement the full Strassen decomposition explicitly.
        
        # For block with columns k0, k1:
        # r1: P1
        U[0, col] = 1; U[3, col] = 1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        
        # r2: P2 = (A[1,0] + A[1,1]) * B[0,k0]
        # Contributes to C[1,k0] with +1 and C[1,k1] with -1
        U[2, col] = 1; U[3, col] = 1
        V[k0, col] = 1
        W[2*k0, col] = 1; W[2*k1, col] = -1
        col += 1
        
        # r3: P3 = A[0,0] * (B[1,k1] - B[0,k0])
        U[0, col] = 1
        V[15+k1, col] = 1; V[k0, col] = -1
        W[2*k1+1, col] = 1; W[2*k1, col] = 1
        col += 1
        
        # r4: P4 = A[1,1] * (B[1,k0] - B[0,k1])
        U[3, col] = 1
        V[15+k0, col] = 1; V[15+k1, col] = -1
        W[2*k0, col] = 1; W[2*k1+1, col] = 1
        col += 1
        
        # r5: P5 = (A[0,0] + A[0,1]) * B[1,k1]
        U[0, col] = 1; U[1, col] = 1
        V[15+k1, col] = 1
        W[2*k0, col] = -1; W[2*k1+1, col] = 1
        col += 1
        
        # r6: P6 = (A[1,0] - A[0,0]) * (B[0,k0] + B[1,k1])
        U[2, col] = 1; U[0, col] = -1
        V[k0, col] = 1; V[15+k1, col] = 1
        W[2*k1+1, col] = 1
        col += 1
        
        # r7: P7 = (A[0,1] - A[1,1]) * (B[1,k0] + B[0,k1])
        U[1, col] = 1; U[3, col] = -1
        V[15+k0, col] = 1; V[15+k1, col] = 1
        W[2*k0, col] = 1
        col += 1
    
    # Last block: single column k = 14
    k = 14
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    
    # r8: A[0,0]*B[0,k] -> C[0,k]
    U[0, col] = 1
    V[k, col] = 1
    W[2*k, col] = 1
    col += 1
    
    # r9: A[0,1]*B[1,k] -> C[0,k]
    U[1, col] = 1
    V[15+k, col] = 1
    W[2*k, col] = 1
    col += 1
    
    # r10: A[1,0]*B[0,k] -> C[1,k]
    U[2, col] = 1
    V[k, col] = 1
    W[2*k+1, col] = 1
    col += 1
    
    # r11: A[1,1]*B[1,k] -> C[1,k]
    U[3, col] = 1
    V[15+k, col] = 1
    W[2*k+1, col] = 1
    col += 1
    
    # Deadline guard: if we've used too much time, skip to fallback
    if time.time() - _start_time > _DEADLINE - 5.0:
        rank_schoolbook = n * m * p
        U_sb = np.zeros((n*m, rank_schoolbook))
        V_sb = np.zeros((m*p, rank_schoolbook))
        W_sb = np.zeros((p*n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        # Check all nonzero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for r in range(rank):
                        val += U[a, r] * V[b, r] * W[c, r]
                    if abs(val - 1.0) > 1e-10:
                        return False
        # Check some zero entries
        for _ in range(500):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            c = random.randint(0, p*n - 1)
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k1 = b % p
            k2 = c // n
            i2 = c % n
            if not (i1 == i2 and j1 == j2):
                val = 0.0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if abs(val) > 1e-10:
                    return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    
    # Fallback to schoolbook if verification fails
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
# EVOLVE-BLOCK-END
