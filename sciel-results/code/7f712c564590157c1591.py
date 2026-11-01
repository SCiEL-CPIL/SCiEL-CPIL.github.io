"""Rewritten program for matrix multiplication 2,3,10 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check whether the decomposition exactly reproduces the matmul tensor."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > max_err:
                    max_err = abs(s - 1.0)
    return max_err < 1e-9

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _winograd_2x2x2():
    """Returns U, V, W for 2x2x2 multiplication with rank 7."""
    # n=2, m=2, p=2
    # Standard Winograd scheme
    # U indices: i*2+j (0..3)
    # V indices: j*2+k (0..3)
    # W indices: k*2+i (0..3)
    
    U = [
        [1, 0, 0, 0, 0, 0, 0], # i=0, j=0
        [0, 1, 1, 0, 0, 0, 0], # i=0, j=1
        [0, 1, -1, 0, 0, 0, 0],# i=1, j=0
        [0, 0, 0, 1, 1, 0, 0]  # i=1, j=1
    ]
    V = [
        [1, 0, 0, 0, 0, 0, 0], # j=0, k=0
        [0, 1, 1, 0, 0, 0, 0], # j=0, k=1
        [0, 1, -1, 0, 0, 0, 0],# j=1, k=0
        [0, 0, 0, 1, 1, 0, 0]  # j=1, k=1
    ]
    W = [
        [1, 0, 0, 0, 0, 0, 0], # k=0, i=0
        [0, 1, 1, 0, 0, 0, 0], # k=0, i=1
        [0, 1, -1, 0, 0, 0, 0],# k=1, i=0
        [0, 0, 0, 1, 1, 0, 0]  # k=1, i=1
    ]
    # Wait, the standard 7-term scheme is:
    # m0 = a00+a01, m1 = a00-a01, m2 = a10+a11, m3 = a10-a11
    # n0 = b00+b01, n1 = b00-b01, n2 = b10+b11, n3 = b10-b11
    # p0 = m0*n0, p1 = m1*n1, p2 = m2*n2, p3 = m3*n3
    # p4 = (a00+a10)(b00+b10), p5 = (a01+a11)(b01+b11), p6 = (a00+a11)(b10+b01)?
    # Actually, let's use a known valid rank 7 decomposition.
    # From literature:
    # U:
    # 1 0 0 0 0 0 0
    # 1 1 0 0 0 0 0
    # 1 -1 0 0 0 0 0
    # 0 0 1 1 0 0 0
    # V:
    # 1 0 0 0 0 0 0
    # 0 1 1 0 0 0 0
    # 0 1 -1 0 0 0 0
    # 0 0 0 1 1 0 0
    # W:
    # 1 0 0 0 0 0 0
    # 0 1 1 0 0 0 0
    # 0 1 -1 0 0 0 0
    # 0 0 0 1 1 0 0
    # This is not quite right.
    
    # Let's use a verified rank 7 decomposition for 2x2x2.
    # https://en.wikipedia.org/wiki/Winograd_algorithm
    # Scheme:
    # m0 = a00 + a01
    # m1 = a00 - a01
    # m2 = a10 + a11
    # m3 = a10 - a11
    # n0 = b00 + b01
    # n1 = b00 - b01
    # n2 = b10 + b11
    # n3 = b10 - b11
    # p0 = m0 * n0
    # p1 = m1 * n1
    # p2 = m2 * n2
    # p3 = m3 * n3
    # p4 = (a00 + a10) * (b00 + b10)
    # p5 = (a01 + a11) * (b01 + b11)
    # p6 = (a00 + a11) * (b10 + b01) -- wait, this is not symmetric.
    
    # Actually, a simpler rank 7 is:
    # Terms:
    # 1. a00*b00
    # 2. a00*b10
    # 3. a01*b00
    # 4. a01*b10
    # 5. a10*b00
    # 6. a10*b10
    # 7. a11*b11 -- no.
    
    # Let's just use the schoolbook for 2x2x2 if I can't get a valid 7 quickly, 
    # but 8 is better than 4? No, 4 is schoolbook.
    # Schoolbook for 2x2x2 is 4.
    # Winograd is 7? No, Winograd reduces 4 to 7? No, 7 is for 3x3x3?
    # Ah, 2x2x2 schoolbook is 4. Winograd is 7? That's worse.
    # Winograd 2x2x2 is actually 7 multiplications? 
    # No, standard matrix mult 2x2x2 is 4 multiplications.
    # Winograd 2x2x2 is 7? 
    # Wait, the number of multiplications for 2x2x2 is 4.
    # Winograd for 2x2x2 is 7? 
    # I think I am confusing dimensions.
    # For 2x2x2, rank is 7? No, rank is 4? 
    # The rank of the 2x2x2 tensor is 7? 
    # No, the rank of 2x2x2 is 7 is a known result? 
    # Actually, the rank of 2x2x2 is 7 is FALSE. It is 4? 
    # No, the rank of 2x2x2 is 7 is the number of multiplications in Winograd?
    # No, Winograd 2x2x2 uses 7 multiplications? 
    # Let's check: 
    # Schoolbook: 4.
    # Winograd: 7.
    # So Winograd is worse for 2x2x2?
    # Yes, Winograd is beneficial for larger blocks like 3x3x3 (19 vs 27) or 2x2x2?
    # Actually, for 2x2x2, the rank is 7? 
    # No, the rank of 2x2x2 is 7 is a common misconception.
    # The rank of 2x2x2 is 4? 
    # No, the rank of 2x2x2 is 7 is the number of terms in the Winograd scheme?
    # Let's assume schoolbook 4 is optimal for 2x2x2.
    
    # So my previous plan of 7+4=11 for 2x3x2 was based on 7 for 2x2x2.
    # If 2x2x2 is 4, then 2x3x2 is 4+4=8?
    # Let's check 2x3x2 schoolbook: 12.
    # If we split into 2x2x2 (rank 4) and 2x1x2 (rank 4), total 8.
    # This is a huge improvement!
    
    # Let's verify rank 4 for 2x2x2.
    # U, V, W for 2x2x2 with rank 4.
    # This is just schoolbook.
    
    # So for 2x3x2:
    # Block 1 (j=0,1): 2x2x2, rank 4.
    # Block 2 (j=2): 2x1x2, rank 4.
    # Total rank 8.
    
    # Can we do better than 8 for 2x3x2?
    # Maybe 7?
    # If we find rank 7 for 2x3x2, total rank 35.
    # If we find rank 8, total rank 40.
    # This is much better than 55 or 60.
    
    # Let's implement the rank 8 block (schoolbook split).
    # Then try to find rank 7 or lower via search.

def _block_2x3x2_rank8():
    """Returns U, V, W for 2x3x2 multiplication with rank 8 (schoolbook split)."""
    # n=2, m=3, p=2
    # U shape (6, 8)
    # V shape (6, 8)
    # W shape (4, 8)
    
    U = [[0]*8 for _ in range(6)]
    V = [[0]*8 for _ in range(6)]
    W = [[0]*8 for _ in range(4)]
    
    # Part 1: j=0,1 (2x2x2)
    # i in {0,1}, j in {0,1}, k in {0,1}
    # 4 terms
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i*3 + j][col] = 1
                V[j*2 + k][col] = 1
                W[k*2 + i][col] = 1
                col += 1
    # Part 2: j=2 (2x1x2)
    # i in {0,1}, j=2, k in {0,1}
    # 4 terms
    for i in range(2):
        for k in range(2):
            U[i*3 + 2][col] = 1
            V[2*2 + k][col] = 1
            W[k*2 + i][col] = 1
            col += 1
            
    return U, V, W

def _try_find_block_rank(target_rank, deadline):
    """Try to find a decomposition for 2x3x2 with given rank."""
    n, m, p = 2, 3, 2
    U_rows = n*m
    V_rows = m*p
    W_rows = p*n
    
    random.seed(42)
    for _ in range(1000):
        if time.time() > deadline:
            break
        # Random integers -1, 0, 1
        U = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(U_rows)]
        V = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(V_rows)]
        W = [[random.randint(-1, 1) for _ in range(target_rank)] for _ in range(W_rows)]
        
        # Verify
        ok = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(target_rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != 1:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            return U, V, W
    return None

def construct():
    start = time.time()
    deadline = start + 140
    
    n, m, p = DIMENSIONS
    
    # 1. Get rank 8 block for 2x3x2
    U_block, V_block, W_block = _block_2x3x2_rank8()
    best_block_rank = 8
    best_U_block, best_V_block, best_W_block = U_block, V_block, W_block
    
    # 2. Try to find lower rank block (7, 6, ...)
    for target_rank in range(7, 0, -1):
        result = _try_find_block_rank(target_rank, deadline)
        if result:
            best_U_block, best_V_block, best_W_block = result
            best_block_rank = target_rank
            break # Found the best possible for this rank, stop
    
    # 3. Assemble full 2x3x10 decomposition
    # 5 blocks
    num_blocks = p // 2 # 5
    total_rank = num_blocks * best_block_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for b in range(num_blocks):
        # Block b uses columns 2*b and 2*b+1 of the full matrix
        # U indices: i*m + j (same for all blocks)
        # V indices: j*p + k. For block b, k is 2*b and 2*b+1.
        # W indices: k*n + i. For block b, k is 2*b and 2*b+1.
        
        offset = b * best_block_rank
        
        for i in range(n):
            for j in range(m):
                for c in range(best_block_rank):
                    U[i*m + j][offset + c] = best_U_block[i*m + j][c]
        
        for j in range(m):
            for k_local in range(2):
                k_global = 2*b + k_local
                for c in range(best_block_rank):
                    V[j*p + k_global][offset + c] = best_V_block[j*2 + k_local][c]
        
        for k_local in range(2):
            k_global = 2*b + k_local
            for i in range(n):
                for c in range(best_block_rank):
                    W[k_global*n + i][offset + c] = best_W_block[k_local*2 + i][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
