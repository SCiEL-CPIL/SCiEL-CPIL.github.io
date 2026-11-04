"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Try Laderman-style block decomposition first
    # Partition 2x5 into [1,1] x [2,3] and 5x6 into [2,3] x [3,3]
    # This gives 4 block-multiplications
    # Use rank-7 for 2x2x2 and rank-5 for 1x1x1 (trivial)
    # Total rank: 7 + 7 + 7 + 7 = 28? No, that's not right.
    
    # Actually, let's try a more systematic approach:
    # The tensor for 2x5 * 5x6 has rank at most 60 (schoolbook)
    # Known lower bound: max(2*5, 5*6, 6*2) = max(10, 30, 12) = 30
    # But for matrix multiplication, the tensor rank is typically close to nm*p
    
    # Let's try to find a decomposition using structured methods
    # First, try the Kronecker product structure
    
    # Method 1: Direct construction using block structure
    # A is 2x5, B is 5x6, C = A*B is 2x6
    # Partition A into rows: A1 (1x5), A2 (1x5)
    # Partition B into columns: B1 (5x3), B2 (5x3)
    # C11 = A1*B1 (1x3), C12 = A1*B2 (1x3)
    # C21 = A2*B1 (1x3), C22 = A2*B2 (1x3)
    
    # Each 1x5 * 5x3 multiplication has rank 15
    # Total: 4 * 15 = 60 (same as schoolbook)
    
    # Try a smarter partition:
    # A: [1,1] x [2,3] where first block is 1x2, second is 1x3
    # B: [2,3] x [3,3] where first block is 2x3, second is 3x3
    # C11 = A[0:1,0:2] * B[0:2,0:3] = 1x2 * 2x3 = 1x3, rank 6
    # C12 = A[0:1,2:5] * B[2:5,0:3] = 1x3 * 3x3 = 1x3, rank 9
    # C21 = A[1:2,0:2] * B[0:2,3:6] = 1x2 * 2x3 = 1x3, rank 6
    # C22 = A[1:2,2:5] * B[2:5,3:6] = 1x3 * 3x3 = 1x3, rank 9
    # Total: 6+9+6+9 = 30
    
    # This gives rank 30! Let's construct it.
    
    # For each block multiplication, use the schoolbook method
    # Block 1: 1x2 * 2x3 -> 1x3, rank 6
    # Block 2: 1x3 * 3x3 -> 1x3, rank 9
    # Block 3: 1x2 * 2x3 -> 1x3, rank 6
    # Block 4: 1x3 * 3x3 -> 1x3, rank 9
    
    # Let's build the full decomposition
    rank = 30
    U = [[0] * rank for _ in range(n * m)]  # 10 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 12 x 30
    
    col = 0
    
    # Block 1: A[0,0:2] * B[0:2,0:3] -> C[0,0:3]
    # A indices: i=0, j=0,1 -> u = 0*5+0=0, 0*5+1=1
    # B indices: j=0,1; k=0,1,2 -> v = 0*6+0=0, 0*6+1=1, 0*6+2=2, 1*6+0=6, 1*6+1=7, 1*6+2=8
    # C indices: i=0, k=0,1,2 -> w = 0*2+0=0, 0*2+1=1, 0*2+2=2
    for j in range(2):
        for k in range(3):
            u = 0 * m + j  # i=0
            v = j * p + k  # j, k
            w = k * n + 0  # k, i=0
            U[u][col] = 1
            V[v][col] = 1
            W[w][col] = 1
            col += 1
    
    # Block 2: A[0,2:5] * B[2:5,0:3] -> C[0,0:3]
    # A indices: i=0, j=2,3,4 -> u = 0*5+2=2, 3, 4
    # B indices: j=2,3,4; k=0,1,2 -> v = 2*6+0=12, 13, 14, 3*6+0=18, 19, 20, 4*6+0=24, 25, 26
    # C indices: i=0, k=0,1,2 -> w = 0, 1, 2
    for j in range(2, 5):
        for k in range(3):
            u = 0 * m + j
            v = j * p + k
            w = k * n + 0
            U[u][col] = 1
            V[v][col] = 1
            W[w][col] = 1
            col += 1
    
    # Block 3: A[1,0:2] * B[0:2,3:6] -> C[1,0:3]
    # A indices: i=1, j=0,1 -> u = 1*5+0=5, 6
    # B indices: j=0,1; k=3,4,5 -> v = 0*6+3=3, 4, 5, 1*6+3=9, 10, 11
    # C indices: i=1, k=0,1,2 -> w = 0*2+1=1, 1*2+1=3, 2*2+1=5
    for j in range(2):
        for k in range(3, 6):
            u = 1 * m + j
            v = j * p + k
            w = (k - 3) * n + 1  # k-3 is 0,1,2; w = (k-3)*2 + 1
            U[u][col] = 1
            V[v][col] = 1
            W[w][col] = 1
            col += 1
    
    # Block 4: A[1,2:5] * B[2:5,3:6] -> C[1,0:3]
    # A indices: i=1, j=2,3,4 -> u = 1*5+2=7, 8, 9
    # B indices: j=2,3,4; k=3,4,5 -> v = 2*6+3=15, 16, 17, 3*6+3=21, 22, 23, 4*6+3=27, 28, 29
    # C indices: i=1, k=0,1,2 -> w = 1, 3, 5
    for j in range(2, 5):
        for k in range(3, 6):
            u = 1 * m + j
            v = j * p + k
            w = (k - 3) * n + 1
            U[u][col] = 1
            V[v][col] = 1
            W[w][col] = 1
            col += 1
    
    # Verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return _schoolbook(n, m, p)


def _verify_decomposition(U, V, W, n, m, p, sample_size=10000):
    """Verify that the decomposition exactly represents matrix multiplication."""
    import random
    
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    # Build the set of tensor entries
    tensor_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                tensor_set.add((u, v, w))
    
    # Check all tensor entries
    for (u, v, w) in tensor_set:
        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
        if abs(val - 1.0) > 1e-9:
            return False
    
    # Check a sample of non-tensor entries
    random.seed(42)
    for _ in range(sample_size):
        u = random.randint(0, nm - 1)
        v = random.randint(0, mp - 1)
        w = random.randint(0, pn - 1)
        if (u, v, w) in tensor_set:
            continue
        val = sum(U[u][r] * V[v][r] * W[w][r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    
    return True


def _schoolbook(n, m, p):
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
# EVOLVE-BLOCK-END
