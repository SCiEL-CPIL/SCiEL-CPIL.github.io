import time
import sys

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 decomposition
    # U is (i,j) flattened, V is (j,k) flattened, W is (k,i) flattened
    # Each entry in rank r contributes to tensor[i,j,k] as U[i,j,r]*V[j,k,r]*W[k,i,r]
    
    strassen_U = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(1,0): 1, (1,1): 1},   # r=1
        {(0,0): 1},              # r=2
        {(1,1): 1},              # r=3
        {(0,0): 1, (0,1): 1},   # r=4
        {(1,0): 1, (0,0): -1},  # r=5
        {(0,1): 1, (1,1): -1},  # r=6
    ]
    strassen_V = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,0): 1},              # r=1
        {(0,1): 1, (1,1): -1},  # r=2
        {(1,0): 1, (0,0): -1},  # r=3
        {(1,1): 1},              # r=4
        {(0,0): 1, (0,1): 1},   # r=5
        {(1,0): 1, (1,1): 1},   # r=6
    ]
    strassen_W = [
        {(0,0): 1, (1,1): 1},   # r=0
        {(0,1): 1, (1,1): -1},  # r=1
        {(1,0): 1, (1,1): 1},   # r=2
        {(0,0): 1, (0,1): 1},   # r=3
        {(0,0): -1, (1,0): 1},  # r=4
        {(1,1): 1},              # r=5
        {(0,0): 1},              # r=6
    ]
    
    def add_block(base_rank, i_base, j_base, k_base):
        """Add a Strassen decomposition for A[i_base:i_base+2, j_base:j_base+2] * 
           B[j_base:j_base+2, k_base:k_base+2] starting at rank index base_rank."""
        for r in range(7):
            cr = base_rank + r
            for (li, lj), val in strassen_U[r].items():
                gi = i_base + li
                gj = j_base + lj
                U[gi * 4 + gj][cr] += val
            
            for (lj, lk), val in strassen_V[r].items():
                gj = j_base + lj
                gk = k_base + lk
                V[gj * 8 + gk][cr] += val
            
            for (lk, li), val in strassen_W[r].items():
                gk = k_base + lk
                gi = i_base + li
                W[gk * 2 + gi][cr] += val
    
    # A0 * Bt0: i_base=0, j_base=0, k_base=0
    add_block(0, 0, 0, 0)
    
    # A0 * Bt1: i_base=0, j_base=0, k_base=2
    add_block(7, 0, 0, 2)
    
    # A0 * Bt2: i_base=0, j_base=0, k_base=4
    add_block(14, 0, 0, 4)
    
    # A0 * Bt3: i_base=0, j_base=0, k_base=6
    add_block(21, 0, 0, 6)
    
    # A1 * Bb0: i_base=0, j_base=2, k_base=0
    add_block(28, 0, 2, 0)
    
    # A1 * Bb1: i_base=0, j_base=2, k_base=2
    add_block(35, 0, 2, 2)
    
    # A1 * Bb2: i_base=0, j_base=2, k_base=4
    add_block(42, 0, 2, 4)
    
    # A1 * Bb3: i_base=0, j_base=2, k_base=6
    add_block(49, 0, 2, 6)
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank by finding linearly dependent terms
    # Flatten the tensor and check for dependencies
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can try to eliminate columns that are linear combinations of others
    
    # For each column r, we can check if it's a linear combination of previous columns
    # This is equivalent to checking if the rank-1 tensor term is in the span of others
    
    # Simple approach: try removing each column and checking if the decomposition still works
    # This is expensive, so we'll do a quick check
    
    # Actually, let's try a different approach: look for pairs of blocks that might share structure
    # In the 2x4x8 case, blocks (0,0,0) and (0,0,2) use the same A submatrix but different B columns
    # Similarly for other pairs
    
    # For now, return the rank-56 decomposition
    # A proper rank reduction would require more sophisticated algebraic geometry techniques
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
