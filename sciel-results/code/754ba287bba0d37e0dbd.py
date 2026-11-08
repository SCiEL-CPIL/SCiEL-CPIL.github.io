"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=4
    
    # We use a block decomposition:
    # Split B (3x4) into B_left (3x2) and B_right (3x2)
    # C = A * B = [A * B_left, A * B_right]
    # Each A*B_block is 3x2 matrix product
    # 
    # For 3x2x2 matrix multiplication, we can use:
    # - Strassen-like for the 2x2 subpart: rank 7
    # - Plus the last row contributions: rank 4 (schoolbook for 1x2*2x2)
    # Total: rank 11 for 3x2x2
    #
    # Two blocks: total rank 22
    # But we can do better by sharing some computations.
    #
    # Actually, let's just use a known good decomposition.
    # For 3x3x4, schoolbook is 36.
    # 
    # Let's try a different approach: use the fact that we can
    # decompose using rank-revealing factorizations.
    #
    # Simplest correct approach: schoolbook rank 36, then try to reduce.
    #
    # Let's implement a rank 30 solution using partial Strassen:
    # Split A into A_top (2x3) and A_bottom (1x3)
    # Split B into B_top (2x4) and B_bottom (1x4) -- wait, B is 3x4
    # 
    # Actually, let me just implement the schoolbook and then apply
    # a simple rank reduction by combining terms.
    #
    # For now, let's implement a clean schoolbook (rank 36) and then
    # try to find a lower rank solution.
    #
    # Let's use a known algorithm: the "Laderman-like" approach adapted.
    #
    # Actually, the simplest reliable improvement is to note that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can group the k dimension. For each fixed (i,j), the contribution
    # to row i of C is A[i,j] * B[j,:]. This is a rank-1 update to a row.
    # 
    # Total terms: 3*3 = 9 such row updates, each contributing 4 values.
    # But in tensor form, each (i,j) pair with all k gives a term.
    # That's still 36 terms.
    #
    # Let me just go with schoolbook rank 36 for correctness, then
    # attempt a search for lower rank if time permits.
    #
    # Actually, let me think about this more carefully.
    # 
    # The tensor for 3x3x4 matrix mult has 3*3*4 = 36 entries.
    # The schoolbook decomposition has rank 36.
    # 
    # Known results: R(3,3,4) is known to be at most 30 (by various
    # decompositions). Let me implement a rank 30 decomposition.
    #
    # One approach: use Strassen on a 2x2x2 sub-block and schoolbook elsewhere.
    # 
    # Let me index: i in {0,1,2} (rows of A/C), j in {0,1,2} (cols of A/rows of B), k in {0,1,2,3} (cols of B/C)
    # 
    # Split: 
    # - Use Strassen for A[0:2, 0:2] * B[0:2, 0:2] (2x2x2, rank 7)
    # - Schoolbook for remaining terms
    #
    # This is getting complex. Let me just implement schoolbook rank 36
    # which is guaranteed correct, and the evaluator will score it.
    # The score is rank, so 36 is the baseline.
    
    # Schoolbook decomposition
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
    
    # Check time - if we have plenty of time, we could try to find
    # a lower rank solution, but for reliability let's return this.
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition using a simple search
    # For now, return the schoolbook result
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
