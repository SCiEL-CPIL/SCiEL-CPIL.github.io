import numpy as np
import time
import itertools

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s
    
    # Strategy: Use 4 disjoint Strassen blocks and schoolbook for the rest.
    # Current best known: Rank 71.
    # We will try to find a better configuration or a lower rank decomposition
    # by searching over block placements.
    
    # Precompute all possible 2x2x2 block placements
    # i0 in 0..1, j0 in 0..3, k0 in 0..3
    possible_blocks = [(i0, j0, k0) for i0 in range(n-1) for j0 in range(m-1) for k0 in range(p-1)]
    
    def get_block_mask(i0, j0, k0):
        """Returns a set of (i,j,k) tuples covered by the block."""
        return {(i0+di, j0+dj, k0+dk) for di in range(2) for dj in range(2) for dk in range(2)}
    
    def is_disjoint(block1, block2):
        mask1 = get_block_mask(*block1)
        mask2 = get_block_mask(*block2)
        return len(mask1.intersection(mask2)) == 0
    
    # Find all sets of 4 disjoint blocks
    # There are 2 * 4 * 4 = 32 possible blocks.
    # We need to choose 4 disjoint ones.
    
    best_rank = 75
    best_blocks = None
    
    # Heuristic: Greedy selection of disjoint blocks
    # Or simply iterate over all combinations if feasible.
    # 32 choose 4 is 35960. This is small enough to iterate.
    
    # However, we also need to check if the remaining terms can be reduced.
    # For now, we assume rank = 4*7 + (75 - 4*8) = 28 + 43 = 71.
    # All disjoint sets of 4 blocks give rank 71.
    # So we need a different strategy to get < 71.
    
    # Let's try to find a rank 70 decomposition by checking if any
    # schoolbook term can be "absorbed" or if a different set of blocks
    # allows for a better remainder.
    
    # Actually, the rank is determined by the number of blocks and the number
    # of uncovered terms.
    # Rank = 7 * (#blocks) + (#uncovered terms).
    # To get rank < 71, we need to either:
    # 1. Use 5 blocks? 5*7 + (75 - 40) = 35 + 35 = 70.
    #    Can we fit 5 disjoint 2x2x2 blocks in 3x5x5?
    #    Volume = 125. 5 blocks * 8 = 40.
    #    i-dim: 3. Can fit 1 block (i=0..1) or 1 block (i=1..2).
    #    If we use i=0..1, we can fit 4 blocks in j,k (2x2 grid of 2x2 blocks).
    #    Can we fit a 5th block?
    #    It would need to be in i=1..2?
    #    Block at i=1..2 overlaps with i=0..1 at i=1.
    #    So no, we cannot have 5 disjoint blocks.
    #    Max disjoint blocks = 4.
    
    # 2. Use 4 blocks but reduce the number of uncovered terms?
    #    Uncovered terms = 75 - 32 = 43.
    #    Can we cover more terms with the Strassen blocks?
    #    No, each block covers exactly 8 terms.
    
    # 3. Use a different decomposition for the remainder?
    #    The remainder is a sum of rank-1 terms.
    #    If any subset of the remainder terms has a lower rank decomposition,
    #    we can save rank.
    
    # Let's check the structure of the remainder.
    # The remainder consists of terms where at least one index is not in a block.
    
    # Let's try a specific configuration:
    # Blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2).
    # This covers i=0,1. j=0,1,2,3. k=0,1,2,3.
    # Uncovered:
    # i=2 (all j,k): 5*5 = 25 terms.
    # i=0,1, j=4 (all k): 2*5 = 10 terms.
    # i=0,1, k=4 (j=0..3): 2*4 = 8 terms.
    # Total uncovered: 25 + 10 + 8 = 43.
    
    # Can we decompose the 25 terms (i=2) with rank < 25?
    # T[2,j,k] = 1 for all j,k.
    # This is a 1x5x5 tensor.
    # It is the outer product of e_2 with the all-ones 5x5 matrix.
    # Rank of 5x5 all-ones matrix is 1.
    # So T[2,j,k] = e_2 \otimes J_5.
    # This can be decomposed as 1 rank-1 term?
    # No, the rank-1 term in tensor decomposition is u \otimes v \otimes w.
    # If we use u = e_2, v = 1_5, w = 1_5, then
    # (e_2 \otimes 1_5 \otimes 1_5)[2,j,k] = 1 * 1 * 1 = 1.
    # This works!
    # So the 25 terms for i=2 can be replaced by 1 term?
    # Wait. The term is U[2*m+j, r] * V[j*p+k, r] * W[k*n+2, r].
    # If we set U[2*m+j, r] = 1 for all j, V[j*p+k, r] = 1 for all j,k, W[k*n+2, r] = 1 for all k.
    # Then for i=2, j, k: 1 * 1 * 1 = 1.
    # For i=0,1: U[i*m+j, r] = 0. So it doesn't interfere.
    # So yes, we can replace the 25 schoolbook terms for i=2 with 1 term!
    
    # Let's check the other uncovered terms.
    # i=0,1, j=4, k=0..4: 10 terms.
    # T[i,4,k] = 1.
    # Can we replace these with 1 term?
    # u = e_0 + e_1 (for i), v = e_4 (for j), w = 1_5 (for k).
    # Term: (e_0+e_1) \otimes e_4 \otimes 1_5.
    # For i=0,1, j=4, k: 1 * 1 * 1 = 1.
    # For other i,j,k: 0.
    # So yes, 10 terms -> 1 term.
    
    # i=0,1, k=4, j=0..3: 8 terms.
    # T[i,j,4] = 1.
    # Term: (e_0+e_1) \otimes (e_0+e_1+e_2+e_3) \otimes e_4.
    # For i=0,1, j=0..3, k=4: 1 * 1 * 1 = 1.
    # So 8 terms -> 1 term.
    
    # Total new rank:
    # 4 Strassen blocks: 28 terms.
    # i=2 part: 1 term.
    # i=0,1, j=4 part: 1 term.
    # i=0,1, k=4, j=0..3 part: 1 term.
    # Total rank: 28 + 1 + 1 + 1 = 31.
    
    # Wait, is this correct?
    # Let's verify the "1 term" decomposition.
    # The tensor is T[i,j,k].
    # We need sum_r U_r[i,j] V_r[j,k] W_r[k,i] = T[i,j,k].
    # My proposed term:
    # U[2*m+j, r] = 1 for all j.
    # V[j*p+k, r] = 1 for all j,k.
    # W[k*n+2, r] = 1 for all k.
    # Product: 1 * 1 * 1 = 1 for i=2, any j,k.
    # For i=0,1: U is 0. Product 0.
    # This seems correct.
    
    # However, we must ensure that this term does not interfere with the Strassen blocks.
    # The Strassen blocks are at i=0,1.
    # So U[0..1, j, r] are non-zero.
    # My new term has U[2, j, r] = 1, U[0..1, j, r] = 0.
    # So no interference.
    
    # Similarly for the other terms.
    
    # So the rank is 31!
    
    # Let's construct this.
    
    U = [[0.0] * 31 for _ in range(n * m)]
    V = [[0.0] * 31 for _ in range(m * p)]
    W = [[0.0] * 31 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1

    # Add 4 Strassen blocks
    blocks = [
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    # Add term for i=2 (all j, k)
    # U[2*m+j, col] = 1 for all j
    # V[j*p+k, col] = 1 for all j, k
    # W[k*n+2, col] = 1 for all k
    for j in range(m):
        U[2*m + j][col] = 1.0
    for j in range(m):
        for k in range(p):
            V[j*p + k][col] = 1.0
    for k in range(p):
        W[k*n + 2][col] = 1.0
    col += 1
    
    # Add term for i=0,1; j=4; all k
    # U[i*m+4, col] = 1 for i=0,1
    # V[4*p+k, col] = 1 for all k
    # W[k*n+i, col] = 1 for all k, i=0,1
    for i in range(2):
        U[i*m + 4][col] = 1.0
    for k in range(p):
        V[4*p + k][col] = 1.0
    for k in range(p):
        for i in range(2):
            W[k*n + i][col] = 1.0
    col += 1
    
    # Add term for i=0,1; j=0..3; k=4
    # U[i*m+j, col] = 1 for i=0,1, j=0..3
    # V[j*p+4, col] = 1 for j=0..3
    # W[4*n+i, col] = 1 for i=0,1
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1.0
    for j in range(4):
        V[j*p + 4][col] = 1.0
    for i in range(2):
        W[4*n + i][col] = 1.0
    col += 1
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(31))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-10:
        # Fallback to schoolbook
        total_rank_fb = n * m * p
        U_fb = [[0.0] * total_rank_fb for _ in range(n * m)]
        V_fb = [[0.0] * total_rank_fb for _ in range(m * p)]
        W_fb = [[0.0] * total_rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][col_fb] = 1.0
                    V_fb[j*p+k][col_fb] = 1.0
                    W_fb[k*n+i][col_fb] = 1.0
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(U.shape[1]))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    print(f"Max error: {max_err}")
