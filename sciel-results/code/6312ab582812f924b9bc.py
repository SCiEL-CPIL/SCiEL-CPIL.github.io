import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    total_terms = n * m * p

    def add_strassen_block(U, V, W, i0, j0, k0, col_offset):
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

    # Search for optimal non-overlapping 2x2x2 block placement
    # Each block is defined by (i0, j0, k0) where i0 in [0, n-2], j0 in [0, m-2], k0 in [0, p-2]
    possible_blocks = list(product(range(n-1), range(m-1), range(p-1)))
    
    best_rank = total_terms  # schoolbook only
    best_blocks = []
    
    # Backtracking search for max set of non-overlapping blocks
    # With n=3, m=5, p=5: possible i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}
    # That's 2*4*4 = 32 possible blocks, but we can only place ~4 non-overlapping
    # Use greedy + local search
    
    def blocks_overlap(b1, b2):
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # Check if 2x2x2 blocks overlap
        if i0a <= i0b+1 and i0b <= i0a+1 and j0a <= j0b+1 and j0b <= j0a+1 and k0a <= k0b+1 and k0b <= k0a+1:
            return True
        return False
    
    # Greedy: try all orderings isn't feasible, so use multiple random starts
    import random
    random.seed(42)
    
    def greedy_placement(block_list):
        placed = []
        remaining = list(block_list)
        while remaining:
            # Pick block that covers most uncovered terms
            best_block = None
            best_count = -1
            for idx, b in enumerate(remaining):
                # Count uncovered terms this block would cover
                count = 0
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i, j, k = b[0]+di, b[1]+dj, b[2]+dk
                            if i < n and j < m and k < p:
                                # Check if already covered
                                covered = False
                                for pb in placed:
                                    if (pb[0] <= i <= pb[0]+1 and pb[1] <= j <= pb[1]+1 and pb[2] <= k <= pb[2]+1):
                                        covered = True
                                        break
                                if not covered:
                                    count += 1
                if count > best_count:
                    best_count = count
                    best_block = idx
            if best_block is None or best_count == 0:
                break
            placed.append(remaining.pop(best_block))
        return placed
    
    # Try multiple random orderings
    best_placement = []
    for trial in range(200):
        shuffled = list(possible_blocks)
        random.shuffle(shuffled)
        placement = greedy_placement(shuffled)
        # Compute rank for this placement
        covered = [[[False]*p for _ in range(m)] for _ in range(n)]
        for b in placement:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered[b[0]+di][b[1]+dj][b[2]+dk] = True
        uncovered = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
        rank = len(placement) * 7 + uncovered
        if rank < best_rank:
            best_rank = rank
            best_placement = placement
    
    # Also try the known good configuration
    known_good = [(0,0,0), (0,2,0), (0,0,2), (0,2,2)]
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for b in known_good:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[b[0]+di][b[1]+dj][b[2]+dk] = True
    uncovered = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered[i][j][k])
    rank = len(known_good) * 7 + uncovered
    if rank < best_rank:
        best_rank = rank
        best_placement = known_good
    
    blocks = best_placement
    num_blocks = len(blocks)
    
    # Determine covered terms
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True

    schoolbook_count = sum(1 for i in range(n) for j in range(m) for k in range(p) if not covered_3d[i][j][k])
    total_rank = num_blocks * 7 + schoolbook_count

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    col = 0
    for i0, j0, k0 in blocks:
        add_strassen_block(U, V, W, i0, j0, k0, col)
        col += 7

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    print(f"Rank: {U.shape[1]}")
    
    # Verify correctness
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                expected = 1.0
                err = abs(val - expected)
                max_err = max(max_err, err)
    print(f"Max error: {max_err}")
