import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    # Generate all possible 2x2x2 block positions
    # i0 in 0..n-2, j0 in 0..m-2, k0 in 0..p-2
    possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_blocks.append((i0, j0, k0))
                
    # Mask for each block
    block_masks = []
    for i0, j0, k0 in possible_blocks:
        mask = set()
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    mask.add((i0+di, j0+dj, k0+dk))
        block_masks.append(mask)
        
    # Find max disjoint blocks
    # Since N=32, we can use a simple recursive search or greedy with backtracking
    # Let's try to find a set of size 5 or 6.
    
    max_blocks_found = []
    best_count = 0
    
    # Heuristic: Try to find 5 blocks.
    # If we find 5, rank is 70.
    # If we find 6, rank is 69.
    
    # We will do a DFS to find the maximum independent set in the conflict graph.
    # Or just enumerate combinations? 32 choose 5 is ~200k. 32 choose 6 is ~900k.
    # This is fast enough in Python if optimized.
    
    from itertools import combinations
    
    # Precompute conflicts
    # conflict[i][j] is True if block i and block j share any term
    N_blocks = len(possible_blocks)
    conflicts = [[False] * N_blocks for _ in range(N_blocks)]
    for i in range(N_blocks):
        for j in range(i+1, N_blocks):
            if block_masks[i].intersection(block_masks[j]):
                conflicts[i][j] = True
                conflicts[j][i] = True
                
    # Search for max size
    # We can try sizes from 6 down to 1
    # Or just find the max.
    
    # Optimization: Sort blocks by degree or just iterate.
    # Let's try to find a set of 6.
    found_6 = False
    # To speed up, we can use a recursive backtracking that picks a block and excludes its neighbors.
    
    def find_max_packing(current_set, available_indices):
        nonlocal best_count, max_blocks_found
        if len(current_set) > best_count:
            best_count = len(current_set)
            max_blocks_found = list(current_set)
            if best_count >= 6: # We are happy with 6
                return True
        if not available_indices:
            return False
            
        # Pick an index from available
        # Heuristic: pick one with fewest conflicts among available?
        # Or just pick the first one.
        idx = available_indices[0]
        rest = available_indices[1:]
        
        # Option 1: Include idx
        new_available = [i for i in rest if not conflicts[idx][i]]
        if find_max_packing(current_set + [idx], new_available):
            return True
            
        # Option 2: Exclude idx
        if find_max_packing(current_set, rest):
            return True
            
        return False

    # Initialize available indices
    all_indices = list(range(N_blocks))
    find_max_packing([], all_indices)
    
    # Use the best set found
    blocks = [possible_blocks[i] for i in max_blocks_found]
    
    # Calculate covered terms
    covered_3d = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered_3d[i0+di][j0+dj][k0+dk] = True
    
    uncovered = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered_3d[i][j][k]:
                    uncovered.append((i, j, k))
    
    total_rank = len(blocks) * 7 + len(uncovered)
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
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

    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    for (i, j, k) in uncovered:
        if time.time() > deadline:
            break
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
