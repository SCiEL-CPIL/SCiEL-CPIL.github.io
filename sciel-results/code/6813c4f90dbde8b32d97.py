"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Start with schoolbook rank 75, then apply Strassen-style
    # block decomposition on 2x2x2 sub-structures to reduce rank.
    # Each 2x2x2 schoolbook block has rank 8; Strassen reduces to rank 7.
    # We need to find non-overlapping 2x2x2 blocks to apply Strassen to.
    
    # A Strassen block operates on:
    # - 2 rows of A (i in {i0, i0+1})
    # - 2 columns of A / 2 rows of B (j in {j0, j0+1})  
    # - 2 columns of B (k in {k0, k0+1})
    
    # The 8 schoolbook terms are:
    # (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1),
    # (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    
    # Strassen replaces these with 7 terms using:
    # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
    # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
    # M3 = A[i0,j0]*(B[j1,k0]-B[j1,k1])
    # M4 = A[i1,j1]*(B[j0,k1]-B[j0,k0])
    # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
    # M6 = (A[i0,j1]-A[i1,j1])*(B[j0,k0]+B[j0,k1])
    # M7 = (A[i1,j0]-A[i0,j0])*(B[j1,k0]+B[j1,k1])
    
    # For each Strassen block, we need to track which schoolbook terms it covers
    # and ensure no two Strassen blocks cover the same schoolbook term.
    
    # Enumerate all possible 2x2x2 blocks:
    # i0 in {0, 1} (since n=3, i0 can be 0 or 1)
    # j0 in {0, 1, 2, 3} (since m=5, j0 can be 0-3)
    # k0 in {0, 1, 2, 3} (since p=5, k0 can be 0-3)
    
    all_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # This block covers 8 schoolbook terms
                terms = set()
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            i, j, k = i0 + di, j0 + dj, k0 + dk
                            terms.add((i, j, k))
                all_blocks.append((i0, j0, k0, terms))
    
    # Find maximum set of non-overlapping blocks using greedy + local search
    # First, try greedy by size (all same size here), then local search
    
    best_rank = n * m * p  # 75
    best_block_set = set()
    
    # Greedy: pick blocks that cover the most uncovered terms
    # Since all blocks cover 8 terms, just pick non-overlapping ones
    # Use a simple greedy: sort by some heuristic, pick if no overlap
    
    # Try multiple random orderings for greedy
    import random
    random.seed(42)
    
    best_count = 0
    for trial in range(100):
        indices = list(range(len(all_blocks)))
        random.shuffle(indices)
        
        chosen = []
        covered = set()
        
        for idx in indices:
            i0, j0, k0, terms = all_blocks[idx]
            if not terms & covered:
                chosen.append(idx)
                covered |= terms
        
        if len(chosen) > best_count:
            best_count = len(chosen)
            best_block_set = set(chosen)
    
    # Local search: try swapping one block out and adding two in
    improved = True
    while improved:
        improved = False
        current_set = set(best_block_set)
        current_covered = set()
        for idx in current_set:
            current_covered |= all_blocks[idx][3]
        
        # Try removing one block and adding up to two others
        for remove_idx in list(current_set):
            new_set = current_set - {remove_idx}
            new_covered = set()
            for idx in new_set:
                new_covered |= all_blocks[idx][3]
            
            # Find blocks that don't overlap with new_covered
            candidates = []
            for idx in range(len(all_blocks)):
                if idx not in new_set:
                    if not all_blocks[idx][3] & new_covered:
                        candidates.append(idx)
            
            # Try adding 1 or 2 candidates
            for add1 in candidates:
                if len(new_set) + 1 > len(current_set):
                    test_set = new_set | {add1}
                    test_covered = new_covered | all_blocks[add1][3]
                    # Check if we can add another
                    for add2 in candidates:
                        if add2 != add1 and not all_blocks[add2][3] & test_covered:
                            if len(test_set) + 1 > len(current_set):
                                current_set = test_set | {add2}
                                current_covered = test_covered | all_blocks[add2][3]
                                improved = True
                    if len(new_set) + 1 > len(current_set):
                        current_set = new_set | {add1}
                        current_covered = new_covered | all_blocks[add1][3]
                        improved = True
        
        best_block_set = current_set
    
    # Number of Strassen blocks applied
    num_strassen = len(best_block_set)
    rank = n * m * p - num_strassen  # Each Strassen saves 1 rank
    
    # Build the decomposition
    # Start with schoolbook, then for each Strassen block, replace 8 terms with 7
    
    # Schoolbook mapping: (i,j,k) -> column index
    schoolbook_col = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_col[(i, j, k)] = col
                col += 1
    
    # For each Strassen block, we'll create 7 new columns
    # First, mark which schoolbook columns are "replaced"
    replaced_terms = set()
    strassen_blocks_info = []
    
    for idx in best_block_set:
        i0, j0, k0, terms = all_blocks[idx]
        strassen_blocks_info.append((i0, j0, k0, terms))
        replaced_terms |= terms
    
    # The final rank: unreplaced schoolbook terms + 7 per Strassen block
    num_unreplaced = n * m * p - len(replaced_terms)
    final_rank = num_unreplaced + 7 * num_strassen
    
    # Assign column indices
    # Unreplaced schoolbook terms get columns 0..num_unreplaced-1
    # Each Strassen block gets 7 columns starting after that
    
    col_assign = {}
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in replaced_terms:
                    col_assign[(i, j, k)] = col
                    col += 1
    
    # Strassen block columns
    strassen_cols = {}
    for bi, (i0, j0, k0, terms) in enumerate(strassen_blocks_info):
        base_col = col
        # 7 Strassen terms
        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        # M3 = A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # M4 = A[i1,j1]*(B[j0,k1]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # M6 = (A[i0,j1]-A[i1,j1])*(B[j0,k0]+B[j0,k1])
        # M7 = (A[i1,j0]-A[i0,j0])*(B[j1,k0]+B[j1,k1])
        
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        
        strassen_cols[bi] = {
            'M1': base_col,
            'M2': base_col + 1,
            'M3': base_col + 2,
            'M4': base_col + 3,
            'M5': base_col + 4,
            'M6': base_col + 5,
            'M7': base_col + 6,
        }
        col += 7
    
    # Build U, V, W matrices
    U = [[0] * final_rank for _ in range(n * m)]
    V = [[0] * final_rank for _ in range(m * p)]
    W = [[0] * final_rank for _ in range(p * n)]
    
    # Fill in unreplaced schoolbook terms
    for (i, j, k), c in col_assign.items():
        U[i * m + j][c] = 1
        V[j * p + k][c] = 1
        W[k * n + i][c] = 1
    
    # Fill in Strassen blocks
    for bi, (i0, j0, k0, terms) in enumerate(strassen_blocks_info):
        i1, j1, k1 = i0 + 1, j0 + 1, k0 + 1
        cols = strassen_cols[bi]
        
        # M1 = (A[i0,j0]+A[i1,j1])*(B[j0,k0]+B[j1,k1])
        # Contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        c = cols['M1']
        U[i0 * m + j0][c] = 1
        U[i1 * m + j1][c] = 1
        V[j0 * p + k0][c] = 1
        V[j1 * p + k1][c] = 1
        W[k0 * n + i0][c] = 1
        W[k1 * n + i0][c] = 1
        W[k0 * n + i1][c] = 1
        W[k1 * n + i1][c] = 1
        
        # M2 = (A[i1,j0]+A[i1,j1])*B[j0,k0]
        # Contributes to C[i1,k0]
        c = cols['M2']
        U[i1 * m + j0][c] = 1
        U[i1 * m + j1][c] = 1
        V[j0 * p + k0][c] = 1
        W[k0 * n + i1][c] = 1
        
        # M3 = A[i0,j0]*(B[j1,k0]-B[j1,k1])
        # Contributes to C[i0,k0] (+), C[i0,k1] (-)
        c = cols['M3']
        U[i0 * m + j0][c] = 1
        V[j1 * p + k0][c] = 1
        V[j1 * p + k1][c] = -1
        W[k0 * n + i0][c] = 1
        W[k1 * n + i0][c] = -1
        
        # M4 = A[i1,j1]*(B[j0,k1]-B[j0,k0])
        # Contributes to C[i1,k1] (+), C[i1,k0] (-)
        c = cols['M4']
        U[i1 * m + j1][c] = 1
        V[j0 * p + k1][c] = 1
        V[j0 * p + k0][c] = -1
        W[k1 * n + i1][c] = 1
        W[k0 * n + i1][c] = -1
        
        # M5 = (A[i0,j0]+A[i0,j1])*B[j1,k1]
        # Contributes to C[i0,k1]
        c = cols['M5']
        U[i0 * m + j0][c] = 1
        U[i0 * m + j1][c] = 1
        V[j1 * p + k1][c] = 1
        W[k1 * n + i0][c] = 1
        
        # M6 = (A[i0,j1]-A[i1,j1])*(B[j0,k0]+B[j0,k1])
        # Contributes to C[i0,k0] (+), C[i0,k1] (+), C[i1,k0] (-), C[i1,k1] (-)
        c = cols['M6']
        U[i0 * m + j1][c] = 1
        U[i1 * m + j1][c] = -1
        V[j0 * p + k0][c] = 1
        V[j0 * p + k1][c] = 1
        W[k0 * n + i0][c] = 1
        W[k1 * n + i0][c] = 1
        W[k0 * n + i1][c] = -1
        W[k1 * n + i1][c] = -1
        
        # M7 = (A[i1,j0]-A[i0,j0])*(B[j1,k0]+B[j1,k1])
        # Contributes to C[i1,k0] (+), C[i1,k1] (+), C[i0,k0] (-), C[i0,k1] (-)
        c = cols['M7']
        U[i1 * m + j0][c] = 1
        U[i0 * m + j0][c] = -1
        V[j1 * p + k0][c] = 1
        V[j1 * p + k1][c] = 1
        W[k0 * n + i1][c] = 1
        W[k1 * n + i1][c] = 1
        W[k0 * n + i0][c] = -1
        W[k1 * n + i0][c] = -1
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        # Check: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        # (since T[i,j,k] = 1 for all i,j,k in the multiplication tensor)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        u = U[i * m + j][c]
                        v = V[j * p + k][c]
                        w = W[k * n + i][c]
                        if u != 0 and v != 0 and w != 0:
                            total += u * v * w
                    if total != 1:
                        return False, (i, j, k, total)
        return True, None
    
    is_valid, err = verify(U, V, W, n, m, p, final_rank)
    
    if not is_valid:
        # Fallback to schoolbook
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
        final_rank = rank
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
