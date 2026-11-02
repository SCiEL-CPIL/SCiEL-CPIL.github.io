"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Hybrid approach:
    # 1. Apply Strassen's algorithm to disjoint 2x2x2 sub-blocks where possible.
    # 2. Use schoolbook multiplication for the remaining terms.
    #
    # Dimensions: 3x5 by 5x5
    # i in {0, 1, 2}, j in {0, 1, 2, 3, 4}, k in {0, 1, 2, 3, 4}
    #
    # We can form 2x2x2 blocks using i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # Since n=3, we can only use i=0,1 for Strassen blocks (i=2 must be schoolbook).
    # For j and k (size 5), we can fit two disjoint 2x2 blocks (indices 0,1 and 2,3).
    # Index 4 in j and k must be schoolbook.
    #
    # Possible Strassen blocks (i0=0, j0 in {0,2}, k0 in {0,2}):
    # 1. i={0,1}, j={0,1}, k={0,1}
    # 2. i={0,1}, j={2,3}, k={0,1}
    # 3. i={0,1}, j={0,1}, k={2,3}
    # 4. i={0,1}, j={2,3}, k={2,3}
    #
    # Total 4 blocks. Each block has rank 7 (Strassen) vs 8 (schoolbook).
    # Savings: 4 * (8 - 7) = 4.
    # Base rank: 3 * 5 * 5 = 75.
    # New rank: 75 - 4 = 71.
    
    blocks = [
        (0, 0, 0),  # i=0, j=0, k=0
        (0, 2, 0),  # i=0, j=2, k=0
        (0, 0, 2),  # i=0, j=0, k=2
        (0, 2, 2),  # i=0, j=2, k=2
    ]
    
    num_blocks = len(blocks)
    strassen_rank = num_blocks * 7
    # Count schoolbook terms:
    # Total terms = 75
    # Terms covered by Strassen = 4 * 8 = 32
    # Remaining terms = 75 - 32 = 43
    schoolbook_rank = 75 - 32
    
    total_rank = strassen_rank + schoolbook_rank
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block at offsets (i0, j0, k0)
    def add_strassen_block(i0, j0, k0, start_col):
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        c = start_col + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M2 = (A10+A11)B00
        c = start_col + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        # M3 = A00(B01-B11)
        c = start_col + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M4 = A11(B10-B00)
        c = start_col + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        # M5 = (A00+A01)B11
        c = start_col + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        # M6 = (A10-A00)(B00+B01)
        c = start_col + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        # M7 = (A01-A11)(B10+B11)
        c = start_col + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        
        return start_col + 7
    
    # Add Strassen blocks
    for (i0, j0, k0) in blocks:
        col = add_strassen_block(i0, j0, k0, col)
    
    # Collect residual terms (not covered by any Strassen block)
    residual_terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                covered = False
                for (i0, j0, k0) in blocks:
                    if i in (i0, i0+1) and j in (j0, j0+1) and k in (k0, k0+1):
                        covered = True
                        break
                if not covered:
                    residual_terms.append((i, j, k))
    
    # Try to find a lower-rank decomposition of the residual tensor
    # using a time-bounded numerical search. The residual is a 3x5x5 tensor
    # with specific sparsity. We attempt to find if any subset of residual
    # terms can be expressed with fewer rank-1 factors by sharing structure.
    
    # For now, use schoolbook for all residual terms as baseline.
    # Attempt: check if the i=2 slice (1x5x5) can be decomposed with rank < 25
    # by exploiting that it's a "slice" — but in general it needs rank 25.
    
    # Instead, try a small local search: can we merge any two residual rank-1
    # terms into a single rank-1 term? This requires their U, V, W supports
    # to be compatible. Since all residual terms are distinct (i,j,k) triples,
    # no two share the same (i,j,k), so direct merging is impossible.
    
    # Alternative: use a 3x3x3 Bini-Capovani-Lotti decomposition on a
    # 3x3x3 sub-block and compare total rank.
    
    # Check time
    elapsed = time.time() - START_TIME
    if elapsed < TIME_LIMIT - SAFETY_MARGIN:
        # Try 3x3x3 BCL block for i={0,1,2}, j={0,1,2}, k={0,1,2}
        # BCL rank-23 decomposition of 3x3x3
        # This is complex; let's try a simpler improvement first:
        # Use a different set of Strassen blocks that might allow
        # better residual packing.
        
        # Try: what if we use blocks at (0,0,0), (0,2,0), (0,0,2), (0,2,2)
        # but also try (0,1,1) as an alternative? No, must be disjoint.
        
        # The current partition is optimal for disjoint 2x2x2 blocks.
        # Let's just use schoolbook for residual.
        pass
    
    # Add schoolbook terms for residual
    for (i, j, k) in residual_terms:
        U[i*m + j][col] += 1
        V[j*p + k][col] += 1
        W[k*n + i][col] += 1
        col += 1
    
    # Trim unused columns (shouldn't happen, but safe)
    if col < total_rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
