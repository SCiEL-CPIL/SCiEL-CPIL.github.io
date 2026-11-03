import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 3, 5, 7

    # --- Strassen Block Definition ---
    # Returns lists of U, V, W vectors (as dicts of index: value) for a 2x2x2 block.
    # Indices are local (0,1) for i, j, k.
    def get_strassen_terms():
        terms = []
        # M1 = (A00+A11)(B00+B11)
        terms.append(({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}))
        # M2 = (A10+A11)B00
        terms.append(({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}))
        # M3 = A00(B01-B11)
        terms.append(({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}))
        # M4 = A11(B10-B00)
        terms.append(({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}))
        # M5 = (A00+A01)B11
        terms.append(({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}))
        # M6 = (A10-A00)(B00+B01)
        terms.append(({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}))
        # M7 = (A01-A11)(B10+B11)
        terms.append(({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}))
        return terms

    strassen_terms = get_strassen_terms()

    # --- Greedy Packing of Strassen Blocks ---
    # We want to place as many 2x2x2 blocks as possible.
    # A block is defined by (i0, j0, k0) where it covers i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}.
    # Since n=3, i0 can be 0 or 1.
    # Since m=5, j0 can be 0..3.
    # Since p=7, k0 can be 0..5.
    
    # We will greedily pick blocks that cover the most uncovered tensor entries.
    # Tensor entry is (i, j, k).
    # Total entries: 105.
    # Each block covers 8 entries.
    # Rank cost: 7 per block. Schoolbook cost: 1 per entry.
    # Saving: 1 per block.
    
    # We need to ensure blocks don't overlap in a way that causes conflicts?
    # No, Strassen blocks can overlap in indices, but we are summing the contributions.
    # Wait, if two blocks overlap in an (i,j,k) entry, the coefficients must sum to 1.
    # In a generic decomposition, we assume the tensor entries are 1.
    # If we use multiple Strassen blocks, we are effectively using a "block Strassen" approach.
    # This requires that the sub-matrices are independent.
    # If we overlap blocks, the linear combinations mix.
    # The standard "block Strassen" works by partitioning the matrix into blocks.
    # If we partition into 2x2 blocks, they are disjoint.
    # So we should only use disjoint blocks to ensure the decomposition is valid without complex interaction terms.
    # Disjoint means the sets of (i,j,k) indices covered are disjoint.
    
    covered = set()
    selected_blocks = []
    
    # Generate all possible block positions
    possible_blocks = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_blocks.append((i0, j0, k0))
                
    # Greedy selection
    # Sort by potential coverage? Or just iterate and pick best?
    # Since all blocks cover 8 entries, and they are disjoint if we enforce it,
    # the order doesn't matter for the count, only for which specific blocks we get.
    # However, we want to maximize the number of blocks.
    # This is a maximum packing problem.
    # For 3x5x7, can we pack more than 6?
    # n=3: only 1 layer of 2x2 in i (either 0,1 or 1,2).
    # If we use i0=0, we use i={0,1}. i=2 is left for schoolbook.
    # If we use i0=1, we use i={1,2}. i=0 is left for schoolbook.
    # Can we mix? No, because a block needs i0 and i0+1.
    # If we use i0=0 for some blocks and i0=1 for others, the i=1 row is shared.
    # The Strassen decomposition relies on the sub-matrices being independent.
    # If we overlap in i, the A sub-matrices share rows.
    # A00 for block (0, j0, k0) is A[0:2, j0:j0+2].
    # A10 for block (1, j1, k1) is A[1:3, j1:j1+2].
    # These share A[1, ...].
    # The linear combinations in U, V, W would interfere.
    # Therefore, to maintain the simple additive structure of the rank, we should probably stick to disjoint blocks in all dimensions.
    # Or, we can verify the final decomposition numerically.
    # Let's try to pack disjoint blocks first.
    
    # Disjoint in i: we can only have blocks with i0=0 OR i0=1. Not both, if we want strict disjointness in i.
    # Actually, if we have block at (0, j0, k0) and (1, j1, k1), they are disjoint in i if {0,1} and {1,2} are disjoint? No, they share 1.
    # So we must choose either i0=0 for all blocks or i0=1 for all blocks to have disjoint i-projections?
    # No, if we use i0=0 for one block and i0=1 for another, they share i=1.
    # The tensor entry (1, j, k) would be touched by both.
    # Block 1 contributes to (1, j, k) via its M terms.
    # Block 2 contributes to (1, j, k) via its M terms.
    # The sum must be 1.
    # This is possible if the contributions cancel or add up correctly.
    # However, Strassen's algorithm is for independent sub-blocks.
    # Let's assume we can only use disjoint blocks to be safe.
    # If we restrict to i0=0, we have 1x5x7 space for j,k.
    # We can pack 2x2 blocks in 5x7.
    # Max disjoint 2x2 in 5x7:
    # 5 rows, 7 cols.
    # We can fit 2 rows of 2x2 blocks?
    # Row 0,1: cols 0,1; 2,3; 4,5. (3 blocks). Cols 6 left.
    # Row 2,3: cols 0,1; 2,3; 4,5. (3 blocks). Cols 6 left.
    # Row 4: no pair.
    # Total 6 blocks.
    # This matches the seed program's 6 blocks.
    
    # Can we do better?
    # What if we don't restrict to disjoint i?
    # Let's try a more general approach:
    # Build the full system and solve for the minimum rank?
    # That's hard.
    
    # Let's stick to the disjoint packing for now, but verify if we can mix.
    # Actually, let's just use the greedy disjoint packing for i0=0 and i0=1 separately and pick the best?
    # Or just i0=0 is symmetric to i0=1.
    
    # Let's implement the disjoint packing for a fixed i0.
    def pack_disjoint_blocks(i0):
        jk_covered = [[False for _ in range(p)] for _ in range(m)]
        blocks = []
        # Greedy: pick blocks that cover most uncovered cells.
        # Since all are 2x2, just pick any valid one.
        # To maximize count, we should pack tightly.
        # Iterate j0, k0.
        # Better: iterate all possible, pick first valid, mark covered, repeat.
        # To ensure max count, we might need backtracking, but 5x7 is small.
        # Greedy usually works well for grid packing if we order carefully.
        # Let's try to pack as many as possible.
        
        # Sort candidates?
        # Just iterate j0 from 0 to m-2, k0 from 0 to p-2.
        # This might leave gaps.
        # Example: 5x7.
        # (0,0) covers (0,0),(0,1),(1,0),(1,1).
        # (0,2) covers (0,2),(0,3),(1,2),(1,3).
        # (0,4) covers (0,4),(0,5),(1,4),(1,5).
        # (2,0) covers (2,0),(2,1),(3,0),(3,1).
        # (2,2) covers (2,2),(2,3),(3,2),(3,3).
        # (2,4) covers (2,4),(2,5),(3,4),(3,5).
        # 6 blocks.
        # What about (1,0)? No, (1,0) overlaps with (0,0) in row 1.
        # So we can only use even j0 if we want to stack vertically?
        # If we pick (0,0), we cover rows 0,1.
        # Next available row is 2.
        # So (2,0) is valid.
        # This yields 3 blocks in rows 0,1 and 3 blocks in rows 2,3.
        # Total 6.
        
        # Is it possible to get 7?
        # 7 blocks * 4 cells = 28 cells.
        # 5*7 = 35 cells.
        # If we could pack 7, we'd cover 28/35.
        # But we need disjoint 2x2.
        # Max number of disjoint 2x2 in 5x7 is floor(5/2)*floor(7/2) = 2*3 = 6.
        # So 6 is the max for a fixed i0.
        
        # So we are capped at 6 Strassen blocks if we use disjoint packing.
        # Rank = 105 - 6 = 99.
        
        # Can we do better than 99?
        # Maybe by using non-disjoint blocks and verifying?
        # Or using a different algorithm?
        # The seed program achieves 99.
        # The inspiration 1 achieved 1.0714, which corresponds to rank 96?
        # 105 / 1.0714 = 98.0?
        # Score = 105 / Rank.
        # If score is 1.0714, Rank = 105 / 1.0714 = 98.
        # So rank 98 is possible.
        # How to get rank 98?
        # Saving 7 over schoolbook.
        # 6 Strassen blocks save 6.
        # We need 1 more saving.
        # Maybe a 3x3x3 block?
        # Rank of 3x3x3 is 23. Schoolbook 27. Saving 4.
        # If we use one 3x3x3 block, we save 4.
        # Remaining:
        # i: 0,1,2 covered.
        # j: 0,1,2 covered.
        # k: 0,1,2 covered.
        # Remaining j: 3,4 (2 cols).
        # Remaining k: 3,4,5,6 (4 cols).
        # We can put Strassen blocks in the remaining space?
        # The remaining space is not a full 3x5x7.
        # The tensor is the sum of the 3x3x3 block and the rest.
        # The rest is a "frame" around the 3x3x3 block.
        # Can we apply Strassen to the frame?
        # The frame consists of:
        # 1. i in {0,1,2}, j in {3,4}, k in {0,1,2}
        # 2. i in {0,1,2}, j in {0,1,2}, k in {3,4,5,6}
        # 3. i in {0,1,2}, j in {3,4}, k in {3,4,5,6}
        
        # This is getting complicated.
        # Let's try to implement the 3x3x3 block.
        # I need the 23 terms for 3x3x3 matrix multiplication.
        # This is a known result.
        # I will hardcode the 3x3x3 decomposition if I can find/generate it.
        # Generating it from scratch is risky.
        
        # Alternative: Can we improve the packing?
        # What if we use 5 Strassen blocks (saving 5) and 1 other trick?
        
        # Let's look at the "non-disjoint" idea again.
        # If we allow blocks to share indices, we might pack more "effective" blocks.
        # But the verification step will catch errors.
        # Let's try to pack 7 blocks by allowing overlap in i.
        # e.g. 3 blocks with i0=0 and 4 blocks with i0=1?
        # But they must be disjoint in (j,k) for each i-layer?
        # No, if they share i=1, they interact.
        
        # Let's stick to the verified 99 rank for now, but try to find a 98 rank solution.
        # One way to save 1 more rank:
        # If we have a 2x2x2 block, it saves 1.
        # If we have a 2x2x3 block?
        # Rank of 2x2x3?
        # Schoolbook: 12.
        # Strassen 2x2x2 + 2x2x1?
        # 2x2x2 saves 1 (rank 7 vs 8).
        # 2x2x1 is rank 4 (schoolbook).
        # Total 11. Saving 1.
        # So a 2x2x3 block saves 1.
        # A 2x2x2 block saves 1.
        
        # So the saving is 1 per 2x2 "slice" in the third dimension?
        # No, 2x2x2 is one block.
        
        # Let's try to construct a solution with rank 98.
        # We need to save 7.
        # 6 Strassen blocks save 6.
        # We need 1 more saving.
        # Can we find a linear dependency in the remaining schoolbook terms?
        # Unlikely.
        
        # What if we use 7 Strassen blocks?
        # Is it possible to pack 7 disjoint 2x2x2 blocks in 3x5x7?
        # We established max 6 for fixed i0.
        # What if we use i0=0 for 3 blocks and i0=1 for 4 blocks?
        # They must be disjoint in (j,k) for the shared i=1?
        # No, if they share i=1, they are not disjoint.
        
        # Let's try a different strategy:
        # Use the 3x3x3 block (saving 4).
        # Then use Strassen on the remaining parts.
        # Remaining parts:
        # j=3,4 with k=0,1,2 (3x2x3 sub-tensor? No, i=0,1,2)
        # j=0,1,2 with k=3,4,5,6
        # j=3,4 with k=3,4,5,6
        
        # Let's just implement the 3x3x3 block and see.
        # I will use a known 3x3x3 decomposition.
        # Reference: https://www.cs.cornell.edu/courses/cs6820/2004fa/lectures/strassen.pdf
        # Or generate it via search? No time.
        
        # Let's try to improve the packing to 7 blocks by relaxing disjointness in i.
        # We will build the matrices and verify.
        # If it fails, we fall back to 99.
        
        U_cols = []
        V_cols = []
        W_cols = []
        covered = set()
        
        def add_strassen_block(i0, j0, k0):
            # Add 7 terms for block at (i0, j0, k0)
            # Local indices 0,1 map to global i0, i0+1 etc.
            for u_local, v_local, w_local in strassen_terms:
                U_dict = {}
                V_dict = {}
                W_dict = {}
                for (li, lj), val in u_local.items():
                    U_dict[(i0 + li) * m + (j0 + lj)] = val
                for (lj, lk), val in v_local.items():
                    V_dict[(j0 + lj) * p + (k0 + lk)] = val
                for (lk, li), val in w_local.items():
                    W_dict[(k0 + lk) * n + (i0 + li)] = val
                U_cols.append(U_dict)
                V_cols.append(V_dict)
                W_cols.append(W_dict)
            
            # Mark covered
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))

        # Try to pack 7 blocks.
        # We will try to find 7 blocks that are "compatible".
        # Compatibility: The sum of contributions to each (i,j,k) must be 1.
        # For disjoint blocks, this is trivially satisfied.
        # For overlapping blocks, we need to check.
        
        # Let's try a specific configuration:
        # 3 blocks with i0=0: (0,0,0), (0,2,0), (0,4,0) -> covers j=0,1,2,3,4,5? No, j=0,1; 2,3; 4,5. j=6? m=5, so j=0..4.
        # m=5. j0 can be 0,1,2,3.
        # (0,0,0) covers j=0,1.
        # (0,2,0) covers j=2,3.
        # (0,4,0) covers j=4,5? No, j=4 is last. j0=3 covers j=3,4.
        # So (0,0,0), (0,2,0), (0,3,0)?
        # (0,0,0) covers (0,1).
        # (0,2,0) covers (2,3).
        # (0,3,0) covers (3,4). Overlaps with (0,2,0) at j=3.
        
        # Let's just use the 6 disjoint blocks and hope for the best.
        # Rank 99 is a solid baseline.
        
        # To get 98, we need a 3x3x3 block.
        # I will hardcode the 3x3x3 terms.
        # There are 23 terms.
        # This is too long to type out manually without error.
        
        # Alternative: Use the 6 Strassen blocks (rank 99) and try to remove 1 term via linear algebra?
        # If the 99 terms are linearly dependent, we can reduce rank.
        # But they are structured to be independent.
        
        # Let's try to pack 7 blocks with overlap and verify.
        # If verification fails, fall back to 6.
        
        # Strategy:
        # 1. Generate 6 disjoint blocks.
        # 2. Try to add a 7th block that overlaps.
        # 3. Verify.
        # 4. If fails, remove 7th and verify 6.
        
        # For simplicity and safety, I will stick to the 6 disjoint blocks.
        # This guarantees rank 99.
        
        # To improve to 98, I need a better algorithm.
        # Let's try to use the 3x3x3 block by searching for its terms?
        # No, that's too slow.
        
        # I will output the rank 99 solution.
        
        # Packing 6 disjoint blocks with i0=0.
        add_strassen_block(0, 0, 0)
        add_strassen_block(0, 0, 2)
        add_strassen_block(0, 0, 4)
        add_strassen_block(0, 2, 0)
        add_strassen_block(0, 2, 2)
        add_strassen_block(0, 2, 4)
        
        # Note: j=4 is covered by j0=2 (covers 2,3). j=4 is NOT covered by j0=2.
        # j0=2 covers j=2,3.
        # We need to cover j=4.
        # j0=3 covers j=3,4.
        # So (0,3,0) would cover j=3,4.
        # But (0,2,0) covers j=2,3. Overlap at j=3.
        # So we can't have both (0,2,0) and (0,3,0) if we want disjoint j.
        
        # Let's re-pack for m=5.
        # j=0,1: block at j0=0.
        # j=2,3: block at j0=2.
        # j=4: left over.
        # So we can have 2 blocks in j-dimension for i0=0.
        # k=0,1: k0=0.
        # k=2,3: k0=2.
        # k=4,5: k0=4.
        # k=6: left over.
        # So we can have 3 blocks in k-dimension.
        # Total blocks = 2 * 3 = 6.
        # This covers j=0,1,2,3 and k=0,1,2,3,4,5.
        # Uncovered: j=4, k=6.
        
        # Let's verify the packing.
        # Blocks:
        # (0,0,0), (0,0,2), (0,0,4)
        # (0,2,0), (0,2,2), (0,2,4)
        
        # This covers:
        # j in {0,1,2,3}, k in {0,1,2,3,4,5}.
        # i in {0,1}.
        
        # Remaining entries:
        # i=2 (all j,k)
        # i in {0,1}, j=4 (all k)
        # i in {0,1}, j in {0,1,2,3}, k=6
        
        # We will add schoolbook terms for these.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U_cols.append({i * m + j: 1})
                        V_cols.append({j * p + k: 1})
                        W_cols.append({k * n + i: 1})
        
        rank = len(U_cols)
        
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        
        for c in range(rank):
            for idx, val in U_cols[c].items():
                U[idx, c] = val
            for idx, val in V_cols[c].items():
                V[idx, c] = val
            for idx, val in W_cols[c].items():
                W[idx, c] = val
        
        # Verify
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        
        T_comp = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.sum((T_comp - T) ** 2)
        
        if error > 1e-9:
            # Fallback to schoolbook
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1.0
                        V[j * p + k, col] = 1.0
                        W[k * n + i, col] = 1.0
                        col += 1
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

