"""Program to find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication."""
import time

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    n, m, p = DIMENSIONS
    
    # For 3x5x7 matrix multiplication:
    # A is 3x5, B is 5x7, C = A*B is 3x7
    # U shape: (n*m, rank) = (15, rank)
    # V shape: (m*p, rank) = (35, rank) 
    # W shape: (p*n, rank) = (21, rank)
    # W[k*n+i, c] convention: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # We'll use a Strassen-based decomposition for 2x2x2 blocks.
    # The idea: partition the matrices into 2x2x2 sub-blocks where possible.
    # 
    # For 3x5x7, we can try to identify disjoint 2x2x2 blocks:
    # i in {0,1,2}, j in {0,1,2,3,4}, k in {0,1,2,3,4,5,6}
    # 
    # A 2x2x2 block needs: 2 rows of A, 2 cols of A (and 2 rows of B), 2 cols of B (and 2 cols of C)
    # 
    # Let's think about this differently. The tensor has indices (i, j, k) with i in [0,3), j in [0,5), k in [0,7).
    # Total terms: 3*5*7 = 105.
    # 
    # We want to replace some of these 105 rank-1 terms with Strassen 2x2x2 decompositions.
    # Each 2x2x2 Strassen decomposition uses 7 terms instead of 8, saving 1.
    #
    # Strategy: Find disjoint sets of 2x2x2 index triples that can each be replaced by a Strassen block.
    # A "2x2x2 block" consists of 8 terms: (i0,j0,k0), (i0,j0,k1), (i0,j1,k0), (i0,j1,k1), 
    #                                          (i1,j0,k0), (i1,j0,k1), (i1,j1,k0), (i1,j1,k1)
    # where i0!=i1, j0!=j1, k0!=k1.
    #
    # For these 8 terms to form a valid Strassen block, we need them to be "disjoint" in the sense
    # that no two different Strassen blocks share any (i,j,k) triple.
    
    # Let's find a maximum set of disjoint 2x2x2 blocks.
    # We can use a greedy approach or try specific patterns.
    
    # One pattern: use i pairs (0,1), j pairs (0,1) and (2,3), k pairs (0,1) and (4,5)
    # But we need to be careful about overlaps.
    
    # Let me define a function to check if a set of 2x2x2 blocks are disjoint.
    # Each block is defined by (i0,i1,j0,j1,k0,k1) and covers 8 triples.
    
    # Let's try to construct a set of disjoint blocks manually.
    # Available i: {0,1,2}, j: {0,1,2,3,4}, k: {0,1,2,3,4,5,6}
    
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={2,3}, k={2,3}
    # Block 3: i={0,1}, j={4,?} - but j only goes to 4, so we can't form a pair with 4 unless we reuse
    # Actually j={0,1,2,3,4} has 5 elements, so we can form at most 2 disjoint pairs: {0,1} and {2,3}, leaving 4.
    
    # Let's try:
    # Block 1: i={0,1}, j={0,1}, k={0,1}
    # Block 2: i={0,1}, j={2,3}, k={2,3}
    # Block 3: i={2,?} - i={0,1,2}, so we can't form another pair with i=2 without reusing.
    # Hmm, i has only 3 elements. So we can form at most 1 disjoint i-pair: {0,1}, leaving 2.
    
    # So with i={0,1} for all blocks, we can have multiple blocks as long as j and k pairs are disjoint.
    # j has 5 elements: can form 2 disjoint pairs {0,1},{2,3} (leaving 4)
    # k has 7 elements: can form 3 disjoint pairs {0,1},{2,3},{4,5} (leaving 6)
    
    # So we can have up to min(2, 3) = 2 blocks with i={0,1}.
    # But we could also use i={1,2} for some blocks.
    
    # Let's try to maximize the number of disjoint blocks.
    # Each block uses 2 i's, 2 j's, 2 k's.
    # Total i capacity: 3, so floor(3/2) = 1 full pair, 1 leftover.
    # But we can reuse i's across different blocks as long as the (i,j,k) triples are disjoint.
    
    # Actually, two blocks can share an i value as long as their (i,j,k) triples don't overlap.
    # E.g., Block1 uses i=0 with j={0,1},k={0,1}, and Block2 uses i=0 with j={2,3},k={2,3}.
    # These are disjoint in terms of triples.
    
    # So the constraint is on the (i,j,k) triples, not on individual indices.
    # Each block covers 8 specific (i,j,k) triples. Two blocks are compatible if their 8 triples are disjoint.
    
    # Let's define blocks and check for disjointness.
    
    def block_triples(i0, i1, j0, j1, k0, k1):
        return set()
    
    def make_block(i0, i1, j0, j1, k0, k1):
        triples = set()
        for ii in [i0, i1]:
            for jj in [j0, j1]:
                for kk in [k0, k1]:
                    triples.add((ii, jj, kk))
        return triples
    
    # Try to find a large set of disjoint blocks.
    # We'll use a greedy approach: generate candidate blocks and add them if disjoint.
    
    candidate_blocks = []
    for i0 in range(n):
        for i1 in range(i0+1, n):
            for j0 in range(m):
                for j1 in range(j0+1, m):
                    for k0 in range(p):
                        for k1 in range(k0+1, p):
                            candidate_blocks.append((i0, i1, j0, j1, k0, k1))
    
    # Sort candidates to prefer certain patterns
    # Let's just try greedy: pick blocks that cover triples not yet covered
    used_triples = set()
    selected_blocks = []
    
    for block in candidate_blocks:
        i0, i1, j0, j1, k0, k1 = block
        triples = make_block(i0, i1, j0, j1, k0, k1)
        if triples.isdisjoint(used_triples):
            selected_blocks.append(block)
            used_triples.update(triples)
            if time.time() > deadline - 30:
                break
    
    # For each selected block, we'll use Strassen decomposition (7 terms) instead of schoolbook (8 terms)
    # For remaining triples, use schoolbook (1 term each)
    
    # Let's identify which triples are covered by Strassen blocks
    strassen_triples = set()
    for block in selected_blocks:
        i0, i1, j0, j1, k0, k1 = block
        for ii in [i0, i1]:
            for jj in [j0, j1]:
                for kk in [k0, k1]:
                    strassen_triples.add((ii, jj, kk))
    
    # Now build U, V, W
    # First, assign column indices
    # Strassen blocks: 7 columns each
    # Remaining triples: 1 column each
    
    rank = 0
    block_col_start = {}
    for idx, block in enumerate(selected_blocks):
        block_col_start[block] = rank
        rank += 7
    
    remaining_triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_triples:
                    remaining_triples.append((i, j, k))
    
    triple_col_start = {}
    for idx, triple in enumerate(remaining_triples):
        triple_col_start[triple] = rank
        rank += 1
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill Strassen blocks
    # Strassen 2x2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A11 - A00)(B00 + B01)
    # M7 = (A00 - A10)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # Local indices: A has rows i0,i1 and cols j0,j1
    #   A00 = A[i0,j0], A01 = A[i0,j1], A10 = A[i1,j0], A11 = A[i1,j1]
    # B has rows j0,j1 and cols k0,k1
    #   B00 = B[j0,k0], B01 = B[j0,k1], B10 = B[j1,k0], B11 = B[j1,k1]
    # C has rows i0,i1 and cols k0,k1
    #   C00 = C[i0,k0], C01 = C[i0,k1], C10 = C[i1,k0], C11 = C[i1,k1]
    #
    # U vector for term t: U[i*m+j, col] = coefficient of A[i,j] in the A-factor of Mt
    # V vector for term t: V[j*p+k, col] = coefficient of B[j,k] in the B-factor of Mt
    # W vector for term t: W[k*n+i, col] = coefficient of C[i,k] in the C-contribution of Mt
    
    for block in selected_blocks:
        i0, i1, j0, j1, k0, k1 = block
        base_col = block_col_start[block]
        
        # M1: A-factor = A00 + A11, B-factor = B00 + B11
        # C-contribution: C00 += 1, C11 += 1
        U[i0*m + j0, base_col + 0] += 1
        U[i1*m + j1, base_col + 0] += 1
        V[j0*p + k0, base_col + 0] += 1
        V[j1*p + k1, base_col + 0] += 1
        W[k0*n + i0, base_col + 0] += 1  # C00
        W[k1*n + i1, base_col + 0] += 1  # C11
        
        # M2: A-factor = A10 + A11, B-factor = B00
        # C-contribution: C10 += 1, C11 += -1
        U[i1*m + j0, base_col + 1] += 1
        U[i1*m + j1, base_col + 1] += 1
        V[j0*p + k0, base_col + 1] += 1
        W[k0*n + i1, base_col + 1] += 1   # C10
        W[k1*n + i1, base_col + 1] += -1  # C11
        
        # M3: A-factor = A00, B-factor = B01 - B11
        # C-contribution: C01 += 1, C11 += 1
        U[i0*m + j0, base_col + 2] += 1
        V[j0*p + k1, base_col + 2] += 1
        V[j1*p + k1, base_col + 2] += -1
        W[k1*n + i0, base_col + 2] += 1   # C01
        W[k1*n + i1, base_col + 2] += 1   # C11
        
        # M4: A-factor = A11, B-factor = B10 - B00
        # C-contribution: C00 += 1, C10 += 1
        U[i1*m + j1, base_col + 3] += 1
        V[j1*p + k0, base_col + 3] += 1
        V[j0*p + k0, base_col + 3] += -1
        W[k0*n + i0, base_col + 3] += 1   # C00
        W[k0*n + i1, base_col + 3] += 1   # C10
        
        # M5: A-factor = A00 + A10, B-factor = B11
        # C-contribution: C00 += -1, C01 += 1
        U[i0*m + j0, base_col + 4] += 1
        U[i1*m + j0, base_col + 4] += 1
        V[j1*p + k1, base_col + 4] += 1
        W[k0*n + i0, base_col + 4] += -1  # C00
        W[k1*n + i0, base_col + 4] += 1   # C01
        
        # M6: A-factor = A11 - A00, B-factor = B00 + B01
        # C-contribution: C11 += 1
        U[i1*m + j1, base_col + 5] += 1
        U[i0*m + j0, base_col + 5] += -1
        V[j0*p + k0, base_col + 5] += 1
        V[j0*p + k1, base_col + 5] += 1
        W[k1*n + i1, base_col + 5] += 1   # C11
        
        # M7: A-factor = A00 - A10, B-factor = B10 + B11
        # C-contribution: C00 += 1
        U[i0*m + j0, base_col + 6] += 1
        U[i1*m + j0, base_col + 6] += -1
        V[j1*p + k0, base_col + 6] += 1
        V[j1*p + k1, base_col + 6] += 1
        W[k0*n + i0, base_col + 6] += 1   # C00
    
    # Fill remaining triples with schoolbook
    for triple in remaining_triples:
        i, j, k = triple
        col = triple_col_start[triple]
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
