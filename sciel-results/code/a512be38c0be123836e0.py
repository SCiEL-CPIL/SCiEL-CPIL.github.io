"""Rewritten program using Strassen-based block decomposition for 3x4x8 matrix multiplication."""
import time

DIMENSIONS = [3, 4, 8]

def construct():
    """Construct a valid tensor decomposition for 3x4x8 matrix multiplication.
    
    Uses a block-partitioning strategy:
    - Split n=3 into [2, 1]
    - Split m=4 into [2, 2] 
    - Split p=8 into [4, 4]
    
    For the 2x2x4 block (n=2, m=2, p=4), use Strassen's algorithm extended to rectangular case.
    For the 1x2x4 block (n=1, m=2, p=4), use schoolbook.
    
    Total rank: 7 (Strassen 2x2x2) * 2 (for p=4, two 2x2x2 blocks) + 1*2*4 (schoolbook) = 14 + 8 = 22
    Wait, let me reconsider. For 2x2x4, we can do two independent 2x2x2 Strassen decompositions, 
    each of rank 7, so total rank 14 for that block.
    For 1x2x4, schoolbook gives 1*2*4 = 8.
    Total rank = 14 + 8 = 22.
    
    But wait, we need to be more careful. Let me think about this differently.
    
    Actually, let me use a simpler approach: use Strassen for 2x2x2 sub-blocks wherever possible,
    and schoolbook for the rest.
    
    The key insight is that for a 2x2x2 block, Strassen gives rank 7 instead of schoolbook rank 8.
    
    Let me partition as follows:
    - n=3: split into [2, 1]
    - m=4: split into [2, 2]
    - p=8: split into [2, 2, 2, 2]
    
    Then we have blocks:
    - (2,2,2) blocks: 2*2*4 = 16 blocks, each of rank 7 (Strassen) -> 16*7 = 112
    - (1,2,2) blocks: 1*2*4 = 8 blocks, each of rank 4 (schoolbook) -> 8*4 = 32
    Total: 144, which is worse than schoolbook 96.
    
    That's not good. Let me try a different approach.
    
    Actually, the simplest improvement is to just use Strassen for 2x2x2 sub-blocks when all three 
    dimensions are at least 2. Let me think about what gives the best improvement.
    
    For n=3, m=4, p=8:
    Schoolbook rank = 96
    
    If I can find any 2x2x2 sub-block, I can replace its 8 rank-1 terms with 7, saving 1 per block.
    
    How many disjoint 2x2x2 blocks can I fit?
    - n=3: can fit 1 full 2-row block (using rows 0,1), leaving row 2
    - m=4: can fit 2 full 2-col blocks (cols 0,1 and 2,3)
    - p=8: can fit 4 full 2-col blocks (cols 0,1 and 2,3 and 4,5 and 6,7)
    
    So I can fit 1*2*4 = 8 disjoint 2x2x2 blocks.
    Each saves 1 rank, so total saving = 8.
    New rank = 96 - 8 = 88.
    
    This matches the knowledge from the prompt: "deterministic rank-88 block decomposition using 
    Strassen's algorithm for 2x2x4 subblocks" - wait, it says 2x2x4, not 2x2x2.
    
    Let me re-read: "deterministic rank-88 block decomposition using Strassen's algorithm for 2x2x4 subblocks"
    
    Hmm, 2x2x4 doesn't make sense for Strassen. Strassen is for 2x2x2. Let me just implement the 
    straightforward approach: use Strassen for all possible 2x2x2 sub-blocks.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # Start with schoolbook rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We'll build the decomposition block by block.
    # For each 2x2x2 sub-block, use Strassen (rank 7) instead of schoolbook (rank 8).
    # For remaining blocks, use schoolbook.
    
    # Track which (i,j,k) triples have been covered by Strassen blocks
    covered = set()
    col = 0
    
    # Define Strassen's 7 multiplications for 2x2x2
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    #
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    #
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # For each possible 2x2x2 block starting at (i0, j0, k0)
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                # Check if this block is already covered
                block_triples = [(i0, j0, k0), (i0, j0, k0+1), (i0, j0+1, k0), (i0, j0+1, k0+1),
                                 (i0+1, j0, k0), (i0+1, j0, k0+1), (i0+1, j0+1, k0), (i0+1, j0+1, k0+1)]
                if any(t in covered for t in block_triples):
                    continue
                
                # All 8 triples in this block will be handled by Strassen
                # Mark them as covered
                for t in block_triples:
                    covered.add(t)
                
                # Add 7 rank-1 terms for Strassen
                # A indices: (i0,j0)=0, (i0,j0+1)=1, (i0+1,j0)=2, (i0+1,j0+1)=3
                # B indices: (j0,k0)=0, (j0,k0+1)=1, (j0+1,k0)=2, (j0+1,k0+1)=3
                # C indices: (k0,i0)=0, (k0+1,i0)=1, (k0,i0+1)=2, (k0+1,i0+1)=3
                
                # M1 = (a00+a11)(b00+b11) -> c00 += 1, c11 += 1
                # U: a00=1, a11=1
                # V: b00=1, b11=1
                # W: c00=1, c11=1
                U[i0 * m + j0][col] = 1
                U[(i0+1) * m + (j0+1)][col] = 1
                V[j0 * p + k0][col] = 1
                V[(j0+1) * p + (k0+1)][col] = 1
                W[k0 * n + i0][col] = 1
                W[(k0+1) * n + (i0+1)][col] = 1
                col += 1
                
                # M2 = (a10+a11)b00 -> c10 += 1, c11 -= 1
                # U: a10=1, a11=1
                # V: b00=1
                # W: c10=1, c11=-1
                U[(i0+1) * m + j0][col] = 1
                U[(i0+1) * m + (j0+1)][col] = 1
                V[j0 * p + k0][col] = 1
                W[k0 * n + (i0+1)][col] = 1
                W[(k0+1) * n + (i0+1)][col] = -1
                col += 1
                
                # M3 = a00(b10-b11) -> c01 += 1, c11 += 1
                # U: a00=1
                # V: b10=1, b11=-1
                # W: c01=1, c11=1
                U[i0 * m + j0][col] = 1
                V[(j0+1) * p + k0][col] = 1
                V[(j0+1) * p + (k0+1)][col] = -1
                W[(k0+1) * n + i0][col] = 1
                W[(k0+1) * n + (i0+1)][col] = 1
                col += 1
                
                # M4 = a11(b01-b00) -> c00 += 1, c10 += 1
                # U: a11=1
                # V: b01=1, b00=-1
                # W: c00=1, c10=1
                U[(i0+1) * m + (j0+1)][col] = 1
                V[j0 * p + (k0+1)][col] = 1
                V[j0 * p + k0][col] = -1
                W[k0 * n + i0][col] = 1
                W[k0 * n + (i0+1)][col] = 1
                col += 1
                
                # M5 = (a00+a01)b11 -> c00 -= 1, c01 += 1
                # U: a00=1, a01=1
                # V: b11=1
                # W: c00=-1, c01=1
                U[i0 * m + j0][col] = 1
                U[i0 * m + (j0+1)][col] = 1
                V[(j0+1) * p + (k0+1)][col] = 1
                W[k0 * n + i0][col] = -1
                W[(k0+1) * n + i0][col] = 1
                col += 1
                
                # M6 = (a10-a00)(b00+b10) -> c11 += 1
                # U: a10=1, a00=-1
                # V: b00=1, b10=1
                # W: c11=1
                U[(i0+1) * m + j0][col] = 1
                U[i0 * m + j0][col] = -1
                V[j0 * p + k0][col] = 1
                V[(j0+1) * p + k0][col] = 1
                W[(k0+1) * n + (i0+1)][col] = 1
                col += 1
                
                # M7 = (a01-a11)(b01+b11) -> c00 += 1
                # U: a01=1, a11=-1
                # V: b01=1, b11=1
                # W: c00=1
                U[i0 * m + (j0+1)][col] = 1
                U[(i0+1) * m + (j0+1)][col] = -1
                V[j0 * p + (k0+1)][col] = 1
                V[(j0+1) * p + (k0+1)][col] = 1
                W[k0 * n + i0][col] = 1
                col += 1
    
    # Now add schoolbook terms for all uncovered (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
