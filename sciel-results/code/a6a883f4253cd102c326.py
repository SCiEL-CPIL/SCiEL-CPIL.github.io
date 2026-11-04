"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition correctly computes matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _make_strassen_222():
    """
    Returns (U, V, W) for 2x2x2 matrix multiplication using Strassen's 7-term algorithm.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Strassen terms:
    # p1 = (a11+a22)(b11+b22)
    # p2 = (a21+a22)b11
    # p3 = a11(b12-b22)
    # p4 = a22(b21-b11)
    # p5 = (a11+a12)b22
    # p6 = (a21-a11)(b11+b12)
    # p7 = (a12-a22)(b21+b22)
    
    # U columns (a coefficients)
    U[:, 0] = [1, 0, 1, 1]  # a11+a22
    U[:, 1] = [0, 0, 1, 1]  # a21+a22
    U[:, 2] = [1, 0, 0, 0]  # a11
    U[:, 3] = [0, 0, 0, 1]  # a22
    U[:, 4] = [1, 1, 0, 0]  # a11+a12
    U[:, 5] = [-1, 1, 0, 0] # a21-a11
    U[:, 6] = [0, 1, -1, 0] # a12-a22
    
    # V columns (b coefficients)
    V[:, 0] = [1, 1, 1, 1]  # b11+b22
    V[:, 1] = [1, 0, 0, 0]  # b11
    V[:, 2] = [0, 1, 0, -1] # b12-b22
    V[:, 3] = [-1, 0, 1, 0] # b21-b11
    V[:, 4] = [0, 0, 0, 1]  # b22
    V[:, 5] = [1, 1, 0, 0]  # b11+b12
    V[:, 6] = [0, 0, 1, 1]  # b21+b22
    
    # W columns (c coefficients)
    # c11 = p1 - p4 - p5 + p7
    # c12 = p3 + p5
    # c21 = p2 + p4
    # c22 = p1 - p2 + p3 + p6
    W[:, 0] = [1, 0, 0, -1, -1, 0, 1]   # c11
    W[:, 1] = [0, 0, 1, 0, 1, 0, 0]     # c12
    W[:, 2] = [0, 1, 0, 1, 0, 0, 0]     # c21
    W[:, 3] = [1, -1, 1, 0, 0, 1, 0]    # c22
    
    return U, V, W

def _make_strassen_233():
    """
    Returns (U, V, W) for 2x3x3 matrix multiplication using an 18-term algorithm.
    This is a known optimal decomposition for 2x3x3.
    U shape: (6, 18), V shape: (9, 18), W shape: (6, 18)
    Indexing: U[i*3+j], V[j*3+k], W[k*2+i]
    """
    # For brevity and reliability, we will construct this via Kronecker of smaller blocks
    # if a direct 18-term formula is not immediately recalled, but 18 is the known rank.
    # Actually, let's use a simpler approach: if we don't have a hardcoded 2x3x3,
    # we can fall back to tiling 2x3x3 into 2x2x2 + 2x1x3 or similar.
    # However, to keep the code robust and avoid bugs in complex manual formulas,
    # we will implement a generic tiling strategy that only uses 2x2x2 (Strassen) and 1x1x1 (Schoolbook).
    # This is safer than hardcoding obscure 3x3x3 or 2x3x3 decompositions which might have sign errors.
    # The prompt asks for a DIFFERENT strategy. The current one is fixed Strassen on specific pairs.
    # The new strategy will be a RECURSIVE TILING SEARCH that finds the best combination of 2x2x2 blocks.
    
    # For now, return None to indicate "not directly available, use tiling"
    return None

def _make_333():
    """
    Placeholder for 3x3x3 decomposition. Rank is 23.
    Since implementing a correct 23-term 3x3x3 decomposition manually is error-prone,
    we will rely on the tiling strategy to break 3x3x3 into 2x2x2 + 1x1x1 blocks.
    """
    return None

def _embed_block(U_block, V_block, W_block, i_off, j_off, k_off, n, m, p):
    """
    Embed a block decomposition into the global matrices.
    U_block: (ni*mj, rank)
    V_block: (mj*pk, rank)
    W_block: (pk*ni, rank)
    i_off, j_off, k_off: starting indices for this block
    """
    ni, mj, pk = U_block.shape[0]//mj, mj, W_block.shape[0]//ni
    # Wait, dimensions must be consistent.
    # U is (n*m, R), V is (m*p, R), W is (p*n, R)
    
    rank = U_block.shape[1]
    for c in range(rank):
        # Fill U
        for i in range(ni):
            for j in range(mj):
                U[(i_off+i)*m + (j_off+j), c] = U_block[i*mj+j, c]
        
        # Fill V
        for j in range(mj):
            for k in range(pk):
                V[(j_off+j)*p + (k_off+k), c] = V_block[j*pk+k, c]
        
        # Fill W
        for k in range(pk):
            for i in range(ni):
                W[(k_off+k)*n + (i_off+i), c] = W_block[k*ni+i, c]

def _try_tiling(n, m, p, tiles):
    """
    Attempt to tile the n x m x p space with the given list of tile sizes.
    Each tile is (ni, mj, pk) and a corresponding decomposition function.
    Returns (U, V, W) if a valid tiling is found, else None.
    """
    # We need to partition the ranges [0,n), [0,m), [0,p) into blocks.
    # This is a 3D packing problem. For small dimensions, we can try simple grid tilings.
    
    # For simplicity, we will try to tile with 2x2x2 blocks where possible,
    # and 1x1x1 for the rest.
    # This is a specific instantiation of the "recursive tiling" idea.
    
    # Let's define a set of available blocks:
    # 2x2x2: Strassen (rank 7)
    # 1x1x1: Schoolbook (rank 1)
    
    # We will greedily place 2x2x2 blocks where they fit, and fill the rest with 1x1x1.
    # To be more general, we can try different alignments or orders.
    
    U = np.zeros((n*m, n*m*p)) # Max possible rank is n*m*p
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    used = np.zeros((n, m, p), dtype=bool)
    
    # Try to place 2x2x2 blocks
    # We iterate over possible starting positions
    # To ensure we cover everything, we can use a simple scan.
    # However, a simple scan might not be optimal in terms of rank if overlaps were possible,
    # but blocks must be disjoint.
    
    # Let's try a few different scanning orders to see if we get different coverages (though for disjoint tiling, coverage is determined by the set of blocks).
    # Actually, for a fixed set of block sizes, the number of blocks is fixed if they tile perfectly.
    # If they don't tile perfectly, the "leftover" is tiled by 1x1x1.
    
    # Strategy: Place 2x2x2 blocks at (0,0,0), (0,0,2), etc.
    # We'll just do a single pass of 2x2x2 placement, then fill 1x1x1.
    
    for i in range(0, n-1, 2):
        for j in range(0, m-1, 2):
            for k in range(0, p-1, 2):
                # Check if 2x2x2 block at (i,j,k) fits and is unused
                if not used[i:i+2, j:j+2, k:k+2].any():
                    Ub, Vb, Wb = _make_strassen_222()
                    _embed_block(Ub, Vb, Wb, i, j, k, n, m, p)
                    # Update col
                    col += 7
                    used[i:i+2, j:j+2, k:k+2] = True
                    # Note: _embed_block writes to global U, V, W at column 'col' implicitly?
                    # No, _embed_block writes to U, V, W passed in. But it doesn't know which column to write to.
                    # I need to modify _embed_block to accept a column offset or return the new column.
                    # Let's refactor: _embed_block should return the number of columns used.
                    # Actually, it writes to U[...], which is a view? No, U is a local variable in _try_tiling.
                    # I need to pass the column index to _embed_block.
    
    # This implementation of _try_tiling is flawed because _embed_block doesn't handle column indexing correctly in the current draft.
    # Let's rewrite the tiling logic to be explicit about columns.
    
    return None # Placeholder, will be implemented below properly

def _embed_block_at_col(U, V, W, U_block, V_block, W_block, i_off, j_off, k_off, n, m, p, col):
    ni = U_block.shape[0] // U_block.shape[1] # This is wrong, shape is (ni*mj, rank)
    # We need to know ni, mj, pk to index correctly.
    # Let's pass ni, mj, pk explicitly.
    pass

def _construct_via_tiling(n, m, p):
    """
    Constructs a decomposition by tiling the space with 2x2x2 Strassen blocks and 1x1x1 Schoolbook blocks.
    """
    # Max rank is n*m*p
    max_rank = n * m * p
    U = np.zeros((n*m, max_rank))
    V = np.zeros((m*p, max_rank))
    W = np.zeros((p*n, max_rank))
    
    col = 0
    used = np.zeros((n, m, p), dtype=bool)
    
    # Place 2x2x2 blocks
    # We can try different alignments. For now, standard grid.
    for i in range(0, n, 2):
        for j in range(0, m, 2):
            for k in range(0, p, 2):
                if i+2 <= n and j+2 <= m and k+2 <= p:
                    if not used[i:i+2, j:j+2, k:k+2].any():
                        Ub, Vb, Wb = _make_strassen_222()
                        # Embed
                        for c in range(7):
                            # U
                            for di in range(2):
                                for dj in range(2):
                                    U[(i+di)*m + (j+dj), col+c] = Ub[di*2+dj, c]
                            # V
                            for dj in range(2):
                                for dk in range(2):
                                    V[(j+dj)*p + (k+dk), col+c] = Vb[dj*2+dk, c]
                            # W
                            for dk in range(2):
                                for di in range(2):
                                    W[(k+dk)*n + (i+di), col+c] = Wb[dk*2+di, c]
                        col += 7
                        used[i:i+2, j:j+2, k:k+2] = True
    
    # Fill remaining with 1x1x1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not used[i, j, k]:
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
    
    # Trim to actual rank
    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]
    
    return U, V, W, col

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 100
    
    # Strategy: Dynamic Tiling with 2x2x2 Strassen blocks.
    # For n=3, m=4, p=8:
    # i-dim 3: can fit one 2-block (0,1) and one 1-block (2).
    # j-dim 4: can fit two 2-blocks (0,1) and (2,3).
    # k-dim 8: can fit four 2-blocks.
    
    # The tiling will place 2x2x2 blocks where possible.
    # i=0,1: j=0,1,2,3: k=0,1,2,3,4,5,6,7
    #   Blocks at i=0:
    #     j=0,1: k=0,1,2,3,4,5,6,7 -> 4 blocks of 2x2x2? No, k is split into 4 pairs.
    #     So for i=0,1, j=0,1, k=0,1: 1 block (rank 7)
    #     i=0,1, j=0,1, k=2,3: 1 block (rank 7)
    #     i=0,1, j=0,1, k=4,5: 1 block (rank 7)
    #     i=0,1, j=0,1, k=6,7: 1 block (rank 7)
    #     i=0,1, j=2,3, k=0,1: 1 block (rank 7)
    #     i=0,1, j=2,3, k=2,3: 1 block (rank 7)
    #     i=0,1, j=2,3, k=4,5: 1 block (rank 7)
    #     i=0,1, j=2,3, k=6,7: 1 block (rank 7)
    #   Total 8 blocks of 2x2x2 for i=0,1. Rank = 8 * 7 = 56.
    
    # i=2:
    #   j=0,1: k=0,1,2,3,4,5,6,7
    #     Since i is size 1, we can't use 2x2x2. We use 1x2x2 or 1x1x1?
    #     My tiling loop only places 2x2x2. So i=2 will be filled by 1x1x1.
    #     Number of 1x1x1 blocks for i=2: 1 * 4 * 8 = 32. Rank = 32.
    
    # Total Rank = 56 + 32 = 88.
    # This matches the current program's rank.
    
    # To IMPROVE, we need a better decomposition for the "leftover" parts.
    # The leftover is i=2 (size 1), j=0..3 (size 4), k=0..7 (size 8).
    # This is a 1x4x8 matrix multiplication.
    # Rank of 1x4x8?
    # It is equivalent to 4x8 matrix multiplication (since A is 1x4, B is 4x8, C is 1x8).
    # Actually, C[i,k] = sum_j A[i,j] B[j,k]. With i=0 fixed, it's just a vector-matrix product.
    # The tensor is T_{j,k} = B_{j,k}. This is just the identity-like structure?
    # No, the multiplication tensor for 1xm x mxp has rank mp?
    # Let's check. T_{j,k} = 1 if we consider the coefficients.
    # Actually, the rank of the matrix multiplication tensor M_{1,m,p} is m*p.
    # Why? Because each output element C_k depends on all m inputs A_j via B_jk.
    # Wait, if A is 1xm, B is mxp. C_k = sum_j A_j B_jk.
    # The tensor is T_{j,k} = B_jk.
    # This tensor has rank p? No.
    # Consider m=1, p=1. Rank 1.
    # m=1, p=2. C1 = A1 B11, C2 = A1 B12.
    # T_{1,1} = 1, T_{1,2} = 1? No, T_{j,k} is the coefficient of A_j B_k in C.
    # For 1x1 x 1x2: C1 = A1 B11, C2 = A1 B12.
    # Terms: (A1)(B11)(C1), (A1)(B12)(C2). Rank 2.
    # So rank of 1xm x mxp is m*p.
    
    # So the schoolbook rank for the leftover i=2 is 1*4*8 = 32.
    # The Strassen part is 88-32=56.
    # Total 88.
    
    # Can we do better than 88?
    # Known lower bounds:
    # R(3x4x8) >= R(2x2x2) * ...
    # R(3x4x8) is at least R(2x2x2) * 2 * 4 = 7 * 8 = 56? No, that's not a valid lower bound.
    # R(3x4x8) >= R(2x2x2) * R(1x2x4)?
    # Actually, R(3x4x8) >= R(2x2x2) * R(1x2x4) is not how submultiplicativity works directly like that.
    # R(n*m x m*p x p*n) <= R(n1*m1 x m1*p1 x p1*n1) * R(n2*m2 x m2*p2 x p2*n2)
    
    # Let's try a different tiling:
    # What if we use 3x3x3 blocks? Rank 23.
    # n=3, m=4, p=8.
    # We can fit one 3x3x3 block?
    # i=0,1,2; j=0,1,2; k=0,1,2.
    # Rank 23.
    # Leftover:
    # i=0,1,2; j=3; k=0,1,2 (1x1x3 block, rank 3)
    # i=0,1,2; j=0,1,2; k=3,4,5,6,7 (3x3x5 block)
    # i=0,1,2; j=3; k=3,4,5,6,7 (1x1x5 block, rank 5)
    
    # This seems messy. The 2x2x2 tiling is very regular.
    
    # Is there a known decomposition for 3x4x8 with rank < 88?
    # R(3x4x8) <= R(3x4x4) * R(1x1x2)? No.
    # R(3x4x8) <= R(3x4x4) * 2?
    # R(3x4x4) is known to be <= 38? (Schoolbook 48).
    # If R(3x4x4) = 38, then R(3x4x8) <= 38 * 2 = 76?
    # Wait, R(3x4x8) is not R(3x4x4) * 2.
    # But R(3x4x8) <= R(3x4x4) * R(1x1x2) is false.
    # However, R(3x4x8) <= R(3x4x4) * 2 is TRUE if we can double the k-dimension.
    # Doubling k-dimension: if we have a decomposition for p=4, we can use it for p=8 by splitting k into two halves?
    # No, the rank would double.
    # So if R(3x4x4) is 38, R(3x4x8) <= 76.
    
    # What is R(3x4x4)?
    # Schoolbook: 3*4*4 = 48.
    # Strassen on 2x2x2 blocks:
    # i=0,1 (2x2x2 blocks for j=0,1,2,3 and k=0,1,2,3? No, k is 4).
    # i=0,1: j=0,1,2,3; k=0,1,2,3.
    #   2x2x2 blocks: i=0,1; j=0,1; k=0,1 (Rank 7)
    #                  i=0,1; j=0,1; k=2,3 (Rank 7)
    #                  i=0,1; j=2,3; k=0,1 (Rank 7)
    #                  i=0,1; j=2,3; k=2,3 (Rank 7)
    #   Total 28.
    # i=2: 1x4x4. Rank 16.
    # Total 28 + 16 = 44.
    
    # Can we do better for 3x4x4?
    # Maybe use 3x3x3?
    # i=0,1,2; j=0,1,2; k=0,1,2. Rank 23.
    # Leftover:
    # i=0,1,2; j=3; k=0,1,2. Rank 3.
    # i=0,1,2; j=0,1,2; k=3. Rank 3.
    # i=0,1,2; j=3; k=3. Rank 1.
    # Total 23 + 3 + 3 + 1 = 30.
    
    # If R(3x4x4) = 30, then R(3x4x8) <= 60.
    
    # Let's implement the 3x3x3 tiling strategy for 3x4x8.
    # We need a 3x3x3 decomposition.
    # The 23-term algorithm for 3x3x3 is complex.
    # However, we can use a known 23-term decomposition.
    # Or, we can stick to 2x2x2 and 1x1x1 if 3x3x3 is too risky to hardcode.
    # But 88 is the current score. If we can get 60, that's a huge jump.
    
    # Let's try to find a 3x3x3 decomposition online or in memory.
    # The 23-term algorithm by Bini et al. is standard.
    # Given the risk of error, I will implement a "safe" 3x3x3 by tiling it with 2x2x2 + 1x1x1.
    # 3x3x3 tiled with 2x2x2:
    # One 2x2x2 block (rank 7).
    # Leftover:
    # i=2, j=0,1,2; k=0,1,2 (1x2x3? No, 1x3x3). Rank 9.
    # i=0,1,2; j=2; k=0,1,2 (3x1x3). Rank 9.
    # i=2, j=2; k=0,1,2 (1x1x3). Rank 3.
    # Wait, the overlaps are tricky.
    # 3x3x3:
    # Block (0,0,0) size 2x2x2.
    # Remaining:
    # i=2, j=0,1,2; k=0,1,2 -> 1x3x3 (Rank 9)
    # i=0,1,2; j=2; k=0,1,2 -> 3x1x3 (Rank 9)
    # i=2, j=2; k=0,1,2 -> 1x1x3 (Rank 3)
    # But i=2, j=2, k=0,1,2 is counted in both 1x3x3 and 3x1x3?
    # No, 1x3x3 covers i=2, j=0,1,2, k=0,1,2.
    # 3x1x3 covers i=0,1,2, j=2, k=0,1,2.
    # Intersection: i=2, j=2, k=0,1,2.
    # So we must be careful not to double count.
    # Better to just use 1x1x1 for the leftover of 3x3x3.
    # 2x2x2 (Rank 7) + 19 * 1x1x1 (Rank 19) = 26.
    # 26 > 23. So tiling 3x3x3 with 2x2x2 is worse than the known 23.
    
    # Since I cannot reliably hardcode the 23-term 3x3x3, I will stick to the 2x2x2 tiling which yields 88.
    # But wait, the current program ALREADY yields 88.
    # The task is to find a DIFFERENT strategy.
    # If I implement the same 88 rank via a different method (tiling vs manual), it's a structural change but same score.
    # To IMPROVE score, I need rank < 88.
    
    # Is there a way to get < 88 with 2x2x2 blocks?
    # No, 88 is the minimum for 2x2x2 + 1x1x1 tiling.
    
    # What if I use 2x3x3 blocks? Rank 18.
    # n=3, m=4, p=8.
    # Can we tile with 2x3x3?
    # i=0,1; j=0,1,2; k=0,1,2. Rank 18.
    # i=0,1; j=0,1,2; k=3,4,5. Rank 18.
    # i=0,1; j=3; k=0,1,2. Rank 3.
    # i=0,1; j=3; k=3,4,5. Rank 3.
    # i=0,1; j=0,1,2; k=6,7. Rank 2*3*2 = 12? (1x3x2 is 6, 2x3x2 is 12? No, 2x3x2 rank is 12? Schoolbook 12. Strassen 2x2x2 is 7. 2x3x2 can be tiled 2x2x2 + 2x1x2. 7 + 4 = 11?)
    # This is getting complicated.
    
    # Given the constraints and the need for a robust program, I will implement the 2x2x2 tiling strategy.
    # It is structurally different (generic tiling engine) even if the rank is the same.
    # If I can find a 3x3x3 decomposition, I will use it.
    # For now, I will return the 88-rank solution via tiling.
    
    U, V, W, rank = _construct_via_tiling(n, m, p)
    
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
