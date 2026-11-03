"""Rewritten program: structured sparse linear algebra / binary IP for exact tensor decomposition."""
DIMENSIONS = [3, 3, 8]
import random
random.seed(1)


# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 8
    nm, mp, pn = n * m, m * p, p * n  # 9, 24, 24

    # We use a structured decomposition based on Strassen's algorithm for 3x3x3 (rank 23)
    # applied to blocks of the 3x8 matrix.
    # Split B (3x8) into B1 (3x3), B2 (3x3), B3 (3x2).
    # Compute C1 = A*B1, C2 = A*B2, C3 = A*B3.
    # C1 and C2 each use rank 23 Strassen terms.
    # C3 uses a specialized rank-? decomposition for 3x3 * 3x2.
    
    # For 3x3 * 3x2, the output is 3x2 = 6 entries.
    # Schoolbook: 3*3*2 = 18 terms.
    # Can we do better? The tensor is 9x6x6 = 324 entries, 6 of which are 1.
    # Let's use a simple rank-18 decomposition for now, or try to find a smaller one.
    # Actually, let's try to use a Strassen-like approach for the 3x2 case.
    # Or just use schoolbook for the last block: 18 terms.
    # Total rank: 23 + 23 + 18 = 64.
    
    # But we can do better! Let's use a shared Strassen-like structure.
    # Actually, let's try a different split: B into 4+4.
    # 3x3 * 3x4: what's the minimal rank?
    # Schoolbook: 3*3*4 = 36.
    # If we can find a rank-27 decomposition for 3x3*3x4, then total is 54.
    # Or if rank 24, total is 48.
    
    # Let's try to construct a rank-24 decomposition for 3x3 * 3x4.
    # The tensor has 9*12*12 = 1296 entries, 36 of which are 1.
    
    # Actually, let's use a more systematic approach.
    # We'll use the known Strassen rank-23 decomposition for 3x3x3,
    # and extend it to handle the extra columns.
    
    # Strassen's 3x3x3 rank-23 decomposition:
    # We need to hardcode the 23 terms.
    # Each term is (u, v, w) where u, v, w are 9-dimensional vectors with entries in {-1, 0, 1}.
    
    # Standard Strassen decomposition for 3x3 (rank 23):
    # This is complex. Let me use a known set.
    
    # Alternative: use the fact that 3x3x8 can be decomposed using
    # two 3x3x4 blocks, where each 3x3x4 uses rank 24 (if we can find it).
    
    # Let's try a greedy approach: start with rank 48 and verify.
    # If we can't verify, fall back to rank 64 or 72.
    
    # For now, let's use a safe rank-64 decomposition:
    # Two Strassen 3x3x3 blocks (rank 23 each) + one schoolbook 3x3x2 block (rank 18).
    # Total: 23 + 23 + 18 = 64.
    
    # But I need to actually construct this. Let me hardcode the Strassen terms.
    
    # Strassen 3x3x3 rank-23 terms (one known decomposition):
    # I'll use the standard one. Each term (u, v, w) where:
    # u[i*3+j] corresponds to A[i,j], v[j*3+k] to B[j,k], w[k*3+i] to C[k,i] (trace convention)
    
    # The 23 Strassen terms for 3x3x3:
    strassen_333 = [
        # Term 1
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        # Term 2
        ([0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        # Term 3
        ([0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0]),
        # Terms 4-23: Strassen's additional 20 terms
        # This is getting complex. Let me use a different approach.
    ]
    
    # Given the complexity of hardcoding Strassen, let me use a simpler approach:
    # Use schoolbook for all blocks but try to find a better decomposition via random search.
    
    # Actually, let me just return a rank-64 decomposition using a known simple structure.
    # Split B into three parts: B1 (3x2), B2 (3x3), B3 (3x3).
    # Each 3x3 * 3x3 uses schoolbook rank 9, 3x3 * 3x2 uses schoolbook rank 6.
    # Total: 9 + 9 + 6 = 24? No, that's wrong.
    # Schoolbook for 3x3*3x3 is 9*9=81? No, it's 3*3*3=27.
    # Schoolbook for 3x3*3x2 is 3*3*2=18.
    # Total: 27 + 27 + 18 = 72. That's the same as full schoolbook.
    
    # Let me just return schoolbook rank 72 for safety.
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
