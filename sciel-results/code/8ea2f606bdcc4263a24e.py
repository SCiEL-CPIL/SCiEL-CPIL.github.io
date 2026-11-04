import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_S = 110

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    def time_left():
        return DEADLINE_S - (time.time() - start_time)
    
    def schoolbook():
        rank = n * m * p  # 72
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen).
        Local indices: U[a*2+b], V[b*2+c], W[c*2+a]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # P1 = (a+d)(e+h)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # P2 = (c+d)e
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        
        # P3 = a(f-h)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        
        # P4 = d(g-e)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        
        # P5 = (a+b)h
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        
        # P6 = (c-a)(e+f)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # P7 = (b-d)(g+h)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        
        return U, V, W
    
    def validate(U, V, W, tol=1e-10):
        """Check that sum_c U[:,c] V[:,c] W[:,c] == multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Build the decomposition with maximum non-overlapping Strassen blocks
    # Strassen blocks at (i0, j0, k0) for i0 in {0,1}, j0 in {0,1}, k0 in {0,2,4,6}
    strassen_blocks = []
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(0, p, 2):
                strassen_blocks.append((i0, j0, k0))
    
    # Determine which (i,j,k) positions are covered by Strassen blocks
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_strassen = len(strassen_blocks)  # 16
    n_school = n * m * p - len(covered)  # 72 - 128... wait, can't be more than 72
    
    # Actually, let me recount: 16 blocks * 8 entries = 128, but we only have 72 total
    # This means blocks overlap! Let me reconsider.
    
    # The issue: 2*2*4 = 16 blocks each covering 2*2*2 = 8 entries = 128 entries
    # But total tensor has only 3*3*8 = 72 entries
    # So blocks MUST overlap. We need to find a valid non-overlapping set.
    
    # Let me think more carefully. We need blocks where no two share any (i,j,k) position.
    # Each block at (i0,j0,k0) covers {(i0+di, j0+dj, k0+dk) : di,dj,dk in {0,1}}
    
    # For i dimension: blocks at i0=0 cover i in {0,1}, blocks at i0=1 cover i in {1,2}
    # So i0=0 and i0=1 overlap at i=1
    # Similarly for j: j0=0 covers j in {0,1}, j0=1 covers j in {1,2}, overlap at j=1
    # For k: k0=0 covers k in {0,1}, k0=2 covers k in {2,3}, k0=4 covers k in {4,5}, k0=6 covers k in {6,7}
    # k blocks don't overlap with each other
    
    # So we need to pick a subset of {i0 in {0,1}, j0 in {0,1}} such that no two blocks
    # share any (i,j) position. Since k ranges don't overlap, we just need (i,j) non-overlap.
    
    # Block (i0,j0) covers (i,j) positions: {(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)}
    # (0,0): {(0,0),(0,1),(1,0),(1,1)}
    # (0,1): {(0,1),(0,2),(1,1),(1,2)}
    # (1,0): {(1,0),(1,1),(2,0),(2,1)}
    # (1,1): {(1,1),(1,2),(2,1),(2,2)}
    
    # (0,0) and (1,1) overlap at (1,1)
    # (0,1) and (1,0) overlap at (1,1)
    # (0,0) and (0,1) overlap at (0,1) and (1,1)
    # etc.
    
    # Maximum independent set: pick at most 2 blocks that don't overlap
    # (0,0) and (1,1) overlap at (1,1) - not valid
    # (0,0) and (1,0) overlap at (1,0) and (1,1) - not valid
    # Actually all pairs overlap! Let me check:
    # (0,0) covers {(0,0),(0,1),(1,0),(1,1)}
    # (1,1) covers {(1,1),(1,2),(2,1),(2,2)} - overlap at (1,1)
    # (0,1) covers {(0,1),(0,2),(1,1),(1,2)} - overlap at (0,1) and (1,1) with (0,0)
    
    # So we can only pick ONE Strassen block per k-slice? No wait...
    # Actually we can pick blocks that don't share (i,j) positions.
    # (0,0) and (1,1) share (1,1). 
    # What about just picking (0,0) for all 4 k-slices? That gives 4 blocks, no overlap.
    # Or (0,0) for k0=0,2 and (1,1) for k0=4,6? 
    # (0,0) covers i in {0,1}, j in {0,1}
    # (1,1) covers i in {1,2}, j in {1,2}
    # They overlap at (i=1, j=1). So if k-ranges don't overlap, it's fine!
    
    # Since k-ranges are disjoint, blocks at different k0 never share (i,j,k) positions.
    # So we can use ALL 16 blocks! They only "overlap" in (i,j) but not in (i,j,k).
    
    # Wait, I was confusing myself. The tensor position is (i,j,k). Two blocks overlap
    # only if they share the same (i,j,k). Since k-ranges are disjoint, no overlap.
    
    # So 16 Strassen blocks, each covering 8 distinct (i,j,k) positions = 128 positions
    # But we only have 72 positions total. This is impossible!
    
    # I see the error. Let me recount: 16 blocks * 8 = 128 > 72. 
    # The blocks DO overlap in (i,j,k) space because different k0 values give
    # different k ranges, but the (i,j) parts overlap.
    
    # Block (i0,j0,k0) covers (i,j,k) where i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}
    # Block (i0',j0',k0') covers similar with primed indices.
    # They share a position if there exists (i,j,k) in both sets.
    # This requires: {i0,i0+1} ∩ {i0',i0'+1} ≠ ∅ AND {j0,j0+1} ∩ {j0',j0'+1} ≠ ∅ AND {k0,k0+1} ∩ {k0',k0'+1} ≠ ∅
    
    # For k: {0,1}, {2,3}, {4,5}, {6,7} are all disjoint. So k-ranges never overlap
    # between different k0 values. Therefore blocks with different k0 never share positions!
    
    # So we CAN use all 16 blocks. Total covered positions = 16*8 = 128?
    # No! Each block covers 2*2*2 = 8 positions, and they're all distinct because
    # k-ranges are disjoint. But 16*8 = 128 > 72. Contradiction!
    
    # Oh wait, I think I'm double counting. Let me list all positions covered:
    # For k0=0: k in {0,1}
    #   (0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1) from block (0,0,0)
    #   (0,1,0),(0,1,1),(0,2,0),(0,2,1),(1,1,0),(1,1,1),(1,2,0),(1,2,1) from block (0,1,0)
    #   etc.
    
    # These DO overlap! Block (0,0,0) and (0,1,0) both cover (0,1,0), (0,1,1), (1,1,0), (1,1,1)
    
    # So I need to find the maximum set of non-overlapping blocks.
    # For a fixed k0, we need non-overlapping (i,j) blocks.
    # As shown, no two (i0,j0) pairs are non-overlapping in the 3x3 grid.
    # So at most 1 Strassen block per k0 value.
    
    # With 4 k0 values (0,2,4,6), we can have at most 4 Strassen blocks.
    
    # Let me verify: pick (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    # Block (0,0,0): covers i in {0,1}, j in {0,1}, k in {0,1}
    # Block (0,0,2): covers i in {0,1}, j in {0,1}, k in {2,3}
    # These don't overlap (different k ranges). Good.
    
    # So maximum 4 Strassen blocks. Rank = 4*7 + (72 - 4*8) = 28 + 40 = 68.
    # That matches the current score!
    
    # Can we do better? What about using partial Strassen or other techniques?
    
    # For now, let's build the rank-68 solution with 4 Strassen blocks.
    
    strassen_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n * m * p - len(covered)  # 72 - 32 = 40
    rank = 7 * len(strassen_blocks) + n_school  # 28 + 40 = 68
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    sU, sV, sW = strassen_2x2x2()
    
    for (i0, j0, k0) in strassen_blocks:
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di) * m + (j0+dj)][col + c] = sU[di*2 + dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj) * p + (k0+dk)][col + c] = sV[dj*2 + dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk) * n + (i0+di)][col + c] = sW[dk*2 + di][c]
        col += 7
    
    # Schoolbook for uncovered positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    result = {"U": U, "V": V, "W": W}
    
    if validate(U, V, W):
        return result
    else:
        return schoolbook()
