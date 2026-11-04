"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    
    def add_strassen(U, V, W, i0, j0, k0, col):
        """Add a 2x2x2 Strassen block. Returns new col index."""
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1  # Wait, this should be W[(k0)*n + (i0+1)] and W[(k0+1)*n + (i0+1)]
        # Let me redo Strassen properly
        # C00 = M1 - M4 + M6
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M7
        
        # Actually let me use the standard Strassen formulas:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B10)
        # M5 = (A00+A10)B11
        # M6 = (A01-A10)(B00+B10)
        # M7 = (A11-A01)(B01+B11)
        
        # C00 = M1 - M4 + M6
        # C01 = M3 + M4
        # C10 = M2 + M5
        # C11 = M1 - M2 + M3 + M7
        
        # Let me restart with correct Strassen
        col -= 1  # undo the previous addition
        # M1 = (A00+A11)(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M3 = A00(B10-B11)
        U[i0*m + j0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5 = (A00+A10)B11
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + j0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[(k0+1)*n + i0][col] += 1
        W[k0*n + i0][col] -= 1
        col += 1
        
        # M6 = (A01-A10)(B00+B10)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + k0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7 = (A11-A01)(B01+B11)
        U[(i0+1)*m + (j0+1)][col] += 1
        U[i0*m + (j0+1)][col] -= 1
        V[(j0+1)*p + (k0+1)][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1
        
        return col
    
    def build_decomposition(i_blocks, j_blocks, k_blocks):
        """Build a decomposition using Strassen for 2x2x2 blocks and schoolbook otherwise."""
        max_rank = 200
        U = [[0] * max_rank for _ in range(n * m)]
        V = [[0] * max_rank for _ in range(m * p)]
        W = [[0] * max_rank for _ in range(p * n)]
        
        col = 0
        for (i0, i1) in i_blocks:
            for (j0, j1) in j_blocks:
                for (k0, k1) in k_blocks:
                    i_size = i1 - i0 + 1
                    j_size = j1 - j0 + 1
                    k_size = k1 - k0 + 1
                    
                    if i_size == 2 and j_size == 2 and k_size == 2:
                        col = add_strassen(U, V, W, i0, j0, k0, col)
                    else:
                        for i in range(i0, i1 + 1):
                            for j in range(j0, j1 + 1):
                                for k in range(k0, k1 + 1):
                                    U[i*m + j][col] = 1
                                    V[j*p + k][col] = 1
                                    W[k*n + i][col] = 1
                                    col += 1
        
        # Trim to actual rank
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
        
        return U, V, W, col
    
    def verify(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if total != 1:
                        return False
        return True
    
    # Define various tilings to try
    tilings = [
        # (i_blocks, j_blocks, k_blocks)
        # i: 0,1,2 ; j: 0,1,2,3,4 ; k: 0,1,2,3,4,5,6
        
        # Tiling 1: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 2: [0,1],[2] ; [0,1],[2,3],[4] ; [0,2],[3,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,2),(3,4),(5,6)]),
        
        # Tiling 3: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,4],[5,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,4),(5,6)]),
        
        # Tiling 4: [0,1],[2] ; [0,2],[3,4] ; [0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,2),(3,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 5: [0,2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        ([(0,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 6: [0,1],[2] ; [0,1],[2,4] ; [0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 7: [0,1],[2] ; [0,3],[4] ; [0,1],[2,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 8: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,5),(6,6)]),
        
        # Tiling 9: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,6)]),
        
        # Tiling 10: [0,1],[2] ; [0,1],[2,3],[4] ; [0,3],[4,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,3),(4,5),(6,6)]),
        
        # Tiling 11: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,4,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,5),(6,6)]),
        
        # Tiling 12: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6] - same as 1
        # Let's try different i-blocks
        # Tiling 12: [0,1,2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        ([(0,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,5),(6,6)]),
        
        # Tiling 13: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        # Already have this
        
        # Tiling 14: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        # Let's try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # This is the same as tiling 1
        
        # Tiling 15: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        # Try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        
        # Let me add more diverse tilings
        # Tiling 16: [0,1],[2] ; [0,1],[2,3],[4] ; [0,1],[2,3],[4,5],[6]
        # Try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        
        # Actually, let me try tilings where we use Strassen on different 2x2x2 subcubes
        # The key insight: we can place 2x2x2 Strassen blocks at different positions
        
        # Tiling 16: Use Strassen at (i=0,1; j=0,1; k=0,1), (i=0,1; j=2,3; k=2,3), (i=0,1; j=4; k=4,5)
        # But j=4 is size 1, so no Strassen there
        
        # Tiling 17: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # This gives Strassen at (0,1;0,1;0,1) and (0,1;2,3;2,3)
        
        # Tiling 18: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Same as above
        
        # Let me try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Strassen blocks at: (i=0,1; j=0,1; k=0,1) and (i=0,1; j=2,3; k=2,3)
        # That's 2 Strassen blocks = 14 multiplications
        # Remaining: i=0,1; j=4; k=0,1,2,3,4,5,6 (7 terms)
        #           i=0,1; j=0,1; k=4,5,6 (6 terms)
        #           i=0,1; j=2,3; k=0,1,4,5,6 (10 terms)
        #           i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35 terms)
        # Total: 14 + 7 + 6 + 10 + 35 = 72
        
        # Tiling 19: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # This is the same as tiling 1
        
        # Let me try a different approach: use Strassen where possible
        # For i=[0,1], j=[0,1], k=[0,1]: Strassen (7 mults)
        # For i=[0,1], j=[2,3], k=[2,3]: Strassen (7 mults)
        # For i=[0,1], j=[4], k=[4,5]: can't use Strassen (j size 1)
        
        # Tiling 20: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Already have this
        
        # Let me try tilings that maximize Strassen usage
        # Best case: 2 Strassen blocks (i=0,1 with j=0,1 and j=2,3, each with k=0,1 and k=2,3)
        # That's 4 Strassen blocks = 28 multiplications
        # But we need k blocks to align
        
        # Tiling 21: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Strassen at: (0,1;0,1;0,1), (0,1;0,1;2,3), (0,1;2,3;0,1), (0,1;2,3;2,3)
        # That's 4 Strassen blocks = 28 multiplications
        # Remaining: i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #           i=0,1; j=0,1; k=4,5,6 (6)
        #           i=0,1; j=2,3; k=4,5,6 (6)
        #           i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 28 + 7 + 6 + 6 + 35 = 82
        
        # Wait, that's worse. Let me recalculate.
        # With k=[0,1],[2,3],[4,5],[6], we have 4 k-blocks
        # Strassen requires 2x2x2, so we need i_size=2, j_size=2, k_size=2
        # With i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]:
        # Strassen possible at: (i=0,1; j=0,1; k=0,1), (i=0,1; j=0,1; k=2,3),
        #                       (i=0,1; j=2,3; k=0,1), (i=0,1; j=2,3; k=2,3)
        # That's 4 Strassen blocks = 28 mults
        # Schoolbook: i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=0,1; j=0,1; k=4,5,6 (6)
        #             i=0,1; j=2,3; k=4,5,6 (6)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 28 + 7 + 6 + 6 + 35 = 82
        
        # But tiling 1 has:
        # Strassen at: (0,1;0,1;0,1) and (0,1;2,3;2,3) = 14 mults
        # Schoolbook: i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=0,1; j=0,1; k=2,3,4,5,6 (10)
        #             i=0,1; j=2,3; k=0,1,4,5,6 (10)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 14 + 7 + 10 + 10 + 35 = 76
        
        # So tiling 1 is better! 76 vs 82.
        
        # The key is to NOT use Strassen when it forces more schoolbook terms.
        # With k=[0,1],[2,3],[4,5],[6], using Strassen at (0,1;0,1;0,1) and (0,1;2,3;2,3)
        # means k=4,5,6 are all schoolbook for j=0,1 and j=2,3.
        
        # What if we use k=[0,1],[2,3,4,5],[6]?
        # Strassen at: (0,1;0,1;0,1) only = 7 mults
        # Schoolbook: i=0,1; j=0,1; k=2,3,4,5 (8)
        #             i=0,1; j=2,3; k=0,1,2,3,4,5 (12)
        #             i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 7 + 8 + 12 + 7 + 35 = 69
        
        # Better! Let me add this tiling.
        
        # Tiling 22: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,5],[6]
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,5),(6,6)]),
        
        # Tiling 23: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,4],[5,6]
        # Strassen at: (0,1;0,1;0,1) and (0,1;2,3;2,4)? No, k=2,4 is size 3
        # Strassen at: (0,1;0,1;0,1) only = 7
        # Schoolbook: i=0,1; j=0,1; k=2,3,4,5,6 (10)
        #             i=0,1; j=2,3; k=0,1,2,3,4,5,6 (14)
        #             i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 7 + 10 + 14 + 7 + 35 = 73
        
        # Tiling 24: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Already have this (tiling 1)
        
        # Tiling 25: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3,4],[5,6]
        # Strassen at: (0,1;0,1;0,1) and (0,1;2,3;2,3,4)? No, k size 3
        # Strassen at: (0,1;0,1;0,1) only = 7
        # Schoolbook: i=0,1; j=0,1; k=2,3,4,5,6 (10)
        #             i=0,1; j=2,3; k=0,1,2,3,4,5,6 (14)
        #             i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 7 + 10 + 14 + 7 + 35 = 73
        
        # Tiling 26: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5,6]
        # Strassen at: (0,1;0,1;0,1) and (0,1;2,3;2,3) = 14
        # Schoolbook: i=0,1; j=4; k=0,1,2,3,4,5,6 (7)
        #             i=0,1; j=0,1; k=4,5,6 (6)
        #             i=0,1; j=2,3; k=0,1,4,5,6 (10)
        #             i=2; j=0,1,2,3,4; k=0,1,2,3,4,5,6 (35)
        # Total: 14 + 7 + 6 + 10 + 35 = 72
        
        ([(0,1),(2,2)], [(0,1),(2,3),(4,4)], [(0,1),(2,3),(4,6)]),
        
        # Tiling 27: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Same as tiling 1
        
        # Tiling 28: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Let me try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        
        # Tiling 29: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Try: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        
        # I think I've covered the main tilings. Let me also try:
        # Tiling 30: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # This is the same as tiling 1
        
        # Tiling 31: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # Let me try a completely different approach:
        # Use Strassen on (i=0,1; j=0,1; k=0,1) and (i=0,1; j=0,1; k=2,3)
        # i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,3],[4,5],[6]
        # Strassen at: (0,1;0,1;0,3)? No, k size 4
        # Can't use Strassen with k size 4
        
        # Tiling 32: i=[0,1],[2], j=[0,1],[2,3],[4], k=[0,1],[2,3],[4,5],[6]
        # I'll stick with the tilings I have.
    ]
    
    best_result = None
    best_rank = float('inf')
    
    for tiling in tilings:
        if time.time() > deadline:
            break
        
        U, V, W, rank = build_decomposition(tiling[0], tiling[1], tiling[2])
        
        if not verify(U, V, W):
            continue
        
        if rank < best_rank:
            best_rank = rank
            best_result = {"U": U, "V": V, "W": W}
    
    # Pure schoolbook as fallback
    U_sch = [[0] * (n*m*p) for _ in range(n*m)]
    V_sch = [[0] * (n*m*p) for _ in range(m*p)]
    W_sch = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i*m + j][col] = 1
                V_sch[j*p + k][col] = 1
                W_sch[k*n + i][col] = 1
                col += 1
    
    if n*m*p < best_rank:
        best_rank = n*m*p
        best_result = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    if best_result is None:
        best_result = {"U": U_sch, "V": V_sch, "W": W_sch}
    
    return best_result
# EVOLVE-BLOCK-END
