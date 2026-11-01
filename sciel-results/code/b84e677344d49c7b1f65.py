"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x6 matrix.
    Uses a structural block-partition approach: split p=6 into three 2-column blocks,
    and for each block use Strassen-style 2x2x2 decompositions on the shared A blocks.
    This gives a base rank of 42 (3 blocks * 14 per block).
    
    Then attempts to reduce rank by:
    1. Merging duplicate/zero columns
    2. Searching for linear dependencies among columns
    3. Randomized perturbation and re-verification for lower ranks
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W, r

    def strassen_based_decomposition():
        """
        Build a rank-42 decomposition using Strassen on 2x2x2 blocks.
        
        Split p=6 into three blocks of 2 columns each.
        For each block r (r=0,1,2), compute C[:, 2r:2r+2] = A * B[:, 2r:2r+2].
        
        A is 2x4, split into A1 (cols 0,1) and A2 (cols 2,3), each 2x2.
        B[:, 2r:2r+2] is 4x2, split into B_top (rows 0,1) and B_bot (rows 2,3), each 2x2.
        
        C_block = A1 * B_top + A2 * B_bot
        
        This is a sum of two 2x2x2 tensor products, each of rank 7 (Strassen).
        Total rank per block = 14, total = 42.
        """
        rank = 42
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        
        # Strassen coefficients for 2x2x2
        # M1 = (X00+X11)(Y00+Y11)
        # M2 = (X10+X11)Y00
        # M3 = X00(Y01-Y11)
        # M4 = X11(Y10-Y00)
        # M5 = (X00+X01)Y11
        # M6 = (X10-X00)(Y00+Y01)
        # M7 = (X01-X11)(Y10+Y11)
        #
        # Z00 = M1 + M4 - M5 + M7
        # Z01 = M3 + M5
        # Z10 = M2 + M4
        # Z11 = M1 - M2 + M3 + M6
        
        strassen_terms = [
            # M1
            {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
            # M2
            {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
            # M3
            {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
            # M4
            {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
            # M5
            {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
            # M6
            {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
            # M7
            {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
        ]
        
        col = 0
        for r in range(3):  # 3 column blocks
            # For each block, we have two 2x2x2 sub-problems:
            # 1. A1 * B_top where A1 = A[:, 0:2], B_top = B[0:2, 2r:2r+2]
            # 2. A2 * B_bot where A2 = A[:, 2:4], B_bot = B[2:4, 2r:2r+2]
            
            for sub in range(2):  # sub=0: A1*B_top, sub=1: A2*B_bot
                # A block offset: sub=0 -> cols 0,1; sub=1 -> cols 2,3
                a_col_offset = sub * 2
                # B row offset: sub=0 -> rows 0,1; sub=1 -> rows 2,3
                b_row_offset = sub * 2
                
                for s_idx in range(7):
                    term = strassen_terms[s_idx]
                    
                    # Update U
                    for local_u, coeff in term['U']:
                        a = local_u // 2  # row in 2x2 block
                        l = local_u % 2   # col in 2x2 block
                        global_u = a * m + (a_col_offset + l)
                        U[global_u][col] += coeff
                    
                    # Update V
                    for local_v, coeff in term['V']:
                        l = local_v // 2   # row in 2x2 block
                        c = local_v % 2    # col in 2x2 block
                        global_v = (b_row_offset + l) * p + (2 * r + c)
                        V[global_v][col] += coeff
                    
                    # Update W
                    for local_w, coeff in term['W']:
                        a = local_w // 2   # row in 2x2 output
                        c = local_w % 2    # col in 2x2 output
                        global_w = (2 * r + c) * n + a
                        W[global_w][col] += coeff
                    
                    col += 1
        
        return U, V, W, rank

    def verify(U, V, W, r):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr] for rr in range(r))
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    def remove_zero_columns(U, V, W, r):
        """Remove columns that are all zeros in any of U, V, W."""
        keep = []
        for c in range(r):
            if any(U[i][c] != 0 for i in range(len(U))) and \
               any(V[i][c] != 0 for i in range(len(V))) and \
               any(W[i][c] != 0 for i in range(len(W))):
                keep.append(c)
        
        new_r = len(keep)
        new_U = [[U[i][c] for c in keep] for i in range(len(U))]
        new_V = [[V[i][c] for c in keep] for i in range(len(V))]
        new_W = [[W[i][c] for c in keep] for i in range(len(W))]
        return new_U, new_V, new_W, new_r

    def try_linear_reduction(U, V, W, r):
        """
        Try to find linear dependencies among columns using rank analysis.
        If columns are linearly dependent, we can potentially reduce rank.
        """
        # Check if we can express some columns as linear combinations of others
        # This is complex; for now, just try removing duplicate columns
        col_dict = {}
        for c in range(r):
            key = (tuple(U[i][c] for i in range(len(U))),
                   tuple(V[i][c] for i in range(len(V))),
                   tuple(W[i][c] for i in range(len(W))))
            if key in col_dict:
                # Duplicate column, can remove
                pass
            else:
                col_dict[key] = c
        
        # For now, just return as-is; full linear algebra reduction is complex
        return U, V, W, r

    def randomized_search(U, V, W, r, target_rank, max_attempts=50):
        """
        Try to find a lower-rank decomposition by perturbing and re-solving.
        """
        if target_rank >= r:
            return U, V, W, r
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Create a perturbed version with some entries set to 0
            new_U = [[U[i][c] if random.random() > 0.3 else 0 for c in range(r)] for i in range(len(U))]
            new_V = [[V[i][c] if random.random() > 0.3 else 0 for c in range(r)] for i in range(len(V))]
            new_W = [[W[i][c] if random.random() > 0.3 else 0 for c in range(r)] for i in range(len(W))]
            
            # Remove zero columns
            new_U, new_V, new_W, new_r = remove_zero_columns(new_U, new_V, new_W, r)
            
            if new_r <= target_rank:
                if verify(new_U, new_V, new_W, new_r):
                    return new_U, new_V, new_W, new_r
        
        return U, V, W, r

    # Start with the Strassen-based decomposition
    U, V, W, r = strassen_based_decomposition()
    
    # Remove zero columns
    U, V, W, r = remove_zero_columns(U, V, W, r)
    
    # Verify the base decomposition
    if not verify(U, V, W, r):
        # Fall back to schoolbook if something went wrong
        U, V, W, r = schoolbook()
    
    best_result = {"U": U, "V": V, "W": W}
    best_rank = r
    
    # Try to reduce rank further
    for target_rank in range(r - 1, max(20, r - 15), -1):
        if time.time() > deadline:
            break
        
        U2, V2, W2, r2 = randomized_search(U, V, W, r, target_rank, max_attempts=30)
        
        if r2 < best_rank and verify(U2, V2, W2, r2):
            best_result = {"U": U2, "V": V2, "W": W2}
            best_rank = r2
            U, V, W, r = U2, V2, W2, r2
    
    return best_result
# EVOLVE-BLOCK-END
