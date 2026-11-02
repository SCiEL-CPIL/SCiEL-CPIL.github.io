"""Bootstrap SEARCH for exact tensor decomposition of 3x4x7 matrix multiplication."""
import time
import itertools
import numpy as np
import scipy.sparse.linalg as spla
import scipy.linalg as sla

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """Return the best exact tensor decomposition found within a time budget.
    The interface is construct() -> {'U', 'V', 'W'} with integer/half-integer entries.
    """
    start = time.time()
    deadline = start + 140.0  # safety margin below 150s kill

    n, m, p = DIMENSIONS

    def _schoolbook():
        """Return the classical rank-nmp decomposition."""
        R = n * m * p
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _strassen_embedded(num_blocks=6):
        """
        Construct a decomposition by replacing multiple 2x2x2 schoolbook blocks
        with Strassen's rank-7 decomposition. Each 2x2x2 block saves 1 rank.
        """
        def strassen_block(i_off, j_off, k_off, U, V, W, col):
            i0, i1 = i_off, i_off + 1
            j0, j1 = j_off, j_off + 1
            k0, k1 = k_off, k_off + 1
            
            # Term 1: M1 = (A00+A11)(B00+B11)
            U[i0*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            # Term 2: M2 = (A10+A11)B00
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = 1
            V[j0*p + k0][col] = 1
            W[k0*n + i1][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 3: M3 = A00(B10-B11)
            U[i0*m + j0][col] = 1
            V[j1*p + k0][col] = 1
            V[j1*p + k1][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 4: M4 = A11(B01-B00)
            U[i1*m + j1][col] = 1
            V[j0*p + k1][col] = 1
            V[j0*p + k0][col] = -1
            W[k0*n + i0][col] = -1
            W[k1*n + i0][col] = 1
            col += 1
            
            # Term 5: M5 = (A00+A10)B11
            U[i0*m + j0][col] = 1
            U[i1*m + j0][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k0*n + i1][col] = 1
            col += 1
            
            # Term 6: M6 = (A01-A00)(B00+B10)
            U[i0*m + j1][col] = 1
            U[i0*m + j0][col] = -1
            V[j0*p + k0][col] = 1
            V[j1*p + k0][col] = 1
            W[k1*n + i1][col] = -1
            col += 1
            
            # Term 7: M7 = (A10-A11)(B01+B11)
            U[i1*m + j0][col] = 1
            U[i1*m + j1][col] = -1
            V[j0*p + k1][col] = 1
            V[j1*p + k1][col] = 1
            W[k0*n + i0][col] = 1
            W[k1*n + i1][col] = 1
            col += 1
            
            return col
        
        blocks = [
            (0, 0, 0),
            (0, 2, 0),
            (0, 0, 2),
            (0, 2, 2),
            (0, 0, 4),
            (0, 2, 4),
        ]
        
        num_blocks = len(blocks)
        R = num_blocks * 7 + (n * m * p - num_blocks * 8)
        
        U = [[0] * R for _ in range(n * m)]
        V = [[0] * R for _ in range(m * p)]
        W = [[0] * R for _ in range(p * n)]
        
        col = 0
        
        for i_off, j_off, k_off in blocks:
            col = strassen_block(i_off, j_off, k_off, U, V, W, col)
        
        covered = set()
        for i_off, j_off, k_off in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i_off + di, j_off + dj, k_off + dk))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in covered:
                        continue
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        
        if col != R:
            return _schoolbook()
            
        return {"U": U, "V": V, "W": W}

    def _verify(U, V, W):
        """Verify that the decomposition exactly reconstructs the matrix multiplication."""
        R = len(U[0])
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        rng = np.random.RandomState(42)
        for trial in range(5):
            A = rng.randn(n, m)
            B = rng.randn(m, p)
            C_true = A @ B
            C_approx = np.zeros((n, p))
            A_flat = A.reshape(-1)
            B_flat = B.reshape(-1)
            for r in range(R):
                ua = Ua[:, r]
                va = Va[:, r]
                wa = Wa[:, r]
                u_dot_a = ua @ A_flat
                v_dot_b = va @ B_flat
                C_approx += (wa * (u_dot_a * v_dot_b)).reshape(p, n).T
            if not np.allclose(C_true, C_approx, atol=1e-10):
                return False
        return True

    def _winograd_cyclic():
        """
        Winograd-like approach: embed the 3x4x7 matmul into a cyclic convolution
        of length L = m + p - 1 = 10, then use FFT-like decomposition over integers.
        
        Key insight: For fixed i, the row i of C is a sum over j of A[i,j] * B[j,:].
        This is equivalent to a convolution if we pad appropriately.
        
        We'll try to find a lower-rank decomposition by exploiting the structure.
        """
        # Try different block structures
        # For 3x4x7, we can try to decompose into overlapping 2x2x2 Strassen blocks
        # plus remaining schoolbook terms
        
        best_rank = n * m * p
        best_decomp = None
        
        # Strategy: Try all ways to tile with 2x2x2 Strassen blocks
        # We have n=3 rows, m=4 cols, p=7 cols in B
        
        # Possible block placements (i_off, j_off, k_off) where:
        # 0 <= i_off <= 1 (since n=3, need 2 rows)
        # 0 <= j_off <= 2 (since m=4, need 2 cols)
        # 0 <= k_off <= 5 (since p=7, need 2 cols)
        
        # Generate all valid block placements
        all_blocks = []
        for i_off in range(n - 1):
            for j_off in range(m - 1):
                for k_off in range(p - 1):
                    all_blocks.append((i_off, j_off, k_off))
        
        # Try subsets of blocks that don't overlap
        # Use greedy: try to pack as many non-overlapping 2x2x2 blocks as possible
        
        def can_add_block(blocks_set, new_block):
            """Check if new_block can be added without overlapping existing blocks."""
            i0, j0, k0 = new_block
            for (i1, j1, k1) in blocks_set:
                # Check overlap: blocks overlap if they share any (i,j,k) position
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            pos = (i0 + di, j0 + dj, k0 + dk)
                            for di2 in range(2):
                                for dj2 in range(2):
                                    for dk2 in range(2):
                                        pos2 = (i1 + di2, j1 + dj2, k1 + dk2)
                                        if pos == pos2:
                                            return False
            return True
        
        # Greedy packing: try all orderings and pick the one with most blocks
        # Since there are many possibilities, use a simpler approach:
        # Try specific patterns
        
        patterns = [
            # Pattern 1: Current approach
            [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2), (0, 0, 4), (0, 2, 4)],
            # Pattern 2: Different arrangement
            [(0, 0, 0), (0, 2, 0), (1, 0, 0), (1, 2, 0), (0, 0, 2), (0, 2, 2)],
            # Pattern 3: Try to cover more
            [(0, 0, 0), (0, 2, 0), (1, 0, 0), (1, 2, 0), (0, 0, 2), (1, 0, 2)],
            # Pattern 4: Spread out
            [(0, 0, 0), (0, 2, 2), (1, 0, 4), (0, 0, 4), (1, 2, 0), (0, 2, 4)],
        ]
        
        for pattern in patterns:
            # Check validity: no overlaps
            valid = True
            for idx1, block1 in enumerate(pattern):
                for idx2, block2 in enumerate(pattern):
                    if idx1 >= idx2:
                        continue
                    i0, j0, k0 = block1
                    i1, j1, k1 = block2
                    for di in range(2):
                        for dj in range(2):
                            for dk in range(2):
                                pos1 = (i0 + di, j0 + dj, k0 + dk)
                                for di2 in range(2):
                                    for dj2 in range(2):
                                        for dk2 in range(2):
                                            pos2 = (i1 + di2, j1 + dj2, k1 + dk2)
                                            if pos1 == pos2:
                                                valid = False
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Count rank: each 2x2x2 block uses 7 ranks, uncovered positions use 1 rank each
            num_blocks = len(pattern)
            covered = set()
            for i_off, j_off, k_off in pattern:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            covered.add((i_off + di, j_off + dj, k_off + dk))
            
            num_uncovered = n * m * p - len(covered)
            rank = num_blocks * 7 + num_uncovered
            
            if rank < best_rank:
                # Construct this decomposition
                R = rank
                U = [[0] * R for _ in range(n * m)]
                V = [[0] * R for _ in range(m * p)]
                W = [[0] * R for _ in range(p * n)]
                
                col = 0
                for i_off, j_off, k_off in pattern:
                    col = strassen_block(i_off, j_off, k_off, U, V, W, col)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if (i, j, k) in covered:
                                continue
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
                
                if col == R:
                    best_rank = rank
                    best_decomp = {"U": U, "V": V, "W": W}
        
        if best_decomp is not None:
            return best_decomp
        return None

    # Try the Winograd-like structured approach first
    winograd_result = _winograd_cyclic()
    if winograd_result is not None and _verify(winograd_result["U"], winograd_result["V"], winograd_result["W"]):
        return winograd_result
    
    # Try the Strassen-embedded decomposition with 6 blocks
    best = _strassen_embedded(6)
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback: schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
