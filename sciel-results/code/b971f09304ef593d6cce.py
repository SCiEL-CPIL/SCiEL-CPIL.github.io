"""Matrix multiplication 4,5,5 tensor decomposition."""
import time
from fractions import Fraction

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Return an exact tensor decomposition of the matrix multiplication tensor
    for multiplying a 4x5 matrix by a 5x5 matrix.
    
    The decomposition is of the form:
        T[i, j, k] = sum_c U[i, c] * V[j, c] * W[k, c]
    
    where T is the matrix multiplication tensor (T[i,j,k] = 1 for all i,j,k
    in the appropriate ranges, following the trace convention).
    
    We use a combination of Strassen's 7-term decomposition for 2x2x2 blocks
    and schoolbook terms for the remaining positions.
    """
    start_time = time.time()
    deadline = start_time + 115  # 15s safety margin before 130s kill
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Strassen's 7-term decomposition for 2x2x2
    # Each term is (u, v, w) where u, v, w are 4-element lists
    # representing coefficients for the four sub-matrices
    strassen_terms = [
        ([1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1]),  # (A11+A22)(B11+B22)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),   # (A21+A22)B11
        ([1, 0, 0, 0], [0, 1, -1, 0], [0, 1, 0, 1]),   # A11(B21-B11)
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # A22(B12-B22)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 1]),   # (A11+A12)B22
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # (A21-A11)(B11+B12)
        ([0, 1, -1, 0], [0, 0, 1, 1], [1, 0, 0, 0]),   # (A12-A22)(B21+B22)
    ]
    
    def verify(U, V, W):
        """Verify the decomposition exactly using Fraction arithmetic."""
        rank = len(U[0]) if U else 0
        if rank == 0:
            return n * m * p == 0
        
        # Check a sample of positions first for speed
        import random
        rng = random.Random(42)
        
        # Always check corners
        test_positions = set()
        test_positions.add((0, 0, 0))
        test_positions.add((n-1, m-1, p-1))
        test_positions.add((0, 0, p-1))
        test_positions.add((n-1, m-1, 0))
        test_positions.add((n-1, 0, 0))
        test_positions.add((0, m-1, p-1))
        
        # Add random samples
        for _ in range(min(50, n*m*p)):
            test_positions.add((rng.randint(0, n-1), rng.randint(0, m-1), rng.randint(0, p-1)))
        
        for i, j, k in test_positions:
            s = Fraction(0)
            for c in range(rank):
                uc = U[i * m + j][c]
                vc = V[j * p + k][c]
                wc = W[k * n + i][c]
                if uc != 0 and vc != 0 and wc != 0:
                    s += Fraction(uc) * Fraction(vc) * Fraction(wc)
            if s != 1:
                return False
        
        # If sample passes, do full verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        uc = U[i * m + j][c]
                        vc = V[j * p + k][c]
                        wc = W[k * n + i][c]
                        if uc != 0 and vc != 0 and wc != 0:
                            s += Fraction(uc) * Fraction(vc) * Fraction(wc)
                    if s != 1:
                        return False
        
        return True
    
    def build_decomposition(blocks):
        """
        Build decomposition given a set of non-overlapping 2x2x2 blocks.
        Each block is (i0, j0, k0) covering i0..i0+1, j0..j0+1, k0..k0+1.
        """
        # Determine which (i,j,k) are covered by blocks
        covered = set()
        for (i0, j0, k0) in blocks:
            for di in range(2):
                for dj in range(2):
                    for dk in range(2):
                        covered.add((i0 + di, j0 + dj, k0 + dk))
        
        # Count schoolbook terms (not in blocks)
        num_schoolbook = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        num_schoolbook += 1
        
        num_strassen = len(blocks) * 7
        rank = num_schoolbook + num_strassen
        
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        
        # Schoolbook terms for uncovered (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
        
        # Strassen terms for each block
        for (i0, j0, k0) in blocks:
            # Define indices for the four sub-matrices
            # U indices: (i, j) pairs
            a_idx = [(i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)]
            # V indices: (j, k) pairs
            b_idx = [(j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)]
            # W indices: (k, i) pairs (trace convention: W[k*n+i])
            c_idx = [(k0, i0), (k0, i0+1), (k0+1, i0), (k0+1, i0+1)]
            
            for (u_c, v_c, w_c) in strassen_terms:
                for la, cf in enumerate(u_c):
                    if cf != 0:
                        i, j = a_idx[la]
                        U[i * m + j][col] += cf
                for lb, cf in enumerate(v_c):
                    if cf != 0:
                        j, k = b_idx[lb]
                        V[j * p + k][col] += cf
                for lc, cf in enumerate(w_c):
                    if cf != 0:
                        k, i = c_idx[lc]
                        W[k * n + i][col] += cf
                col += 1
        
        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}
    
    # Find optimal non-overlapping 2x2x2 blocks
    # For n=4, m=5, p=5:
    # i0 can be 0 or 2 (since i0+1 < 4, so i0 < 3)
    # j0 can be 0, 1, 2, 3 (since j0+1 < 5, so j0 < 4)
    # k0 can be 0, 1, 2, 3 (since k0+1 < 5, so k0 < 4)
    
    # Non-overlap conditions:
    # Two blocks (i0a, j0a, k0a) and (i0b, j0b, k0b) overlap if:
    # |i0a - i0b| < 2 AND |j0a - j0b| < 2 AND |k0a - k0b| < 2
    
    # For n=4: i0=0 covers {0,1}, i0=2 covers {2,3} -> no overlap
    # For m=5: j0=0 covers {0,1}, j0=1 covers {1,2}, j0=2 covers {2,3}, j0=3 covers {3,4}
    #   Non-overlapping pairs: (0,2), (0,3), (1,3)
    # For p=5: same as m
    
    # Maximum set of non-overlapping blocks:
    # i: {0, 2} -> 2 blocks
    # j: {0, 2} or {1, 3} or {0, 3} -> up to 2 blocks
    # k: {0, 2} or {1, 3} or {0, 3} -> up to 2 blocks
    # Total: 2 * 2 * 2 = 8 blocks
    
    # Let's enumerate all valid combinations of non-overlapping blocks
    # and find the one with maximum count
    
    def blocks_overlap(b1, b2):
        i0a, j0a, k0a = b1
        i0b, j0b, k0b = b2
        # Check if the 2x2x2 regions overlap
        i_overlap = (i0a <= i0b + 1) and (i0b <= i0a + 1)
        j_overlap = (j0a <= j0b + 1) and (j0b <= j0a + 1)
        k_overlap = (k0a <= k0b + 1) and (k0b <= k0a + 1)
        return i_overlap and j_overlap and k_overlap
    
    # All possible block positions
    all_blocks = []
    for i0 in range(0, n - 1):  # i0+1 < n
        for j0 in range(0, m - 1):  # j0+1 < m
            for k0 in range(0, p - 1):  # k0+1 < p
                all_blocks.append((i0, j0, k0))
    
    # Find maximum independent set using greedy + backtracking
    # Since the number of blocks is small, we can do a simple search
    
    best_blocks = []
    best_count = 0
    
    # Try all combinations of i0 values: {0}, {2}, {0,2}
    # For each, try non-overlapping j0 and k0 values
    
    i_options = [0, 2]  # Only these two don't overlap with each other
    
    # For each pair of i values, find max non-overlapping j and k
    for i_set in [[0], [2], [0, 2]]:
        # Find max non-overlapping j values
        j_options = list(range(m - 1))  # 0, 1, 2, 3
        # Greedy: sort by position, pick greedily
        # Actually for m=5, the max independent set in the path graph
        # j0=0,1,2,3 with edges between adjacent values
        # Max independent set: {0,2} or {0,3} or {1,3} -> size 2
        
        # Similarly for k
        k_options = list(range(p - 1))
        
        # Try all subsets of j of size 2
        from itertools import combinations
        
        best_jk_for_i = None
        best_jk_size = 0
        
        for j_subset in combinations(j_options, 2):
            # Check if j_subset is independent
            if blocks_overlap((0, j_subset[0], 0), (0, j_subset[1], 0)):
                continue
            for k_subset in combinations(k_options, 2):
                # Check if k_subset is independent
                if blocks_overlap((0, 0, k_subset[0]), (0, 0, k_subset[1])):
                    continue
                # Now check all 4 blocks are mutually non-overlapping
                candidate_blocks = []
                for i0 in i_set:
                    for j0 in j_subset:
                        for k0 in k_subset:
                            candidate_blocks.append((i0, j0, k0))
                
                # Check mutual non-overlap
                valid = True
                for a in range(len(candidate_blocks)):
                    for b in range(a + 1, len(candidate_blocks)):
                        if blocks_overlap(candidate_blocks[a], candidate_blocks[b]):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid and len(candidate_blocks) > best_jk_size:
                    best_jk_size = len(candidate_blocks)
                    best_jk_for_i = candidate_blocks
        
        if best_jk_for_i is not None and best_jk_size > best_count:
            best_count = best_jk_size
            best_blocks = best_jk_for_i
    
    # Also try single i values with potentially more j,k blocks
    for i0 in i_options:
        # With single i0, we can have more j and k blocks
        # But since i is fixed, j and k are independent
        # Max j blocks: 2, max k blocks: 2, total: 4
        # This is less than 8, so skip
        
        pass
    
    # Build the decomposition with the best blocks found
    if best_blocks and len(best_blocks) > 0:
        candidate = build_decomposition(best_blocks)
        if verify(candidate["U"], candidate["V"], candidate["W"]):
            return candidate
    
    # Fallback: try with fewer blocks if the above fails
    # Try just 4 blocks: (0,0,0), (0,2,0), (2,0,0), (2,2,0)
    fallback_blocks = [(0, 0, 0), (0, 2, 0), (2, 0, 0), (2, 2, 0)]
    candidate = build_decomposition(fallback_blocks)
    if verify(candidate["U"], candidate["V"], candidate["W"]):
        return candidate
    
    # Final fallback: pure schoolbook
    rank = n * m * p
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
# EVOLVE-BLOCK-END
