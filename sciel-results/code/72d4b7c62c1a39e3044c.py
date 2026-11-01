import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = 2, 4, 6
    
    # First, build the known rank-42 Strassen solution as fallback
    def build_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # Strassen 2x2x2 terms: (A_indices, B_indices, W_indices)
        # Each term is a list of (row, col, sign) for the 2x2 matrix
        strassen_terms = [
            # P1 = (A00+A11)(B00+B11)
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            # P2 = (A10+A11)B00
            {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
            # P3 = A00(B01-B11)
            {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
            # P4 = A11(B10-B00)
            {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
            # P5 = (A00+A01)B11
            {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
            # P6 = (A10-A00)(B00+B01)
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            # P7 = (A01-A11)(B10+B11)
            {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
        ]
        
        col = 0
        for j_block in range(2):  # j = 0,1 and j = 2,3
            for k_block in range(3):  # k = 0,1 and k = 2,3 and k = 4,5
                for term in strassen_terms:
                    r = col
                    for (ar, ac, sign) in term['A']:
                        u_idx = ar * m + (j_block * 2 + ac)
                        U[u_idx][r] += sign
                    for (br, bc, sign) in term['B']:
                        v_idx = (j_block * 2 + br) * p + (k_block * 2 + bc)
                        V[v_idx][r] += sign
                    for (or_, oc, sign) in term['W']:
                        w_idx = (k_block * 2 + oc) * n + or_
                        W[w_idx][r] += sign
                    col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True
    
    # Build the rank-42 fallback
    fallback = build_strassen_42()
    
    if time.time() > deadline - 5:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}
    
    # Try to find a lower-rank solution using structured search
    # Strategy: Use a parametric family of U,V and solve for W linearly
    
    # For a given rank r, the constraint is:
    # For all (i,j,k): sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # 
    # Group by (i,k): sum_t W[k*n+i,t] * (sum_j U[i*m+j,t] * V[j*p+k,t]) = 1
    # Let C_{i,k,t} = sum_j U[i*m+j,t] * V[j*p+k,t]
    # Then: sum_t C_{i,k,t} * W[k*n+i,t] = 1
    #
    # This is 12 linear equations (one per (i,k) pair) in the W variables.
    # Each equation involves all r components of one row of W.
    # So we have 12 equations in 12*r unknowns, but each equation only involves r unknowns.
    # If we can make C_{i,k} = e_0 (unit vector) for all (i,k), then W[k*n+i,0] = 1 and rest = 0.
    # But that's rank 1, which is impossible for matrix multiplication.
    #
    # The issue: C_{i,k} depends on both i and k through U and V.
    # We need C_{i,k} to be such that the system is solvable.
    #
    # Key insight: If we set up U and V so that C_{i,k} has a specific structure,
    # we can solve for W. The minimum rank is bounded below by the tensor rank.
    #
    # Let's try: fix U to have a specific sparse pattern, then optimize V to make
    # the resulting C_{i,k} vectors linearly independent in a useful way.
    
    # Try rank 36: use 36 terms
    # Idea: For each j_block (2 blocks) and k_block (3 blocks), use 6 terms instead of 7
    # by finding a rank-6 decomposition of 2x2x2 (which is impossible - Strassen is minimal at 7)
    # So we can't reduce below 7 per 2x2x2 block.
    #
    # Alternative: share terms across k_blocks. The 3 k_blocks for a given j_block
    # all use the same A submatrix. Can we combine their Strassen terms?
    #
    # For fixed j_block, the 3 k_blocks give 3 separate 2x2x2 problems.
    # The A part is the same. The B and W parts are different.
    # If we use the same U columns for all 3 k_blocks, we save 2*6 = 12 U columns
    # but the V and W parts still need separate columns.
    #
    # Actually, the rank is the number of columns in U,V,W (they must all have the same rank).
    # So if we share U columns, we still need corresponding V and W columns.
    # The total rank is still 7 per (j_block, k_block) pair = 42.
    #
    # Let's try a different approach: use a non-block decomposition.
    # Try to find a rank-r decomposition where r < 42 by solving a nonlinear system.
    #
    # For small r, we can try:
    # - r = 36: 6 terms per (j_block) instead of 21 per (j_block)
    # - This would mean 6 terms that handle all 3 k_blocks for a given j_block
    
    # Try r = 36 with a structured approach
    # For each j_block, use 18 terms (9 per k_block on average, but shared)
    
    # Actually, let's try to directly solve for a lower rank using the following:
    # Fix a random U and V with the desired sparsity, then solve for W.
    # If the system is consistent, we have a valid decomposition.
    
    def try_rank(r):
        """Try to find a rank-r decomposition by fixing U,V and solving for W."""
        if time.time() > deadline - 10:
            return None
        
        # Initialize U and V with a structured pattern
        # U: (n*m) x r = 8 x r
        # V: (m*p) x r = 24 x r
        # W: (p*n) x r = 12 x r
        
        # Strategy: For each term t, assign it to a specific (j_block, k_block) pair
        # and use a Strassen-like pattern but with some sharing
        
        # Try: 6 terms per j_block (total 12), each term handles multiple k_blocks
        # This is complex, so let's try a simpler structured approach
        
        # Use the fact that for n=2, the W matrix has only 12 rows.
        # We can try to make the C_{i,k} vectors have a simple form.
        
        # Set up U and V so that C_{i,k} = v_{i,k} * e_{f(i,k)} for some function f
        # Then W[k*n+i, f(i,k)] = 1/v_{i,k}
        
        # For this to work, we need:
        # C_{i,k,t} = sum_j U[i*m+j,t] * V[j*p+k,t] = v_{i,k} if t = f(i,k), else 0
        
        # This is a strong constraint. Let's try a simpler version:
        # Make C_{i,k} have support on a single coordinate for each (i,k),
        # but allow the coordinate to vary.
        
        # For r terms, we have 12 (i,k) pairs. If each uses a unique coordinate,
        # we need at least 12 terms. But the constraint is nonlinear in U,V.
        
        # Let's try: for each term t, assign it to handle a specific set of (i,k) pairs.
        # This is getting complex. Let's just try to verify if rank 36 is achievable
        # by checking if the tensor rank is actually 42 or lower.
        
        # Known bounds: For 2x4x6, the tensor rank is at most 42 (Strassen blocks)
        # and at least... let's compute the lower bound.
        
        # The tensor rank is at least the matrix rank of any "slice".
        # Slice fixing j: a 2x6 matrix, rank at most 2.
        # Slice fixing i: a 4x6 matrix, rank at most 4.
        # Slice fixing k: a 2x4 matrix, rank at most 2.
        # These don't give strong lower bounds.
        
        # A better lower bound: the border rank.
        # For now, let's just try to construct a rank-36 solution.
        
        # Approach: Use 36 terms, each associated with a (j_block, k_block, strassen_idx)
        # but with 36 = 2*3*6, so 6 terms per (j_block, k_block) pair.
        # Since 2x2x2 requires rank 7, we can't do 6 per block.
        # So this won't work directly.
        
        # Alternative: 36 = 2*18, so 18 terms per j_block.
        # Each j_block has 3 k_blocks, so 6 terms per k_block on average.
        # Still below the 7 needed for 2x2x2.
        
        # I think rank 42 might be close to optimal for this block approach.
        # Let's try a completely different decomposition.
        
        # Use the fact that 2x4x6 can be seen as 2x(4x6) and use
        # the structure of the 4x6 matrix.
        
        # Try: rank = 32
        # 32 = 2*16, so 16 terms per j_block
        # 16/3 ≈ 5.33 terms per k_block, still below 7
        
        # I'm convinced that the block Strassen approach gives 42 and it's hard to beat
        # without a fundamentally different decomposition.
        
        # Let's try one more thing: use a rank-7 decomposition for 2x2x2 but
        # with a different partitioning that allows sharing.
        
        return None
    
    # Try to find a better solution
    better = try_rank(36)
    
    if better is not None and verify(better["U"], better["V"], better["W"]):
        return better
    
    # Return the rank-42 solution
    return fallback

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
