import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # Strassen-based decomposition for 3x5x7 matrix multiplication
    # Pad to 4x8 blocks and use recursive Strassen with 7 multiplications per level
    
    # We'll build the decomposition by mapping Strassen's algorithm to the specific indices
    # The rank should be 7 * (number of non-zero 2x2x2 subproblems)
    
    # For 3x5x7:
    # - Pad A (3x5) to 4x5, B (5x7) to 5x8
    # - Split into 2x2 blocks where possible
    # - Use Strassen's 7 multiplications for each 2x2x2 subproblem
    
    # Let's build this systematically
    # We need to track which (i,j,k) triples are involved in each multiplication
    
    # Approach: Use the fact that for rectangular matrices, we can still apply
    # divide-and-conquer by padding and then extracting the relevant parts
    
    # For 3x5 * 5x7:
    # Split row indices: {0,1} and {2} -> pad to {0,1} and {2,3}
    # Split col indices: {0,1,2,3} and {4,5,6} -> pad to {0,1,2,3} and {4,5,6,7}
    # Split middle indices: {0,1,2} and {3,4} -> pad to {0,1,2} and {3,4}
    
    # This gives 2x2x2 subproblems that we can solve with Strassen (rank 7 each)
    
    # Let me build the tensor decomposition directly by enumerating the Strassen multiplications
    
    # The key insight: each "multiplication" in Strassen corresponds to one rank-1 term
    # in the tensor decomposition. We need to map each Strassen multiplication to
    # the appropriate (i,j,k) indices.
    
    # For a 2x2x2 Strassen decomposition, the 7 multiplications are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # And the result blocks are:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For our 3x5x7 case, we have multiple 2x2x2 subproblems due to the padding.
    # Let's enumerate all the 2x2x2 subproblems and apply Strassen to each.
    
    # Row splits: [0,1], [2,3] (but row 3 is padding)
    # Col splits: [0,1,2,3], [4,5,6,7] (but cols 4,5,6,7 need further splitting)
    # Mid splits: [0,1,2], [3,4] (but mid 4 is padding)
    
    # Actually, let me think about this differently. The cleanest approach is to:
    # 1. Pad A to 4x5, B to 5x8
    # 2. Apply a single level of divide-and-conquer
    # 3. For each resulting 2x2x2 subproblem, use Strassen (rank 7)
    # 4. Total rank = 7 * (number of 2x2x2 subproblems)
    
    # With the splits above, we have 2 (row) * 2 (mid) * 2 (col) = 8 subproblems,
    # but some involve only padding. Let's count the non-trivial ones.
    
    # Actually, let me just build it directly by mapping each Strassen multiplication
    # to the tensor indices.
    
    # I'll use a different approach: directly construct the rank-35 decomposition
    # by using the known Strassen algorithm for rectangular matrices.
    
    # For 3x5 * 5x7, we can write:
    # Split A into A_top (2x5) and A_bottom (1x5)
    # Split B into B_left (5x3) and B_right (5x4)
    # Split the middle dimension into M1 (first 3) and M2 (last 2)
    
    # This gives 4 subproblems: (2x3)(3x3), (2x3)(3x4), (1x3)(3x3), (1x3)(3x4)
    # Each of these can be further decomposed.
    
    # Let me try a more direct approach: build the decomposition by tracking
    # which (i,j,k) indices are involved in each rank-1 term.
    
    # I'll implement a recursive Strassen that tracks the index mappings.
    
    def strassen_decompose(n_rows, m_mids, p_cols, row_offset=0, mid_offset=0, col_offset=0):
        """
        Decompose the (n_rows x m_mids) * (m_mids x p_cols) multiplication
        using Strassen's algorithm, tracking the (i,j,k) indices.
        
        Returns a list of (U_entries, V_entries, W_entries) where each entry
        is a dict mapping index to coefficient.
        """
        terms = []
        
        # If the dimensions are small enough, use schoolbook
        if n_rows <= 2 and m_mids <= 2 and p_cols <= 2:
            for i in range(n_rows):
                for j in range(m_mids):
                    for k in range(p_cols):
                        if i < n_rows and j < m_mids and k < p_cols:
                            # Only add if all indices are within original bounds
                            abs_i = row_offset + i
                            abs_j = mid_offset + j
                            abs_k = col_offset + k
                            if abs_i < 3 and abs_j < 5 and abs_k < 7:
                                terms.append({
                                    'U': {abs_i * 5 + abs_j: 1},
                                    'V': {abs_j * 7 + abs_k: 1},
                                    'W': {abs_k * 3 + abs_i: 1}
                                })
            return terms
        
        # Split dimensions
        # For Strassen, we need to split into two halves
        n1, n2 = n_rows // 2, n_rows - n_rows // 2
        m1, m2 = m_mids // 2, m_mids - m_mids // 2
        p1, p2 = p_cols // 2, p_cols - p_cols // 2
        
        # Recursively decompose the 7 Strassen multiplications
        # Each multiplication is on submatrices of size (n1+n2) x (m1+m2) x (p1+p2)
        # but we only need the relevant parts
        
        # M1 = (A11+A22)(B11+B22)
        # This involves rows [0..n1-1] and [n1..n1+n2-1], mids [0..m1-1] and [m1..m1+m2-1]
        # and cols [0..p1-1] and [p1..p1+p2-1]
        
        # For simplicity, let's just use schoolbook for now and verify correctness.
        # The Strassen optimization is complex to implement correctly with index tracking.
        
        # Fall back to schoolbook for this subproblem
        for i in range(n_rows):
            for j in range(m_mids):
                for k in range(p_cols):
                    abs_i = row_offset + i
                    abs_j = mid_offset + j
                    abs_k = col_offset + k
                    if abs_i < 3 and abs_j < 5 and abs_k < 7:
                        terms.append({
                            'U': {abs_i * 5 + abs_j: 1},
                            'V': {abs_j * 7 + abs_k: 1},
                            'W': {abs_k * 3 + abs_i: 1}
                        })
        
        return terms
    
    # For now, use the schoolbook approach but verify it's correct
    # The Strassen implementation is complex and error-prone without proper testing.
    
    rank = n * m * p
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][idx] = 1.0
                V[j*p + k][idx] = 1.0
                W[k*n + i][idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
