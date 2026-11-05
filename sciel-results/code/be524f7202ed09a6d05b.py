import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank decomposition for 5x5 matrix multiplication.
    Uses Strassen's algorithm for 2x2 blocks where applicable.
    Returns the best valid decomposition found.
    """
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    n, m, p = DIMENSIONS
    
    def build_strassen_2x2(terms, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """
        Add Strassen 2x2 multiplication terms.
        A is 2x2 at rows [a_row_offset, a_row_offset+1], cols [a_col_offset, a_col_offset+1]
        B is 2x2 at rows [b_row_offset, b_row_offset+1], cols [b_col_offset, b_col_offset+1]
        C is 2x2 at rows [c_row_offset, c_row_offset+1], cols [c_col_offset, c_col_offset+1]
        
        Strassen formulas:
        M1 = (A11 + A22)(B11 + B22)
        M2 = (A21 + A22)B11
        M3 = A11(B21 - B22)
        M4 = A22(B12 - B11)
        M5 = (A11 + A12)B22
        M6 = (A21 - A11)(B11 + B12)
        M7 = (A12 - A22)(B21 + B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        # Define the 7 Strassen multiplications
        # Each is a product of linear combinations of A entries and B entries
        # We need to expand each into individual U,V,W terms
        
        # M1 = (A11 + A22)(B11 + B22)
        # This expands to 4 terms: A11*B11, A11*B22, A22*B11, A22*B22
        # But we can factor: (A11+A22) * (B11+B22)
        # In tensor decomposition, we need to represent this as sum of rank-1 terms
        # The trick: we create intermediate "virtual" entries
        
        # For each Strassen multiplication Mi = (sum of A entries) * (sum of B entries),
        # we create one term per pair of (A entry, B entry) in the sums
        
        # M1: (A11+A22) * (B11+B22)
        m1_a = [(0,0,1), (1,1,1)]  # (row_offset, col_offset, coeff)
        m1_b = [(0,0,1), (1,1,1)]
        
        # M2: (A21+A22) * B11
        m2_a = [(1,0,1), (1,1,1)]
        m2_b = [(0,0,1)]
        
        # M3: A11 * (B21-B22)
        m3_a = [(0,0,1)]
        m3_b = [(1,0,1), (1,1,-1)]
        
        # M4: A22 * (B12-B11)
        m4_a = [(1,1,1)]
        m4_b = [(0,1,1), (0,0,-1)]
        
        # M5: (A11+A12) * B22
        m5_a = [(0,0,1), (0,1,1)]
        m5_b = [(1,1,1)]
        
        # M6: (A21-A11) * (B11+B12)
        m6_a = [(1,0,1), (0,0,-1)]
        m6_b = [(0,0,1), (0,1,1)]
        
        # M7: (A12-A22) * (B21+B22)
        m7_a = [(0,1,1), (1,1,-1)]
        m7_b = [(1,0,1), (1,1,1)]
        
        # For each Mi, create terms for each (a_entry, b_entry) pair
        strassen_mults = [
            (m1_a, m1_b, [
                # C11 = M1 + M4 - M5 + M7
                (0,0,1),  # M1 contributes +1 to C11
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                (1,1,1),  # M1 contributes +1 to C22
            ]),
            (m2_a, m2_b, [
                # C11: 0
                # C12: 0
                # C21 = M2 + M4: M2 contributes +1 to C21
                (1,0,1),  # M2 contributes +1 to C21
                # C22 = M1 - M2 + M3 + M6: M2 contributes -1 to C22
                (1,1,-1),  # M2 contributes -1 to C22
            ]),
            (m3_a, m3_b, [
                # C11: 0
                # C12 = M3 + M5: M3 contributes +1 to C12
                (0,1,1),  # M3 contributes +1 to C12
                # C21: 0
                # C22 = M1 - M2 + M3 + M6: M3 contributes +1 to C22
                (1,1,1),  # M3 contributes +1 to C22
            ]),
            (m4_a, m4_b, [
                # C11 = M1 + M4 - M5 + M7: M4 contributes +1 to C11
                (0,0,1),  # M4 contributes +1 to C11
                # C12: 0
                # C21 = M2 + M4: M4 contributes +1 to C21
                (1,0,1),  # M4 contributes +1 to C21
                # C22: 0
            ]),
            (m5_a, m5_b, [
                # C11 = M1 + M4 - M5 + M7: M5 contributes -1 to C11
                (0,0,-1),  # M5 contributes -1 to C11
                # C12 = M3 + M5: M5 contributes +1 to C12
                (0,1,1),  # M5 contributes +1 to C12
                # C21: 0
                # C22: 0
            ]),
            (m6_a, m6_b, [
                # C11: 0
                # C12: 0
                # C21: 0
                # C22 = M1 - M2 + M3 + M6: M6 contributes +1 to C22
                (1,1,1),  # M6 contributes +1 to C22
            ]),
            (m7_a, m7_b, [
                # C11 = M1 + M4 - M5 + M7: M7 contributes +1 to C11
                (0,0,1),  # M7 contributes +1 to C11
                # C12: 0
                # C21: 0
                # C22: 0
            ]),
        ]
        
        for m_a, m_b, c_contribs in strassen_mults:
            for a_row, a_col, a_coeff in m_a:
                for b_row, b_col, b_coeff in m_b:
                    u = (a_row_offset + a_row) * 5 + (a_col_offset + a_col)
                    v = (b_row_offset + b_row) * 5 + (b_col_offset + b_col)
                    coeff = a_coeff * b_coeff
                    for c_row, c_col, c_coeff in c_contribs:
                        w = (c_col_offset + c_col) * 5 + (c_row_offset + c_row)
                        terms.append((u, v, w, coeff * c_coeff))
    
    def build_schoolbook_block(terms, a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, a_size, b_size, c_size):
        """Add schoolbook multiplication terms for blocks of given sizes."""
        for ii in range(a_size):
            for jj in range(b_size):
                for kk in range(c_size):
                    u = (a_row_offset + ii) * 5 + (a_col_offset + jj)
                    v = (b_row_offset + jj) * 5 + (b_col_offset + kk)
                    w = (c_col_offset + kk) * 5 + (c_row_offset + ii)
                    terms.append((u, v, w, 1.0))
    
    def build_strassen_partition():
        """
        Build decomposition using 2+2+1 block partition with Strassen for 2x2 blocks.
        Blocks: A rows [0:2], [2:4], [4:5]; A cols [0:2], [2:4], [4:5]
                B rows [0:2], [2:4], [4:5]; B cols [0:2], [2:4], [4:5]
                C rows [0:2], [2:4], [4:5]; C cols [0:2], [2:4], [4:5]
        
        Block sizes: [2, 2, 1]
        Offsets: [0, 2, 4, 5]
        
        For each output block C[i,k], sum over j: A[i,j] * B[j,k]
        - If A[i,j] and B[j,k] are both 2x2, use Strassen (7 terms instead of 8)
        - Otherwise use schoolbook
        """
        terms = []
        block_sizes = [2, 2, 1]
        offsets = [0, 2, 4, 5]
        
        for i in range(3):  # C row block
            for k in range(3):  # C col block
                for j in range(3):  # sum over A col / B row block
                    a_row_off = offsets[i]
                    a_col_off = offsets[j]
                    b_row_off = offsets[j]
                    b_col_off = offsets[k]
                    c_row_off = offsets[i]
                    c_col_off = offsets[k]
                    
                    a_size = block_sizes[i]
                    a_col_size = block_sizes[j]
                    b_row_size = block_sizes[j]
                    b_col_size = block_sizes[k]
                    
                    # Check if we can use Strassen: need a_size==2, a_col_size==2, b_row_size==2, b_col_size==2
                    if a_size == 2 and a_col_size == 2 and b_row_size == 2 and b_col_size == 2:
                        build_strassen_2x2(terms, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off)
                    else:
                        build_schoolbook_block(terms, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, 
                                             a_size, a_col_size, b_col_size)
        
        return terms
    
    def verify_and_build(terms):
        """Build U,V,W from terms and verify. Return (rank, U, V, W) or None if invalid."""
        rank = len(terms)
        
        U = np.zeros((25, rank), dtype=np.float64)
        V = np.zeros((25, rank), dtype=np.float64)
        W = np.zeros((25, rank), dtype=np.float64)
        
        for r, (u, v, w, coeff) in enumerate(terms):
            U[u, r] = coeff
            V[v, r] = coeff
            W[w, r] = coeff
        
        # Verify: for all (a_row, a_col, b_row, b_col), 
        # sum_r U[a_row*5+a_col, r] * V[b_row*5+b_col, r] * W[b_col*5+a_row, r] = 1 if a_col==b_row else 0
        
        max_err = 0.0
        for a_row in range(5):
            for a_col in range(5):
                for b_row in range(5):
                    for b_col in range(5):
                        expected = 1.0 if a_col == b_row else 0.0
                        val = 0.0
                        for r in range(rank):
                            val += U[a_row*5+a_col, r] * V[b_row*5+b_col, r] * W[b_col*5+a_row, r]
                        err = abs(val - expected)
                        if err > max_err:
                            max_err = err
                        if max_err > 1e-10:
                            return None
        
        return (rank, U, V, W)
    
    # Try Strassen partition
    best_result = None
    best_rank = 126
    
    if time.time() < deadline - safety_margin:
        terms = build_strassen_partition()
        result = verify_and_build(terms)
        
        if result is not None:
            rank, U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_result = (U, V, W)
    
    if best_result is None:
        # Fallback to schoolbook
        rank = 125
        U = np.zeros((25, rank), dtype=np.float64)
        V = np.zeros((25, rank), dtype=np.float64)
        W = np.zeros((25, rank), dtype=np.float64)
        r = 0
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    U[i*5+j, r] = 1.0
                    V[j*5+k, r] = 1.0
                    W[k*5+i, r] = 1.0
                    r += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
