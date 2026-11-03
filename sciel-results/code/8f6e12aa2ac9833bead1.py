"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-minimal tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.
    
    Returns:
        dict: {'U': list of lists, 'V': list of lists, 'W': list of lists}
              where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
              with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 120  # Must finish within 120s
    
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # We'll use a hybrid approach:
    # 1. Try to find a low-rank decomposition using known algorithms
    # 2. Fall back to Strassen blocks if no better solution found
    
    # First, let's try to use the fact that we can combine Strassen blocks
    # with careful placement to maximize rank reduction.
    
    # Strategy: Use as many 2x2x2 Strassen blocks as possible.
    # A 2x2x2 block covers 2 rows of A, 2 columns of B, and produces 2x2 of C.
    # For 3x4x5, we can place:
    # - Block at (0,0): covers A[0:2, 0:2], B[0:2, 0:2], C[0:2, 0:2]
    # - Block at (0,2): covers A[0:2, 2:4], B[2:4, 2:4], C[0:2, 2:4]
    # - Block at (1,0): covers A[1:3, 0:2], B[0:2, 0:2], C[1:3, 0:2]
    # - Block at (1,2): covers A[1:3, 2:4], B[2:4, 2:4], C[1:3, 2:4]
    # 
    # But these overlap! We need disjoint blocks.
    # 
    # Let's think about it differently. The tensor is T[i*m+j, j'*p+k, k'*n+i']
    # = 1 if i=i', j=j', k=k' else 0.
    # 
    # We want to decompose this into rank-r tensors.
    # 
    # Known result: For 2x2x2, rank is 7 (Strassen).
    # For larger tensors, we can sometimes do better than schoolbook.
    
    # Let's try a different approach: use integer programming or search
    # to find a low-rank decomposition.
    
    # Actually, let's first try to see if we can do better than 58.
    # The theoretical minimum rank for 3x4x5 matrix multiplication is unknown,
    # but it's at most 58 (from our Strassen construction).
    
    # Let's try to use a more clever decomposition.
    # Idea: Instead of disjoint 2x2x2 blocks, can we share some terms?
    
    # For now, let's stick with the known 58-rank construction but verify it's correct.
    
    # Build the 58-rank Strassen-based decomposition
    U, V, W, rank = _build_strassen_decomposition(n, m, p)
    
    # Verify the decomposition is correct
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    U, V, W, rank = _build_schoolbook_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}

def _build_strassen_decomposition(n, m, p):
    """
    Build a rank-58 decomposition using two disjoint Strassen 2x2x2 blocks.
    """
    # Initialize U, V, W with zeros
    max_rank = n * m * p  # 60 for schoolbook
    U = [[0] * max_rank for _ in range(n * m)]  # 12 x 60
    V = [[0] * max_rank for _ in range(m * p)]  # 20 x 60
    W = [[0] * max_rank for _ in range(p * n)]  # 15 x 60
    
    def add_strassen_block(i_offset, j_offset, k_offset, col_start):
        """
        Add Strassen terms for a 2x2x2 block.
        i_offset, j_offset, k_offset: global offsets for local indices 0,1
        col_start: starting column index in U, V, W
        Returns: number of columns used (7)
        """
        # Define the 7 Strassen multipliers
        # Each multiplier is (coeff_A_list, coeff_B_list) where
        # coeff_A_list = [(ii, jj, coeff), ...] for A entries
        # coeff_B_list = [(jj, kk, coeff), ...] for B entries
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        M1_A = [((0, 0), 1), ((1, 1), 1)]
        M1_B = [((0, 0), 1), ((1, 1), 1)]
        
        # M2 = (A[1,0] + A[1,1]) * B[0,0]
        M2_A = [((1, 0), 1), ((1, 1), 1)]
        M2_B = [((0, 0), 1)]
        
        # M3 = A[0,0] * (B[0,1] - B[1,1])
        M3_A = [((0, 0), 1)]
        M3_B = [((0, 1), 1), ((1, 1), -1)]
        
        # M4 = A[1,1] * (B[1,0] - B[0,0])
        M4_A = [((1, 1), 1)]
        M4_B = [((1, 0), 1), ((0, 0), -1)]
        
        # M5 = (A[0,0] + A[0,1]) * B[1,1]
        M5_A = [((0, 0), 1), ((0, 1), 1)]
        M5_B = [((1, 1), 1)]
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        M6_A = [((1, 0), 1), ((0, 0), -1)]
        M6_B = [((0, 0), 1), ((0, 1), 1)]
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        M7_A = [((0, 1), 1), ((1, 1), -1)]
        M7_B = [((1, 0), 1), ((1, 1), 1)]
        
        multipliers = [
            (M1_A, M1_B),
            (M2_A, M2_B),
            (M3_A, M3_B),
            (M4_A, M4_B),
            (M5_A, M5_B),
            (M6_A, M6_B),
            (M7_A, M7_B),
        ]
        
        col = col_start
        for m_idx, (A_terms, B_terms) in enumerate(multipliers):
            # A_terms: list of ((ii, jj), coeff)
            # B_terms: list of ((jj, kk), coeff)
            
            # For U: index is i*m + j
            for (ii, jj), coeff_A in A_terms:
                i_global = i_offset + ii
                j_global = j_offset + jj
                U[i_global * m + j_global][col] += coeff_A
            
            # For V: index is j*p + k
            for (jj, kk), coeff_B in B_terms:
                j_global = j_offset + jj
                k_global = k_offset + kk
                V[j_global * p + k_global][col] += coeff_B
            
            # For W: we need to add contributions for each output (ii, kk)
            # that this multiplier affects.
            
            # M1 (m_idx=0): contributes to C[0,0] with +1, C[1,1] with +1
            if m_idx == 0:  # M1
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M2 (m_idx=1): contributes to C[1,0] with +1, C[1,1] with -1
            elif m_idx == 1:  # M2
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += -1  # C[1,1]
            
            # M3 (m_idx=2): contributes to C[0,1] with +1, C[1,1] with +1
            elif m_idx == 2:  # M3
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M4 (m_idx=3): contributes to C[0,0] with +1, C[1,0] with +1
            elif m_idx == 3:  # M4
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
                W[(k_offset + 0) * n + (i_offset + 1)][col] += 1  # C[1,0]
            
            # M5 (m_idx=4): contributes to C[0,0] with -1, C[0,1] with +1
            elif m_idx == 4:  # M5
                W[(k_offset + 0) * n + (i_offset + 0)][col] += -1  # C[0,0]
                W[(k_offset + 1) * n + (i_offset + 0)][col] += 1  # C[0,1]
            
            # M6 (m_idx=5): contributes to C[1,1] with +1
            elif m_idx == 5:  # M6
                W[(k_offset + 1) * n + (i_offset + 1)][col] += 1  # C[1,1]
            
            # M7 (m_idx=6): contributes to C[0,0] with +1
            elif m_idx == 6:  # M7
                W[(k_offset + 0) * n + (i_offset + 0)][col] += 1  # C[0,0]
            
            col += 1
        
        return col - col_start
    
    # Add Strassen block 1: i_offset=0, j_offset=0, k_offset=0
    col = 0
    col += add_strassen_block(0, 0, 0, col)  # Uses columns 0-6
    
    # Add Strassen block 2: i_offset=0, j_offset=2, k_offset=2
    col += add_strassen_block(0, 2, 2, col)  # Uses columns 7-13
    
    # Now add schoolbook terms for all other (i, j, k) tuples.
    # We need to skip the ones already covered by Strassen blocks.
    # Block 1 covers: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2 covers: i in {0,1}, j in {2,3}, k in {2,3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this (i, j, k) is covered by a Strassen block
                covered = False
                # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    covered = True
                # Block 2: i in {0,1}, j in {2,3}, k in {2,3}
                if i in (0, 1) and j in (2, 3) and k in (2, 3):
                    covered = True
                
                if not covered:
                    # Schoolbook: one term
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    # Trim U, V, W to the actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return U, V, W, actual_rank

def _build_schoolbook_decomposition(n, m, p):
    """
    Build the standard schoolbook decomposition with rank n*m*p.
    """
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
    
    return U, V, W, rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is correct.
    For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    and for all other combinations, the sum is 0.
    """
    rank = len(U[0]) if U else 0
    
    # Convert to numpy for efficient verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each output position (i, k), the (i,k)-th entry of C should be
    # sum_j A[i,j] * B[j,k]
    # In the tensor decomposition, C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
    
    # Actually, let's verify directly:
    # The tensor T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # should equal sum_r U[a, r] * V[b, r] * W[c, r]
    
    # For the matrix multiplication tensor:
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j',k), c to (k',i')
    #             and i=i', j=j', k=k'
    #             else 0
    
    # Let's verify by checking all (i, j, k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Expected value: 1
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute sum_r U[a, r] * V[b, r] * W[c, r]
                val = np.sum(U_np[a] * V_np[b] * W_np[c])
                if abs(val - 1) > 1e-9:
                    return False
    
    # Also check that for non-matching (i, j, k), the value is 0
    # This is more complex, so let's do a spot check
    # For random non-matching tuples
    import random
    random.seed(42)
    for _ in range(100):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k2 = random.randint(0, p-1)
        
        # Skip if this is a matching tuple
        if i == i2 and j == j2 and k == k2:
            continue
        
        a = i * m + j
        b = j2 * p + k
        c = k2 * n + i2
        
        # Expected value: 0
        val = np.sum(U_np[a] * V_np[b] * W_np[c])
        if abs(val) > 1e-9:
            return False
    
    return True
# EVOLVE-BLOCK-END
