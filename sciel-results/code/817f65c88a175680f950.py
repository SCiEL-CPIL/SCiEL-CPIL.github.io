import time
import random
import itertools
from fractions import Fraction

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W, n, m, p):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1
                actual = 0
                for c in range(rank):
                    actual += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if actual != expected:
                    return False
    return True

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7).
    
    Order of A elements: a00, a01, a10, a11
    Order of B elements: b00, b01, b10, b11
    Order of C elements: c00, c01, c10, c11
    
    Strassen products:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b01-b11)
    m4 = a11(b10-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a00)(b00+b01)
    m7 = (a01-a11)(b10+b11)
    
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # a00: m1+m3+m5-m6
        [0, 0, 0, 0, 1, 0, 1],    # a01: m5+m7
        [0, 1, 0, 0, 0, 1, 0],    # a10: m2+m6
        [1, 1, 0, 1, 0, 0, -1]    # a11: m1+m2+m4-m7
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # b00: m1+m2-m4+m6
        [0, 0, 1, 0, 0, 1, 0],    # b01: m3+m6
        [0, 0, 0, 1, 0, 0, 1],    # b10: m4+m7
        [1, 0, -1, 0, 1, 0, 1]    # b11: m1-m3+m5+m7
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # c00: m1+m4-m5+m7
        [0, 0, 1, 0, 1, 0, 0],    # c01: m3+m5
        [0, 1, 0, 1, 0, 0, 0],    # c10: m2+m4
        [1, -1, 1, 0, 0, 1, 0]    # c11: m1-m2+m3+m6
    ]
    return U, V, W

def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4).
    
    A is 1x2 (a0, a1), B is 2x2 (b00,b01,b10,b11), C is 1x2 (c0, c1)
    c0 = a0*b00 + a1*b10
    c1 = a0*b01 + a1*b11
    
    U (2x4): rows for a0, a1
    V (4x4): rows for b00, b01, b10, b11
    W (2x4): rows for c0, c1
    """
    U = [
        [1, 0, 1, 0],  # a0: contributes to c0 (via b00) and c1 (via b01)
        [0, 1, 0, 1]   # a1: contributes to c0 (via b10) and c1 (via b11)
    ]
    V = [
        [1, 0, 0, 0],  # b00
        [0, 1, 0, 0],  # b01
        [0, 0, 1, 0],  # b10
        [0, 0, 0, 1]   # b11
    ]
    W = [
        [1, 0, 1, 0],  # c0: m1 + m3 where m1=a0*b00, m3=a1*b10
        [0, 1, 0, 1]   # c1: m2 + m4 where m2=a0*b01, m4=a1*b11
    ]
    return U, V, W

def _build_3x4x8_strassen():
    """
    Build 3x4x8 decomposition using Strassen blocks for rows 0,1 and schoolbook for row 2.
    
    Strategy:
    - For i=0,1: Use 2x2x2 Strassen blocks.
      Split j (0..3) into two pairs: (0,1) and (2,3).
      Split k (0..7) into four pairs: (0,1), (2,3), (4,5), (6,7).
      This gives 2 * 4 = 8 Strassen blocks. Rank = 8 * 7 = 56.
    - For i=2: Use 1x2x2 schoolbook blocks.
      Split j (0..3) into two pairs: (0,1) and (2,3).
      Split k (0..7) into four pairs: (0,1), (2,3), (4,5), (6,7).
      This gives 2 * 4 = 8 schoolbook blocks. Rank = 8 * 4 = 32.
    
    Total rank = 56 + 32 = 88.
    """
    n, m, p = 3, 4, 8
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    
    # Calculate total rank
    rank_strassen = 8 * 7  # 56
    rank_school = 8 * 4    # 32
    total_rank = rank_strassen + rank_school
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Process Strassen blocks for i=0 and i=1
    for i_block in range(2):  # i=0, i=1
        for jp in range(2):   # j pairs: (0,1), (2,3)
            for kp in range(4): # k pairs: (0,1), (2,3), (4,5), (6,7)
                j0 = 2 * jp
                j1 = 2 * jp + 1
                k0 = 2 * kp
                k1 = 2 * kp + 1
                
                # U rows: (i, j0), (i, j1) for the 2x2 block
                # In our indexing, U row index is i*m + j
                u_rows = [i_block * m + j0, i_block * m + j1]
                
                # V rows: (j0, k0), (j0, k1), (j1, k0), (j1, k1)
                # In our indexing, V row index is j*p + k
                v_rows = [
                    j0 * p + k0, j0 * p + k1,
                    j1 * p + k0, j1 * p + k1
                ]
                
                # W rows: (k0, i), (k1, i) for the 2x2 block output
                # In our indexing, W row index is k*n + i
                w_rows = [
                    k0 * n + i_block, k1 * n + i_block
                ]
                
                # Add Strassen block
                for r in range(2):
                    for c in range(7):
                        # U contribution
                        U[u_rows[r]][col_offset + c] += U2[r][c]
                        
                        # V contribution
                        # U2 has 4 rows, V2 has 4 rows
                        # The mapping is:
                        # U2[0] -> a00, U2[1] -> a01, U2[2] -> a10, U2[3] -> a11
                        # V2[0] -> b00, V2[1] -> b01, V2[2] -> b10, V2[3] -> b11
                        # So for row r of U2 (which is 0 or 1 for our 2x1 block of A),
                        # we need to map to the correct V2 row.
                        # Wait, the Strassen block is 2x2x2.
                        # A is 2x2, B is 2x2, C is 2x2.
                        # Our A block is 2x2 (rows i_block, i_block+1? No, i_block is fixed row).
                        # Ah, the Strassen block handles 2 rows of A, 2 cols of A, 2 rows of B, 2 cols of B.
                        # So A block is 2x2.
                        # U2 has 4 rows corresponding to a00, a01, a10, a11.
                        # Our u_rows only has 2 entries? No.
                        # The Strassen decomposition is for a 2x2 matrix multiplication.
                        # It uses 7 scalar multiplications.
                        # The inputs are 2x2 matrices.
                        # So we need to map the 4 elements of the A block to the 4 rows of U2.
                        # The 4 elements of the A block are:
                        # A[i_block, j0], A[i_block, j1] -> these are row 0 of the 2x2 block?
                        # No, the block is defined by rows {i_block, i_block+1} and cols {j0, j1}.
                        # But i_block is 0 or 1. If i_block=0, rows are 0,1. If i_block=1, rows are 1,2?
                        # No, the tiling is:
                        # i=0,1 use Strassen. This implies the A block is 2x2 (rows 0,1).
                        # But we are iterating i_block in range(2).
                        # If i_block=0, we handle rows 0? Or rows 0,1?
                        # The Strassen block is 2x2x2. It multiplies a 2x2 A by a 2x2 B.
                        # So it consumes 2 rows of A and 2 cols of A.
                        # In our loop:
                        # for i_block in range(2):
                        #   for jp in range(2):
                        #     for kp in range(4):
                        # This structure suggests that for each i_block, we are handling a specific part.
                        # But Strassen handles 2 rows of A at once.
                        # So we should only iterate i_block once? Or does i_block select the row pair?
                        # If we have 3 rows (0,1,2), we can form one 2x2 block (rows 0,1) and one 1x2 block (row 2).
                        # So the loop `for i_block in range(2)` is wrong if it implies 2 separate Strassen blocks covering disjoint rows.
                        # It should be:
                        # 1 Strassen block for rows 0,1.
                        # 1 Schoolbook block for row 2.
                        
                        # Let's re-evaluate the tiling.
                        # A is 3x4.
                        # Block 1: Rows 0,1; Cols 0,1,2,3. This is a 2x4 block.
                        # We can split this 2x4 block into two 2x2 blocks: (rows 0,1; cols 0,1) and (rows 0,1; cols 2,3).
                        # Each 2x2 A block is multiplied by a 2x2 B block?
                        # B is 4x8.
                        # For A block (rows 0,1; cols 0,1), we multiply by B block (rows 0,1; cols k0,k1).
                        # This produces C block (rows 0,1; cols k0,k1).
                        # This is a 2x2x2 multiplication.
                        # So for each jp (j col pair) and kp (k col pair), we have a 2x2x2 multiplication.
                        # The A block is always rows 0,1.
                        # So we don't loop over i_block for Strassen. We just use rows 0,1.
                        # The loop `for i_block in range(2)` in the previous thought process was flawed.
                        
                        # Correct Tiling:
                        # Strassen blocks:
                        # A rows: 0, 1 (fixed)
                        # A cols: (0,1) or (2,3) -> jp in {0,1}
                        # B rows: (0,1) or (2,3) -> must match A cols!
                        # Wait, A[i,j] * B[j,k].
                        # If A block uses cols 0,1, then B block must use rows 0,1.
                        # So jp determines both A cols and B rows.
                        # B cols: (0,1), (2,3), (4,5), (6,7) -> kp in {0,1,2,3}
                        # C block: rows 0,1; cols k0,k1.
                        
                        # So there are 2 (jp) * 4 (kp) = 8 Strassen blocks.
                        # Each uses A rows 0,1.
                        
                        # Schoolbook blocks:
                        # A rows: 2 (fixed)
                        # A cols: (0,1) or (2,3) -> jp in {0,1}
                        # B rows: (0,1) or (2,3) -> must match A cols!
                        # B cols: (0,1), (2,3), (4,5), (6,7) -> kp in {0,1,2,3}
                        # C block: row 2; cols k0,k1.
                        
                        # So there are 2 (jp) * 4 (kp) = 8 Schoolbook blocks.
                        # Each uses A row 2.
                        
                        # Total Rank: 8*7 + 8*4 = 56 + 32 = 88.
                        
                        pass # Placeholder to exit the incorrect loop logic below

    # Reset matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # --- Strassen Blocks (Rows 0,1 of A) ---
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # A Block: Rows 0,1; Cols j0,j1
            # Elements: A[0,j0], A[0,j1], A[1,j0], A[1,j1]
            # U2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
            # Global U indices:
            u_idx = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            
            # B Block: Rows j0,j1; Cols k0,k1
            # Elements: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
            # V2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
            # Global V indices:
            v_idx = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            
            # C Block: Rows 0,1; Cols k0,k1
            # Elements: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
            # W2 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
            # Global W indices (k*n + i):
            w_idx = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            
            for r in range(4):
                for c in range(7):
                    U[u_idx[r]][col_offset + c] += U2[r][c]
                    V[v_idx[r]][col_offset + c] += V2[r][c]
                    W[w_idx[r]][col_offset + c] += W2[r][c]
            
            col_offset += 7
    
    # --- Schoolbook Blocks (Row 2 of A) ---
    for jp in range(2):
        for kp in range(4):
            j0 = 2 * jp
            j1 = 2 * jp + 1
            k0 = 2 * kp
            k1 = 2 * kp + 1
            
            # A Block: Row 2; Cols j0,j1
            # Elements: A[2,j0], A[2,j1]
            # U1 indices: 0:(0), 1:(1) relative to block (1x2)
            # Global U indices:
            u_idx = [
                2 * m + j0, 2 * m + j1
            ]
            
            # B Block: Rows j0,j1; Cols k0,k1
            # Elements: B[j0,k0], B[j0,k1], B[j1,k0], B[j1,k1]
            # V1 indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) relative to block
            # Global V indices:
            v_idx = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            
            # C Block: Row 2; Cols k0,k1
            # Elements: C[2,k0], C[2,k1]
            # W1 indices: 0:(0), 1:(1) relative to block (1x2)
            # Global W indices:
            w_idx = [
                k0 * n + 2, k1 * n + 2
            ]
            
            # Map U1 (2 rows) to u_idx (2 rows)
            for r in range(2):
                for c in range(4):
                    U[u_idx[r]][col_offset + c] += U1[r][c]
                    W[w_idx[r]][col_offset + c] += W1[r][c]
            
            # Map V1 (4 rows) to v_idx (4 rows)
            for r in range(4):
                for c in range(4):
                    V[v_idx[r]][col_offset + c] += V1[r][c]
            
            col_offset += 4
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try the Strassen-based construction (Rank 88)
    try:
        result = _build_3x4x8_strassen()
        if _verify(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    # Fallback to schoolbook (Rank 96)
    return _schoolbook()

# EVOLVE-BLOCK-END
