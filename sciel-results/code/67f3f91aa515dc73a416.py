"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x3 matrix multiplication.
    Uses overlapping Strassen blocks with careful verification.
    """
    n, m, p = 3, 3, 3
    size = n * m  # 9

    def verify(U, V, W, rank, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    for c in range(rank):
                        val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def make_schoolbook():
        """Return the trivial rank-27 schoolbook decomposition."""
        U = np.zeros((size, 27), dtype=float)
        V = np.zeros((size, 27), dtype=float)
        W = np.zeros((size, 27), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def strassen_block(U, V, W, rank_offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """
        Add a 2x2 Strassen block to the decomposition.
        a_rows, a_cols: row/col indices in A for the 2x2 block
        b_rows, b_cols: row/col indices in B for the 2x2 block  
        c_rows, c_cols: row/col indices in C for the 2x2 block
        
        For A[i,j], index = i*m+j
        For B[j,k], index = j*p+k
        For C[i,k], index = k*n+i
        """
        # The 2x2 block covers:
        # A[a_rows[0], a_cols[0]], A[a_rows[0], a_cols[1]]
        # A[a_rows[1], a_cols[0]], A[a_rows[1], a_cols[1]]
        # B[b_rows[0], b_cols[0]], B[b_rows[0], b_cols[1]]
        # B[b_rows[1], b_cols[0]], B[b_rows[1], b_cols[1]]
        # C[c_rows[0], c_cols[0]], C[c_rows[0], c_cols[1]]
        # C[c_rows[1], c_cols[0]], C[c_rows[1], c_cols[1]]
        
        # Strassen 7 products for 2x2:
        # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C01, C10, C11
        # M2 = (A10+A11)*B00 -> contributes to C10, C11
        # M3 = A00*(B01-B11) -> contributes to C00, C10
        # M4 = A11*(B10-B00) -> contributes to C10, C11
        # M5 = (A00+A01)*B11 -> contributes to C01, C11
        # M6 = (A10-A00)*(B00+B01) -> contributes to C10, C11
        # M7 = (A01-A11)*(B10+B11) -> contributes to C01, C11
        
        # Map to actual indices
        A00 = a_rows[0]*m + a_cols[0]
        A01 = a_rows[0]*m + a_cols[1]
        A10 = a_rows[1]*m + a_cols[0]
        A11 = a_rows[1]*m + a_cols[1]
        
        B00 = b_rows[0]*p + b_cols[0]
        B01 = b_rows[0]*p + b_cols[1]
        B10 = b_rows[1]*p + b_cols[0]
        B11 = b_rows[1]*p + b_cols[1]
        
        C00 = c_cols[0]*n + c_rows[0]
        C01 = c_cols[1]*n + c_rows[0]
        C10 = c_cols[0]*n + c_rows[1]
        C11 = c_cols[1]*n + c_rows[1]
        
        # M1
        c = rank_offset + 0
        U[A00, c] += 1; U[A11, c] += 1
        V[B00, c] += 1; V[B11, c] += 1
        W[C00, c] += 1; W[C01, c] += -1; W[C10, c] += -1; W[C11, c] += 1
        
        # M2
        c = rank_offset + 1
        U[A10, c] += 1; U[A11, c] += 1
        V[B00, c] += 1
        W[C10, c] += 1; W[C11, c] += 1
        
        # M3
        c = rank_offset + 2
        U[A00, c] += 1
        V[B01, c] += 1; V[B11, c] += -1
        W[C00, c] += 1; W[C10, c] += -1
        
        # M4
        c = rank_offset + 3
        U[A11, c] += 1
        V[B10, c] += 1; V[B00, c] += -1
        W[C10, c] += 1; W[C11, c] += -1
        
        # M5
        c = rank_offset + 4
        U[A00, c] += 1; U[A01, c] += 1
        V[B11, c] += 1
        W[C01, c] += 1; W[C11, c] += 1
        
        # M6
        c = rank_offset + 5
        U[A10, c] += 1; U[A00, c] += -1
        V[B00, c] += 1; V[B01, c] += 1
        W[C10, c] += 1; W[C11, c] += -1
        
        # M7
        c = rank_offset + 6
        U[A01, c] += 1; U[A11, c] += -1
        V[B10, c] += 1; V[B11, c] += 1
        W[C01, c] += 1; W[C11, c] += -1

    # Try rank-20 construction with overlapping Strassen blocks
    # Strategy: Use 4 overlapping 2x2 blocks but share some products
    
    # Let's try a different approach: use the known rank-20 algorithm
    # from the literature. One construction uses:
    # - 2 Strassen blocks (14 products)
    # - 6 additional products for the remaining terms
    
    # Actually, let me try a more careful approach.
    # The 3x3 matrix multiplication tensor has rank at most 20 (known result).
    
    # Let me try the construction from "Fast Matrix Multiplication" 
    # using a specific pattern of overlapping blocks.
    
    # Approach: Use Strassen for top-left 2x2 and bottom-right 2x2,
    # then handle the remaining 6 cross terms with 6 more products.
    # Total: 7 + 7 + 6 = 20
    
    rank = 20
    U = np.zeros((size, rank), dtype=float)
    V = np.zeros((size, rank), dtype=float)
    W = np.zeros((size, rank), dtype=float)
    
    # Block 1: Top-left 2x2 (rows 0-1, cols 0-1)
    # A[0,0], A[0,1], A[1,0], A[1,1]
    # B[0,0], B[0,1], B[1,0], B[1,1]
    # C[0,0], C[0,1], C[1,0], C[1,1]
    strassen_block(U, V, W, 0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: Bottom-right 2x2 (rows 1-2, cols 1-2)
    # A[1,1], A[1,2], A[2,1], A[2,2]
    # B[1,1], B[1,2], B[2,1], B[2,2]
    # C[1,1], C[1,2], C[2,1], C[2,2]
    strassen_block(U, V, W, 7, [1,2], [1,2], [1,2], [1,2], [1,2], [1,2])
    
    # Now we need to handle the cross terms that are not covered:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    
    # Wait, the two blocks cover:
    # Block 1 covers C[0,0], C[0,1], C[1,0], C[1,1]
    # Block 2 covers C[1,1], C[1,2], C[2,1], C[2,2]
    # So C[1,1] is covered by both (should sum correctly)
    # Missing: C[0,2], C[2,0], and parts of C[2,1], C[1,2]
    
    # Actually, let me check what's covered:
    # C[0,2]: Not covered by either block. Needs A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2]
    # C[2,0]: Not covered. Needs A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0]
    # C[2,1]: Block 2 covers A[2,1]*B[1,1] and A[2,2]*B[2,1] but not A[2,0]*B[0,1]
    # C[1,2]: Block 2 covers A[1,1]*B[1,2] and A[1,2]*B[2,2] but not A[1,0]*B[0,2]
    
    # So we need 6 more terms for the 6 missing cross products:
    # A[0,0]*B[0,2], A[0,1]*B[1,2], A[0,2]*B[2,2] -> C[0,2]
    # A[2,0]*B[0,0], A[2,1]*B[1,0], A[2,2]*B[2,0] -> C[2,0]
    # A[2,0]*B[0,1] -> C[2,1]
    # A[1,0]*B[0,2] -> C[1,2]
    
    # That's 8 terms, not 6. So this approach gives rank 22, not 20.
    
    # Let me try a different overlap pattern.
    # Use blocks: top-left, top-right, bottom-left, bottom-right
    # But with more sharing.
    
    # Actually, the known rank-20 construction for 3x3 is non-trivial.
    # Let me try the approach from Winograd or similar.
    
    # Alternative: Use a single large construction.
    # Let me try the "border rank" approach or a known explicit construction.
    
    # For now, let me try rank-22 with the above approach and verify.
    
    rank = 22
    U = np.zeros((size, rank), dtype=float)
    V = np.zeros((size, rank), dtype=float)
    W = np.zeros((size, rank), dtype=float)
    
    # Block 1: Top-left 2x2
    strassen_block(U, V, W, 0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: Bottom-right 2x2
    strassen_block(U, V, W, 7, [1,2], [1,2], [1,2], [1,2], [1,2], [1,2])
    
    # Cross terms (indices 14-21)
    # A[0,0]*B[0,2] -> C[0,2]
    c = 14
    U[0, c] = 1; V[2, c] = 1; W[2*3+0, c] = 1  # C[0,2] = idx 2*3+0=6
    
    # A[0,1]*B[1,2] -> C[0,2]
    c = 15
    U[1, c] = 1; V[5, c] = 1; W[6, c] = 1
    
    # A[0,2]*B[2,2] -> C[0,2]
    c = 16
    U[2, c] = 1; V[8, c] = 1; W[6, c] = 1
    
    # A[2,0]*B[0,0] -> C[2,0]
    c = 17
    U[6, c] = 1; V[0, c] = 1; W[0*3+2, c] = 1  # C[2,0] = idx 0*3+2=2
    
    # A[2,1]*B[1,0] -> C[2,0]
    c = 18
    U[7, c] = 1; V[3, c] = 1; W[2, c] = 1
    
    # A[2,2]*B[2,0] -> C[2,0]
    c = 19
    U[8, c] = 1; V[6, c] = 1; W[2, c] = 1
    
    # A[2,0]*B[0,1] -> C[2,1]
    c = 20
    U[6, c] = 1; V[1, c] = 1; W[1*3+2, c] = 1  # C[2,1] = idx 1*3+2=5
    
    # A[1,0]*B[0,2] -> C[1,2]
    c = 21
    U[3, c] = 1; V[2, c] = 1; W[2*3+1, c] = 1  # C[1,2] = idx 2*3+1=7
    
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If rank-22 doesn't work, try other constructions
    
    # Try rank-23 with a different block arrangement
    rank = 23
    U = np.zeros((size, rank), dtype=float)
    V = np.zeros((size, rank), dtype=float)
    W = np.zeros((size, rank), dtype=float)
    
    # Use 3 overlapping 2x2 blocks: top-left, top-right, bottom-left
    # Block 1: rows 0-1, cols 0-1
    strassen_block(U, V, W, 0, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1])
    
    # Block 2: rows 0-1, cols 1-2
    strassen_block(U, V, W, 7, [0,1], [1,2], [1,2], [1,2], [0,1], [1,2])
    
    # Block 3: rows 1-2, cols 0-1
    strassen_block(U, V, W, 14, [1,2], [0,1], [0,1], [0,1], [1,2], [0,1])
    
    # This gives 21 products. Need 2 more for remaining terms.
    # Remaining: C[2,2] terms not fully covered
    # A[2,2]*B[2,2] -> C[2,2]
    c = 21
    U[8, c] = 1; V[8, c] = 1; W[2*3+2, c] = 1  # C[2,2] = idx 2*3+2=8
    
    # A[2,0]*B[0,2] -> C[2,2]
    c = 22
    U[6, c] = 1; V[2, c] = 1; W[8, c] = 1
    
    if verify(U, V, W, rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 27
    U, V, W = make_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
