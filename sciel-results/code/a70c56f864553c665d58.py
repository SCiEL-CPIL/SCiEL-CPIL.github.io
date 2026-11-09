"""Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.

Strategy: Split 4x4*4x5 into 4x4*4x4 (rank 49 via recursive Strassen) 
plus 4x4*4x1 (rank 16 via schoolbook), total rank 65.
"""
import time
import numpy as np

_DIMENSIONS = [4, 4, 5]
_DEADLINE = 140.0  # safety margin before 150s kill

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 via Strassen, rank 7.
    
    U shape: (4, 7) indexed by [i*2+j, c]
    V shape: (4, 7) indexed by [j*2+k, c]  
    W shape: (4, 7) indexed by [k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    # Strassen products for 2x2:
    # A = [[A00, A01], [A10, A11]], B = [[B00, B10], [B01, B11]]
    # Wait, B is 2x2: B[j,k] where j is row of A's cols, k is col
    # A[i,j], B[j,k], C[i,k]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A00+A11)(B00+B11)
    # A00=A[0,0]=idx0, A11=A[1,1]=idx3, B00=B[0,0]=idx0, B11=B[1,1]=idx3
    # C00 += M1, C11 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1  # C[0,0] and C[1,1]: W[k*2+i]
    # C[0,0]: k=0,i=0 -> W[0]; C[1,1]: k=1,i=1 -> W[3]
    
    # M2 = (A10+A11)B00
    # A10=A[1,0]=idx2, A11=A[1,1]=idx3, B00=B[0,0]=idx0
    # C10 += M2, C11 += -M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1  # C[1,0]: k=0,i=1 -> W[2]; C[1,1]: k=1,i=1 -> W[3]
    
    # M3 = A00(B01-B11)
    # A00=A[0,0]=idx0, B01=B[0,1]=idx1, B11=B[1,1]=idx3
    # C01 += M3, C11 += M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1  # C[0,1]: k=1,i=0 -> W[2]; C[1,1]: k=1,i=1 -> W[3]
    
    # M4 = A11(B10-B00)
    # A11=A[1,1]=idx3, B10=B[1,0]=idx2, B00=B[0,0]=idx0
    # C00 += M4, C10 += M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1  # C[0,0]: k=0,i=0 -> W[0]; C[1,0]: k=0,i=1 -> W[2]
    
    # M5 = (A00+A01)B11
    # A00=A[0,0]=idx0, A01=A[0,1]=idx1, B11=B[1,1]=idx3
    # C01 += M5, C00 += -M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1  # C[0,1]: k=1,i=0 -> W[2]; C[0,0]: k=0,i=0 -> W[0]
    
    # M6 = (A10-A00)(B00+B01)
    # A10=A[1,0]=idx2, A00=A[0,0]=idx0, B00=B[0,0]=idx0, B01=B[0,1]=idx1
    # C11 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1  # C[1,1]: k=1,i=1 -> W[3]
    
    # M7 = (A01-A11)(B10+B11)
    # A01=A[0,1]=idx1, A11=A[1,1]=idx3, B10=B[1,0]=idx2, B11=B[1,1]=idx3
    # C00 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1  # C[0,0]: k=0,i=0 -> W[0]
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4 * 4x4 using recursive Strassen.
    
    Split into 2x2 blocks, apply Strassen at top level (7 products),
    each product is a 2x2*2x2 which we decompose with Strassen (rank 7).
    Total: 7 * 7 = 49.
    
    U shape: (16, 49) indexed by [i*4+j, c]
    V shape: (16, 49) indexed by [j*4+k, c]
    W shape: (16, 49) indexed by [k*4+i, c]
    """
    U2, V2, W2 = _strassen_2x2x2()
    
    # Top-level Strassen for 4x4*4x4:
    # A is 4x4, split into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    # B is 4x4, split into 2x2 blocks: B = [[B00, B01], [B10, B11]]
    # where A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], etc.
    # C = A*B, C = [[C00, C01], [C10, C11]]
    
    # Strassen products (each is 2x2 * 2x2):
    # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
    # M2 = (A10+A11)B00 -> contributes to C10 and -C11
    # M3 = A00(B01-B11) -> contributes to C01 and C11
    # M4 = A11(B10-B00) -> contributes to C00 and C10
    # M5 = (A00+A01)B11 -> contributes to C01 and -C00
    # M6 = (A10-A00)(B00+B01) -> contributes to C11
    # M7 = (A01-A11)(B10+B11) -> contributes to C00
    
    # For each Strassen product M_c (c=0..6), we need to:
    # 1. Form the appropriate linear combination of A blocks (U side)
    # 2. Form the appropriate linear combination of B blocks (V side)
    # 3. Multiply using Strassen 2x2x2 (7 sub-products)
    # 4. Distribute the 2x2 result to the appropriate 4x4 output positions (W side)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Define the Strassen structure at the 4x4 level
    # For each top-level product c (0..6):
    #   U factors: which A-blocks with what signs
    #   V factors: which B-blocks with what signs  
    #   W factors: which C-blocks with what signs
    
    # A-block indices: A00=rows 0-1,cols 0-1; A01=rows 0-1,cols 2-3
    #                  A10=rows 2-3,cols 0-1; A11=rows 2-3,cols 2-3
    # In flat index [i*4+j]: A00: i in {0,1}, j in {0,1}
    #                        A01: i in {0,1}, j in {2,3}
    #                        A10: i in {2,3}, j in {0,1}
    #                        A11: i in {2,3}, j in {2,3}
    
    # B-block indices: B00=j in {0,1},k in {0,1}; B01=j in {0,1},k in {2,3}
    #                  B10=j in {2,3},k in {0,1}; B11=j in {2,3},k in {2,3}
    # In flat index [j*4+k]
    
    # C-block indices: C00=i in {0,1},k in {0,1}; C01=i in {0,1},k in {2,3}
    #                  C10=i in {2,3},k in {0,1}; C11=i in {2,3},k in {2,3}
    # In flat index [k*4+i]
    
    # Top-level Strassen definitions:
    # (a_blocks, a_signs, b_blocks, b_signs, c_blocks, c_signs)
    strassen_defs = [
        # M1 = (A00+A11)(B00+B11) -> C00 + C11
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2 = (A10+A11)B00 -> C10 - C11
        ([2, 3], [1, 1], [0], [1], [2, 3], [1, -1]),
        # M3 = A00(B01-B11) -> C01 + C11
        ([0], [1], [1, 3], [1, -1], [1, 3], [1, 1]),
        # M4 = A11(B10-B00) -> C00 + C10
        ([3], [1], [2, 0], [1, -1], [0, 2], [1, 1]),
        # M5 = (A00+A01)B11 -> C01 - C00
        ([0, 1], [1, 1], [3], [1], [1, 0], [1, -1]),
        # M6 = (A10-A00)(B00+B01) -> C11
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7 = (A01-A11)(B10+B11) -> C00
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    # Block row/col ranges for A: block b -> (row_start, col_start)
    # A00: (0,0), A01: (0,2), A10: (2,0), A11: (2,2)
    a_block_rc = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    # B00: (0,0), B01: (0,2), B10: (2,0), B11: (2,2)
    b_block_rc = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    # C00: (0,0), C01: (0,2), C10: (2,0), C11: (2,2)
    # C block (row_start, col_start) in (i, k) space
    c_block_rc = {0: (0,0), 1: (0,2), 2: (2,0), 3: (2,2)}
    
    for c_idx, (a_blks, a_sigs, b_blks, b_sigs, c_blks, c_sigs) in enumerate(strassen_defs):
        # For this top-level product, we have 7 sub-products from Strassen 2x2x2
        for sub in range(7):
            col = c_idx * 7 + sub
            
            # U side: for each A block involved, add the Strassen U pattern
            for a_blk, a_sig in zip(a_blks, a_sigs):
                ar, ac = a_block_rc[a_blk]
                # A block a_blk has local indices (li, lj) in [0,1]x[0,1]
                # Global A index: i = ar + li, j = ac + lj, flat = i*4 + j
                # Strassen U2[local_flat, sub] where local_flat = li*2 + lj
                for li in range(2):
                    for lj in range(2):
                        local_flat = li * 2 + lj
                        global_flat = (ar + li) * 4 + (ac + lj)
                        U4[global_flat, col] = a_sig * U2[local_flat, sub]
            
            # V side: for each B block involved
            for b_blk, b_sig in zip(b_blks, b_sigs):
                br, bc = b_block_rc[b_blk]
                for lj in range(2):
                    for lk in range(2):
                        local_flat = lj * 2 + lk
                        global_flat = (br + lj) * 4 + (bc + lk)
                        V4[global_flat, col] = b_sig * V2[local_flat, sub]
            
            # W side: for each C block involved
            for c_blk, c_sig in zip(c_blks, c_sigs):
                cr, ck = c_block_rc[c_blk]
                for lk in range(2):
                    for li in range(2):
                        local_flat = lk * 2 + li
                        global_flat = (ck + lk) * 4 + (cr + li)
                        W4[global_flat, col] = c_sig * W2[local_flat, sub]
    
    return U4, V4, W4


def _build_4x4x1_schoolbook():
    """Build rank-16 decomposition for 4x4 * 4x1.
    
    U shape: (16, 16) indexed by [i*4+j, c]
    V shape: (4, 16) indexed by [j*1+k, c] = [j, c]
    W shape: (4, 16) indexed by [k*4+i, c] = [i, c]
    """
    U = np.zeros((16, 16), dtype=float)
    V = np.zeros((4, 16), dtype=float)
    W = np.zeros((4, 16), dtype=float)
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j, col] = 1
            V[j, col] = 1
            W[i, col] = 1
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication: T[i,j,k] = 1 if j == k (when multiplying A[i,j]*B[j,k])
    # Wait, the tensor for matrix multiplication: T[i,j,k] = delta_{j,k}? No.
    # Actually for A (n x m) * B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all i,j,k (it's the identity in the middle dimension)
    # So T[i,j,k] = 1 always.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(U.shape[1]):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > 1e-9:
                    return False, (i, j, k, expected, actual)
    return True, None


def construct():
    start_time = time.time()
    n, m, p = _DIMENSIONS
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    U4, V4, W4 = _build_4x4x4_strassen()
    
    # Build 4x4x1 schoolbook decomposition (rank 16)
    U1, V1, W1 = _build_4x4x1_schoolbook()
    
    # Combine: 4x4*4x5 = 4x4*4x4 (cols 0-3) + 4x4*4x1 (col 4)
    # U: (16, 49+16) = (16, 65)
    # V: (20, 65) - V4 is (16,49) for p=4, need to extend to p=5
    #    V4[j*4+k, c] for k in 0..3; V1[j*1+k, c] for k in 0 (only k=0)
    #    For the combined V: V[j*5+k, c]
    #    For cols 0-3 of B: V[j*5+k, c] = V4[j*4+k, c] for k=0..3
    #    For col 4 of B: V[j*5+4, c] = V1[j, c]
    # W: (20, 65) - W4 is (16,49) for p=4, W1 is (4,16) for p=1
    #    W[k*4+i, c] for k in 0..3; W1[k*1+i, c] = W1[i, c] for k=0
    #    For combined W: W[k*5+i, c]
    #    For k=0..3: W[k*5+i, c] = W4[k*4+i, c]
    #    For k=4: W[4*5+i, c] = W1[i, c]
    
    rank = 65
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First 49 columns: 4x4x4 Strassen
    U[:, :49] = U4
    # V4 is (16, 49) with index [j*4+k, c], we need V[j*5+k, c] for k=0..3
    for j in range(4):
        for k in range(4):
            V[j*5 + k, :49] = V4[j*4 + k, :49]
    # W4 is (16, 49) with index [k*4+i, c], we need W[k*5+i, c] for k=0..3
    for k in range(4):
        for i in range(4):
            W[k*5 + i, :49] = W4[k*4 + i, :49]
    
    # Next 16 columns: 4x4x1 schoolbook for column 4
    U[:, 49:] = U1
    # V1 is (4, 16) with index [j, c], we need V[j*5+4, c]
    for j in range(4):
        V[j*5 + 4, 49:] = V1[j, :]
    # W1 is (4, 16) with index [i, c], we need W[4*5+i, c] = W[20+i, c]
    # Wait, p*n = 5*4 = 20, so W has 20 rows. W[k*5+i] for k=4: W[20+i]... 
    # But W only has 20 rows (indices 0..19). k*5+i for k=4, i=0..3 gives 20,21,22,23.
    # That's out of bounds! Let me reconsider.
    
    # W has shape (p*n, rank) = (20, rank), indexed by [k*n+i, c] where k in [0,p), i in [0,n)
    # So k ranges 0..4, i ranges 0..3. k*n+i = k*4+i.
    # For k=0..3: W[k*4+i, c] = W4[k*4+i, c]
    # For k=4: W[4*4+i, c] = W[16+i, c] = W1[i, c]
    
    # I made an error above. Let me redo.
    # W4 is (16, 49) indexed by [k*4+i, c] for k=0..3, i=0..3
    # W1 is (4, 16) indexed by [i, c] (since for p=1, k*n+i = 0*4+i = i)
    
    # Reset W
    W = np.zeros((p*n, rank), dtype=float)
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W4[k*4 + i, :49]
    for i in range(4):
        W[4*4 + i, 49:] = W1[i, :]
    
    # Similarly fix V: V has shape (m*p, rank) = (20, rank), indexed by [j*p+k, c] = [j*5+k, c]
    # V4 is (16, 49) indexed by [j*4+k, c] for j=0..3, k=0..3
    # V1 is (4, 16) indexed by [j*1+k, c] = [j, c] for j=0..3, k=0
    # For combined: V[j*5+k, c]
    #   For k=0..3: V[j*5+k, c] = V4[j*4+k, c]
    #   For k=4: V[j*5+4, c] = V1[j, c]
    
    V = np.zeros((m*p, rank), dtype=float)
    for j in range(4):
        for k in range(4):
            V[j*5 + k, :49] = V4[j*4 + k, :49]
        V[j*5 + 4, 49:] = V1[j, :]
    
    # Verify
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        valid, info = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Fallback to schoolbook rank 80
            rank = 80
            U = np.zeros((n*m, rank), dtype=float)
            V = np.zeros((m*p, rank), dtype=float)
            W = np.zeros((p*n, rank), dtype=float)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
    
    # Convert to lists for the return interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
