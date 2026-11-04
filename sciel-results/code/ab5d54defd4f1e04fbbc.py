"""Tensor decomposition for 4x4x9 matrix multiplication via structured Strassen blocks + local search."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] equals the matmul tensor."""
    rank = len(U[0])
    # Build tensor T[i, j, k] = delta for A[i,j] * B[j,k] -> C[i,k]
    # Convention: U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute reconstructed tensor
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T_recon[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Use einsum
    T_recon = np.einsum('ar,br,cr->abc', Ua, Va, Wa)
    return np.allclose(T_recon, T, atol=1e-9)

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2x2 matmul, rank 7, with integer entries.
    U shape (4,7), V shape (4,7), W shape (4,7)
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Strassen:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # U[i*2+j, r] = coefficient of A_ij in M_r
    # V[j*2+k, r] = coefficient of B_jk in M_r
    # W[k*2+i, r] = coefficient of M_r in C_ki  (note: C index is [i,k], W index is [k*n+i])
    
    # For 2x2: n=2, m=2, p=2
    # W[k*n+i, r] where n=2, so W[2*k+i, r]
    # C00 = M1+M4-M5+M7 -> W[2*0+0, r] = W[0, r]
    # C01 = M3+M5 -> W[2*1+0, r] = W[2, r]
    # C10 = M2+M4 -> W[2*0+1, r] = W[1, r]
    # C11 = M1-M2+M3+M6 -> W[2*1+1, r] = W[3, r]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C00 and C11
    
    # M2: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1  # C10 and C11
    
    # M3: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C01 and C11
    
    # M4: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C00 and C10
    
    # M5: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C00 and C01
    
    # M6: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C11
    
    # M7: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1  # C00
    
    return U, V, W

def _build_structured(n=4, m=4, p=9):
    """Build structured decomposition using Kronecker product of Strassen blocks.
    
    4x4x9 = (2x2) ⊗ (2x2x9)
    2x2x9: split output 9 into [4, 4, 1]
      - First 4 cols: 2x2x4 = (2x2x2) ⊗ (1x1x2) -> rank 7*2 = 14? No.
      Actually let me think differently.
    
    Better: 4x4x9 as (2x2) ⊗ (2x2x9)
    For 2x2x9: use 3 independent 2x2x3 blocks? No, 9 isn't 3*3.
    
    Let me use: 4x4x9 = (2x2) ⊗ (2x2x9)
    2x2x9: decompose as 2x2x4 + 2x2x4 + 2x2x1
      2x2x4 = (2x2x2) ⊗ (1x1x2): rank = 7 * 2 = 14? No, that's not right either.
    
    Actually the Kronecker product of tensors: if T1 has rank r1 and T2 has rank r2,
    T1⊗T2 has rank r1*r2.
    
    4x4x9 = (2x2) ⊗ (2x2x9)
    2x2x9: Let's decompose the 9 as 4+4+1.
      2x2x4 = (2x2x2) ⊗ (1x1x2) -> rank 7*2 = 14
      2x2x1 = rank 4 (schoolbook for 2x2x1)
    So 2x2x9 has rank 14+14+4 = 32
    Then 4x4x9 has rank 7*32 = 224? That's worse than schoolbook.
    
    Hmm, that's not good. Let me reconsider.
    
    Actually: 4x4x9 = (2x2) ⊗ (2x2x9)
    But the rank of tensor product is at most r1*r2, and can be less.
    
    Alternative: 4x4x9 = (2x2x4) ⊗ (2x1x9/2)... this is getting complicated.
    
    Let me just use the schoolbook but with Strassen on the 4x4x4 blocks:
    4x4x9: split into 4x4x4 + 4x4x4 + 4x4x1
    4x4x4: use Strassen recursively: 4x4x4 = (2x2)⊗(2x2x4)... 
    
    Actually the simplest: 4x4x4 can be done with rank 49 (Strassen applied to 4x4).
    4x4x1 has rank 16 (schoolbook).
    Total: 49 + 49 + 16 = 114.
    
    Let me build 4x4x4 with Strassen:
    4x4x4 = (2x2) ⊗ (2x2x4)
    2x2x4 = (2x2x2) ⊗ (1x1x2) -> rank 7*2 = 14
    So 4x4x4 rank = 7*14 = 98? No, that's the tensor product rank which is an upper bound.
    
    Actually for 4x4x4, the known best is 49 (Strassen). Let me just construct it directly.
    """
    # 4x4x4 Strassen decomposition, rank 49
    # 4x4x4 = (2x2) ⊗ (2x2x4)
    # 2x2x4: decompose 4 columns as 2+2
    #   2x2x2: Strassen rank 7
    #   2x2x2: Strassen rank 7
    #   So 2x2x4 rank = 14
    # 4x4x4 rank = 7 * 14 = 98? That's not 49.
    
    # The issue: tensor product rank is r1*r2, but 4x4x4 is NOT (2x2)⊗(2x2x4) in a way that gives 49.
    # 4x4x4 is a single tensor, and Strassen gives rank 49 by a different construction.
    
    # Let me just use the known 49-rank decomposition for 4x4x4.
    # 4x4x4: partition into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bj is 2x2
    # C = A*B, each Ci is 2x2
    # Apply Strassen at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    # Each Mr is a 2x2 matrix product, which itself needs rank 7 (Strassen for 2x2).
    # So total rank = 7 * 7 = 49.
    
    # Now for 4x4x9: partition 9 columns as 4+4+1
    # 4x4x4 (cols 0-3): rank 49
    # 4x4x4 (cols 4-7): rank 49
    # 4x4x1 (col 8): rank 16
    # Total: 114
    
    # Build 4x4x4 decomposition
    def _build_4x4x4():
        """Build 4x4x4 Strassen decomposition, rank 49."""
        n = m = p = 4
        sU, sV, sW = _strassen_2x2x2()
        
        # Block structure:
        # A indices: i in {0,1}, j in {0,1} for block; within block: ii in {0,1}, jj in {0,1}
        # A[i*2+ii, j*2+jj]
        # B[j*2+jj, k*2+kk]
        # C[i*2+ii, k*2+kk], W index: (k*2+kk)*4 + (i*2+ii)
        
        # Strassen at block level gives 7 block-products, each is 2x2x2 (rank 7)
        # Total: 7*7 = 49
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        # Block-level Strassen factors
        # For block product r (0..6):
        # U_block[i*2+j, r] from sU
        # V_block[j*2+k, r] from sV
        # W_block[k*2+i, r] from sW
        
        # Within each block product, apply 2x2x2 Strassen
        for r in range(7):
            for rr in range(7):
                col = r * 7 + rr
                # A factor: sU[i*2+j, r] * sU[ii*2+jj, rr]
                for i in range(2):
                    for j in range(2):
                        a_coeff = sU[i * 2 + j][r]
                        if a_coeff == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                a_inner = sU[ii * 2 + jj][rr]
                                if a_inner == 0:
                                    continue
                                U[(i * 2 + ii) * 4 + (j * 2 + jj)][col] += a_coeff * a_inner
                # B factor: sV[j*2+k, r] * sV[jj*2+kk, rr]
                for j in range(2):
                    for k in range(2):
                        b_coeff = sV[j * 2 + k][r]
                        if b_coeff == 0:
                            continue
                        for jj in range(2):
                            for kk in range(2):
                                b_inner = sV[jj * 2 + kk][rr]
                                if b_inner == 0:
                                    continue
                                V[(j * 2 + jj) * 4 + (k * 2 + kk)][col] += b_coeff * b_inner
                # C factor: sW[k*2+i, r] * sW[kk*2+ii, rr]
                for k in range(2):
                    for i in range(2):
                        c_coeff = sW[k * 2 + i][r]
                        if c_coeff == 0:
                            continue
                        for kk in range(2):
                            for ii in range(2):
                                c_inner = sW[kk * 2 + ii][rr]
                                if c_inner == 0:
                                    continue
                                W[(k * 2 + kk) * 4 + (i * 2 + ii)][col] += c_coeff * c_inner
        
        return U, V, W
    
    def _build_4x4x1():
        """Build 4x4x1 schoolbook decomposition, rank 16."""
        n = m = p = 4
        rank = 16
        U = [[0]*rank for _ in range(16)]
        V = [[0]*rank for _ in range(16)]
        W = [[0]*rank for _ in range(4)]
        col = 0
        for i in range(4):
            for j in range(4):
                U[i * 4 + j][col] = 1
                V[j * 4 + 0][col] = 1
                W[0 * 4 + i][col] = 1
                col += 1
        return U, V, W
    
    # Build the full 4x4x9
    n, m, p = 4, 4, 9
    U4_4_4, V4_4_4, W4_4_4 = _build_4x4x4()
    U4_4_1, V4_4_1, W4_4_1 = _build_4x4x1()
    
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # First 4x4x4 block (cols 0-3 of B, cols 0-3 of C)
    for i in range(16):
        for c in range(49):
            U[i][c] = U4_4_4[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 4 + k][c] = V4_4_4[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k * 4 + i][c] = W4_4_4[k * 4 + i][c]
    
    # Second 4x4x4 block (cols 4-7 of B, cols 4-7 of C)
    offset = 49
    for i in range(16):
        for c in range(49):
            U[i][offset + c] = U4_4_4[i][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j * 4 + (k + 4)][offset + c] = V4_4_4[j * 4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[(k + 4) * 4 + i][offset + c] = W4_4_4[k * 4 + i][c]
    
    # 4x4x1 block (col 8 of B, col 8 of C)
    offset = 98
    for i in range(16):
        for c in range(16):
            U[i][offset + c] = U4_4_1[i][c]
    for j in range(4):
        for c in range(16):
            V[j * 4 + 8][offset + c] = V4_4_1[j * 4][c]
    for i in range(4):
        for c in range(16):
            W[8 * 4 + i][offset + c] = W4_4_1[i][c]
    
    return U, V, W

def _local_search(U, V, W, n, m, p, time_limit=100):
    """Try to reduce rank by deleting or merging columns."""
    start = time.time()
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    rank = len(U[0])
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    def try_delete(U, V, W, col):
        """Try deleting column col. Returns (success, new_U, new_V, new_W)."""
        rank = len(U[0])
        # Remove column col
        new_U = [row[:col] + row[col+1:] for row in U]
        new_V = [row[:col] + row[col+1:] for row in V]
        new_W = [row[:col] + row[col+1:] for row in W]
        if _verify(new_U, new_V, new_W, n, m, p):
            return True, new_U, new_V, new_W
        return False, None, None
    
    def try_merge(U, V, W, c1, c2):
        """Try merging columns c1 and c2 (add c2 to c1, remove c2)."""
        rank = len(U[0])
        # Add column c2 to column c1
        new_U = [row[:] for row in U]
        new_V = [row[:] for row in V]
        new_W = [row[:] for row in W]
        for i in range(len(new_U)):
            new_U[i][c1] += new_U[i][c2]
        for i in range(len(new_V)):
            new_V[i][c1] += new_V[i][c2]
        for i in range(len(new_W)):
            new_W[i][c1] += new_W[i][c2]
        # Remove column c2
        new_U = [row[:c2] + row[c2+1:] for row in new_U]
        new_V = [row[:c2] + row[c2+1:] for row in new_V]
        new_W = [row[:c2] + row[c2+1:] for row in new_W]
        if _verify(new_U, new_V, new_W, n, m, p):
            return True, new_U, new_V, new_W
        return False, None, None
    
    # Try deletions
    cols_to_try = list(range(rank))
    random.shuffle(cols_to_try)
    
    for col in cols_to_try:
        if time.time() - start > time_limit:
            break
        if col >= len(U[0]):
            continue
        success, nU, nV, nW = try_delete(U, V, W, col)
        if success:
            U, V, W = nU, nV, nW
            rank = len(U[0])
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            # Reset order
            cols_to_try = list(range(rank))
            random.shuffle(cols_to_try)
    
    # Try merges (limited)
    if time.time() - start < time_limit - 10:
        for _ in range(min(200, (time_limit - (time.time() - start)) * 10)):
            if time.time() - start > time_limit:
                break
            c1 = random.randint(0, rank - 1)
            c2 = random.randint(0, rank - 1)
            if c1 == c2:
                continue
            success, nU, nV, nW = try_merge(U, V, W, c1, c2)
            if success:
                U, V, W = nU, nV, nW
                rank = len(U[0])
                best_rank = rank
                best_U, best_V, best_W = U, V, W
    
    return best_U, best_V, best_W

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Build structured decomposition
    U, V, W = _build_structured(4, 4, 9)
    
    # Verify
    if not _verify(U, V, W, 4, 4, 9):
        # Fallback to schoolbook
        n, m, p = 4, 4, 9
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Local search to reduce rank
    remaining = deadline - time.time()
    if remaining > 5:
        U, V, W = _local_search(U, V, W, 4, 4, 9, time_limit=min(remaining - 5, 100))
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
