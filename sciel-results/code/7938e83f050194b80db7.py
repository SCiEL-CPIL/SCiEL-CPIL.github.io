"""
Structural change: Explicit Kronecker/Strassen construction for rank-101 decomposition.
Instead of random search + exact linear solves (which stall at schoolbook rank 120),
we directly construct a rank-101 decomposition using:
  - 4x4x4 Strassen block (rank 49)
  - 4x4x2 Strassen block (rank 28)  
  - 4x1x6 schoolbook block (rank 24)
Total: 49 + 28 + 24 = 101

This is a deterministic construction that should verify exactly.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_tensors():
    """
    Return the 7 rank-1 tensors for 2x2x2 Strassen.
    Each tensor is a triple (u, v, w) where u, v, w are length-4 vectors
    (for the 2x2 matrices flattened in row-major order).
    
    Strassen formulas for C = A * B where A, B, C are 2x2:
    A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    C = [[c11, c12], [c21, c22]]
    
    p1 = (a11 + a22)(b11 + b22)
    p2 = (a21 + a22)b11
    p3 = a11(b12 - b22)
    p4 = a22(b21 - b11)
    p5 = (a11 + a12)b22
    p6 = (a21 - a11)(b11 + b12)
    p7 = (a12 - a22)(b21 + b22)
    
    c11 = p1 + p4 - p5 + p7
    c12 = p3 + p5
    c21 = p2 + p4
    c22 = p1 - p2 + p3 + p6
    """
    # u vectors (for A, flattened row-major: [a11, a12, a21, a22])
    # v vectors (for B, flattened row-major: [b11, b12, b21, b22])
    # w vectors (for C, flattened row-major: [c11, c12, c21, c22])
    
    tensors = []
    
    # p1 = (a11 + a22)(b11 + b22)
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]  # contributes to c11 and c22
    tensors.append((u1, v1, w1))
    
    # p2 = (a21 + a22)b11
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]  # contributes to c21
    tensors.append((u2, v2, w2))
    
    # p3 = a11(b12 - b22)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0]  # contributes to c12
    tensors.append((u3, v3, w3))
    
    # p4 = a22(b21 - b11)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [1, 0, 1, 0]  # contributes to c11 and c21
    tensors.append((u4, v4, w4))
    
    # p5 = (a11 + a12)b22
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]  # contributes to c11 (neg) and c12
    tensors.append((u5, v5, w5))
    
    # p6 = (a21 - a11)(b11 + b12)
    u6 = [-1, 0, 1, 0]
    v6 = [1, 1, 0, 0]
    w6 = [0, 0, 0, 1]  # contributes to c22
    tensors.append((u6, v6, w6))
    
    # p7 = (a12 - a22)(b21 + b22)
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [1, 0, 0, 0]  # contributes to c11
    tensors.append((u7, v7, w7))
    
    return tensors

def _kron_strassen_block(n_rows_a, n_rows_b, n_cols_c, base_tensors):
    """
    Expand a base set of rank-1 tensors (for 2x2x2) into a larger block.
    
    For a block that computes C[i:i+na, k:k+nc] = A[i:i+na, j:j+nb] * B[j:j+nb, k:k+nc],
    we need to tile the 2x2 Strassen pattern.
    
    Actually, let's just build the blocks explicitly for our specific cases.
    """
    pass

def _build_4x4x4_strassen():
    """
    Build rank-1 tensors for 4x4x4 matrix multiplication using Strassen.
    We split 4x4 into four 2x2 blocks and use Strassen recursively.
    
    A is 4x4, B is 4x4, C is 4x4.
    Split: A = [[A11, A12], [A21, A22]] where each Ai is 2x2.
           B = [[B11, B12], [B21, B22]]
           C = [[C11, C12], [C21, C22]]
    
    Using Strassen at the 2x2 block level:
    p1 = (A11+A22)(B11+B22)
    p2 = (A21+A22)B11
    p3 = A11(B12-B22)
    p4 = A22(B21-B11)
    p5 = (A11+A12)B22
    p6 = (A21-A11)(B11+B12)
    p7 = (A12-A22)(B21+B22)
    
    Each pi is a 2x2 matrix product of 2x2 matrices.
    Each such product requires 7 more rank-1 tensors (by Strassen).
    So total = 7 * 7 = 49 tensors.
    
    We need to map these to the global 4x4x4 indices.
    """
    base_tensors = _strassen_2x2_tensors()
    result = []  # list of (u, v, w) where each is length 16
    
    # For each of the 7 top-level Strassen products:
    for pi_idx in range(7):
        u_top, v_top, w_top = base_tensors[pi_idx]
        
        # Determine which 2x2 sub-blocks of A, B, C are involved
        # u_top tells us which 2x2 blocks of A are combined
        # v_top tells us which 2x2 blocks of B are combined  
        # w_top tells us which 2x2 blocks of C receive the result
        
        # For each of the 7 bottom-level Strassen tensors:
        for pj_idx in range(7):
            u_bot, v_bot, w_bot = base_tensors[pj_idx]
            
            # Build full u, v, w vectors of length 16
            u_full = [0] * 16
            v_full = [0] * 16
            w_full = [0] * 16
            
            # Map top-level block indices to global indices
            # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3)
            #           A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
            # Flattened: A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
            
            # u_top has 4 entries for the 4 2x2 blocks of A
            # Block 0: A11, Block 1: A12, Block 2: A21, Block 3: A22
            a_block_offsets = [0, 4, 8, 12]
            for blk in range(4):
                coeff_top = u_top[blk]
                if coeff_top == 0:
                    continue
                # Within this 2x2 block, apply bottom-level Strassen
                for local_idx in range(4):
                    coeff_bot = u_bot[local_idx]
                    if coeff_bot == 0:
                        continue
                    u_full[a_block_offsets[blk] + local_idx] = coeff_top * coeff_bot
            
            # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3)
            #           B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
            # Flattened: B11: 0-3, B12: 4-7, B21: 8-11, B22: 12-15
            b_block_offsets = [0, 4, 8, 12]
            for blk in range(4):
                coeff_top = v_top[blk]
                if coeff_top == 0:
                    continue
                for local_idx in range(4):
                    coeff_bot = v_bot[local_idx]
                    if coeff_bot == 0:
                        continue
                    v_full[b_block_offsets[blk] + local_idx] = coeff_top * coeff_bot
            
            # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3)
            #           C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
            # Flattened: C11: 0-3, C12: 4-7, C21: 8-11, C22: 12-15
            c_block_offsets = [0, 4, 8, 12]
            for blk in range(4):
                coeff_top = w_top[blk]
                if coeff_top == 0:
                    continue
                for local_idx in range(4):
                    coeff_bot = w_bot[local_idx]
                    if coeff_bot == 0:
                        continue
                    w_full[c_block_offsets[blk] + local_idx] = coeff_top * coeff_bot
            
            # Check if this tensor is non-trivial
            if any(x != 0 for x in u_full) and any(x != 0 for x in v_full) and any(x != 0 for x in w_full):
                result.append((u_full, v_full, w_full))
    
    return result

def _build_4x4x2_strassen():
    """
    Build rank-1 tensors for computing C[:, 4:6] = A[:, 0:4] * B[0:4, 4:6].
    A is 4x4, B is 4x2, C is 4x2.
    
    Split A into 2x2 blocks: A = [[A11, A12], [A21, A22]]
    Split B into B_top (2x2, rows 0-1) and B_bottom (2x2, rows 2-3)
    C = [C_top; C_bottom] where:
      C_top = A11*B_top + A12*B_bottom
      C_bottom = A21*B_top + A22*B_bottom
    
    Each 2x2*2x2=2x2 product uses 7 Strassen tensors.
    Four such products: 4 * 7 = 28 tensors total.
    
    Local indices:
    u: length 16 (A flattened row-major: A11[0:4], A12[4:8], A21[8:12], A22[12:16])
    v: length 8  (B flattened row-major: B_top[0:4], B_bottom[4:8])
    w: length 8  (C flattened row-major: C_top[0:4], C_bottom[4:8])
    """
    base_tensors = _strassen_2x2_tensors()
    tensors = []
    
    # A block offsets in u (length 16):
    # A11: 0-3, A12: 4-7, A21: 8-11, A22: 12-15
    a_offsets = [0, 4, 8, 12]
    
    # B block offsets in v (length 8):
    # B_top: 0-3, B_bottom: 4-7
    b_offsets = [0, 4]
    
    # C block offsets in w (length 8):
    # C_top: 0-3, C_bottom: 4-7
    c_offsets = [0, 4]
    
    # Four products:
    # 1. A11 * B_top -> C_top
    # 2. A12 * B_bottom -> C_top
    # 3. A21 * B_top -> C_bottom
    # 4. A22 * B_bottom -> C_bottom
    
    products = [
        (0, 0, 0),  # A11, B_top, C_top
        (1, 1, 0),  # A12, B_bottom, C_top
        (2, 0, 1),  # A21, B_top, C_bottom
        (3, 1, 1),  # A22, B_bottom, C_bottom
    ]
    
    for (a_blk, b_blk, c_blk) in products:
        for (u_bot, v_bot, w_bot) in base_tensors:
            u_full = [0] * 16
            v_full = [0] * 8
            w_full = [0] * 8
            
            for local_idx in range(4):
                if u_bot[local_idx] != 0:
                    u_full[a_offsets[a_blk] + local_idx] = u_bot[local_idx]
                if v_bot[local_idx] != 0:
                    v_full[b_offsets[b_blk] + local_idx] = v_bot[local_idx]
                if w_bot[local_idx] != 0:
                    w_full[c_offsets[c_blk] + local_idx] = w_bot[local_idx]
            
            if any(x != 0 for x in u_full) and any(x != 0 for x in v_full) and any(x != 0 for x in w_full):
                tensors.append((u_full, v_full, w_full))
    
    return tensors

def _build_4x1x6_schoolbook():
    """
    Build rank-1 tensors for computing C[:, 0:6] = A[:, 4] * B[4, 0:6].
    A is 4x1 (column vector), B is 1x6 (row vector), C is 4x6.
    This is an outer product: C = a * b^T.
    Rank = 4*6 = 24 with schoolbook.
    """
    tensors = []
    for i in range(4):  # row of C / row of A
        for k in range(6):  # col of C / col of B
            u = [0] * 20  # 4*5
            v = [0] * 30  # 5*6
            w = [0] * 24  # 6*4
            u[i * 5 + 4] = 1  # A[i, 4]
            v[4 * 6 + k] = 1  # B[4, k]
            w[k * 4 + i] = 1  # C[k, i] (note: W[k*n+i] convention)
            tensors.append((u, v, w))
    return tensors

def _assemble_decomposition():
    """
    Assemble the full decomposition from the three blocks.
    
    Block 1: 4x4x4 (A cols 0-3, B rows 0-3, C cols 0-3) - rank 49
    Block 2: 4x4x2 (A cols 0-3, B rows 0-3, C cols 4-5) - rank 32
    Block 3: 4x1x6 (A col 4, B row 4, C cols 0-5) - rank 24
    
    Total rank: 49 + 28 + 24 = 101
    """
    U = []  # will be list of lists, shape (20, rank)
    V = []  # shape (30, rank)
    W = []  # shape (24, rank)
    
    # Block 1: 4x4x4
    block1 = _build_4x4x4_strassen()
    for (u, v, w) in block1:
        # u is length 16 (4x4 flattened)
        # Map to global U indices: U[i*5+j] for i in 0..3, j in 0..3
        u_full = [0] * 20
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            u_full[i * 5 + j] = u[idx]
        
        # v is length 16 (4x4 flattened)
        # Map to global V indices: V[j*6+k] for j in 0..3, k in 0..3
        v_full = [0] * 30
        for idx in range(16):
            j = idx // 4
            k = idx % 4
            v_full[j * 6 + k] = v[idx]
        
        # w is length 16 (4x4 flattened)
        # Map to global W indices: W[k*4+i] for k in 0..3, i in 0..3
        w_full = [0] * 24
        for idx in range(16):
            k = idx // 4
            i = idx % 4
            w_full[k * 4 + i] = w[idx]
        
        U.append(u_full)
        V.append(v_full)
        W.append(w_full)
    
    # Block 2: 4x4x2
    block2 = _build_4x4x2_strassen()
    for (u, v, w) in block2:
        # u is length 16 (4x4 flattened)
        u_full = [0] * 20
        for idx in range(16):
            i = idx // 4
            j = idx % 4
            u_full[i * 5 + j] = u[idx]
        
        # v is length 8 (4x2 flattened: B_top rows 0-1, B_bottom rows 2-3, cols 4-5)
        # v[0:4] = B_top (rows 0-1, cols 4-5), v[4:8] = B_bottom (rows 2-3, cols 4-5)
        v_full = [0] * 30
        for idx in range(8):
            row = idx // 2  # local row 0-3
            col = idx % 2   # local col 0-1
            global_row = row
            global_col = 4 + col
            v_full[global_row * 6 + global_col] = v[idx]
        
        # w is length 8 (4x2 flattened: C_top rows 0-1, C_bottom rows 2-3, cols 4-5)
        # w[0:4] = C_top (rows 0-1, cols 4-5), w[4:8] = C_bottom (rows 2-3, cols 4-5)
        w_full = [0] * 24
        for idx in range(8):
            row = idx // 2   # local row 0-3
            col = idx % 2    # local col 0-1
            global_row = row
            global_col = 4 + col
            # W convention: W[k*n + i] where k=col, n=4, i=row
            w_full[global_col * 4 + global_row] = w[idx]
        
        U.append(u_full)
        V.append(v_full)
        W.append(w_full)
    
    # Block 3: 4x1x6
    block3 = _build_4x1x6_schoolbook()
    for (u, v, w) in block3:
        # u, v, w are already in global form
        U.append(u)
        V.append(v)
        W.append(w)
    
    rank = len(U)
    return U, V, W, rank

def _verify_numpy(U, V, W, n, m, p):
    if np is None:
        return False
    R = len(U[0]) if U else 0
    Ua = np.array(U, dtype=np.float64).T  # shape (rank, n*m)
    Va = np.array(V, dtype=np.float64).T  # shape (rank, m*p)
    Wa = np.array(W, dtype=np.float64).T  # shape (rank, p*n)
    
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[c, k * n + i]
                if wc == 0:
                    continue
                uc = Ua[c, i * m: i * m + m]
                vc = Va[c, k * p: k * p + m]
                M += np.outer(uc, vc) * wc
            target = np.eye(m, dtype=np.float64)
            if not np.allclose(M, target, atol=1e-9):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    U, V, W, rank = _assemble_decomposition()
    
    # Verify
    if _verify_numpy(U, V, W, 4, 5, 6):
        return {"U": U, "V": V, "W": W}
    
    # If verification fails, fall back to schoolbook
    R = 4 * 5 * 6
    U_school = [[0] * R for _ in range(20)]
    V_school = [[0] * R for _ in range(30)]
    W_school = [[0] * R for _ in range(24)]
    col = 0
    for i in range(4):
        for j in range(5):
            for k in range(6):
                U_school[i * 5 + j][col] = 1
                V_school[j * 6 + k][col] = 1
                W_school[k * 4 + i][col] = 1
                col += 1
    
    return {"U": U_school, "V": V_school, "W": W_school}

# EVOLVE-BLOCK-END
