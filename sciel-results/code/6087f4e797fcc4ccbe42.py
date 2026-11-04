"""Matrix multiplication 4x4 * 4x6 tensor decomposition.

Strategy:
  1. Build a rank-77 decomposition using a block-level Kronecker structure:
     - Split A (4x4) into 2x2 blocks, B (4x6) into 2x2 blocks (3 pairs of cols).
     - For each of the 3 column-pairs of B, apply Strassen (rank 7) on the 2x2 A-blocks
       combined with the 2x2 B-block, plus schoolbook (rank 4) for the cross terms.
     - Actually, the cleaner construction: treat the problem as 4x4 * 4x6.
       Split into two 4x4 * 4x2 subproblems? No, let's use the Kronecker idea.
       
     Better approach from inspiration 1 (score 1.2468): 
     Build rank-44 by tiling 4x4*4x6 into four 2x2*2x3 blocks, each using
     Strassen(7) for 2x2 part + schoolbook(4) for 2x1 part = 11 terms per block.
     But wait, 4 blocks * 11 = 44? Let me re-derive.
     
     Actually the key insight: 
     4x4 * 4x6 can be seen as:
       A (4x4) * B (4x6) = C (4x6)
     Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]]
     Split B into 2x2 blocks (3 column pairs): B = [B0 | B1 | B2] where each Bj is 2x2... 
     No, B is 4x6, so split columns into 3 pairs: B = [B^(0) | B^(1) | B^(2)] each 4x2.
     Then C = [A*B^(0) | A*B^(1) | A*B^(2)], each A*B^(j) is 4x2.
     
     For A (4x4) * B^(j) (4x2):
       Split A into 2x2 blocks, B^(j) into 2x1 blocks (top 2 rows, bottom 2 rows).
       B^(j) = [B_top; B_bot] where B_top is 2x2, B_bot is 2x2.
       C_top = A00*B_top + A01*B_bot  (2x2 = 2x2*2x2 + 2x2*2x2... no)
       
     Let me use the direct Kronecker product structure:
     The 4x4*4x6 multiplication tensor can be decomposed as:
     For each of the 7 Strassen terms on the 2x2 A-blocks, combined with
     appropriate B-blocks, plus additional terms for the extra 2 columns.
     
     Simplest valid construction: use the known rank-77 from the cell findings.
     The cell finding says: "11 outer terms (7 Strassen + 4 schoolbook) each expanded
     via rank-7 inner Strassen gives 77 rank"
     
     Let me build this properly:
     - The 4x4*4x6 product can be viewed as a Kronecker-like structure.
     - Split the 4x4 A into 2x2 blocks, 4x6 B into 2x2 blocks (3 col pairs).
     - For the first 4 columns of B (2 pairs of 2x2), use Strassen on the 2x2 A-blocks:
       This gives 7 terms for the 4x4*4x4 part (using 2x2 Strassen blocks).
     - For the last 2 columns, use schoolbook: 4 terms.
     - Total outer: 7 + 4 = 11 terms.
     - Each outer term, when expanded to full 4x4*4x6, needs inner structure.
     
     Actually, let me just build a clean rank-77 construction:
     
     Method: 
     4x4 * 4x6 = (2x2 block matrix) * (2x2 block matrix with 3 col blocks)
     
     Let A = [[A00, A01], [A10, A11]] (2x2 blocks of 2x2)
     Let B = [[B00, B01, B02], [B10, B11, B12]] (2x3 blocks of 2x2)
     Let C = [[C00, C01, C02], [C10, C11, C12]] (2x3 blocks of 2x2)
     
     C_ab = A_a0 * B_0b + A_a1 * B_1b for a,b in {0,1,2}
     
     For the 4x4*4x4 part (b=0,1), we can use Strassen on the 2x2 block level:
     7 terms compute C_a0 and C_a1 for a=0,1.
     For the 4x4*4x2 part (b=2), we use schoolbook: 4 terms.
     
     Each "term" at the block level is a product of a linear combination of 2x2 A-blocks
     times a linear combination of 2x2 B-blocks, resulting in a 2x2 C-block.
     But each such block product, when expanded to scalar entries, is a 2x2*2x2 = 4 scalar
     products. However, in tensor decomposition, each block-level term corresponds to
     4 scalar-level terms (one per entry of the 2x2 result).
     
     Wait, that's not right either. In tensor decomposition, each rank-1 term is
     U[:,r] * V[:,r] * W[:,r] which is a single scalar multiplication contributing
     to one entry of the output tensor.
     
     Let me think about this more carefully. The total number of scalar multiplications
     in schoolbook is 4*4*6 = 96. A rank-r decomposition means we can do it with r
     scalar multiplications.
     
     For the Kronecker structure: if we have a rank-7 decomposition of 2x2*2x2 (Strassen),
     and we "tile" it, the rank multiplies by the number of tiles.
     
     Actually, the cleanest way: 
     4x4*4x6 can be decomposed by viewing it as:
     - First, compute the 4x4*4x4 part using Strassen at the 2x2 block level: rank 7*4 = 28? No.
     
     Let me just use the explicit construction:
     For the 4x4*4x4 part (first 4 columns of B):
       Use 2x2 block Strassen: 7 block-multiplications, each block-multiplication is
       a 2x2*2x2 product which in scalar terms is 4 multiplications.
       But in tensor rank, each block-multiplication contributes 4 rank terms.
       So 7 * 4 = 28 rank terms for the 4x4*4x4 part.
     
     For the 4x4*4x2 part (last 2 columns of B):
       Schoolbook: 4*4*2 = 32 rank terms.
     
     Total: 28 + 32 = 60. That's better than 77!
     
     Wait, but the cell finding says 77. Let me re-read: "11 outer terms (7 Strassen + 4 schoolbook)
     each expanded via rank-7 inner Strassen gives 77 rank". So 11 * 7 = 77. That means each
     outer term is expanded using inner Strassen (rank 7), not schoolbook (rank 4).
     
     Hmm, that doesn't make sense for schoolbook terms. Let me just build the 60-rank construction
     and verify it.
     
     Actually, I realize the issue. When I say "7 block-multiplications, each is 2x2*2x2", 
     in tensor decomposition each 2x2*2x2 product has rank 4 (schoolbook) or 7 (Strassen).
     If I use schoolbook for the inner 2x2*2x2, then 7*4 = 28. If I use Strassen, 7*7 = 49.
     
     So for the 4x4*4x4 part: 28 rank terms (using schoolbook inner).
     For the 4x4*4x2 part: 32 rank terms (schoolbook).
     Total: 60.
     
     Let me build this.
"""
import numpy as np
import time

DIMENSIONS = [4, 4, 6]

# Strassen 2x2*2x2 coefficients
alpha = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 0, 1: 0, 2: 1, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
]
beta = [
    {0: 1, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 0, 2: 0, 3: 0},
    {0: 0, 1: 1, 2: 0, 3: -1},
    {0: -1, 1: 0, 2: 1, 3: 0},
    {0: 0, 1: 0, 2: 0, 3: 1},
    {0: 1, 1: 1, 2: 0, 3: 0},
    {0: 0, 1: 0, 2: 1, 3: 1},
]
recon = [
    [(0, 0, 1), (1, 1, 1)],
    [(1, 0, 1), (1, 1, -1)],
    [(0, 1, 1), (1, 1, 1)],
    [(0, 0, 1), (1, 0, 1)],
    [(0, 0, -1), (0, 1, 1)],
    [(1, 1, 1)],
    [(0, 0, 1)],
]


def _build_rank60():
    """Build rank-60 decomposition: 28 for 4x4*4x4 (Strassen blocks, schoolbook inner) + 32 for 4x4*4x2."""
    n, m, p = DIMENSIONS
    rank = 60
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    # Part 1: 4x4 * 4x4 (first 4 columns of B)
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    # Each block is 2x2. Use Strassen at block level, schoolbook at scalar level.
    for s in range(7):
        # A-block coefficients
        a_coeffs = alpha[s]  # dict mapping block_idx -> coeff
        # B-block coefficients
        b_coeffs = beta[s]
        # C-block reconstruction
        c_recon = recon[s]
        
        # For each of the 4 scalar positions in the 2x2 result
        for ai in range(2):
            for aj in range(2):
                # A-block (bi, bj) where bi, bj in {0,1}
                a_block = bi * 2 + bj if False else None  # We need to iterate over A-blocks
                pass
        
        # Actually, let me be more explicit.
        # A-blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3),
        #           A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
        # Block index: 0=A00, 1=A01, 2=A10, 3=A11
        # B-blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3),
        #           B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
        # C-blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3),
        #           C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)
        
        # For each scalar position (i,j) in the 2x2 result of the block product:
        # The block product M_s = (sum_a a_coeffs[a] * A_block_a) * (sum_b b_coeffs[b] * B_block_b)
        # = sum_{a,b} a_coeffs[a] * b_coeffs[b] * A_block_a * B_block_b
        # Each A_block_a * B_block_b is a 2x2*2x2 product, which in schoolbook is 4 scalar products.
        # So M_s contributes to the C-blocks via the recon coefficients.
        
        # For each scalar position (i,j) in the 2x2 output:
        for i in range(2):
            for j in range(2):
                # U: A entries
                for a_block in range(4):
                    ca = a_coeffs.get(a_block, 0)
                    if ca == 0:
                        continue
                    a_row_base = (a_block // 2) * 2
                    a_col_base = (a_block % 2) * 2
                    U[(a_row_base + i) * m + (a_col_base + j), col] += ca
                
                # V: B entries
                for b_block in range(4):
                    cb = b_coeffs.get(b_block, 0)
                    if cb == 0:
                        continue
                    b_row_base = (b_block // 2) * 2
                    b_col_base = (b_block % 2) * 2
                    V[(b_row_base + i) * p + (b_col_base + j), col] += cb
                
                # W: C entries
                for (c_a, c_b, gamma) in c_recon:
                    c_row_base = c_a * 2
                    c_col_base = c_b * 2
                    W[(c_col_base + j) * n + (c_row_base + i), col] += gamma
                
                col += 1
    
    # Part 2: 4x4 * 4x2 (last 2 columns of B, cols 4-5)
    # Schoolbook: for each i (0-3), j (0-3), k (4-5): one rank term
    for i in range(n):
        for j in range(m):
            for k in range(4, 6):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    return U, V, W


def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _err_of(U, V, W, T):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.sum((T - recon) ** 2)


def _check(U, V, W, T, atol=1e-9):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(recon, T, atol=atol)


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    T = _build_target_tensor()
    
    # Build rank-60 decomposition
    U60, V60, W60 = _build_rank60()
    
    # Verify
    assert _check(U60, V60, W60, T, atol=1e-6), "Rank-60 verification failed"
    
    best_rank = 60
    best_U, best_V, best_W = U60.copy(), V60.copy(), W60.copy()
    
    # Try to reduce rank via optimization
    for target_rank in range(59, 45, -1):
        if time.time() > deadline - 20:
            break
        
        for restart in range(2):
            if time.time() > deadline - 20:
                break
            
            U = U60[:, :target_rank].copy()
            V = V60[:, :target_rank].copy()
            W = W60[:, :target_rank].copy()
            
            scale = 0.05
            U += np.random.randn(n * m, target_rank) * scale
            V += np.random.randn(m * p, target_rank) * scale
            W += np.random.randn(p * n, target_rank) * scale
            
            err = _err_of(U, V, W, T)
            
            for it in range(300):
                if time.time() > deadline - 15:
                    break
                if err < 1e-12:
                    break
                
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                diff = T - recon
                
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
                
                lr = 0.005
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                new_err = _err_of(U, V, W, T)
                if new_err < err:
                    err = new_err
                else:
                    U += lr * grad_U
                    V += lr * grad_V
                    W += lr * grad_W
                    lr *= 0.5
                    if lr < 1e-7:
                        break
        
        if err < 1e-10:
            best_rank = target_rank
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
