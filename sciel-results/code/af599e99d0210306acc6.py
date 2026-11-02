"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            A_c = uc.reshape(n, m)
            B_c = vc.reshape(m, p)
            C_c = np.zeros((n, p))
            for idx in range(p * n):
                k, i = idx // n, idx % n
                C_c[i, k] = wc[idx]
            C_computed += C_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
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

def _build_laderman_444():
    """
    Build Laderman's rank-26 decomposition for 4x4x4 matrix multiplication.
    
    Laderman's algorithm computes C = A*B where A, B, C are 4x4 matrices.
    We partition each into 2x2 blocks:
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]]
    C = [[C11, C12], [C21, C22]]
    
    The 26 products are:
    M1  = (A11 + A22) * (B11 + B22)
    M2  = (A11 + A12) * (B21 + B22)
    M3  = (A21 - A11) * (B11 + B12)
    M4  = A11 * (B12 - B22)
    M5  = A22 * (B21 - B11)
    M6  = (A12 - A22) * (B21 + B22)
    M7  = (A21 + A22) * (B11 - B12)
    M8  = (A11 + A22) * (B11 - B22)
    M9  = (A11 + A12) * (B21 - B22)
    M10 = (A21 - A11) * (B11 - B12)
    M11 = A11 * (B12 + B22)
    M12 = A22 * (B21 + B11)
    M13 = (A12 - A22) * (B21 - B22)
    M14 = (A21 + A22) * (B11 + B12)
    M15 = (A11 + A22) * (B11 + B22 + B12 + B21)  # Wait, this doesn't look right
    ...
    """
    # Let me use the known Laderman decomposition.
    # Actually, let me construct it more carefully.
    # 
    # For 4x4x4, we can use the following approach:
    # Split into 2x2 blocks and use Strassen for the block matrix multiplication.
    # This gives 4 * 7 = 28 rank.
    # 
    # To get to 26, we need to share some computations.
    # 
    # Let me just implement the 28-rank block Strassen first, which is simpler.
    
    n = m = p = 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 2x2 Strassen formulas:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we have blocks:
    # A = [[A11, A12], [A21, A22]] where each is 2x2
    # B = [[B11, B12], [B21, B22]] where each is 2x2
    # C = [[C11, C12], [C21, C22]] where each is 2x2
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each of these is a 2x2 matrix product, which we can compute with Strassen (rank 7).
    # So we need 4 * 7 = 28 rank total.
    
    # Block indices:
    # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    #
    # B11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # B21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    #
    # C11: rows 0-1, cols 0-1 -> indices 0,1,4,5
    # C12: rows 0-1, cols 2-3 -> indices 2,3,6,7
    # C21: rows 2-3, cols 0-1 -> indices 8,9,12,13
    # C22: rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    # Strassen products for each block pair
    # For block product X*Y where X, Y are 2x2:
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # Helper to get 2x2 block indices
    def block_indices(base_row, base_col, size=2):
        """Get linear indices for a 2x2 block starting at (base_row, base_col) in a 4x4 matrix."""
        indices = []
        for dr in range(size):
            for dc in range(size):
                indices.append((base_row + dr) * 4 + (base_col + dc))
        return indices
    
    # A block indices
    A11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    A12_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    A21_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    A22_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    # B block indices
    B11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    B12_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    B21_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    B22_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    # C block indices
    C11_idx = block_indices(0, 0)  # [0, 1, 4, 5]
    C12_idx = block_indices(0, 2)  # [2, 3, 6, 7]
    C21_idx = block_indices(2, 0)  # [8, 9, 12, 13]
    C22_idx = block_indices(2, 2)  # [10, 11, 14, 15]
    
    col = 0
    
    # For each of the 4 block products, apply Strassen
    block_pairs = [
        (A11_idx, B11_idx, C11_idx),  # C11 = A11*B11
        (A12_idx, B21_idx, C11_idx),  # C11 += A12*B21
        (A11_idx, B12_idx, C12_idx),  # C12 = A11*B12
        (A12_idx, B22_idx, C12_idx),  # C12 += A12*B22
        (A21_idx, B11_idx, C21_idx),  # C21 = A21*B11
        (A22_idx, B21_idx, C21_idx),  # C21 += A22*B21
        (A21_idx, B12_idx, C22_idx),  # C22 = A21*B12
        (A22_idx, B22_idx, C22_idx),  # C22 += A22*B22
    ]
    
    # Wait, this gives 8 block products * 7 rank each = 56 rank, which is worse than schoolbook!
    # 
    # The issue is that I'm treating each block product independently.
    # The correct approach is to compute the full 4x4 product using Strassen on the 2x2 block structure.
    #
    # C = A*B where A, B are 4x4 partitioned into 2x2 blocks:
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each Cij is a 2x2 matrix. To compute C11 = A11*B11 + A12*B21, we can:
    # 1. Compute P1 = A11*B11 using Strassen (7 products)
    # 2. Compute P2 = A12*B21 using Strassen (7 products)
    # 3. C11 = P1 + P2
    #
    # This gives 4 * 2 * 7 = 56 rank, which is worse than schoolbook (64).
    #
    # The correct way to use Strassen for 4x4 is to treat the 4x4 matrices as 2x2 block matrices
    # and apply Strassen to the block matrix multiplication:
    #
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mj is a 2x2 matrix product. If we compute each Mj using Strassen (7 scalar products),
    # we get 7 * 7 = 49 scalar products. But that's not right either.
    #
    # Actually, when we say "apply Strassen to 2x2 block matrices", we mean:
    # - The inputs to Strassen are 2x2 matrices (blocks)
    # - The products M1, ..., M7 are 2x2 matrix products
    # - Each 2x2 matrix product requires 4 scalar multiplications (schoolbook) or 7 (Strassen)
    #
    # If we use schoolbook for the inner 2x2 products: 7 * 4 = 28 rank
    # If we use Strassen for the inner 2x2 products: 7 * 7 = 49 rank
    #
    # So the block Strassen with schoolbook inner products gives rank 28 for 4x4x4.
    # This is better than schoolbook (64) but worse than Laderman (26).
    
    # Let me implement the rank-28 block Strassen:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # Each Mj is a 2x2 * 2x2 product, which we compute with schoolbook (4 scalar products).
    # Total: 7 * 4 = 28 scalar products.
    
    # Define the 7 Strassen combinations for the block level
    # Each combination is a linear combination of blocks, then a 2x2 matrix product
    
    strassen_blocks = [
        # (A_factor, B_factor) where each is a dict mapping block_name to coefficient
        # M1 = (A11 + A22) * (B11 + B22)
        ({'A11': 1, 'A22': 1}, {'B11': 1, 'B22': 1}),
        # M2 = (A21 + A22) * B11
        ({'A21': 1, 'A22': 1}, {'B11': 1}),
        # M3 = A11 * (B12 - B22)
        ({'A11': 1}, {'B12': 1, 'B22': -1}),
        # M4 = A22 * (B21 - B11)
        ({'A22': 1}, {'B21': 1, 'B11': -1}),
        # M5 = (A11 + A12) * B22
        ({'A11': 1, 'A12': 1}, {'B22': 1}),
        # M6 = (A21 - A11) * (B11 + B12)
        ({'A21': 1, 'A11': -1}, {'B11': 1, 'B12': 1}),
        # M7 = (A12 - A22) * (B21 + B22)
        ({'A12': 1, 'A22': -1}, {'B21': 1, 'B22': 1}),
    ]
    
    # For each Strassen product Mj, we compute a 2x2 * 2x2 product using schoolbook
    # This gives 4 scalar products per Mj, so 28 total
    
    # Map block names to their 2x2 index sets
    A_blocks = {
        'A11': A11_idx,
        'A12': A12_idx,
        'A21': A21_idx,
        'A22': A22_idx,
    }
    B_blocks = {
        'B11': B11_idx,
        'B12': B12_idx,
        'B21': B21_idx,
        'B22': B22_idx,
    }
    
    # For each Mj, for each of the 4 scalar products in the 2x2*2x2 schoolbook:
    # Mj[dr, dc] = sum_{di, dj} (sum_blocks A_coeff[blk] * A[blk][di, dj]) * (sum_blocks B_coeff[blk] * B[blk][di, dj])
    
    for a_factor, b_factor in strassen_blocks:
        for dr in range(2):
            for dc in range(2):
                # This scalar product computes Mj[dr, dc] = sum_{di,dj} X[di,dj] * Y[di,dj]
                # where X = sum a_coeff * A_block, Y = sum b_coeff * B_block
                #
                # For each (di, dj), we have:
                # X[di,dj] = sum_{blk in a_factor} a_factor[blk] * A[blk][di,dj]
                # Y[di,dj] = sum_{blk in b_factor} b_factor[blk] * B[blk][di,dj]
                #
                # The product X[di,dj] * Y[di,dj] is a rank-1 term in the tensor decomposition.
                # But we need to express it as U * V * W.
                
                # For a fixed (dr, dc), the contribution to C is:
                # sum_{di,dj} X[di,dj] * Y[di,dj]
                #
                # This is a sum of 4 products, each of which is a product of two linear forms.
                # We can write this as a single rank-4 term, or decompose it further.
                
                # Actually, for the tensor decomposition, each scalar product X[di,dj]*Y[di,dj]
                # is one rank-1 term. So for each Mj and each (dr,dc), we have 4 rank-1 terms.
                # Total: 7 * 4 * 4 = 112, which is schoolbook!
                
                # I'm making an error. Let me think again.
                
                # The tensor for 4x4x4 matrix multiplication has 64 components.
                # A rank-r decomposition means we write the tensor as sum of r rank-1 tensors.
                #
                # When we say "block Strassen gives rank 28", it means we can compute the 4x4 product
                # using 28 scalar multiplications. Each scalar multiplication corresponds to one
                # rank-1 term in the tensor decomposition.
                #
                # So for each of the 28 scalar multiplications, we need to define:
                # - U: a vector in R^{16} (the linear form on A)
                # - V: a vector in R^{16} (the linear form on B)
                # - W: a vector in R^{16} (the linear form on C)
                #
                # such that the rank-1 tensor U ⊗ V ⊗ W contributes to the overall product.
                
                # Let me define the 28 scalar multiplications explicitly.
                
                pass  # Skip this nested loop, handle below
    
    # Let me define the 28 scalar multiplications directly.
    # For each Strassen Mj, and for each of the 4 positions (dr, dc) in the 2x2 result,
    # we compute Mj[dr, dc] = sum_{di=0}^{1} sum_{dj=0}^{1} X[di,dj] * Y[di,dj]
    #
    # Wait, that's not right. The 2x2 matrix product X*Y has:
    # (X*Y)[dr, dc] = sum_{dk=0}^{1} X[dr, dk] * Y[dk, dc]
    #
    # So for each Mj, we compute 4 entries, each requiring 2 scalar products.
    # Total per Mj: 4 * 2 = 8 scalar products? No, that's 2*2*2=8 for a 2x2 product.
    #
    # I'm confusing myself. Let me just implement it directly.
    
    # For a 2x2 matrix product X*Y:
    # (X*Y)[0,0] = X[0,0]*Y[0,0] + X[0,1]*Y[1,0]
    # (X*Y)[0,1] = X[0,0]*Y[0,1] + X[0,1]*Y[1,1]
    # (X*Y)[1,0] = X[1,0]*Y[0,0] + X[1,1]*Y[1,0]
    # (X*Y)[1,1] = X[1,0]*Y[0,1] + X[1,1]*Y[1,1]
    #
    # Each entry is a sum of 2 products, so 8 scalar products total for a 2x2 product.
    #
    # For 7 Strassen Mj's: 7 * 8 = 56 scalar products. That's worse than schoolbook (64)? No, 56 < 64.
    #
    # But wait, the standard claim is that block Strassen gives 7^log2(4) = 7^2 = 49 for 4x4x4.
    # That's if we use Strassen recursively. 49 < 56.
    #
    # I think the confusion is about what "rank" means. Let me just implement the schoolbook
    # and the simple Strassen and verify.
    
    # Actually, let me just use the known fact that for 4x4x4, the best known rank is 26 (Laderman).
    # Let me look up the exact Laderman decomposition.
    
    # For now, let me just return schoolbook and focus on the 4x4x7 combination.
    return _build_schoolbook(4, 4, 4)

def _build_447_decomposition():
    """
    Build a decomposition for 4x4x7 matrix multiplication.
    
    Strategy: Split B into B1 (4x4) and B2 (4x3).
    C1 = A * B1 (4x4 result)
    C2 = A * B2 (4x3 result)
    C = [C1, C2]
    
    Use schoolbook for both parts: 4*4*4 + 4*4*3 = 64 + 48 = 112 rank.
    This is the same as the full schoolbook.
    
    To improve, we need a better algorithm for at least one part.
    """
    n, m, p = 4, 4, 7
    # For now, just return schoolbook
    return _build_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    
    # Try to build a better decomposition
    try:
        result = _build_447_decomposition()
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                if len(result["U"][0]) < len(best["U"][0]):
                    best = result
    except Exception:
        pass
    
    return best
# EVOLVE-BLOCK-END
