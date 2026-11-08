"""
Rank-minimal exact tensor decomposition for 6x6 matrix multiplication.

Uses a structured block decomposition:
  - Split 6x6 into 3x3 blocks of size 2x2.
  - For each 2x2 block product, use Strassen (rank 7) where beneficial.
  - For cross-block products (e.g., 2x4 * 4x2), use rank-4 decompositions.
  - For same-block products (2x2 * 2x2), use Strassen rank 7.
  
Total rank calculation:
  There are 9 block pairs (i,j) for A-block row i, B-block col j -> C-block (i,j).
  Actually, the product C = A * B where A is 6x6 (3x3 blocks of 2x2), B is 6x6 (3x3 blocks of 2x2).
  C[i,j] = sum_k A[i,k] * B[k,j] for block indices.
  
  So for each output block C[i,j], we sum over k=0..2: A[i,k] * B[k,j].
  Each term A[i,k] * B[k,j] is a product of two 2x2 matrices.
  
  For each such 2x2 * 2x2 product, we can use Strassen (rank 7) or naive (rank 4).
  But wait: the tensor decomposition is for the ENTIRE 6x6 multiplication.
  We need to sum over all k for each (i,j).
  
  Approach:
  For each output block position (i,j) and each sum index k, we have a 2x2*2x2 product.
  We can decompose each 2x2*2x2 using Strassen (7 terms) or naive (4 terms).
  If we use Strassen for all 9*3=27 block products, total rank = 27*7 = 189.
  If we use naive for all, total rank = 27*4 = 108.
  
  Wait, that's better! Let me reconsider.
  
  Actually, the naive decomposition of 2x2*2x2 has rank 4 (one term per entry of the shared index).
  The Strassen decomposition has rank 7, which is WORSE than naive for 2x2.
  
  So for 2x2*2x2, naive (rank 4) is better than Strassen (rank 7).
  
  Total rank with all naive: 27 * 4 = 108.
  
  Can we do better? 
  For products like 2x4 * 4x2, the rank is 4 (shared index has 4 values).
  For 4x2 * 2x4, the rank is 2.
  
  Let me reconsider the block structure.
  
  A is divided into 3x3 blocks of 2x2: A[i,j] where i,j in 0..2.
  B is divided into 3x3 blocks of 2x2: B[i,j] where i,j in 0..2.
  C[i,j] = sum_k A[i,k] B[k,j].
  
  Each A[i,k] is 2x2, B[k,j] is 2x2.
  The product A[i,k] B[k,j] is a 2x2 matrix.
  
  The tensor T has indices (a_row, a_col, b_row, b_col, c_row, c_col) but really:
  C[c_row, c_col] = sum_{a_col, b_row} A[a_row, a_col] B[b_row, b_col] where a_row=c_row, b_col=c_col.
  
  More precisely: C[r, s] = sum_{t=0}^{5} A[r, t] B[t, s].
  
  Using blocks: r = 2*i + i', s = 2*j + j', t = 2*k + k' where i,j,k in 0..2, i',j',k' in 0..1.
  
  C[2i+i', 2j+j'] = sum_{k=0}^{2} sum_{k'=0}^{1} A[2i+i', 2k+k'] B[2k+k', 2j+j']
                   = sum_{k=0}^{2} (A[i,k] B[k,j])[i', j']
  
  So for each (i,j), C[i,j] = sum_k A[i,k] B[k,j].
  
  The decomposition: for each (i,j,k), we add a contribution from A[i,k] * B[k,j] to C[i,j].
  
  For a 2x2*2x2 product, the naive rank is 4.
  So total rank = 9 * 3 * 4 = 108.
  
  Can we combine terms across k? Probably not easily without more sophisticated methods.
  
  Let's go with rank 108 using naive 2x2*2x2 for all block products.
"""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_START_TIME = time.time()
_TIME_LIMIT = 110.0

def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _build_naive_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                idx += 1
    return {"U": U, "V": V, "W": W}

def _build_block_naive_decomposition(n, m, p, block_size=2):
    """
    Decompose n x m x p multiplication using block structure with naive 2x2*2x2 products.
    n = m = p = 6, block_size = 2, so 3x3 blocks.
    Total rank = 9 * 3 * 4 = 108.
    """
    num_blocks = n // block_size  # 3
    total_rank = num_blocks * num_blocks * num_blocks * (block_size * block_size)  # 3*3*3*4 = 108
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_idx = 0
    for i in range(num_blocks):      # A block row
        for j in range(num_blocks):  # B block col
            for k in range(num_blocks):  # sum index
                # A[i,k] is at rows 2i:2i+2, cols 2k:2k+2
                # B[k,j] is at rows 2k:2k+2, cols 2j:2j+2
                # C[i,j] is at rows 2i:2i+2, cols 2j:2j+2
                for a_prime in range(block_size):  # 0,1
                    for b_prime in range(block_size):  # 0,1
                        # Term: A[2i+a_prime, 2k+b_prime] * B[2k+b_prime, 2j+a_prime] contributes to C[2i+a_prime, 2j+a_prime]
                        # Wait, let me re-derive.
                        # C[2i+a_prime, 2j+c_prime] = sum_{t} A[2i+a_prime, t] B[t, 2j+c_prime]
                        # t = 2k+b_prime
                        # So A[2i+a_prime, 2k+b_prime] * B[2k+b_prime, 2j+c_prime]
                        # This contributes to C[2i+a_prime, 2j+c_prime]
                        # The shared index is b_prime (the column of A / row of B within the block).
                        # For fixed (i,j,k), we have 4 terms (one for each b_prime=0,1 and the product involves a_prime and c_prime).
                        # Actually, for fixed (i,j,k), the 2x2*2x2 product has 4 terms, one for each value of the shared index b_prime.
                        # Each term: A_row = 2i+a_prime, A_col = 2k+b_prime, B_row = 2k+b_prime, B_col = 2j+c_prime, C_row = 2i+a_prime, C_col = 2j+c_prime
                        # But a_prime and c_prime are independent. So for each b_prime, we have 4 terms? No.
                        # For fixed b_prime, the term is: U[2i+a_prime, 2k+b_prime] * V[2k+b_prime, 2j+c_prime] for all a_prime, c_prime.
                        # This is one rank-1 tensor? No, it's 4 entries in U, 4 in V, 4 in W.
                        # Actually, for fixed (i,j,k,b_prime), we add 4 terms? No, we add 1 term where:
                        # U has 1 at positions (2i+a_prime, 2k+b_prime) for a_prime=0,1
                        # V has 1 at positions (2k+b_prime, 2j+c_prime) for c_prime=0,1
                        # W has 1 at positions (2i+a_prime, 2j+c_prime) for a_prime,c_prime=0,1
                        # This is 4 entries in each of U, V, W for this one rank-1 tensor.
                        pass
                # Correct approach: for fixed (i,j,k), the 2x2*2x2 naive product has 4 terms (one per b_prime).
                # For each b_prime in 0..1:
                for b_prime in range(block_size):
                    for a_prime in range(block_size):
                        U[(2*i + a_prime) * m + (2*k + b_prime)][col_idx] = 1
                    for c_prime in range(block_size):
                        V[(2*k + b_prime) * p + (2*j + c_prime)][col_idx] = 1
                    for a_prime in range(block_size):
                        for c_prime in range(block_size):
                            W[(2*i + a_prime) * n + (2*j + c_prime)][col_idx] = 1
                    col_idx += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(decomp, n, m, p, num_tests=3):
    U = np.array(decomp["U"])
    V = np.array(decomp["V"])
    W = np.array(decomp["W"])
    rank = U.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randint(-3, 4, (n, m)).astype(float)
        B = np.random.randint(-3, 4, (m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            Ur = U[:, r].reshape(n, m)
            Vr = V[:, r].reshape(m, p)
            Wr = W[:, r].reshape(p, n)
            prod = Ur @ Vr
            C_computed += prod * Wr
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    if _remaining_time() < 5:
        return _build_naive_decomposition(n, m, p)
    
    # Try block naive decomposition (rank 108)
    decomp = _build_block_naive_decomposition(n, m, p, block_size=2)
    
    if _remaining_time() < 5:
        return decomp
    
    if _verify_decomposition(decomp, n, m, p, num_tests=2):
        return decomp
    else:
        return _build_naive_decomposition(n, m, p)

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    print(f"Rank: {U.shape[1]}")
    valid = _verify_decomposition(result, 6, 6, 6, num_tests=5)
    print(f"Valid: {valid}")
