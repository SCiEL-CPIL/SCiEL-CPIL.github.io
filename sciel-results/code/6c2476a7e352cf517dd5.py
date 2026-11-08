import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We decompose into:
    # 1. A_1 B_11 (4x4 * 4x4) -> Rank 49
    # 2. A_1 B_12 (4x4 * 4x1) -> Rank 16
    # 3. A_2 B_21 (4x1 * 1x4) -> Rank 16
    # 4. A_2 B_22 (4x1 * 1x1) -> Rank 4
    # Total Rank 85
    
    rank = 85
    U = [[0] * rank for _ in range(n * m)] # 20 x 85
    V = [[0] * rank for _ in range(m * p)] # 25 x 85
    W = [[0] * rank for _ in range(p * n)] # 20 x 85
    
    # Indices:
    # A[i, j] -> U[i*m + j]
    # B[j, k] -> V[j*p + k]
    # C[i, k] -> W[k*n + i]
    
    col = 0
    
    # Part 1: A_1 B_11 (4x4 * 4x4)
    # A_1 is A[0:4, 0:4], B_11 is B[0:4, 0:4]
    # Output C[0:4, 0:4]
    # Use Strassen for 4x4x4
    # We can call a helper to get the 49 terms for 4x4x4
    # For simplicity, let's use schoolbook for now to ensure correctness, 
    # but we know Strassen is 49. 
    # Actually, let's just use schoolbook for all parts to be safe, 
    # but that gives 100. 
    # We need to implement Strassen for 4x4x4.
    
    # Let's implement Strassen for 2x2x2 and recurse.
    # Or just hardcode the 49 terms? 
    # Let's use a recursive function to generate the U, V, W columns for Strassen.
    
    def strassen_decomp(n_size, U_sub, V_sub, W_sub, offset):
        # This is complex to do in-place.
        # Let's just use schoolbook for 4x4x4? Rank 64.
        # Then total rank 64 + 16 + 16 + 4 = 100. No improvement.
        # We MUST use Strassen for 4x4x4 to get 49.
        
        pass
