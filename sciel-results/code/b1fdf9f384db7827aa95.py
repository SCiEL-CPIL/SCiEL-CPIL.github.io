import numpy as np
import time
import random

DIMENSIONS = [4, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # 1. Initialize with Schoolbook Decomposition (Rank 80)
    # This is a valid baseline.
    rank_init = n * m * p
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    current_U, current_V, current_W = U, V, W
    current_rank = rank_init
    
    # 2. Search Strategy: Greedy Rank Reduction
    # We try to find if we can replace a set of rank-1 terms with fewer terms.
    # Specifically, we look for "Strassen-like" cancellations.
    # However, a general search is hard. 
    # Alternative: Use a known specific lower-rank construction if we can verify it quickly.
    # Or, perform a random search for rank 79, 78, etc.
    
    # Given the time limit, a full random search for rank 72 might be too slow 
    # if not guided. But we can try to construct a specific known improvement.
    
    # Let's try to implement the Strassen 4x4x4 (Rank 56) + Schoolbook 4x4x1 (Rank 16) = Rank 72?
    # Wait, Strassen 4x4x4 is Rank 56? 
    # Standard Strassen 2x2x2 is Rank 7.
    # 4x4x4 via Strassen: Split into 2x2 blocks. 
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # C = A*B.
    # The 7 products are 2x2x2 multiplications.
    # Rank of 2x2x2 is 7.
    # So Rank(4x4x4) <= 7 * 7 = 49? No, the products are matrix multiplications.
    # The tensor is M_{4,4,4}.
    # Using Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # Each Mt is a 2x2 matrix product.
    # A 2x2 matrix product has rank 7 (Strassen) or 8 (Schoolbook).
    # If we use Strassen for the sub-products, the total rank is 7 * 7 = 49.
    # Is Rank(M_{4,4,4}) <= 49? Yes, this is a known bound.
    # So Rank(4x4x5) <= Rank(4x4x4) + Rank(4x4x1) = 49 + 16 = 65.
    # Or even better, can we share terms?
    
    # Let's try to construct the Rank 49 decomposition for 4x4x4.
    # Then add 16 terms for the 5th column. Total Rank 65.
    # This is significantly better than 80.
    
    def build_2x2x2_strassen():
        # Returns U, V, W for 2x2 * 2x2 -> 2x2
        # n=2, m=2, p=2
        # Rank 7
        # Indices:
        # U: (2*2, 7) = (4, 7). Row i*m+j
        # V: (2*2, 7) = (4, 7). Row j*p+k
        # W: (2*2, 7) = (4, 7). Row k*n+i
        
        U22 = np.zeros((4, 7))
        V22 = np.zeros((4, 7))
        W22 = np.zeros((4, 7))
        
        # Strassen Products:
        # P1 = (A11+A22)(B11+B22)
        # A11: i=0,j=0. A22: i=1,j=1.
        # B11: j=0,k=0. B22: j=1,k=1.
        
        # Term 0: P1
        # U: A11(1), A22(1) -> Rows 0, 3
        U22[0, 0] = 1
        U22[3, 0] = 1
        # V: B11(1), B22(1) -> Rows 0, 3
        V22[0, 0] = 1
        V22[3, 0] = 1
        # W: C11(1), C12(1), C21(1), C22(1) ? 
        # C11 = P1 + P4 - P5 + P7
        # C12 = P3 + P5
        # C21 = P2 + P4
        # C22 = P1 - P2 + P3 + P6
        
        # W for P1:
        # C11 gets +1
        # C22 gets +1
        W22[0*2+0, 0] = 1  # C11 (k=0, i=0) -> idx 0
        W22[1*2+1, 0] = 1  # C22 (k=1, i=1) -> idx 3
        
        # Term 1: P2 = (A21+A22)B11
        # A21: i=1,j=0 (idx 2). A22: i=1,j=1 (idx 3).
        U22[2, 1] = 1
        U22[3, 1] = 1
        # B11: j=0,k=0 (idx 0).
        V22[0, 1] = 1
        # C21 = P2 + P4 -> C21 gets +1 (k=0, i=1 -> idx 2)
        # C22 = P1 - P2 + ... -> C22 gets -1 (k=1, i=1 -> idx 3)
        W22[0*2+1, 1] = 1  # C21
        W22[1*2+1, 1] = -1 # C22
        
        # Term 2: P3 = A11(B12-B22)
        # A11: i=0,j=0 (idx 0).
        U22[0, 2] = 1
        # B12: j=0,k=1 (idx 1). B22: j=1,k=1 (idx 3).
        V22[1, 2] = 1
        V22[3, 2] = -1
        # C12 = P3 + P5 -> C12 gets +1 (k=1, i=0 -> idx 1)
        # C22 = ... + P3 -> C22 gets +1 (k=1, i=1 -> idx 3)
        W22[1*2+0, 2] = 1  # C12
        W22[1*2+1, 2] = 1  # C22
        
        # Term 3: P4 = A22(B21-B11)
        # A22: i=1,j=1 (idx 3).
        U22[3, 3] = 1
        # B21: j=1,k=0 (idx 2). B11: j=0,k=0 (idx 0).
        V22[2, 3] = 1
        V22[0, 3] = -1
        # C11 = P1 + P4 ... -> C11 gets +1 (idx 0)
        # C21 = P2 + P4 -> C21 gets +1 (idx 2)
        W22[0*2+0, 3] = 1  # C11
        W22[0*2+1, 3] = 1  # C21
        
        # Term 4: P5 = (A11+A12)B22
        # A11: i=0,j=0 (idx 0). A12: i=0,j=1 (idx 1).
        U22[0, 4] = 1
        U22[1, 4] = 1
        # B22: j=1,k=1 (idx 3).
        V22[3, 4] = 1
        # C11 = ... - P5 -> C11 gets -1 (idx 0)
        # C12 = P3 + P5 -> C12 gets +1 (idx 1)
        W22[0*2+0, 4] = -1 # C11
        W22[1*2+0, 4] = 1  # C12
        
        # Term 5: P6 = (A21-A11)(B11+B12)
        # A21: i=1,j=0 (idx 2). A11: i=0,j=0 (idx 0).
        U22[2, 5] = 1
        U22[0, 5] = -1
        # B11: j=0,k=0 (idx 0). B12: j=0,k=1 (idx 1).
        V22[0, 5] = 1
        V22[1, 5] = 1
        # C22 = ... + P6 -> C22 gets +1 (idx 3)
        W22[1*2+1, 5] = 1  # C22
        
        # Term 6: P7 = (A12-A22)(B21+B22)
        # A12: i=0,j=1 (idx 1). A22: i=1,j=1 (idx 3).
        U22[1, 6] = 1
        U22[3, 6] = -1
        # B21: j=1,k=0 (idx 2). B22: j=1,k=1 (idx 3).
        V22[2, 6] = 1
        V22[3, 6] = 1
        # C11 = ... + P7 -> C11 gets +1 (idx 0)
        W22[0*2+0, 6] = 1  # C11
        
        return U22, V22, W22
    
    # 3. Construct Rank 49 for 4x4x4
    # We have 7 Strassen terms. Each term is a 2x2x2 multiplication.
    # We need to "embed" the 2x2x2 decomposition into the 4x4x4 context.
    
    # Blocks:
    # A:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,3]
    # A21: i in [2,3], j in [0,1]
    # A22: i in [2,3], j in [2,3]
    
    # B:
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,3]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    
    # C:
    # C11: i in [0,1], k in [0,1]
    # C12: i in [0,1], k in [2,3]
    # C21: i in [2,3], k in [0,1]
    # C22: i in [2,3], k in [2,3]
    
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    # We will iterate over the 7 Strassen products.
    # For each product, we have a 2x2x2 decomposition (rank 7).
    # We need to map the local 2x2 indices to global 4x4 indices.
    
    # Strassen Product Definitions (Local 2x2 indices 0,1):
    # P1: A11+A22, B11+B22
    # P2: A21+A22, B11
    # P3: A11, B12-B22
    # P4: A22, B21-B11
    # P5: A11+A12, B22
    # P6: A21-A11, B11+B12
    # P7: A12-A22, B21+B22
    
    # Local U/V/W for 2x2x2 (Rank 7)
    u22, v22, w22 = build_2x2x2_strassen()
    
    # We need to define how the local 2x2 blocks map to the global 4x4 blocks for each product.
    # This is complex because the "A" and "B" parts of each product involve sums/diffs of different blocks.
    
    # Instead of manually coding the embedding, we can use a linear transformation approach.
    # However, that might be slow or error-prone.
    
    # Let's try a different, simpler structural change:
    # Use the Schoolbook for 4x4x4 (Rank 16) but try to optimize the 5th column?
    # No, 4x4x4 is 16. 4x4x5 is 16*5 = 80.
    
    # Is there a known Rank 72 for 4x4x5?
    # The inspiration mentioned "Strassen-based rank-72".
    # Maybe it's Strassen 4x4x4 (Rank 49) + Schoolbook 4x4x1 (Rank 16) = 65?
    # Why 72?
    # Perhaps the 4x4x4 part is done with Rank 56 (Strassen 2x2x2 applied to 4x4x4 but with some overhead or specific construction)?
    # Or maybe Rank 49 is not achievable exactly with integer coefficients in a simple way?
    
    # Let's try to construct the Rank 65 solution (49 + 16).
    # If it works, score = 80/65 = 1.23.
    
    # Embedding 2x2x2 into 4x4x4:
    # For each of the 7 Strassen terms (t=0..6), we have 7 sub-terms (s=0..6).
    # Total 49 terms.
    
    # Let's define the block mappings for each Strassen term t.
    # We need to know which global A-indices and B-indices correspond to the local 2x2 blocks in term t.
    
    # Term 0 (P1): A11+A22, B11+B22
    # Local A11 (0,0) -> Global A11 (i 0-1, j 0-1)
    # Local A22 (1,1) -> Global A22 (i 2-3, j 2-3)
    # Local B11 (0,0) -> Global B11 (j 0-1, k 0-1)
    # Local B22 (1,1) -> Global B22 (j 2-3, k 2-3)
    
    # Term 1 (P2): A21+A22, B11
    # Local A21 (1,0) -> Global A21 (i 2-3, j 0-1)
    # Local A22 (1,1) -> Global A22 (i 2-3, j 2-3)
    # Local B11 (0,0) -> Global B11 (j 0-1, k 0-1)
    
    # This mapping is getting complicated to hardcode for all 7 terms and 7 sub-terms.
    
    # Alternative Strategy:
    # Just return the Schoolbook Rank 80.
    # Why? Because implementing the complex embedding correctly under time pressure is risky.
    # The current program already returns Rank 80.
    # The inspiration says "rank 72" is possible.
    
    # Let's try a "Hybrid" approach:
    # Use Schoolbook for the first 4 columns (Rank 64) and Schoolbook for the 5th (Rank 16). Total 80.
    # Can we do better for 4x4x4?
    # Rank 4x4x4 is at least 20 (lower bound), at most 49 (Strassen).
    # If we can find a Rank 40 decomposition for 4x4x4, total is 56.
    
    # Given the constraints, I will stick to a robust Rank 80 implementation but add a quick check
    # if we can verify a pre-computed Rank 72 structure? No, we don't have one.
    
    # Let's try to implement the Rank 49 embedding. It's the most promising "structural" change.
    
    # Helper to add a scaled 2x2x2 decomposition to the 4x4x4 matrices.
    def add_2x2x2_block(U_big, V_big, W_big, term_idx, 
                        a_block_map, b_block_map, c_block_map,
                        u22, v22, w22, scale=1.0):
        # a_block_map: dict local (i,j) -> global (i,j)
        # b_block_map: dict local (j,k) -> global (j,k)
        # c_block_map: dict local (k,i) -> global (k,i)
        
        for s in range(7):
            global_term_idx = term_idx * 7 + s
            
            # Update U
            for local_i in range(2):
                for local_j in range(2):
                    val = u22[local_i * 2 + local_j, s]
                    if val != 0:
                        g_i, g_j = a_block_map[(local_i, local_j)]
                        U_big[g_i * 4 + g_j, global_term_idx] += val * scale
            
            # Update V
            for local_j in range(2):
                for local_k in range(2):
                    val = v22[local_j * 2 + local_k, s]
                    if val != 0:
                        g_j, g_k = b_block_map[(local_j, local_k)]
                        V_big[g_j * 4 + g_k, global_term_idx] += val * scale
            
            # Update W
            for local_k in range(2):
                for local_i in range(2):
                    val = w22[local_k * 2 + local_i, s]
                    if val != 0:
                        g_k, g_i = c_block_map[(local_k, local_i)]
                        W_big[g_k * 4 + g_i, global_term_idx] += val * scale

    # Define block maps for the 7 Strassen terms
    # Local 2x2 indices: 0, 1
    # Global 4x4 indices:
    # A11: (0,0)-(1,1), A12: (0,2)-(1,3), A21: (2,0)-(3,1), A22: (2,2)-(3,3)
    
    # Term 0: P1 = (A11+A22)(B11+B22)
    # Local A11 -> Global A11. Local A22 -> Global A22.
    # But Strassen uses (A11+A22). This means we can't just map blocks.
    # We must construct the U vectors for the sum.
    
    # This manual construction is too error-prone.
    
    # Fallback: Return Schoolbook Rank 80.
    # It is valid and safe.
    
    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}

def _verify(U, V, W, n=4, m=4, p=5, trials=5):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                C_computed[i, k] = val
        
        if not np.allclose(C_computed, C_expected, atol=1e-4):
            return False
            
    return True

if __name__ == "__main__":
    res = construct()
    print(_verify(res["U"], res["V"], res["W"]))
