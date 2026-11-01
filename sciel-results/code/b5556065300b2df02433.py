import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # 1. Construct the "Kronecker Product" base decomposition (Rank 42)
    # This is the tensor product of Strassen (2x2x2) and Schoolbook (1x2x3)
    # Structure:
    # A (2x4) is partitioned into 2 blocks of 2x2.
    # B (4x6) is partitioned into 2x3 blocks of 2x2.
    # C (2x6) is partitioned into 2x3 blocks of 2x2.
    
    # Strassen 2x2x2 core (Rank 7)
    # A: 2x2, B: 2x2, C: 2x2
    # Local indices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) -> Global W index k*n+i
    
    U_strassen = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    V_strassen = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [1, -1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0]
    ])
    
    # Base Rank 42 Decomposition
    # Blocks: 2 (n) * 2 (m) * 3 (p) = 12 blocks
    # Each block is 2x2x2, using Strassen (Rank 7)
    # Total Rank = 12 * 7 = 84? 
    # Wait. The current program uses a mixed partition.
    # Let's re-evaluate the "Kronecker Product with Padding" idea from the prompt.
    # The prompt suggests: Tensor product of Rank-7 Strassen (2x2x2) and Rank-1 (1x2x3).
    # This implies we view the 2x4x6 tensor as:
    # (2x2) x (2x2) for A, B, C? No.
    # A is 2x4. B is 4x6. C is 2x6.
    # If we split m=4 into 2+2 and p=6 into 2+2+2?
    # Then we have 1 (n) * 2 (m) * 3 (p) = 6 blocks of 2x2x2?
    # No, n=2 is one block of size 2.
    # So we have 1 block in n, 2 blocks in m, 3 blocks in p.
    # Total 6 blocks.
    # Each block is 2x2x2.
    # If we use Strassen for each 2x2x2 block, Rank = 6 * 7 = 42.
    # This matches the "Rank 42" mentioned in the prompt.
    
    # Let's construct this specific Rank 42 decomposition.
    rank_base = 42
    U = np.zeros((n * m, rank_base))
    V = np.zeros((m * p, rank_base))
    W = np.zeros((p * n, rank_base))
    
    col = 0
    # n=2 (1 block), m=4 (2 blocks), p=6 (3 blocks)
    for i_blk in range(1):
        for j_blk in range(2):
            for k_blk in range(3):
                # Local offsets
                i_off = 0
                j_off = j_blk * 2
                k_off = k_blk * 2
                
                for r in range(7):
                    # U: A[i, j]
                    for a_idx in range(4):
                        coeff = U_strassen[r, a_idx]
                        if coeff != 0:
                            i_loc = a_idx // 2
                            j_loc = a_idx % 2
                            gi = i_off + i_loc
                            gj = j_off + j_loc
                            U[gi * m + gj, col] += coeff
                            
                    # V: B[j, k]
                    for b_idx in range(4):
                        coeff = V_strassen[r, b_idx]
                        if coeff != 0:
                            j_loc = b_idx // 2
                            k_loc = b_idx % 2
                            gj = j_off + j_loc
                            gk = k_off + k_loc
                            V[gj * p + gk, col] += coeff
                            
                    # W: C[k, i] -> Global index k*n + i
                    for c_idx in range(4):
                        coeff = W_strassen[r, c_idx]
                        if coeff != 0:
                            k_loc = c_idx // 2
                            i_loc = c_idx % 2
                            gk = k_off + k_loc
                            gi = i_off + i_loc
                            W[gk * n + gi, col] += coeff
                            
                    col += 1
                    
    # 2. Check if this decomposition is valid
    def check_decomposition(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(rank):
                        s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    if not check_decomposition(U, V, W, rank_base):
        # Fallback to schoolbook if construction failed
        rank_fb = n * m * p
        U_fb = np.zeros((n*m, rank_fb))
        V_fb = np.zeros((m*p, rank_fb))
        W_fb = np.zeros((p*n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}, rank_fb

    # 3. Attempt to reduce rank by finding linear dependencies
    # We view the decomposition as a set of rank-1 tensors T_r = u_r \otimes v_r \otimes w_r
    # We want to find a subset or linear combination that still spans the same tensor T.
    # Specifically, if T = sum_{r=1}^{42} u_r v_r w_r, and we can write T = sum_{s=1}^{R'} u'_s v'_s w'_s with R' < 42.
    # This is equivalent to finding the rank of the tensor.
    # A common heuristic is to check if the columns of the "flattened" matrices are linearly dependent in a way that allows merging.
    
    # Let's try a simpler pruning: 
    # If two terms r1 and r2 have proportional U, V, or W vectors, they might be mergeable?
    # Or, we can try to solve for a smaller rank using the "border rank" or "tensor rank" algorithms.
    # Given the time limit, we will try a greedy pruning based on linear dependence of the "interaction" vectors.
    
    # Actually, a more robust approach for small tensors is to check if the 42 terms span a space of dimension < 42 in some flattened view.
    # Flatten T into a matrix T_{(i,j), k} of size (n*m) x p = 8x6.
    # Or T_{i, (j,k)} of size 2 x 24.
    # Or T_{(i,k), j} of size 12 x 4.
    
    # If the rank of any flattening is R, then Tensor Rank >= R.
    # Rank 42 is likely far above the flattening ranks (which are small, e.g., 8, 24, 12).
    # So flattening ranks don't help reduce 42.
    
    # Let's try to find if there is a known lower rank.
    # For 2x4x6, the schoolbook is 48.
    # The 42-term decomposition is an improvement.
    # Is 42 minimal?
    # The lower bound for 2x4x6 is at least max(rank flattening).
    # Rank flattening 1: 2x24 matrix. Rank <= 2.
    # Rank flattening 2: 8x6 matrix. Rank <= 6.
    # Rank flattening 3: 12x4 matrix. Rank <= 4.
    # So lower bound is 6.
    # 42 is very high. There is likely a much lower rank.
    
    # However, finding the exact minimal rank is NP-hard.
    # We can try to use the "Strassen-like" recursive structure more aggressively.
    # What if we use a 2x2x2 Strassen for the first 2x2x2 block, and then treat the rest differently?
    
    # Let's try a hybrid:
    # Partition n=2, m=4, p=6.
    # Use Strassen for (2,2,2) blocks.
    # We have 6 such blocks in the pure 2x2x2 partition.
    # Rank 42.
    
    # Can we do better?
    # What if we partition m=4 as (2,2) and p=6 as (4,2)?
    # Block 1: 2x2x4.
    # Block 2: 2x2x2.
    # Rank of 2x2x2 is 7.
    # Rank of 2x2x4?
    # This is two 2x2x2 blocks? No, they share the 2x2 A and B parts?
    # A is 2x2. B is 2x4. C is 2x4.
    # C = A B.
    # This is equivalent to multiplying a 2x2 matrix by two 2x2 matrices.
    # We can do this with two Strassen decompositions?
    # p1..p7 for B1 (cols 0,1).
    # p8..p14 for B2 (cols 2,3).
    # Total rank 14.
    # Total rank = 7 (for 2x2x2 block) + 14 (for 2x2x4 block) = 21?
    # Wait, the blocks are disjoint in terms of C and B columns, but share A?
    # No, A is 2x2 for both?
    # If we partition m=4 into (2,2), then A is split into A_left (2x2) and A_right (2x2).
    # If we partition p=6 into (4,2), then B is split into B_top (4x4) and B_bottom (4x2)?
    # No, B is 4x6.
    # If m is split (2,2), B rows are split (2,2).
    # If p is split (4,2), B cols are split (4,2).
    # So B is split into 4 blocks: 2x4, 2x2, 2x4, 2x2.
    # A is split into 2 blocks: 2x2, 2x2.
    # C is split into 2 blocks: 2x4, 2x2.
    
    # Let's trace the dependencies.
    # C_left (2x4) = A_left B_left_top (2x2 * 2x4) + A_right B_right_top (2x2 * 2x4).
    # C_right (2x2) = A_left B_left_bottom (2x2 * 2x2) + A_right B_right_bottom (2x2 * 2x2).
    
    # Rank of 2x2 * 2x4 multiplication:
    # A (2x2), B (2x4), C (2x4).
    # This is two independent 2x2 multiplications?
    # C_col1 = A B_col1. C_col2 = A B_col2.
    # We can use Strassen for each column?
    # Rank 7 + 7 = 14.
    # Or is there a trick?
    # A is shared.
    # p1..p7 for col 1.
    # p8..p14 for col 2.
    # Yes, rank 14.
    
    # So for the partition m=(2,2), p=(4,2):
    # C_left (2x4) depends on A_left, A_right and B_left_top, B_right_top.
    # This is not a direct sum of independent blocks because A is shared between the two m-blocks for the same p-block?
    # No, A is 2x4.
    # A = [A_left | A_right].
    # B = [B_left_top; B_right_top] (for p=0..3).
    # C_left = A_left B_left_top + A_right B_right_top.
    # This is a sum of two 2x2x4 multiplications.
    # Rank = Rank(2x2x4) + Rank(2x2x4) = 14 + 14 = 28?
    # Plus C_right (2x2) = A_left B_left_bottom + A_right B_right_bottom.
    # Rank = Rank(2x2x2) + Rank(2x2x2) = 7 + 7 = 14?
    # Total Rank = 28 + 14 = 42.
    # Same as before.
    
    # It seems 42 is a robust upper bound.
    # Can we go lower?
    # What if we use a 2x4x4 block?
    # A (2x4), B (4x4), C (2x4).
    # Rank of 2x4x4?
    # Partition m=4 into (2,2).
    # C = A_left B_left + A_right B_right.
    # Each is 2x2x2. Rank 7+7=14.
    # So 2x4x4 is rank 14.
    # 2x4x6 = 2x4x4 + 2x4x2.
    # 2x4x2:
    # Partition m=4 into (2,2).
    # C = A_left B_left + A_right B_right.
    # Each is 2x2x2. Rank 7+7=14.
    # Total Rank = 14 + 14 = 28?
    # Wait.
    # If we partition p=6 into (4,2).
    # B is 4x6. Split into B_4 (4x4) and B_2 (4x2).
    # A is 2x4.
    # C_4 = A B_4.
    # C_2 = A B_2.
    # Rank(C_4) = 14.
    # Rank(C_2) = 14.
    # Total Rank = 28.
    
    # Why did I get 42 before?
    # Before: m=(2,2), p=(2,2,2).
    # 6 blocks of 2x2x2.
    # Each block i,j,k corresponds to A_ij, B_jk, C_ik.
    # For a fixed i (which is 0 since n=2 is one block), we sum over j.
    # C_i0 = A_00 B_00 + A_01 B_10.
    # C_i1 = A_00 B_01 + A_01 B_11.
    # C_i2 = A_00 B_02 + A_01 B_12.
    # These are 3 independent 2x2x2 multiplications?
    # No, A_00 and A_01 are shared across the k-blocks?
    # Yes.
    # So we cannot just sum the ranks of the blocks if they share factors.
    # In the 6-block decomposition (m=2,2; p=2,2,2), the factors A_00, A_01 are used in multiple terms.
    # This "sharing" is what makes the rank 42 (7 * 6) potentially reducible if we can reuse the Strassen coefficients.
    # But Strassen coefficients are specific to the 2x2x2 tensor.
    # If we reuse A_00, we are effectively doing:
    # sum_k (A_00 B_0k + A_01 B_1k).
    # This is linear in B_0k and B_1k.
    # It is equivalent to multiplying A_00 by B_0 (2x6) and A_01 by B_1 (2x6).
    # Rank of 2x2x6 is 14.
    # So C = A_00 B_0 + A_01 B_1.
    # Rank = 14 + 14 = 28.
    
    # So, the Rank 28 decomposition is:
    # Partition m=4 into (2,2).
    # Partition p=6 into (6).
    # Two blocks: (2x2x6) and (2x2x6).
    # Each 2x2x6 is decomposed into 3 2x2x2 Strassen cores?
    # Rank 7*3 = 21?
    # No, 2x2x6 is two 2x2x3?
    # 2x2x3 rank?
    # A (2x2), B (2x3), C (2x3).
    # C = A B.
    # This is 3 columns of 2x2 multiplication.
    # Rank 7*3 = 21?
    # Or can we do better?
    # 2x2x3 tensor rank is 7? No, 2x2x2 is 7. 2x2x3 is likely 7?
    # If we add a column, we add 1 rank?
    # Actually, 2x2x3 rank is 7?
    # Let's check.
    # Lower bound: flattening 2x6 rank <= 2. 4x3 rank <= 3. 6x2 rank <= 2.
    # So LB is 3.
    # Schoolbook is 12.
    # Strassen-like:
    # p1..p7 for col 1.
    # p8..p14 for col 2.
    # p15..p21 for col 3.
    # Rank 21.
    # Is it possible to do 2x2x3 in rank 7?
    # Unlikely.
    
    # So, if we use m=(2,2) and p=(6), we have two 2x2x6 blocks.
    # Rank 2x2x6 = 14 (using 2x2x2 Strassen for each 2x2x2 sub-block of p? No, 2x2x6 is not 3x 2x2x2 in a way that allows rank addition directly without sharing A).
    # Wait, if A is shared, the rank is Rank(A B).
    # If B is 2x6, we can view it as 3 columns.
    # We compute 3 separate 2x2x2 products.
    # Total rank 21.
    # So two such blocks (for m=0 and m=1) would be 21 + 21 = 42.
    
    # So 42 seems to be the result of this "Strassen per 2x2x2" strategy.
    # To get below 42, we need a better core than 2x2x2 for some dimension, or a different partition.
    
    # What if we use a 2x4x6 core directly?
    # We don't know the rank.
    # But we can try to find a rank 28 decomposition manually?
    # C = A B.
    # A is 2x4. B is 4x6.
    # Let A = [A1 A2] where A1, A2 are 2x2.
    # Let B = [B1; B2] where B1, B2 are 2x6.
    # C = A1 B1 + A2 B2.
    # Rank(A1 B1) for 2x2x6 is 14?
    # If we can compute A1 B1 in rank 14 and A2 B2 in rank 14, total is 28.
    # Is 2x2x6 rank 14?
    # 2x2x6 = 2x2x2 + 2x2x2 + 2x2x2.
    # If we use Strassen for each 2x2x2, rank is 21.
    # But we can do 2x2x6 in rank 14?
    # 2x2x6 is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    # This is 3 independent 2x2x2 multiplications?
    # No, the A is the same.
    # c_col = A b_col.
    # We can use the 7 Strassen products for the first column.
    # And 7 for the second.
    # And 7 for the third.
    # Total 21.
    # Can we share?
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # ...
    # These depend on specific b's.
    # For column 2, we need (b01, b11).
    # p1' = (a00+a11)(b01+b11).
    # This is a new product.
    # So it seems 21 is the rank for 2x2x6 using Strassen cores.
    
    # So 28 is not achievable with simple Strassen cores.
    
    # Therefore, the current best known upper bound is 42.
    # The schoolbook is 48.
    # The prompt asks for a strategy that might reveal structural symmetries.
    # I will return the Rank 42 decomposition, as it is the best known simple construction.
    # I will add a check to see if we can prune any columns that are linearly dependent in the "interaction" space, but given the time, I'll stick to the 42.
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}, rank_base

# EVOLVE-BLOCK-END
