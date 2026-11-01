import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Dimensions
    n, m, p = 2, 4, 6
    nm = n * m  # 8
    mp = m * p  # 24
    pn = p * n  # 12
    
    # Rank 42 solution via Strassen blocks
    # We partition A (2x4) into 2 rows of 1x4
    # We partition B (4x6) into 3 columns of 4x2
    # For each pair (A_row, B_col), we compute the 1x2 product using 7 Strassen multiplications
    # Total 2 * 3 * 7 = 42 multiplications
    
    rank = 42
    U = np.zeros((nm, rank), dtype=int)
    V = np.zeros((mp, rank), dtype=int)
    W = np.zeros((pn, rank), dtype=int)
    
    col = 0
    
    # Helper to add a Strassen block for 1x4 * 4x2 -> 1x2
    # A is 1x4: [a0, a1, a2, a3]
    # B is 4x2: [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
    # C is 1x2: [c0, c1]
    # Indices in U: a0..a3 correspond to A indices
    # Indices in V: b00..b31 correspond to B indices
    # Indices in W: c0, c1 correspond to C indices
    
    def add_strassen_1x2(a_idx, b_idx, c_idx, offset):
        # a_idx: list of 4 indices for A
        # b_idx: list of 16 indices for B (flattened 4x2)
        # c_idx: list of 2 indices for C
        # offset: current column index in U, V, W
        
        a0, a1, a2, a3 = a_idx
        b00, b01, b10, b11, b20, b21, b30, b31 = b_idx
        c0, c1 = c_idx
        
        # M1 = (a0+a1) * (b00+b10)  -> contributes to c0 (b00 part) and c1? No, let's map carefully.
        # Standard Strassen for 2x2:
        # M1 = (A11 + A21)(B12 + B22)  -- wait, standard is M1 = (A11+A21)(B12+B22)?
        # Let's use the standard 2x2 Strassen:
        # C11 = M1 - M2 + M4 - M5
        # C12 = M2 + M3
        # C21 = M4 + M5
        # C22 = M1 - M3 + M4 - M6
        # Where:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Here A is 1x4, B is 4x2.
        # We can view A as [A1, A2] where A1=[a0,a1], A2=[a2,a3]? No, Strassen requires square blocks.
        # But we can embed 1x4 * 4x2 into 2x2 * 2x2 by padding?
        # A = [[a0, a1, a2, a3]]
        # B = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
        # C = [c0, c1] = A * B
        # c0 = a0*b00 + a1*b10 + a2*b20 + a3*b30
        # c1 = a0*b01 + a1*b11 + a2*b21 + a3*b31
        
        # We can split A into two 1x2 vectors: A1=[a0,a1], A2=[a2,a3]
        # And B into two 2x2 matrices? No, B is 4x2.
        # Let's split B into top 2x2 and bottom 2x2:
        # B_top = [[b00, b01], [b10, b11]]
        # B_bot = [[b20, b21], [b30, b31]]
        # c0 = [a0, a1] * B_top[:, 0] + [a2, a3] * B_bot[:, 0]
        # c1 = [a0, a1] * B_top[:, 1] + [a2, a3] * B_bot[:, 1]
        
        # This is two 1x2 * 2x2 products? No, it's two 1x2 * 2x1 dot products?
        # Actually, it's simpler to just use the 7-multiplication algorithm for 2x2 * 2x2
        # by viewing the problem as:
        # Let A' = [[a0, a1], [a2, a3]] (2x2)
        # Let B' = [[b00, b20], [b10, b30]]? No, indices don't match.
        
        # Let's stick to the 2x2 Strassen on the "logical" 2x2 blocks if we can reshape.
        # A is 1x4. B is 4x2.
        # We can decompose this into two 1x2 * 2x2 multiplications?
        # No, let's just use the 7 terms derived for 2x2 matrix multiplication,
        # applied to the "sub-blocks" formed by splitting the inner dimension 4 into 2+2.
        
        # Let A1 = [a0, a1], A2 = [a2, a3]
        # Let B1 = [[b00, b01], [b10, b11]] (2x2)
        # Let B2 = [[b20, b21], [b30, b31]] (2x2)
        
        # C_part1 = A1 * B1 (1x2)
        # C_part2 = A2 * B2 (1x2)
        # C = C_part1 + C_part2
        
        # To compute A1 * B1 (1x2 * 2x2), we can pad A1 to 2x2: [[a0, a1], [0, 0]]?
        # Then C_part1[0] = (A1_padded * B1)[0, 0]
        # C_part1[1] = (A1_padded * B1)[0, 1]
        
        # This seems complicated. Let's use a known 7-term decomposition for 1x2 * 2x2?
        # Actually, the rank of 1x2 * 2x2 is 4? No, it's 2x2 multiplication which is rank 7.
        # Wait, 1x2 * 2x2 is just two dot products. Rank is 2?
        # c0 = a0*b00 + a1*b10
        # c1 = a0*b01 + a1*b11
        # This is rank 2 (two independent rank-1 terms).
        # So A1*B1 takes 2 multiplications.
        # A2*B2 takes 2 multiplications.
        # Total 4 multiplications per block?
        # If we use 4 multiplications per block, total rank = 6 * 4 = 24.
        # Is rank 24 possible?
        # The lower bound for 2x4 * 4x6 is not 24.
        # The rank of M_{n,m,p} is at least max(nm, mp, pn). Here max(8, 24, 12) = 24.
        # So rank 24 is a candidate.
        
        # Let's try to construct a rank 24 solution.
        # For each of the 6 blocks (2 A-rows, 3 B-cols), we need to compute 1x4 * 4x2 -> 1x2.
        # This sub-problem M_{1,4,2} has rank?
        # M_{1,4,2}: T[i, j, k] = 1 if i=0, k=0/1, j=0..3.
        # T[0, j, 0] = 1 for j=0..3
        # T[0, j, 1] = 1 for j=0..3
        # This is two vectors of length 4.
        # Rank is 2?
        # sum_r U[0,r] V[j,r] W[k,r]
        # For k=0: sum_r U[0,r] V[j,r] W[0,r] = 1 for all j.
        # This implies V[j,r] W[0,r] must be constant across j?
        # If we set U[0,r]=1, W[0,r]=1, then V[j,r]=1 for all j,r?
        # Then sum_r 1*1*1 = r. We need sum = 1. So r=1?
        # If r=1, U=1, V=1, W=1.
        # Then for k=1: sum_r U[0,r] V[j,r] W[1,r] = 1*1*W[1,r].
        # We need this to be 1 for all j. So W[1,r]=1.
        # So 1 term?
        # U[0,0]=1, V[j,0]=1 for all j, W[0,0]=1, W[1,0]=1.
        # Check: T[0,j,0] = 1*1*1 = 1. T[0,j,1] = 1*1*1 = 1.
        # So M_{1,4,2} has rank 1?
        # Wait, the tensor is defined by T[a,b,c].
        # a is index of A (size 1). b is index of B (size 4). c is index of C (size 2).
        # T[0, j, k] = 1 if valid.
        # Yes, it seems rank 1 is possible for M_{1,4,2}.
        
        # If each block has rank 1, total rank is 6 * 1 = 6?
        # But the blocks are not independent in the global tensor?
        # The global tensor is the sum of these block tensors?
        # No, the global tensor T is sparse.
        # T[a, b, c] is non-zero only if a, b, c are compatible.
        # The block decomposition sums up the contributions.
        
        # Let's verify if rank 6 is possible.
        # If we use 6 terms, one for each (i, k) pair?
        # i in 0..1, k in 0..2 (since p=6, k=0..5? No, p=6, so k=0..5).
        # Wait, p=6. So there are 6 columns in B.
        # My previous "3 blocks of 2 columns" was for p=6.
        # So 2 A-rows, 3 B-blocks (each 2 cols wide).
        # Total 6 blocks.
        # If each block is M_{1,4,2} and has rank 1, total rank 6.
        
        # Let's check the rank of M_{1,4,2} again.
        # T[0, j, k] = 1 for j=0..3, k=0..1.
        # Decomposition: U[0, r] V[j, r] W[k, r].
        # We found U=1, V=1, W=1 works for 1 term.
        # So rank 1.
        
        # Therefore, rank 6 should be possible?
        # But the lower bound is max(nm, mp, pn) = max(8, 24, 12) = 24.
        # Contradiction!
        
        # Why?
        # The block tensors are not simply added.
        # The global tensor T has entries T[a, b, c].
        # a = i*m + j. Here m=4. So a = 4i + j.
        # b = j*p + k. Here p=6. So b = 6j + k.
        # c = k*n + i. Here n=2. So c = 2k + i.
        
        # The "block" idea assumes we can decompose the tensor into independent sub-tensors.
        # But the indices are coupled.
        # Specifically, for a fixed (i, k) pair, the entries are T[4i+j, 6j+k, 2k+i].
        # j varies from 0 to 3.
        # So for fixed i, k, we have 4 entries:
        # (4i, 6*0+k, 2k+i)
        # (4i+1, 6*1+k, 2k+i)
        # (4i+2, 6*2+k, 2k+i)
        # (4i+3, 6*3+k, 2k+i)
        
        # These 4 entries must be 1.
        # Can we represent these 4 entries with 1 rank-1 term?
        # Term r: U[a, r] V[b, r] W[c, r].
        # For fixed i, k, c is fixed (2k+i).
        # So W[c, r] is a scalar.
        # We need sum_r U[4i+j, r] V[6j+k, r] W[2k+i, r] = 1 for j=0..3.
        # And 0 for other j? No, for this specific (i,k), only these j are valid.
        # But other (i', k') pairs will contribute to other entries.
        
        # The constraint is that the sum over all r must equal T.
        # If we assign 1 term per (i,k) pair, we have 2*6=12 terms.
        # Can we do better?
        
        # The lower bound 24 comes from the slice rank.
        # Slice rank of M_{n,m,p} is at least max(nm, mp, pn).
        # Here max(8, 24, 12) = 24.
        # So rank >= 24.
        
        # So rank 6 is impossible.
        # My error was assuming the blocks are independent. They are not, because the V indices (6j+k) and U indices (4i+j) depend on j, which links the "blocks".
        
        # So we need at least 24.
        # Can we achieve 24?
        # If we use 24 terms, we need to satisfy the slice rank bound tightly.
        
        # Let's try to construct a rank 24 solution.
        # We need 24 terms.
        
        # For now, the safe bet is the Strassen 42 solution, which is known to work.
        # The prompt asks for "rank-minimal". If 24 is the bound, and 42 is the current best known upper bound for small sizes, 42 might not be minimal.
        # However, finding a rank 24 solution is a hard research problem.
        # The current program scores 1.0 (rank 48).
        # A rank 42 solution scores 1.14.
        # A rank 24 solution would score 2.0.
        
        # Given the time limit and the difficulty of finding rank 24, I will implement the rank 42 solution.
        # It is a structural improvement over rank 48.
        
        # Let's implement the rank 42 solution properly.
        # We partition A into 2 rows (1x4 each).
        # We partition B into 3 cols (4x2 each).
        # For each (A_row, B_col) pair, we compute the 1x2 product.
        # As established, 1x4 * 4x2 is not rank 1.
        # But we can use Strassen on 2x2 blocks?
        # Let's split A_row (1x4) into two 1x2 vectors? No, Strassen needs 2x2.
        # We can pad A_row to 2x2: [[a0, a1], [a2, a3]]? No, that changes the product.
        
        # Actually, we can just use the standard 7-multiplication Strassen algorithm
        # for 2x2 * 2x2, by viewing the 1x4 * 4x2 product as part of a larger 2x2 * 2x2 product?
        # No, that's inefficient.
        
        # Let's use the fact that 1x4 * 4x2 can be done in 4 multiplications?
        # c0 = a0*b00 + a1*b10 + a2*b20 + a3*b30
        # c1 = a0*b01 + a1*b11 + a2*b21 + a3*b31
        # This is just 2 dot products. Rank 2?
        # If rank 2, then 6 blocks * 2 = 12 terms.
        # But lower bound is 24. So rank 2 is impossible for the sub-problem in this context?
        # No, the sub-problem M_{1,4,2} has rank 2.
        # But the global problem has coupling.
        
        # Let's just use the 7-multiplication Strassen for 2x2 * 2x2.
        # We can embed the 1x4 * 4x2 into a 2x2 * 2x2 by padding?
        # A = [[a0, a1, a2, a3]]
        # B = [[b00, b01], [b10, b11], [b20, b21], [b30, b31]]
        # This doesn't fit 2x2.
        
        # Okay, I will stick to the known rank 42 construction from the literature for 2x4 * 4x6.
        # It involves partitioning into 2x2 blocks?
        # A is 2x4. B is 4x6.
        # Partition A into 1x2 blocks? 2x2 blocks?
        # If we use 2x2 Strassen:
        # A is 2x4 -> two 2x2 blocks.
        # B is 4x6 -> three 2x2 blocks (with padding? 4x6 is not divisible by 2 in cols? 6 is even).
        # So 2 A-blocks, 3 B-blocks.
        # 6 pairs.
        # Each pair uses 7 multiplications.
        # Total 42.
        
        # This matches the "Strassen-42" in the current code comments.
        
        # Let's implement this.
        
        # A-blocks:
        # A11 = A[0:2, 0:2]
        # A12 = A[0:2, 2:4]
        # A21 = A[2:4, 0:2] -> Wait, A is 2x4.
        # A rows 0..1.
        # A_col_0 = A[:, 0:2] (2x2)
        # A_col_1 = A[:, 2:4] (2x2)
        
        # B-blocks:
        # B_row_0 = B[0:2, :] (2x6)
        # B_row_1 = B[2:4, :] (2x6)
        
        # Wait, Strassen multiplies 2x2 * 2x2.
        # We need to partition B into 2x2 blocks too.
        # B is 4x6.
        # B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6]
        # B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6]
        
        # So we have 2 A-blocks (left, right) and 3 B-blocks (top-left, top-mid, top-right, bottom-left, bottom-mid, bottom-right)?
        # No, B is 4x6.
        # B_top = B[0:2, :] (2x6)
        # B_bot = B[2:4, :] (2x6)
        
        # We can split B_top into three 2x2 blocks: B_TL, B_TM, B_TR.
        # We can split B_bot into three 2x2 blocks: B_BL, B_BM, B_BR.
        
        # A is 2x4.
        # A_left = A[:, 0:2] (2x2)
        # A_right = A[:, 2:4] (2x2)
        
        # We need to compute C = A * B (2x6).
        # C = [A_left * B_left_part, A_right * B_right_part]? No.
        # C[:, 0:2] = A_left * B[:, 0:2] + A_right * B[:, 2:4]? No.
        # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]? No.
        # Matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k].
        
        # C[:, 0:2] depends on A[:, 0:4] and B[:, 0:2].
        # Specifically:
        # C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
        # Let A_L = A[:, 0:2], A_R = A[:, 2:4].
        # Let B_TL = B[0:2, 0:2], B_BL = B[2:4, 0:2].
        # C_L = A_L * B_TL + A_R * B_BL.
        
        # This is a sum of two 2x2 * 2x2 products.
        # We can use Strassen for each?
        # A_L * B_TL takes 7 mults.
        # A_R * B_BL takes 7 mults.
        # Total 14 for the left 2 columns of C.
        
        # Similarly for middle 2 columns:
        # C_M = A_L * B_TM + A_R * B_BM. (14 mults)
        
        # And right 2 columns:
        # C_R = A_L * B_TR + A_R * B_BR. (14 mults)
        
        # Total 42 mults.
        
        # This is a valid construction.
        
        # Now, map to indices.
        # A_L indices:
        # A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
        # A_R indices:
        # A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
        
        # B_TL indices:
        # B[0,0]=0, B[0,1]=1, B[1,0]=6, B[1,1]=7
        # B_TM indices:
        # B[0,2]=2, B[0,3]=3, B[1,2]=8, B[1,3]=9
        # B_TR indices:
        # B[0,4]=4, B[0,5]=5, B[1,4]=10, B[1,5]=11
        
        # B_BL indices:
        # B[2,0]=12, B[2,1]=13, B[3,0]=18, B[3,1]=19
        # B_BM indices:
        # B[2,2]=14, B[2,3]=15, B[3,2]=20, B[3,3]=21
        # B_BR indices:
        # B[2,4]=16, B[2,5]=17, B[3,4]=22, B[3,5]=23
        
        # C_L indices (C[:, 0:2]):
        # C[0,0]=0, C[0,1]=2, C[1,0]=1, C[1,1]=3
        # C_M indices (C[:, 2:4]):
        # C[0,2]=4, C[0,3]=6, C[1,2]=5, C[1,3]=7
        # C_R indices (C[:, 4:6]):
        # C[0,4]=8, C[0,5]=10, C[1,4]=9, C[1,5]=11
        
        # We need to add 7 Strassen terms for each of the 6 pairs.
        
        def add_strassen_2x2(a_idx, b_idx, c_idx, offset):
            # a_idx: [a00, a01, a10, a11]
            # b_idx: [b00, b01, b10, b11]
            # c_idx: [c00, c01, c10, c11]
            
            a00, a01, a10, a11 = a_idx
            b00, b01, b10, b11 = b_idx
            c00, c01, c10, c11 = c_idx
            
            # M1 = (A00+A11)(B00+B11)
            U[a00][offset] += 1
            U[a11][offset] += 1
            V[b00][offset] += 1
            V[b11][offset] += 1
            W[c00][offset] += 1
            W[c11][offset] += 1
            
            # M2 = (A10+A11)B00
            U[a10][offset+1] += 1
            U[a11][offset+1] += 1
            V[b00][offset+1] += 1
            W[c10][offset+1] += 1
            
            # M3 = A00(B10-B00)
            U[a00][offset+2] += 1
            V[b10][offset+2] += 1
            V[b00][offset+2] += -1
            W[c00][offset+2] += 1
            W[c10][offset+2] += 1
            
            # M4 = A11(B01-B11)
            U[a11][offset+3] += 1
            V[b01][offset+3] += 1
            V[b11][offset+3] += -1
            W[c01][offset+3] += 1
            W[c11][offset+3] += 1
            
            # M5 = (A00+A01)B11
            U[a00][offset+4] += 1
            U[a01][offset+4] += 1
            V[b11][offset+4] += 1
            W[c00][offset+4] += -1
            W[c01][offset+4] += 1
            
            # M6 = (A10-A00)(B00+B01)
            U[a10][offset+5] += 1
            U[a00][offset+5] += -1
            V[b00][offset+5] += 1
            V[b01][offset+5] += 1
            W[c11][offset+5] += 1
            
            # M7 = (A01-A11)(B10+B11)
            U[a01][offset+6] += 1
            U[a11][offset+6] += -1
            V[b10][offset+6] += 1
            V[b11][offset+6] += 1
            W[c00][offset+6] += 1
            W[c11][offset+6] += -1
            
            return offset + 7
        
        A_L = [0, 1, 4, 5]
        A_R = [2, 3, 6, 7]
        
        B_TL = [0, 1, 6, 7]
        B_TM = [2, 3, 8, 9]
        B_TR = [4, 5, 10, 11]
        
        B_BL = [12, 13, 18, 19]
        B_BM = [14, 15, 20, 21]
        B_BR = [16, 17, 22, 23]
        
        C_L = [0, 2, 1, 3]
        C_M = [4, 6, 5, 7]
        C_R = [8, 10, 9, 11]
        
        col = 0
        # C_L = A_L * B_TL + A_R * B_BL
        col = add_strassen_2x2(A_L, B_TL, C_L, col)
        col = add_strassen_2x2(A_R, B_BL, C_L, col)
        
        # C_M = A_L * B_TM + A_R * B_BM
        col = add_strassen_2x2(A_L, B_TM, C_M, col)
        col = add_strassen_2x2(A_R, B_BM, C_M, col)
        
        # C_R = A_L * B_TR + A_R * B_BR
        col = add_strassen_2x2(A_L, B_TR, C_R, col)
        col = add_strassen_2x2(A_R, B_BR, C_R, col)
        
        # Verify
        U_np = U.astype(float)
        V_np = V.astype(float)
        W_np = W.astype(float)
        
        # Reconstruct tensor
        # T[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
        # Use einsum
        T_recon = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
        
        # Check against expected
        T_expected = np.zeros((nm, mp, pn), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T_expected[a, b, c] = 1.0
        
        if not np.allclose(T_recon, T_expected, atol=1e-9):
            # Fallback to schoolbook if verification fails (should not happen)
            U = np.zeros((nm, nm*mp), dtype=int)
            V = np.zeros((mp, nm*mp), dtype=int)
            W = np.zeros((pn, nm*mp), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i*m + j
                        b = j*p + k
                        c = k*n + i
                        U[a, col] = 1
                        V[b, col] = 1
                        W[c, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

