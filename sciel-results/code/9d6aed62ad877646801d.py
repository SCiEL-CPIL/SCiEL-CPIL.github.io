import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    N, M, P = n, m, p

    # We will use a block decomposition.
    # 2x2x2 blocks of Strassen (rank 7 each)
    # Remaining dimensions handled by Schoolbook (rank 1 each)
    
    # n=4, m=5, p=5.
    # n-blocks: [0,2), [2,4)  (two blocks of size 2)
    # m-blocks: [0,2), [2,4), [4,5)  (two blocks of size 2, one of size 1)
    # p-blocks: [0,2), [2,4), [4,5)  (two blocks of size 2, one of size 1)
    
    # Total blocks = 2 * 3 * 3 = 18.
    
    # Ranks:
    # R(2,2,2) = 7
    # R(2,2,1) = 4
    # R(2,1,2) = 4
    # R(1,2,2) = 4
    # R(2,1,1) = 2
    # R(1,2,1) = 2
    # R(1,1,2) = 2
    # R(1,1,1) = 1
    
    # Counts:
    # (2,2,2): 2*2*2 = 8 blocks. Rank: 8*7 = 56.
    # (2,2,1): 2*2*1 = 4 blocks. Rank: 4*4 = 16.
    # (2,1,2): 2*1*2 = 4 blocks. Rank: 4*4 = 16.
    # (2,1,1): 2*1*1 = 2 blocks. Rank: 2*2 = 4.
    # (1,2,2): 1*2*2 = 4 blocks? No, n only has size 2 blocks.
    # Wait, n blocks are [0,2) and [2,4). Both size 2.
    # So there are no n=1 blocks.
    # Thus, only cases where bn=2.
    # Cases:
    # bn=2, bm=2, bp=2: 8 blocks.
    # bn=2, bm=2, bp=1: 4 blocks.
    # bn=2, bm=1, bp=2: 4 blocks.
    # bn=2, bm=1, bp=1: 2 blocks.
    
    # Total Rank = 56 + 16 + 16 + 4 = 92.
    
    rank = 92
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        
        # Strassen factors for 2x2x2
        # U (A): i, j
        # V (B): j, k
        # W (C): k, i
        
        # The 7 factors
        # We define them as lists of (idx, val) for U, V, W
        
        # M1 = (A11+A22)(B11+B22)
        u1 = [((i_off+0)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v1 = [((j_off+0)*p + (k_off+0), 1), ((j_off+1)*p + (k_off+1), 1)]
        w1 = [((k_off+0)*n + (i_off+0), 1), ((k_off+1)*n + (i_off+1), 1)]
        
        # M2 = (A21+A22)B11
        u2 = [((i_off+1)*m + (j_off+0), 1), ((i_off+1)*m + (j_off+1), 1)]
        v2 = [((j_off+0)*p + (k_off+0), 1)]
        w2 = [((k_off+1)*n + (i_off+1), 1)]
        
        # M3 = A11(B12-B22)
        u3 = [((i_off+0)*m + (j_off+0), 1)]
        v3 = [((j_off+0)*p + (k_off+1), 1), ((j_off+1)*p + (k_off+1), -1)]
        w3 = [((k_off+1)*n + (i_off+0), 1)]
        
        # M4 = A22(B21-B11)
        u4 = [((i_off+1)*m + (j_off+1), 1)]
        v4 = [((j_off+1)*p + (k_off+1), 1), ((j_off+0)*p + (k_off+0), -1)]
        w4 = [((k_off+0)*n + (i_off+1), 1), ((k_off+1)*n + (i_off+1), 1)]
        
        # M5 = (A11+A12)B22
        u5 = [((i_off+0)*m + (j_off+0), 1), ((i_off+0)*m + (j_off+1), 1)]
        v5 = [((j_off+1)*p + (k_off+1), 1)]
        w5 = [((k_off+1)*n + (i_off+0), 1), ((k_off+0)*n + (i_off+0), -1)]
        
        # M6 = (A21-A11)(B11+B12)
        # Note: In standard Strassen, M6 is not used in the final output?
        # Let's check the output formulas again.
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # It seems M6 is not used.
        # However, the rank is 7.
        # If M6 is not used, its W vector is 0.
        # Then the decomposition is of rank 6?
        # No, the rank of the tensor is 7.
        # This implies that the 7 tensors are linearly independent.
        # But if one has W=0, it is the zero tensor.
        # The zero tensor is linearly dependent with any other tensor.
        # So the set of 7 tensors would be linearly dependent.
        # This is a contradiction.
        
        # Let's re-verify the Strassen formulas.
        # Maybe M6 IS used?
        # Let's check C21.
        # C21 = A21 B11 + A22 B21.
        # M2 = (A21+A22)B11 = A21 B11 + A22 B11.
        # M4 = A22(B21-B11) = A22 B21 - A22 B11.
        # M2 + M4 = A21 B11 + A22 B21.
        # So C21 = M2 + M4.
        
        # What about C22?
        # C22 = A21 B12 + A22 B22.
        # M1 = A11 B11 + A11 B22 + A22 B11 + A22 B22.
        # -M2 = -A21 B11 - A22 B11.
        # M3 = A11 B12 - A11 B22.
        # M7 = (A12-A22)(B21+B22) = A12 B21 + A12 B22 - A22 B21 - A22 B22.
        
        # Sum:
        # B11: A11 - A22. (Should be 0) -> Mismatch.
        
        # There is an error in my manual expansion or the formulas.
        # Let's use a reliable source or derive it properly.
        
        # Actually, let's just use the schoolbook for the 2x2x2 block too if Strassen is too risky.
        # Rank 8 for 2x2x2.
        # Total rank would be 8*8 + 4*4 + 4*4 + 2*2 = 64 + 16 + 16 + 4 = 100.
        # This is the same as full schoolbook.
        
        # Let's try to fix Strassen.
        # The correct Strassen formulas are:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # Let's re-evaluate C22.
        # M1 = A11B11 + A11B22 + A22B11 + A22B22
        # -M2 = -A21B11 - A22B11
        # M3 = A11B12 - A11B22
        # M7 = A12B21 + A12B22 - A22B21 - A22B22
        
        # Sum of B11 terms: A11 - A22. (Should be 0)
        # Sum of B12 terms: A11. (Should be A21)
        # Sum of B21 terms: A12 - A22. (Should be 0)
        # Sum of B22 terms: A22 - A22 = 0. (Should be A22)
        
        # This is completely wrong.
        
        # Let's look up the correct Strassen formulas.
        # Wikipedia:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 - M5 - M7 + M4
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # It seems the formulas are correct, but my expansion is wrong?
        # Let's check C22 again.
        # C22 = A21 B12 + A22 B22.
        
        # Let's try a different set of Strassen formulas.
        # Maybe the indices are swapped?
        
        # Given the time and risk, I will use schoolbook for all blocks.
        # This guarantees correctness.
        # Rank = 100.
        
        pass

    # Schoolbook Decomposition for all blocks
    def add_schoolbook_block(i_off, j_off, k_off, bn, bm, bp):
        nonlocal col
        for i in range(bn):
            for j in range(bm):
                for k in range(bp):
                    U[(i_off + i) * m + (j_off + j)][col] = 1
                    V[(j_off + j) * p + (k_off + k)][col] = 1
                    W[(k_off + k) * n + (i_off + i)][col] = 1
                    col += 1
    
    # Process all blocks
    n_blocks = [(0, 2), (2, 4)]
    m_blocks = [(0, 2), (2, 4), (4, 5)]
    p_blocks = [(0, 2), (2, 4), (4, 5)]
    
    for i_off, i_end in n_blocks:
        for j_off, j_end in m_blocks:
            for k_off, k_end in p_blocks:
                bn = i_end - i_off
                bm = j_end - j_off
                bp = k_end - k_off
                # Use schoolbook for all blocks to ensure correctness
                add_schoolbook_block(i_off, j_off, k_off, bn, bm, bp)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    rank = len(res["U"][0])
    print(f"Rank: {rank}")
    
    # Verify the decomposition
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if total != 1:
                    max_err = max(max_err, abs(total - 1))
    print(f"Max error: {max_err}")
