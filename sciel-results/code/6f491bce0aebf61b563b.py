"""Rank-minimal exact tensor decomposition for 4x4 x 4x5 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Target tensor: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1
    
    def verify(U, V, W):
        """Check if U, V, W exactly decompose the tensor."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        est = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(est - target)) < 1e-9

    # Build Strassen 2x2x2 factors (rank 7)
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 0: (A00+A11)(B00+B11)
    U2[0,0]=1; U2[3,0]=1
    V2[0,0]=1; V2[3,0]=1
    W2[0,0]=1; W2[3,0]=1
    
    # Term 1: (A10+A11)B00
    U2[2,1]=1; U2[3,1]=1
    V2[0,1]=1
    W2[1,1]=1; W2[3,1]=-1
    
    # Term 2: A00(B01-B11)
    U2[0,2]=1
    V2[1,2]=1; V2[3,2]=-1
    W2[2,2]=1; W2[3,2]=1
    
    # Term 3: A11(B10-B00)
    U2[3,3]=1
    V2[2,3]=1; V2[0,3]=-1
    W2[0,3]=1; W2[1,3]=1
    
    # Term 4: (A00+A01)B11
    U2[0,4]=1; U2[1,4]=1
    V2[3,4]=1
    W2[0,4]=-1; W2[2,4]=1
    
    # Term 5: (A10-A00)(B00+B01)
    U2[2,5]=1; U2[0,5]=-1
    V2[0,5]=1; V2[1,5]=1
    W2[3,5]=1
    
    # Term 6: (A01-A11)(B10+B11)
    U2[1,6]=1; U2[3,6]=-1
    V2[2,6]=1; V2[3,6]=1
    W2[0,6]=1
    
    # Verify 2x2x2
    T222 = np.zeros((4,4,4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T222[i*2+j, j*2+k, k*2+i] = 1
    est2 = np.einsum('ar,br,cr->abc', U2, V2, W2)
    assert np.allclose(est2, T222), "Strassen 2x2x2 failed"
    
    # Build 4x4x4 Strassen factors (rank 49)
    U444 = np.zeros((16, 49))
    V444 = np.zeros((16, 49))
    W444 = np.zeros((16, 49))
    
    # Block index mappings for 4x4 matrices
    # A: index = i*4+j, block (I,J) has i in {2I,2I+1}, j in {2J,2J+1}
    A_blocks = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    # B: index = j*4+k, block (J,K) has j in {2J,2J+1}, k in {2K,2K+1}
    B_blocks = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    # C: index = k*4+i, block (K,I) has k in {2K,2K+1}, i in {2I,2I+1}
    C_blocks = {
        (0,0): [0,1,4,5],
        (0,1): [2,3,6,7],
        (1,0): [8,9,12,13],
        (1,1): [10,11,14,15]
    }
    
    # Strassen terms: (A_blocks_list, B_blocks_list)
    strassen_terms = [
        ([(0,0),(1,1)], [(0,0),(1,1)]),  # M0
        ([(1,0),(1,1)], [(0,0)]),         # M1
        ([(0,0)], [(0,1),(1,1)]),         # M2
        ([(1,1)], [(1,0),(0,0)]),         # M3
        ([(0,0),(0,1)], [(1,1)]),         # M4
        ([(1,0),(0,0)], [(0,0),(0,1)]),   # M5
        ([(0,1),(1,1)], [(1,0),(1,1)])    # M6
    ]
    
    # C block coefficients: C[K,I] = sum_r coeff[r] * M_r
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    coeff_matrix = {
        (0,0): {0:1, 3:1, 4:-1, 6:1},
        (0,1): {2:1, 4:1},
        (1,0): {1:1, 3:1},
        (1,1): {0:1, 1:-1, 2:1, 5:1}
    }
    
    for r, (a_blocks, b_blocks) in enumerate(strassen_terms):
        for s in range(7):
            col = r * 7 + s
            # Fill U444
            for block in a_blocks:
                for local_idx, global_idx in enumerate(A_blocks[block]):
                    U444[global_idx, col] += U2[local_idx, s]
            # Fill V444
            for block in b_blocks:
                for local_idx, global_idx in enumerate(B_blocks[block]):
                    V444[global_idx, col] += V2[local_idx, s]
            # Fill W444
            for c in range(16):
                k = c // 4
                i = c % 4
                K = k // 2
                I = i // 2
                k_prime = k % 2
                i_prime = i % 2
                local_c = k_prime * 2 + i_prime
                coeffs = coeff_matrix[(K, I)]
                coeff = coeffs.get(r, 0)
                if coeff != 0:
                    W444[c, col] = coeff * W2[local_c, s]
    
    # Verify 4x4x4
    T444 = np.zeros((16,16,16))
    for i in range(4):
        for j in range(4):
            for k in range(4):
                T444[i*4+j, j*4+k, k*4+i] = 1
    est444 = np.einsum('ar,br,cr->abc', U444, V444, W444)
    if not np.allclose(est444, T444):
        # Fallback: use schoolbook for 4x4x4
        U444 = np.zeros((16, 64))
        V444 = np.zeros((16, 64))
        W444 = np.zeros((16, 64))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U444[i*4+j, col] = 1
                    V444[j*4+k, col] = 1
                    W444[k*4+i, col] = 1
                    col += 1
        rank_444 = 64
    else:
        rank_444 = 49
    
    # Build 4x4x1 factors (rank 16)
    # B is 4x1, so b = j*1+k = j+0 = j (since k=0 only)
    # Wait, for 4x4x1: n=4, m=4, p=1
    # a = i*4+j, b = j*1+0 = j, c = 0*4+i = i
    U441 = np.zeros((16, 16))
    V441 = np.zeros((4, 16))
    W441 = np.zeros((4, 16))
    for i in range(4):
        for j in range(4):
            col = i*4+j
            U441[i*4+j, col] = 1
            V441[j, col] = 1
            W441[i, col] = 1
    
    # Verify 4x4x1
    T441 = np.zeros((16, 4, 4))
    for i in range(4):
        for j in range(4):
            T441[i*4+j, j, i] = 1
    est441 = np.einsum('ar,br,cr->abc', U441, V441, W441)
    assert np.allclose(est441, T441), "4x4x1 failed"
    
    # Now combine for 4x4x5
    # B is 4x5: b = j*5+k
    # Split B into B1 (4x4, k=0..3) and B2 (4x1, k=4)
    # C = A*B1 + A*B2
    # 
    # For B1: b = j*5+k where k in {0,1,2,3}
    #   This is NOT the same as 4x4x4 where b = j*4+k
    #   We need to remap the B indices
    #
    # Actually, let me reconsider. The 4x4x4 decomposition assumes B is 4x4 with b=j*4+k.
    # For 4x4x5, B is 4x5 with b=j*5+k.
    # The first 4 columns of B (k=0..3) have indices:
    #   j=0: k=0,1,2,3 -> b=0,1,2,3
    #   j=1: k=0,1,2,3 -> b=5,6,7,8
    #   j=2: k=0,1,2,3 -> b=10,11,12,13
    #   j=3: k=0,1,2,3 -> b=15,16,17,18
    # 
    # In 4x4x4, B indices are:
    #   j=0: k=0,1,2,3 -> b=0,1,2,3
    #   j=1: k=0,1,2,3 -> b=4,5,6,7
    #   j=2: k=0,1,2,3 -> b=8,9,10,11
    #   j=3: k=0,1,2,3 -> b=12,13,14,15
    #
    # So the B index mapping is different. We need to create a new V matrix
    # that maps from the 4x4x4 V indices to the 4x4x5 B indices.
    #
    # Let's create V445 for the 4x4x5 problem.
    # V445 has shape (20, rank) where 20 = m*p = 4*5
    #
    # For the 4x4x4 part, we need to map V444 indices to V445 indices.
    # V444[b4, r] where b4 = j*4+k
    # V445[b5, r] where b5 = j*5+k
    #
    # For k in {0,1,2,3}: b5 = j*5+k
    # For k=4: b5 = j*5+4 (this is the B2 part)
    
    # Build V445 for the 4x4x4 part (first 4 columns)
    V445_part1 = np.zeros((20, rank_444))
    for j in range(4):
        for k in range(4):
            b4 = j*4+k
            b5 = j*5+k
            V445_part1[b5, :] = V444[b4, :]
    
    # Build V445 for the 4x4x1 part (5th column, k=4)
    # For 4x4x1: b = j (since p=1, b=j*1+0=j)
    # For 4x4x5: b = j*5+4
    V445_part2 = np.zeros((20, 16))
    for j in range(4):
        b1 = j  # index in 4x4x1
        b5 = j*5+4  # index in 4x4x5
        V445_part2[b5, :] = V441[b1, :]
    
    # Combine V445
    V445 = np.hstack([V445_part1, V445_part2])
    
    # U445 = U444 for the first part, U441 for the second part
    # But U is the same for both parts since A is the same
    U445 = np.hstack([U444, U441])
    
    # W445: need to map C indices
    # For 4x4x4: c = k*4+i where k in {0,1,2,3}
    # For 4x4x5: c = k*4+i where k in {0,1,2,3,4}
    # The C indices are the same for k in {0,1,2,3}
    # For k=4: c = 4*4+i = 16+i
    
    # W445_part1: W444 mapped to 4x4x5 C indices
    W445_part1 = np.zeros((20, rank_444))
    for c4 in range(16):
        W445_part1[c4, :] = W444[c4, :]
    # c4 = k*4+i for k in {0,1,2,3}, i in {0,1,2,3}
    # c5 = k*4+i for k in {0,1,2,3}, i in {0,1,2,3} -> same indices
    
    # W445_part2: W441 mapped to 4x4x5 C indices
    # W441[i, col] where i is the C index for 4x4x1 (p=1, so c=i)
    # For 4x4x5: c = 4*4+i = 16+i for k=4
    W445_part2 = np.zeros((20, 16))
    for i in range(4):
        W445_part2[16+i, :] = W441[i, :]
    
    # Combine W445
    W445 = np.hstack([W445_part1, W445_part2])
    
    total_rank = rank_444 + 16
    
    # Verify
    if not verify(U445, V445, W445):
        # Fallback to schoolbook
        U445 = np.zeros((16, 80))
        V445 = np.zeros((20, 80))
        W445 = np.zeros((20, 80))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(5):
                    U445[i*4+j, col] = 1
                    V445[j*5+k, col] = 1
                    W445[k*4+i, col] = 1
                    col += 1
        total_rank = 80
    
    return {"U": U445.tolist(), "V": V445.tolist(), "W": W445.tolist()}

# EVOLVE-BLOCK-END
