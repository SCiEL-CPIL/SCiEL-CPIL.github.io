import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a rank-52 decomposition for 3x3x6.
    # Strategy: Use two copies of a rank-26 decomposition for 3x3x3.
    # The rank-26 decomposition for 3x3x3 is derived from the schoolbook (rank 27)
    # by replacing the 2x2x2 sub-block (top-left) with Strassen's rank-7 algorithm.
    
    # 1. Define the 3x3x3 rank-26 decomposition
    # Indices: A is 3x3 (0..2), B is 3x3 (0..2), C is 3x3 (0..2)
    # We split indices into {0, 1} (Strassen part) and {2} (Schoolbook tail).
    
    # Strassen decomposition for 2x2x2 (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Tail schoolbook terms for index 2
    # T1: A02 * B02 -> C02
    # T2: A02 * B12 -> C12
    # T3: A02 * B22 -> C22
    # T4: A12 * B02 -> C02 (Wait, A12*B02 contributes to C10? No, C_{i,k} = sum_j A_{i,j} B_{j,k})
    # Let's be precise.
    # C_{i,k} = sum_{j=0}^2 A_{i,j} B_{j,k}
    
    # Strassen covers i in {0,1}, j in {0,1}, k in {0,1}.
    # The remaining terms involve index 2 in i, j, or k.
    
    # Terms where i=2 (row 2 of A, row 2 of C):
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    # These are 3 * 3 = 9 terms? No, we can group them.
    # Actually, the "tail" terms are those not covered by the 2x2x2 block.
    # The 2x2x2 block covers i in {0,1}, j in {0,1}, k in {0,1}.
    # Total terms in schoolbook 3x3x3: 27.
    # Terms in 2x2x2 block: 2*2*2 = 8.
    # Remaining terms: 27 - 8 = 19.
    # Wait, Strassen replaces the 8 terms with 7.
    # So total rank = 7 + 19 = 26.
    
    # Let's list the 19 tail terms explicitly.
    # They are terms (i, j, k) where at least one index is 2.
    # i in {0,1,2}, j in {0,1,2}, k in {0,1,2}
    # Exclude i in {0,1}, j in {0,1}, k in {0,1}
    
    tail_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i < 2 and j < 2 and k < 2:
                    continue
                tail_terms.append((i, j, k))
                
    # There are 19 such terms.
    # For each tail term (i, j, k), we have:
    # U has 1 at index i*m + j (i*3 + j)
    # V has 1 at index j*p + k (j*3 + k)
    # W has 1 at index k*n + i (k*3 + i)
    
    # Now, let's define the Strassen terms for the 2x2 block (indices 0,1).
    # We need to map the Strassen M's to U, V, W vectors.
    # The U vector for the whole 3x3x3 has size 9.
    # The V vector has size 9.
    # The W vector has size 9.
    
    # Strassen M1: (A00+A11)(B00+B11)
    # U1: A00 + A11 -> indices 0, 3 (since i=0,j=0 is 0; i=1,j=1 is 4? No. i*m+j. m=3.
    # i=0,j=0 -> 0
    # i=1,j=1 -> 1*3+1 = 4
    # So U1 has 1 at 0 and 4.
    # V1: B00 + B11 -> indices 0, 4.
    # W1: contributes to C00 and C11.
    # C00 is k=0, i=0 -> index 0*3+0 = 0.
    # C11 is k=1, i=1 -> index 1*3+1 = 4.
    # So W1 has 1 at 0 and 4.
    
    # Let's define a helper to create the vectors.
    def make_vec(size, indices_values):
        v = [0] * size
        for idx, val in indices_values:
            v[idx] += val
        return v
    
    # Strassen Terms for 2x2 block (local indices 0,1)
    # Global indices for A (i,j): i*3+j
    # Global indices for B (j,k): j*3+k
    # Global indices for C (k,i): k*3+i
    
    strassen_terms = []
    
    # M1: (A00+A11)(B00+B11) -> C00 + C11
    # A: 00(0), 11(4)
    # B: 00(0), 11(4)
    # C: 00(0), 11(4)
    strassen_terms.append({
        'U': [(0, 1), (4, 1)],
        'V': [(0, 1), (4, 1)],
        'W': [(0, 1), (4, 1)]
    })
    
    # M2: (A10+A11)B00 -> C10 + C11 (with -1 for C11? No, C10 = M2+M4, C11 = M1-M2+M3+M6)
    # C10: k=1, i=1 -> index 1*3+1 = 4? No. C index is k*n+i. n=3.
    # C_{i,k}. Wait, the task says W[k*n+i].
    # My previous check: C00 is k=0, i=0 -> 0. C11 is k=1, i=1 -> 4.
    # C10 is k=0, i=1 -> 0*3+1 = 1.
    # C01 is k=1, i=0 -> 1*3+0 = 3.
    
    # Let's re-verify W indices.
    # W[k*n+i]
    # C_{i,k} corresponds to W[k*n+i].
    # M2 contributes to C10 (i=1, k=0) and C11 (i=1, k=1).
    # C10: k=0, i=1 -> idx 1.
    # C11: k=1, i=1 -> idx 4.
    # M2 = (A10+A11)B00
    # A10: i=1, j=0 -> idx 3.
    # A11: i=1, j=1 -> idx 4.
    # B00: j=0, k=0 -> idx 0.
    strassen_terms.append({
        'U': [(3, 1), (4, 1)],
        'V': [(0, 1)],
        'W': [(1, 1), (4, -1)]  # C10 gets +1, C11 gets -1
    })
    
    # M3: A00(B01-B11) -> C01 + C11 (C01 gets +1, C11 gets +1)
    # C01: k=1, i=0 -> idx 3.
    # C11: k=1, i=1 -> idx 4.
    # A00: idx 0.
    # B01: j=0, k=1 -> idx 1.
    # B11: j=1, k=1 -> idx 4.
    strassen_terms.append({
        'U': [(0, 1)],
        'V': [(1, 1), (4, -1)],
        'W': [(3, 1), (4, 1)]
    })
    
    # M4: A11(B10-B00) -> C10 + C00 (C10 gets +1, C00 gets +1)
    # C10: idx 1.
    # C00: idx 0.
    # A11: idx 4.
    # B10: j=1, k=0 -> idx 3.
    # B00: idx 0.
    strassen_terms.append({
        'U': [(4, 1)],
        'V': [(3, 1), (0, -1)],
        'W': [(1, 1), (0, 1)]
    })
    
    # M5: (A00+A01)B11 -> C00 (-1) + C01 (+1)
    # C00: idx 0.
    # C01: idx 3.
    # A00: idx 0.
    # A01: i=0, j=1 -> idx 1.
    # B11: idx 4.
    strassen_terms.append({
        'U': [(0, 1), (1, 1)],
        'V': [(4, 1)],
        'W': [(0, -1), (3, 1)]
    })
    
    # M6: (A10-A00)(B00+B01) -> C11 (+1)
    # C11: idx 4.
    # A10: idx 3.
    # A00: idx 0.
    # B00: idx 0.
    # B01: idx 1.
    strassen_terms.append({
        'U': [(3, 1), (0, -1)],
        'V': [(0, 1), (1, 1)],
        'W': [(4, 1)]
    })
    
    # M7: (A01-A11)(B10+B11) -> C00 (+1)
    # C00: idx 0.
    # A01: idx 1.
    # A11: idx 4.
    # B10: idx 3.
    # B11: idx 4.
    strassen_terms.append({
        'U': [(1, 1), (4, -1)],
        'V': [(3, 1), (4, 1)],
        'W': [(0, 1)]
    })
    
    # Now, construct the full 3x3x3 rank-26 decomposition.
    # We will store U, V, W as lists of lists.
    # Each term is a rank-1 tensor.
    # We will build the matrices U, V, W for the 3x3x3 case.
    
    U33 = []
    V33 = []
    W33 = []
    
    # Add Strassen terms
    for term in strassen_terms:
        u = make_vec(9, term['U'])
        v = make_vec(9, term['V'])
        w = make_vec(9, term['W'])
        U33.append(u)
        V33.append(v)
        W33.append(w)
        
    # Add Tail terms
    for i, j, k in tail_terms:
        u = make_vec(9, [(i*3+j, 1)])
        v = make_vec(9, [(j*3+k, 1)])
        w = make_vec(9, [(k*3+i, 1)])
        U33.append(u)
        V33.append(v)
        W33.append(w)
        
    # Rank of 333 is 26.
    # Now, for 3x3x6, we use two copies of this 333 decomposition.
    # The second copy handles the "second half" of the 6 columns.
    # B is 3x6. We can split B into B_left (3x3) and B_right (3x3).
    # A is 3x3.
    # C is 3x6. C_left (3x3) comes from A * B_left. C_right (3x3) comes from A * B_right.
    
    # So we just duplicate the 333 decomposition twice.
    # The U matrix is the same for both copies (since A is the same).
    # Wait, U has shape (n*m, rank). Here n*m = 9.
    # V has shape (m*p, rank). Here m*p = 18.
    # W has shape (p*n, rank). Here p*n = 18.
    
    # In the 333 case:
    # U is 9x26.
    # V is 9x26.
    # W is 9x26.
    
    # In the 336 case:
    # U is 9x52.
    # V is 18x52.
    # W is 18x52.
    
    # For the first copy (left half):
    # U is the same U33.
    # V is V33 placed in the top 9 rows (indices 0..8 of the 18-row V).
    # W is W33 placed in the top 9 rows (indices 0..8 of the 18-row W).
    
    # For the second copy (right half):
    # U is the same U33.
    # V is V33 placed in the bottom 9 rows (indices 9..17 of the 18-row V).
    #    Note: B_right indices j,k map to global j*6+k.
    #    Wait. B is 3x6. Index is j*6+k.
    #    In the 333 case, B index was j*3+k.
    #    For the left half (k=0,1,2), global index is j*6+k.
    #    For the right half (k=3,4,5), global index is j*6+k.
    #    So we need to map the local 3x3 indices to the global 3x6 indices.
    
    # Let's re-map the indices for V and W.
    # In 333:
    # V index: j*3+k.
    # W index: k*3+i.
    
    # In 336:
    # V index: j*6+k.
    # W index: k*6+i.
    
    # Left half (k in 0..2):
    # V global idx = j*6 + k.
    # W global idx = k*6 + i.
    
    # Right half (k in 3..5):
    # Let k_local = k - 3.
    # V global idx = j*6 + (k_local + 3) = j*6 + k_local + 3.
    # W global idx = (k_local + 3)*6 + i = k_local*6 + 18 + i.
    
    # We will construct U, V, W for 336 directly.
    
    rank_336 = 52
    U_336 = [[0] * rank_336 for _ in range(9)]
    V_336 = [[0] * rank_336 for _ in range(18)]
    W_336 = [[0] * rank_336 for _ in range(18)]
    
    # Fill in the two copies
    for copy in range(2):
        offset = copy * 26
        for r in range(26):
            col = offset + r
            
            # U is the same for both copies
            for i in range(9):
                U_336[i][col] = U33[r][i]
                
            # V
            for local_idx, val in enumerate(V33[r]):
                if val != 0:
                    # local_idx is j*3+k
                    j = local_idx // 3
                    k = local_idx % 3
                    if copy == 0:
                        global_idx = j * 6 + k
                    else:
                        global_idx = j * 6 + (k + 3)
                    V_336[global_idx][col] = val
                    
            # W
            for local_idx, val in enumerate(W33[r]):
                if val != 0:
                    # local_idx is k*3+i
                    k = local_idx // 3
                    i = local_idx % 3
                    if copy == 0:
                        global_idx = k * 6 + i
                    else:
                        global_idx = (k + 3) * 6 + i
                    W_336[global_idx][col] = val

    return {"U": U_336, "V": V_336, "W": W_336}
# EVOLVE-BLOCK-END
