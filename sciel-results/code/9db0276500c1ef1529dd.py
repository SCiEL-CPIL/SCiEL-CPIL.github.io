"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    
    # Strassen 2x2x2 decomposition (rank 7)
    # For 2x2 A, 2x2 B -> 2x2 C:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each Strassen term: list of (a_idx, a_coeff), (b_idx, b_coeff), (c_idx, c_coeff)
    # A indices: (row, col) -> flat index row*m + col for m=2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: (row, col) -> flat index row*p_local + col for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: (row, col) -> flat index row*n + col for 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    strassen_terms = [
        # M1: (A00+A11)(B00+B11) -> C00+, C11+
        ([((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)], [((0,0), 1), ((1,1), 1)]),
        # M2: (A10+A11)B00 -> C10+, C11-
        ([((1,0), 1), ((1,1), 1)], [((0,0), 1)], [((1,0), 1), ((1,1), -1)]),
        # M3: A00(B01-B11) -> C01+, C11+
        ([((0,0), 1)], [((0,1), 1), ((1,1), -1)], [((0,1), 1), ((1,1), 1)]),
        # M4: A11(B10-B00) -> C00+, C10+
        ([((1,1), 1)], [((1,0), 1), ((0,0), -1)], [((0,0), 1), ((1,0), 1)]),
        # M5: (A00+A01)B11 -> C00-, C01+
        ([((0,0), 1), ((0,1), 1)], [((1,1), 1)], [((0,0), -1), ((0,1), 1)]),
        # M6: (A10-A00)(B00+B01) -> C11+
        ([((1,0), 1), ((0,0), -1)], [((0,0), 1), ((0,1), 1)], [((1,1), 1)]),
        # M7: (A01-A11)(B10+B11) -> C00+
        ([((0,1), 1), ((1,1), -1)], [((1,0), 1), ((1,1), 1)], [((0,0), 1)]),
    ]
    
    # For the single 2x2x1 column: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # Schoolbook rank 2:
    single_col_terms = [
        # Term 1: A00 * B00 -> C00
        ([((0,0), 1)], [((0,0), 1)], [((0,0), 1)]),
        # Term 2: A10 * B10 -> C10
        ([((1,0), 1)], [((1,0), 1)], [((1,0), 1)]),
        # Wait, that's not right. For 2x2x1: C[i,0] = sum_j A[i,j]*B[j,0]
        # Schoolbook: for each (i,j), one term
        # i=0,j=0: A00*B00 -> C00
        # i=0,j=1: A01*B10 -> C00
        # i=1,j=0: A10*B00 -> C10
        # i=1,j=1: A11*B10 -> C10
        # That's rank 4 for 2x2x1? No wait...
    ]
    
    # Actually for 2x2x1: C is 2x1. C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # The tensor has indices (i,j,k) where i in {0,1}, j in {0,1}, k in {0} (only one column)
    # Schoolbook rank = 2*2*1 = 4? No, that seems high.
    # Actually the tensor is: T = sum_{i,j} A[i,j] B[j,0] e_i (output)
    # As a 3-tensor: U is 2*2=4 dim (A), V is 2*1=2 dim (B), W is 1*2=2 dim (C)
    # Schoolbook: for each (i,j,k=0): rank 1 term. That's 2*2*1 = 4 terms.
    # But can we do better? C[0] = A00*B0 + A01*B1, C[1] = A10*B0 + A11*B1
    # This is like a 2x2 matrix times a 2x1 vector. The rank of this tensor...
    # It's actually rank 4 by schoolbook, but maybe less?
    # Let's just use schoolbook for the single column: rank 4.
    
    # Hmm, but 6*7 + 4 = 46. Still better than 52.
    
    # Let me reconsider. For the single column (2x2x1), the tensor is:
    # C[i,0] = sum_j A[i,j] * B[j,0]
    # This is a bilinear map. The tensor T in R^{2x2} x R^{2x1} x R^{1x2}
    # T_{(i,j),(j',k),(k',i')} = 1 if j=j' and i=i' and k=k'
    # Actually: T = sum_{i,j} e_i^{(A_row)} e_j^{(A_col)} ⊗ e_j^{(B_row)} e_0^{(B_col)} ⊗ e_0^{(C_row)} e_i^{(C_col)}
    # Wait, I need to be more careful with the indexing convention.
    
    # The convention: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m + j, c] for A[i,j]
    # V[j*p + k, c] for B[j,k]  
    # W[k*n + i, c] for C[k,i]  (note: k*n + i, not i*n + k!)
    
    # So C is indexed as C[k, i] where k is the column of C (which is the column of B), i is the row.
    # Wait, that's unusual. Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has index k*n + i where k is... the column index of C? And i is the row index?
    # C has shape (n, p) = (2, 13). C[i, k] where i in {0,1}, k in {0,...,12}
    # W[k*n + i, r] means the output index is k*n + i, so for C[i,k], the flat index is k*2 + i.
    
    # So the tensor is: T[(i,j), (j,k), (k,i)] = 1 (summing over j)
    # In flat indices: U index = i*m + j = i*2 + j, V index = j*p + k = j*13 + k, W index = k*n + i = k*2 + i
    
    # For the single column k=12:
    # Terms: for i in {0,1}, j in {0,1}:
    #   U[i*2+j] = 1, V[j*13+12] = 1, W[12*2+i] = 1
    # That's 4 terms for the single column.
    
    # Total: 6*7 + 4 = 46.
    
    # Let me build this.
    
    U_rows = n * m  # 4
    V_rows = m * p  # 26
    W_rows = p * n  # 26
    
    # Collect all terms
    all_U = []  # list of (row, coeff)
    all_V = []
    all_W = []
    
    # Process 6 pairs of columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for pair_idx in range(6):
        k1 = pair_idx * 2
        k2 = pair_idx * 2 + 1
        
        for term in strassen_terms:
            a_parts, b_parts, c_parts = term
            u_entries = []
            v_entries = []
            w_entries = []
            
            for (ai, aj), coeff in a_parts:
                u_entries.append((ai * m + aj, coeff))
            
            for (bi, bj), coeff in b_parts:
                # B index: bi is row (0 or 1), bj is column within the 2x2 block
                # Actual column in B: k1 + bj
                v_entries.append((bi * p + (k1 + bj), coeff))
            
            for (ci, cj), coeff in c_parts:
                # C index: ci is row, cj is column within the 2x2 block
                # Actual column in C: k1 + cj
                # W index: (k1 + cj) * n + ci
                w_entries.append(((k1 + cj) * n + ci, coeff))
            
            all_U.append(u_entries)
            all_V.append(v_entries)
            all_W.append(w_entries)
    
    # Process single column k=12
    for i in range(n):
        for j in range(m):
            all_U.append([(i * m + j, 1)])
            all_V.append([(j * p + 12, 1)])
            all_W.append([(12 * n + i, 1)])
    
    rank = len(all_U)
    
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    for c in range(rank):
        for (row, coeff) in all_U[c]:
            U[row][c] = coeff
        for (row, coeff) in all_V[c]:
            V[row][c] = coeff
        for (row, coeff) in all_W[c]:
            W[row][c] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
