import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation
    # 4x4 Strassen: 49
    # 4x1x4 Schoolbook: 16
    # 4x4x1 Schoolbook: 16
    # 4x1x1 Schoolbook: 4
    # 1x4x4 Schoolbook: 16
    # 1x1x4 Schoolbook: 4
    # 1x4x1 Schoolbook: 4
    # 1x1x1 Schoolbook: 1
    total_rank = 49 + 16 + 16 + 4 + 16 + 4 + 4 + 1
    
    U = np.zeros((25, total_rank), dtype=int)
    V = np.zeros((25, total_rank), dtype=int)
    W = np.zeros((25, total_rank), dtype=int)
    
    current_rank = 0
    
    # Helper to add schoolbook decomposition for n x m x p
    # u_rows: list of global row indices in U (size n*m)
    # v_rows: list of global row indices in V (size m*p)
    # w_rows: list of global row indices in W (size p*n)
    def add_schoolbook(n, m, p, u_rows, v_rows, w_rows):
        nonlocal current_rank
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = u_rows[i * m + j]
                    v_idx = v_rows[j * p + k]
                    w_idx = w_rows[k * n + i]
                    U[u_idx, current_rank] = 1
                    V[v_idx, current_rank] = 1
                    W[w_idx, current_rank] = 1
                    current_rank += 1

    # Helper to add Strassen 4x4 decomposition
    # u_rows: list of global row indices in U for the 4x4 A block (size 16)
    # v_rows: list of global row indices in V for the 4x4 B block (size 16)
    # w_rows: list of global row indices in W for the 4x4 C block (size 16)
    def add_strassen_4x4(u_rows, v_rows, w_rows):
        nonlocal current_rank
        
        # Local indices for 4x4 matrix (row-major 0-15)
        # A00: 0,1,4,5
        # A01: 2,3,6,7
        # A10: 8,9,12,13
        # A11: 10,11,14,15
        
        # We define the 7 Strassen products for 4x4
        # Each product is a 2x2 multiplication.
        # We need to map the 2x2 Strassen decomposition to the 4x4 global indices.
        
        # strassen_2x2 returns local U, V, W for 2x2 (size 4x7)
        # Local A indices: 0:00, 1:01, 2:10, 3:11
        # Local B indices: 0:00, 1:01, 2:10, 3:11
        # Local C indices: 0:00, 1:01, 2:10, 3:11
        
        # We will manually construct the 49 rank-1 tensors.
        # This is tedious but robust.
        
        # Define the 7 2x2 multiplications
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B11-B00)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each M, we have a 2x2 multiplication.
        # We use the standard Strassen 2x2 decomposition.
        # U_2x2, V_2x2, W_2x2 are 4x7 matrices.
        
        # Let's get the base Strassen 2x2 tensors
        # We'll define them as a list of (u_vec, v_vec, w_vec)
        # u_vec is length 4, v_vec length 4, w_vec length 4.
        
        s_u = [
            [1, 0, 1, 0, 0, 0, 0], # M1: A00+A11 -> [1,0,0,1]
            [0, 1, 0, 0, 0, 0, 0], # M2: A10+A11? No.
            # Let's re-derive Strassen 2x2 vectors carefully.
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # m1 = (a+d)(e+h)
            # m2 = (c+d)e
            # m3 = a(f-h)
            # m4 = d(g-e)
            # m5 = (a+b)h
            # m6 = (c-d)(f+h)
            # m7 = (b-d)(e+g)
            
            # U vectors (for A):
            # m1: [1,0,0,1]
            # m2: [0,0,1,1]
            # m3: [1,0,0,0]
            # m4: [0,0,0,1]
            # m5: [1,1,0,0]
            # m6: [0,0,1,-1]
            # m7: [0,1,0,-1]
            
            # V vectors (for B):
            # m1: [1,0,0,1]
            # m2: [1,0,0,0]
            # m3: [0,1,0,-1]
            # m4: [0,-1,1,0]
            # m5: [0,0,0,1]
            # m6: [0,1,0,1]
            # m7: [1,0,1,0]
            
            # W vectors (for C):
            # C = [[c11,c12],[c21,c22]]
            # c11 = m1 + m4 - m5 + m7
            # c12 = m3 + m5
            # c21 = m2 + m4
            # c22 = m1 - m2 + m3 + m6
            
            # w for m1: [1,0,0,1]
            # w for m2: [0,0,1,-1]
            # w for m3: [0,1,0,1]
            # w for m4: [1,0,1,0]
            # w for m5: [-1,1,0,0]
            # w for m6: [0,0,0,1]
            # w for m7: [1,0,0,0]
        ]
        
        # We will iterate through the 7 2x2 multiplications.
        # For each, we define the mapping from local 2x2 indices to global 4x4 indices.
        
        # Local 2x2 indices: 0:00, 1:01, 2:10, 3:11
        # Global 4x4 indices:
        # 00: 0,1,4,5
        # 01: 2,3,6,7
        # 10: 8,9,12,13
        # 11: 10,11,14,15
        
        # We need to expand the local U, V, W vectors to the global 4x4 space.
        
        def expand_2x2_to_4x4(local_vec, block_map, sign=1):
            # local_vec: list of 4 values for indices 0,1,2,3
            # block_map: dict {0: [global_indices], 1: [...], ...}
            # Returns a dict {global_index: value}
            res = {}
            for local_idx, val in enumerate(local_vec):
                if val != 0:
                    for g_idx in block_map[local_idx]:
                        res[g_idx] = res.get(g_idx, 0) + val * sign
            return res

        # Define the 7 blocks for A and B for the 7 Strassen products
        # M1: A' = A00+A11, B' = B00+B11
        # A' block map: 0->A00, 1->A01, 2->A10, 3->A11 is NOT correct.
        # The 2x2 Strassen expects a 2x2 matrix.
        # For M1, the "A" matrix of the 2x2 Strassen is A00+A11.
        # But A00+A11 is a 2x2 matrix.
        # So the local index 0 of the 2x2 Strassen corresponds to the (0,0) element of A00+A11.
        # (0,0) of A00+A11 is A00[0][0] + A11[0][0].
        
        # This requires knowing the structure of A00, A11 etc.
        # A00 is 2x2. A11 is 2x2.
        # So A' is 2x2.
        # local_idx 0 (A'[0][0]) = A00[0][0] + A11[0][0]
        # local_idx 1 (A'[0][1]) = A00[0][1] + A11[0][1]
        # local_idx 2 (A'[1][0]) = A00[1][0] + A11[1][0]
        # local_idx 3 (A'[1][1]) = A00[1][1] + A11[1][1]
        
        # Global indices for A00: 0,1,4,5
        # Global indices for A11: 10,11,14,15
        
        # So for M1, A mapping:
        # local 0 -> global 0, 10
        # local 1 -> global 1, 11
        # local 2 -> global 4, 14
        # local 3 -> global 5, 15
        
        # This pattern holds for all sums/differences.
        
        # Let's define the mappings for the 7 products.
        
        # A_blocks:
        # A00: [0,1,4,5]
        # A01: [2,3,6,7]
        # A10: [8,9,12,13]
        # A11: [10,11,14,15]
        
        a00 = [0,1,4,5]
        a01 = [2,3,6,7]
        a10 = [8,9,12,13]
        a11 = [10,11,14,15]
        
        b00 = [0,1,4,5]
        b01 = [2,3,6,7]
        b10 = [8,9,12,13]
        b11 = [10,11,14,15]
        
        c00 = [0,1,4,5]
        c01 = [2,3,6,7]
        c10 = [8,9,12,13]
        c11 = [10,11,14,15]
        
        # M1: (A00+A11)(B00+B11)
        # A map: 0->a00[0]+a11[0], 1->a00[1]+a11[1], 2->a00[2]+a11[2], 3->a00[3]+a11[3]
        # B map: 0->b00[0]+b11[0], 1->b00[1]+b11[1], 2->b00[2]+b11[2], 3->b00[3]+b11[3]
        # C map: 0->c00[0]+c11[0]... NO.
        # The result of M1 is a 2x2 matrix M1.
        # M1 contributes to C00 and C11?
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes to C00 (top-left 2x2 of C) and C11 (bottom-right 2x2 of C).
        # Wait, C is 4x4.
        # C00 is the top-left 2x2 block of C.
        # C11 is the bottom-right 2x2 block of C.
        # So the local C indices of the 2x2 Strassen result map to:
        # local 0 -> global c00[0]
        # local 1 -> global c00[1]
        # local 2 -> global c00[2]
        # local 3 -> global c00[3]
        # AND
        # local 0 -> global c11[0]
        # local 1 -> global c11[1]
        # local 2 -> global c11[2]
        # local 3 -> global c11[3]
        
        # This is correct.
        
        # We will create a list of 7 "sub-decompositions".
        # Each sub-decomposition has:
        # - A_map: list of 4 lists of global indices (for local 0,1,2,3)
        # - B_map: list of 4 lists of global indices
        # - C_map: list of 4 lists of global indices (can have duplicates if contributing to multiple blocks)
        
        sub_decomps = []
        
        # M1
        sub_decomps.append({
            'A': [a00[i] + a11[i] for i in range(4)],
            'B': [b00[i] + b11[i] for i in range(4)],
            'C': [c00[i] + c11[i] for i in range(4)]
        })
        
        # M2: (A10+A11)B00
        sub_decomps.append({
            'A': [a10[i] + a11[i] for i in range(4)],
            'B': b00,
            'C': [c10[i] for i in range(4)] + [c11[i] for i in range(4)] # C21 and C22?
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            # So M2 contributes to C10 (bottom-left) and C11 (bottom-right).
            # Local C indices 0,1,2,3 map to C10[0..3] and C11[0..3].
            # Wait, C10 is 2x2. C11 is 2x2.
            # So yes.
        })
        # Correction: C_map should be a list of 4 lists.
        # For M2, local C index 0 (top-left of 2x2 result) contributes to C10[0] and C11[0].
        # So C_map[0] = [c10[0], c11[0]]
        
        # Let's re-define C_map properly.
        
        sub_decomps[1]['C'] = [c10[i] + c11[i] for i in range(4)]
        
        # M3: A00(B11-B00)
        sub_decomps.append({
            'A': a00,
            'B': [b11[i] - b00[i] for i in range(4)],
            'C': [c01[i] + c11[i] for i in range(4)] # C12 and C22
            # C12 = M3 + M5
            # C22 = M1 - M2 + M3 + M6
        })
        
        # M4: A11(B10-B00)
        sub_decomps.append({
            'A': a11,
            'B': [b10[i] - b00[i] for i in range(4)],
            'C': [c00[i] + c10[i] for i in range(4)] # C11 and C21
            # C11 = M1 + M4 - M5 + M7
            # C21 = M2 + M4
        })
        
        # M5: (A00+A01)B11
        sub_decomps.append({
            'A': [a00[i] + a01[i] for i in range(4)],
            'B': b11,
            'C': [c00[i] + c01[i] for i in range(4)] # C11 and C12
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
        })
        
        # M6: (A10-A00)(B00+B01)
        sub_decomps.append({
            'A': [a10[i] - a00[i] for i in range(4)],
            'B': [b00[i] + b01[i] for i in range(4)],
            'C': [c11[i] for i in range(4)] # C22 only
            # C22 = M1 - M2 + M3 + M6
        })
        
        # M7: (A01-A11)(B10+B11)
        sub_decomps.append({
            'A': [a01[i] - a11[i] for i in range(4)],
            'B': [b10[i] + b11[i] for i in range(4)],
            'C': [c00[i] for i in range(4)] # C11 only
            # C11 = M1 + M4 - M5 + M7
        })
        
        # Now apply the Strassen 2x2 vectors
        # u_vecs, v_vecs, w_vecs are 7 vectors of length 4.
        
        u_vecs = [
            [1,0,0,1],
            [0,0,1,1],
            [1,0,0,0],
            [0,0,0,1],
            [1,1,0,0],
            [0,0,1,-1],
            [0,1,0,-1]
        ]
        
        v_vecs = [
            [1,0,0,1],
            [1,0,0,0],
            [0,1,0,-1],
            [0,-1,1,0],
            [0,0,0,1],
            [0,1,0,1],
            [1,0,1,0]
        ]
        
        w_vecs = [
            [1,0,0,1],
            [0,0,1,-1],
            [0,1,0,1],
            [1,0,1,0],
            [-1,1,0,0],
            [0,0,0,1],
            [1,0,0,0]
        ]
        
        for m_idx in range(7):
            for s_idx in range(7):
                # u_vecs[s_idx] is the vector for the s-th Strassen product of the 2x2 block
                # Wait, the 2x2 Strassen has 7 products.
                # We are iterating over the 7 "big" Strassen products (M1..M7) of the 4x4.
                # For each big product, we use the full 2x2 Strassen decomposition.
                # So for each big product m_idx, we add 7 rank-1 tensors.
                
                u_vec = u_vecs[s_idx]
                v_vec = v_vecs[s_idx]
                w_vec = w_vecs[s_idx]
                
                # Expand to global indices
                u_map = sub_decomps[m_idx]['A']
                v_map = sub_decomps[m_idx]['B']
                w_map = sub_decomps[m_idx]['C']
                
                # u_map is a list of 4 lists of global indices.
                # u_vec[i] is the coefficient for local index i.
                
                for i in range(4):
                    if u_vec[i] != 0:
                        for g_idx in u_map[i]:
                            U[g_idx, current_rank] += u_vec[i]
                            
                for i in range(4):
                    if v_vec[i] != 0:
                        for g_idx in v_map[i]:
                            V[g_idx, current_rank] += v_vec[i]
                            
                for i in range(4):
                    if w_vec[i] != 0:
                        for g_idx in w_map[i]:
                            W[g_idx, current_rank] += w_vec[i]
                            
                current_rank += 1

    # Now call the helpers
    
    # 1. A11*B11 (4x4x4)
    # A11 rows 0-3, cols 0-3.
    # Global U indices for A11: i*5+j for i in 0-3, j in 0-3.
    u_map_4x4 = [i*5+j for i in range(4) for j in range(4)]
    v_map_4x4 = [i*5+j for i in range(4) for j in range(4)]
    w_map_4x4 = [i*5+j for i in range(4) for j in range(4)]
    
    add_strassen_4x4(u_map_4x4, v_map_4x4, w_map_4x4)
    
    # 2. A12*B21 (4x1x4)
    # A12: rows 0-3, col 4.
    # B21: row 4, cols 0-3.
    # C11: rows 0-3, cols 0-3.
    
    # U rows for A12: i*5+4 for i in 0-3.
    u_map_41 = [i*5+4 for i in range(4)]
    # V rows for B21: 4*5+j for j in 0-3.
    v_map_14 = [20+j for j in range(4)]
    # W rows for C11: i*5+j for i in 0-3, j in 0-3.
    w_map_44 = [i*5+j for i in range(4) for j in range(4)]
    
    # Schoolbook 4x1x4
    # n=4, m=1, p=4
    # U size 4, V size 4, W size 16.
    # We need to map the local indices to the global ones.
    # Local U i (0-3) -> global u_map_41[i]
    # Local V j (0-3) -> global v_map_14[j] (Note: m=1, so j is 0. p=4, so k is 0-3. V index is j*p+k = k)
    # Local W k*n+i -> global w_map_44[k*4+i]? No.
    # W is (p*n, rank). Local W index is k*n+i.
    # Global W index for C11 is row i, col j.
    # C11[i][j] = sum_k A12[i][0] * B21[0][j].
    # So term is A12[i] * B21[j].
    # U index: i. V index: j. W index: i*4+j?
    # In schoolbook, W index is k*n+i. Here m=1, so k is the column of B?
    # Let's check schoolbook definition.
    # U[i*m+j], V[j*p+k], W[k*n+i].
    # Here n=4, m=1, p=4.
    # U[i*1+0] = U[i].
    # V[0*4+k] = V[k].
    # W[k*4+i].
    # So for each i, k:
    # U[i] = 1
    # V[k] = 1
    # W[k*4+i] = 1
    
    # Global mapping:
    # U[i] -> u_map_41[i]
    # V[k] -> v_map_14[k]
    # W[k*4+i] -> w_map_44[k*4+i]?
    # w_map_44 is [i*5+j for i in 0-3, j in 0-3].
    # Index in w_map_44 for row i, col j is i*4+j?
    # My list comprehension: [i*5+j for i in range(4) for j in range(4)]
    # i=0, j=0 -> 0
    # i=0, j=1 -> 1
    # i=1, j=0 -> 5
    # So w_map_44[i*4+j] is NOT the global index.
    # The global index IS i*5+j.
    # The local W index is k*4+i.
    # So we need to map local W index `w_idx` to global W index.
    # w_idx = k*4+i.
    # Global index = i*5+k.
    
    # I will write a specific loop for this term to avoid confusion.
    
    for i in range(4):
        for k in range(4):
            U[u_map_41[i], current_rank] = 1
            V[v_map_14[k], current_rank] = 1
            W[i*5+k, current_rank] = 1
            current_rank += 1
            
    # 3. A11*B12 (4x4x1)
    # A11: 4x4. B12: 4x1. C12: 4x1.
    # U rows: u_map_4x4
    # V rows: i*5+4 for i in 0-3.
    v_map_41 = [i*5+4 for i in range(4)]
    # W rows: i*5+4 for i in 0-3.
    w_map_41 = [i*5+4 for i in range(4)]
    
    # Schoolbook 4x4x1
    # n=4, m=4, p=1.
    # U[i*4+j], V[j*1+0]=V[j], W[0*4+i]=W[i].
    # Term: A[i][j] * B[j][0] -> C[i][0].
    # U[i*4+j] = 1
    # V[j] = 1
    # W[i] = 1
    
    for i in range(4):
        for j in range(4):
            U[u_map_4x4[i*4+j], current_rank] = 1
            V[v_map_41[j], current_rank] = 1
            W[w_map_41[i], current_rank] = 1
            current_rank += 1
            
    # 4. A12*B22 (4x1x1)
    # A12: 4x1. B22: 1x1. C12: 4x1.
    # U rows: u_map_41
    # V rows: [24] (row 4, col 4)
    v_map_11 = [24]
    # W rows: w_map_41
    
    for i in range(4):
        U[u_map_41[i], current_rank] = 1
        V[v_map_11[0], current_rank] = 1
        W[w_map_41[i], current_rank] = 1
        current_rank += 1
        
    # 5. A21*B11 (1x4x4)
    # A21: 1x4. B11: 4x4. C21: 1x4.
    # U rows: 20+j for j in 0-3.
    u_map_14 = [20+j for j in range(4)]
    # V rows: v_map_4x4
    # W rows: 20+j for j in 0-3.
    w_map_14 = [20+j for j in range(4)]
    
    # Schoolbook 1x4x4
    # n=1, m=4, p=4.
    # U[0*4+j]=U[j], V[j*4+k], W[k*1+0]=W[k].
    # Term: A[0][j] * B[j][k] -> C[0][k].
    
    for j in range(4):
        for k in range(4):
            U[u_map_14[j], current_rank] = 1
            V[v_map_4x4[j*4+k], current_rank] = 1
            W[w_map_14[k], current_rank] = 1
            current_rank += 1
            
    # 6. A22*B21 (1x1x4)
    # A22: 1x1. B21: 1x4. C21: 1x4.
    # U rows: [24]
    u_map_11 = [24]
    # V rows: v_map_14
    # W rows: w_map_14
    
    for k in range(4):
        U[u_map_11[0], current_rank] = 1
        V[v_map_14[k], current_rank] = 1
        W[w_map_14[k], current_rank] = 1
        current_rank += 1
        
    # 7. A21*B12 (1x4x1)
    # A21: 1x4. B12: 4x1. C22: 1x1.
    # U rows: u_map_14
    # V rows: v_map_41
    # W rows: [24]
    w_map_11 = [24]
    
    for j in range(4):
        U[u_map_14[j], current_rank] = 1
        V[v_map_41[j], current_rank] = 1
        W[w_map_11[0], current_rank] = 1
        current_rank += 1
        
    # 8. A22*B22 (1x1x1)
    # U rows: [24]
    # V rows: [24]
    # W rows: [24]
    
    U[u_map_11[0], current_rank] = 1
    V[v_map_11[0], current_rank] = 1
    W[w_map_11[0], current_rank] = 1
    current_rank += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
