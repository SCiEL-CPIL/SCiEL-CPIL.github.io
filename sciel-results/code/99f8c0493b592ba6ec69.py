import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    deadline = time.time() + 140.0
    R = 38
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()

    def add_term(u, v, w, col):
        for a_idx, val in u.items():
            U_list[a_idx][col] += val
        for b_idx, val in v.items():
            V_list[b_idx][col] += val
        for c_idx, val in w.items():
            W_list[c_idx][col] += val

    def a_idx(i, j):
        return i * 4 + j

    def b_idx(j, k):
        return j * 8 + k

    def c_idx(i, k):
        return k * 4 + i

    # 19 rank-1 terms for 4x4x4 matrix multiplication
    # Each term: (sum of A entries) * (sum of B entries) -> weighted sum of C entries
    terms = [
        # Term 0: (A00+A11+A22+A33)(B00+B11+B22+B33)
        # Actually, let me use a known rank-19 decomposition.
        # For 4x4x4, the known minimal rank is 19.
        # I'll construct it using the 7x7 Strassen recursion but with a known rank-19 construction.
        
        # Actually, let me use a simpler approach: I'll hardcode a known rank-19 decomposition.
        # The 4x4x4 tensor of rank 19 can be built from the 2x2x2 rank-7 Strassen applied
        # with clever sharing. But implementing this from scratch is complex.
        
        # Instead, let me use the fact that rank(4x4x4) = 19 is known and construct
        # a valid decomposition programmatically.
        #
        # Approach: Use the known result that 4x4x4 has rank 19.
        # I'll implement a specific rank-19 construction.
    ]
    
    # Let me use a different, more reliable approach.
    # I'll implement the Strassen-based decomposition which gives rank 49 for 4x4x4,
    # then apply it to both halves for total rank 98.
    # But first, let me try to get rank 38 by using a rank-19 decomposition.
    
    # Known rank-19 decomposition of 4x4x4 matmul tensor:
    # This is complex to hardcode. Let me use a computational approach.
    
    # Alternative: Use the fact that for 4x4x8, we can potentially do better than 98
    # by exploiting structure. But without a known rank-19 formula, I'll implement
    # the Strassen rank-49 for each 4x4x4 block, giving total rank 98.
    
    # Let me implement Strassen 2x2x2 correctly and recurse.
    
    def strassen_2x2x2():
        """
        Returns the 7 rank-1 terms for 2x2x2 Strassen.
        Each term is (U_dict, V_dict, W_dict) where keys are flattened indices.
        For 2x2x2: A is 2x2, B is 2x2, C is 2x2.
        U indices: i*2+j, V indices: j*2+k, W indices: k*2+i
        """
        results = []
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C01 += M1, C10 += M1, C11 += M1
        u = {0: 1, 3: 1}
        v = {0: 1, 3: 1}
        w = {0: 1, 1: 1, 2: 1, 3: 1}
        results.append((u, v, w))
        
        # M2 = (A10+A11)(B00)
        # C10 += M2, C11 += M2
        u = {2: 1, 3: 1}
        v = {0: 1}
        w = {2: 1, 3: 1}
        results.append((u, v, w))
        
        # M3 = (A00)(B10-B11)
        # C01 += M3, C11 += M3
        u = {0: 1}
        v = {2: 1, 3: -1}
        w = {1: 1, 3: 1}
        results.append((u, v, w))
        
        # M4 = (A11)(B01-B00)
        # C00 += M4, C10 += M4
        u = {3: 1}
        v = {1: 1, 0: -1}
        w = {0: 1, 2: 1}
        results.append((u, v, w))
        
        # M5 = (A00+A01)(B11)
        # C00 -= M5, C01 += M5
        u = {0: 1, 1: 1}
        v = {3: 1}
        w = {0: -1, 1: 1}
        results.append((u, v, w))
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        u = {2: 1, 0: -1}
        v = {0: 1, 1: 1}
        w = {3: 1}
        results.append((u, v, w))
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7, C10 += M7
        u = {1: 1, 3: -1}
        v = {2: 1, 3: 1}
        w = {0: 1, 2: 1}
        results.append((u, v, w))
        
        return results
    
    # Build 4x4x4 decomposition from Strassen recursion
    # This gives rank 49 for 4x4x4
    def build_4x4x4_strassen():
        """
        Build rank-49 decomposition for 4x4x4 using Strassen recursion.
        Returns list of 49 (U_dict, V_dict, W_dict) tuples with global indices.
        """
        base_2x2 = strassen_2x2x2()
        all_terms = []
        
        # For 4x4x4, we decompose into 7 intermediate 2x2x2 multiplications
        # P1 = (A00+A11)(B00+B11) -> 2x2 result
        # P2 = (A10+A11)(B00) -> 2x2 result  
        # P3 = (A00)(B10-B11) -> 2x2 result
        # P4 = (A11)(B01-B00) -> 2x2 result
        # P5 = (A00+A01)(B11) -> 2x2 result
        # P6 = (A10-A00)(B00+B01) -> 2x2 result
        # P7 = (A01-A11)(B10+B11) -> 2x2 result
        
        # Where A00, A01, A10, A11 are the 2x2 quadrants of the 4x4 matrix A
        # A = [[A00, A01], [A10, A11]]
        # Similarly for B and C.
        
        # Each Pk is a 2x2x2 multiplication that we decompose with rank 7.
        # So total rank = 7 * 7 = 49.
        
        # For each of the 7 products, we need to define how the input 2x2 blocks
        # map to the full 4x4 matrices.
        
        # Let me define the 7 products more carefully.
        # A quadrants: A00=[0:2,0:2], A01=[0:2,2:4], A10=[2:4,0:2], A11=[2:4,2:4]
        # B quadrants: B00=[0:2,0:2], B01=[0:2,2:4], B10=[2:4,0:2], B11=[2:4,2:4]
        # C quadrants: C00=[0:2,0:2], C01=[0:2,2:4], C10=[2:4,0:2], C11=[2:4,2:4]
        
        # P1 = (A00+A11)(B00+B11)
        # P2 = (A10+A11)(B00)
        # P3 = (A00)(B10-B11)
        # P4 = (A11)(B01-B00)
        # P5 = (A00+A01)(B11)
        # P6 = (A10-A00)(B00+B01)
        # P7 = (A01-A11)(B10+B11)
        
        # C00 = P1 + P4 - P5 + P7
        # C01 = P3 + P5
        # C10 = P2 + P4
        # C11 = P1 - P2 + P3 + P6
        
        # For each product Pk, it's a 2x2x2 matmul.
        # The "A" part of Pk is a linear combination of 2x2 blocks of A.
        # The "B" part of Pk is a linear combination of 2x2 blocks of B.
        # The result is added to certain 2x2 blocks of C.
        
        # Let me define each product's contribution.
        
        products = [
            # (A_blocks_with_coeffs, B_blocks_with_coeffs, C_blocks_with_coeffs)
            # A_blocks: list of (quadrant_idx, coeff) where quadrant_idx 0=A00,1=A01,2=A10,3=A11
            # B_blocks: list of (quadrant_idx, coeff)
            # C_blocks: list of (quadrant_idx, coeff)
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),  # P1
            ([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)]),  # P2
            ([(0,1)], [(2,1),(3,-1)], [(1,1),(3,1)]),  # P3
            ([(3,1)], [(1,1),(0,-1)], [(0,1),(2,1)]),  # P4
            ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),  # P5
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),  # P6
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(2,1)]),  # P7
        ]
        
        for prod_idx, (a_blocks, b_blocks, c_blocks) in enumerate(products):
            # For each of the 7 rank-1 terms in the 2x2x2 Strassen decomposition
            for term_idx, (u_base, v_base, w_base) in enumerate(base_2x2):
                u_full = {}
                v_full = {}
                w_full = {}
                
                # Build U: for each (i,j) in the 2x2 block, multiply by the A coefficient
                for (aq, a_coeff) in a_blocks:
                    for (local_ij, u_val) in u_base.items():
                        li, lj = local_ij // 2, local_ij % 2
                        # Map to global A indices
                        if aq == 0:  # A00
                            gi, gj = li, lj
                        elif aq == 1:  # A01
                            gi, gj = li, lj + 2
                        elif aq == 2:  # A10
                            gi, gj = li + 2, lj
                        else:  # A11
                            gi, gj = li + 2, lj + 2
                        gidx = gi * 4 + gj
                        u_full[gidx] = u_full.get(gidx, 0) + u_val * a_coeff
                
                # Build V: for each (j,k) in the 2x2 block, multiply by the B coefficient
                for (bq, b_coeff) in b_blocks:
                    for (local_jk, v_val) in v_base.items():
                        lj, lk = local_jk // 2, local_jk % 2
                        # Map to global B indices
                        if bq == 0:  # B00
                            gj, gk = lj, lk
                        elif bq == 1:  # B01
                            gj, gk = lj, lk + 2
                        elif bq == 2:  # B10
                            gj, gk = lj + 2, lk
                        else:  # B11
                            gj, gk = lj + 2, lk + 2
                        gidx = gj * 8 + gk
                        v_full[gidx] = v_full.get(gidx, 0) + v_val * b_coeff
                
                # Build W: for each (k,i) in the 2x2 block, multiply by the C coefficient
                for (cq, c_coeff) in c_blocks:
                    for (local_ki, w_val) in w_base.items():
                        lk, li = local_ki // 2, local_ki % 2
                        # Map to global C indices
                        if cq == 0:  # C00
                            gk, gi = lk, li
                        elif cq == 1:  # C01
                            gk, gi = lk, li + 2
                        elif cq == 2:  # C10
                            gk, gi = lk + 2, li
                        else:  # C11
                            gk, gi = lk + 2, li + 2
                        gidx = gk * 4 + gi
                        w_full[gidx] = w_full.get(gidx, 0) + w_val * c_coeff
                
                all_terms.append((u_full, v_full, w_full))
        
        return all_terms
    
    # Now build the full 4x4x8 decomposition
    # We have two independent 4x4x4 multiplications:
    # C[:, 0:4] = A * B[:, 0:4]  (first half)
    # C[:, 4:8] = A * B[:, 4:8]  (second half)
    
    terms_4x4 = build_4x4x4_strassen()
    
    # For the first half (k=0..3), use terms as-is
    # For the second half (k=4..7), shift k indices by 4
    
    all_terms_full = []
    for u, v, w in terms_4x4:
        # First half
        u1 = dict(u)
        v1 = {}
        for k_idx, val in v.items():
            # k_idx = j*8 + k, for first half k < 4
            j = k_idx // 8
            k = k_idx % 8
            v1[k_idx] = val
        w1 = dict(w)
        all_terms_full.append((u1, v1, w1))
        
        # Second half: shift k by 4
        u2 = dict(u)
        v2 = {}
        for k_idx, val in v.items():
            j = k_idx // 8
            k = k_idx % 8
            new_k_idx = j * 8 + (k + 4)
            v2[new_k_idx] = val
        w2 = {}
        for w_idx, val in w.items():
            # w_idx = k*4 + i, for second half k >= 4
            k = w_idx // 4
            i = w_idx % 4
            new_w_idx = (k + 4) * 4 + i
            w2[new_w_idx] = val
        all_terms_full.append((u2, v2, w2))
    
    # Total rank = 49 * 2 = 98
    R = len(all_terms_full)
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for col, (u, v, w) in enumerate(all_terms_full):
        if time.time() > deadline:
            break
        for idx, val in u.items():
            U[idx, col] += val
        for idx, val in v.items():
            V[idx, col] += val
        for idx, val in w.items():
            W[idx, col] += val
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
