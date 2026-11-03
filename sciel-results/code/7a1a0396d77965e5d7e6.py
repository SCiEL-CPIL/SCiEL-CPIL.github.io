"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Tensor slicing strategy: partition output dim p into parts
    # Try partition p=5 into sizes (2,3) and (3,2), and also (1,4), (4,1)
    # For each partition, solve for minimal rank via linear algebra
    
    deadline = time.time() + 120
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (i,k) entry of C
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a,b,c] = 1 iff a%m == b//p (the inner index matches)
    
    nm = n * m  # 8
    mp = m * p  # 20
    pn = p * n  # 10
    
    # Build tensor as a 3D array
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # We want to find rank-r decomposition: T = sum_{r} u_r ⊗ v_r ⊗ w_r
    # This is equivalent to: for each slice c, T[:,:,c] = sum_r u_r * (v_r * w_r[c])
    # Or: the tensor can be viewed as a matrix of size (nm*mp) x pn via flattening
    # But that doesn't directly give rank.
    
    # Alternative approach: use the fact that matrix multiplication tensor has special structure.
    # For fixed output columns k_set, we have a sub-tensor.
    
    # Strategy: partition the p output columns into two groups.
    # Group 1: columns K1, Group 2: columns K2
    # The sub-tensor for group 1 involves B rows and output columns in K1.
    # We can try to find a low-rank decomposition of each sub-tensor.
    
    # Actually, let's use a more direct approach:
    # The tensor T has the property that T[a,b,c] depends only on whether a%m == b//p.
    # 
    # Key insight: We can use the "border rank" or "slice rank" approach.
    # For a given partition of the output dimension, we can try to minimize the rank.
    
    # Let's try a different approach: use known optimal decompositions.
    # For 2x4x5, the schoolbook rank is 2*4*5 = 40.
    # Using Strassen-style blocking: split m=4 into two 2s, and p=5 into parts.
    
    # Current best: rank 36 using two 2x2x5 blocks, each with two 2x2x2 Strassen (7) + one 2x2x1 schoolbook (4) = 18 per block, total 36.
    
    # Can we do better? Let's try to use a smarter decomposition.
    # 
    # Idea: Instead of splitting m into two 2x2 blocks, try other block sizes.
    # Or: use a 2x4x2 Strassen-like decomposition for the first two output columns,
    # and handle the remaining 3 columns differently.
    
    # Let's try: partition p into (2, 3).
    # For the first 2 output columns: we have a 2x4x2 tensor.
    # For the last 3 output columns: we have a 2x4x3 tensor.
    
    # The 2x4x2 tensor: split m=4 into two 2x2 blocks.
    # Each 2x2x2 block can be done in rank 7 (Strassen).
    # Total for 2x4x2: 7 + 7 = 14.
    
    # The 2x4x3 tensor: split m=4 into two 2x2 blocks.
    # Each 2x2x3 block: we need to multiply 2x2 by 2x3.
    # Schoolbook: 2*2*3 = 12 per block, total 24.
    # Can we do better? 
    # 2x2x3: split the 3 columns into 2+1.
    # 2x2x2 (Strassen, rank 7) + 2x2x1 (schoolbook, rank 4) = 11 per block.
    # Total for 2x4x3: 11 + 11 = 22.
    
    # Grand total: 14 + 22 = 36. Same as before.
    
    # Let's try a different partition: p into (3, 2).
    # 2x4x3: 11 + 11 = 22 (as above)
    # 2x4x2: 7 + 7 = 14
    # Total: 36. Same.
    
    # What if we use a non-block-diagonal decomposition?
    # The key is that the m dimension (inner product) is shared across all output columns.
    # 
    # Alternative: use the fact that for 2x4x5, we can potentially share some rank-1 terms
    # across different output columns.
    
    # Let's try to find the minimal rank using a more systematic approach.
    # 
    # The tensor T can be viewed as a matrix M of size (nm*mp) x pn by flattening the first two indices.
    # M[a*mp + b, c] = T[a, b, c]
    # The rank of M is an upper bound on the tensor rank.
    
    M = T.reshape(nm * mp, pn)
    rank_M = np.linalg.matrix_rank(M)
    # rank_M is the "slice rank" or a related quantity.
    
    # Actually, the tensor rank is at least the rank of any matrix flattening.
    # Let's compute the rank of different flattenings.
    
    # Flattening 1: (nm*mp) x pn
    r1 = np.linalg.matrix_rank(M)
    
    # Flattening 2: (nm*pn) x mp
    M2 = T.reshape(nm, pn, mp).transpose(0, 2, 1).reshape(nm * pn, mp)
    r2 = np.linalg.matrix_rank(M2)
    
    # Flattening 3: (mp*pn) x nm
    M3 = T.reshape(nm, mp, pn).transpose(1, 2, 0).reshape(mp * pn, nm)
    r3 = np.linalg.matrix_rank(M3)
    
    # The tensor rank is at least max(r1, r2, r3).
    # For the standard matrix multiplication tensor, the tensor rank is known to be
    # at least the geometric mean of the three flattening ranks.
    
    # Let's try to find a decomposition of rank max(r1, r2, r3) or close to it.
    
    # Use SVD to get a low-rank approximation, then round to integers.
    
    # For flattening 1: M = U1 @ S1 @ V1^T
    # This gives M[a*mp+b, c] = sum_r U1[a*mp+b, r] * V1[c, r]
    # But we need T[a,b,c] = sum_r u[a,r] * v[b,r] * w[c,r], which is more constrained.
    
    # The flattening approach doesn't directly give a tensor decomposition.
    
    # Let's stick with the blocking approach but try to optimize it.
    
    # Current: rank 36.
    # Can we reduce by using a smarter decomposition of 2x2x3?
    # 
    # 2x2x3: multiply 2x2 by 2x3.
    # Let A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # C = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # 
    # Schoolbook: 12 multiplications.
    # Can we do it in fewer?
    # 
    # Try: 
    # m1 = (a+d)(e+h) = ae + ah + de + dh
    # m2 = (a+b)(e+f) = ae + af + be + bf
    # m3 = (c+d)(e+h) = ce + ch + de + dh
    # m4 = (c+d)(g+i) = cg + ci + dg + di
    # m5 = (a+b)(g+i) = ag + ai + bg + bi
    # m6 = (c-a)(e-h) = ce - ch - ae + ah
    # m7 = (d-b)(f-i) = df - di - bf + bi
    # ... this is getting complicated.
    
    # Known result: the tensor rank of 2x2x3 is 11 (I believe).
    # Let me verify: for 2x2xn, the rank is known to be 2n-1 for n>=2? No.
    # Actually, for 2x2x2 it's 7, for 2x2x3 it should be around 11-12.
    
    # Let's just use the known decomposition and see if we can get rank 35 or lower.
    
    # Try: use Strassen for 2x2x2 parts, and a custom 2x2x1 part.
    # Current: 7+7+4 = 18 per 2x2x5 block, total 36.
    
    # What if we use a different split of p=5?
    # Split into 1+1+1+1+1: each 2x2x1 is rank 4, total 5*4 = 20 per block, total 40. Worse.
    # Split into 2+1+2: 7+4+7 = 18 per block, total 36. Same.
    # Split into 3+2: 11+7 = 18 per block, total 36. Same.
    
    # It seems 36 is the best we can get with this blocking strategy.
    
    # Let's try a completely different approach: use the fact that n=2 is small.
    # For each pair of output columns (k1, k2), we have a 2x4x2 sub-tensor.
    # The full tensor is the sum over all pairs? No, that's not right.
    
    # Actually, let's just return the rank-36 decomposition we already have,
    # but implement it more cleanly and verify it.
    
    rank = 36
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    
    def add_rank1_term(u_dict, v_dict, w_dict):
        nonlocal col
        if col >= rank:
            raise ValueError("Rank exceeded")
        for idx, val in u_dict.items():
            U[idx][col] = val
        for idx, val in v_dict.items():
            V[idx][col] = val
        for idx, val in w_dict.items():
            W[idx][col] = val
        col += 1
    
    # Strassen for 2x2 * 2x2
    def add_strassen_block(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local // 2) * p + (b_base_j + local % 2)
        def map_c(local):
            i = c_base_i + local // 2
            k = c_base_j + local % 2
            return k * n + i
        
        # M1 = (A00+A11)(B00+B11)
        u = {map_a(0): 1, map_a(3): 1}
        v = {map_b(0): 1, map_b(3): 1}
        w = {map_c(0): 1, map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M2 = (A10+A11)B00
        u = {map_a(2): 1, map_a(3): 1}
        v = {map_b(0): 1}
        w = {map_c(2): 1, map_c(3): -1}
        add_rank1_term(u, v, w)
        
        # M3 = A00(B01-B11)
        u = {map_a(0): 1}
        v = {map_b(1): 1, map_b(3): -1}
        w = {map_c(1): 1, map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M4 = A11(B10-B00)
        u = {map_a(3): 1}
        v = {map_b(2): 1, map_b(0): -1}
        w = {map_c(0): 1, map_c(2): 1}
        add_rank1_term(u, v, w)
        
        # M5 = (A00+A01)B11
        u = {map_a(0): 1, map_a(1): 1}
        v = {map_b(3): 1}
        w = {map_c(0): -1, map_c(1): 1}
        add_rank1_term(u, v, w)
        
        # M6 = (A10-A00)(B00+B01)
        u = {map_a(2): 1, map_a(0): -1}
        v = {map_b(0): 1, map_b(1): 1}
        w = {map_c(3): 1}
        add_rank1_term(u, v, w)
        
        # M7 = (A01-A11)(B10+B11)
        u = {map_a(1): 1, map_a(3): -1}
        v = {map_b(2): 1, map_b(3): 1}
        w = {map_c(0): 1}
        add_rank1_term(u, v, w)
    
    # Schoolbook for 2x2 * 2x1
    def add_schoolbook_2x2_1(a_base_i, a_base_j, b_base_i, b_base_j, c_base_i, c_base_j):
        def map_a(local):
            return (a_base_i + local // 2) * m + (a_base_j + local % 2)
        def map_b(local):
            return (b_base_i + local) * p + b_base_j
        def map_c(local):
            i = c_base_i + local
            k = c_base_j
            return k * n + i
        
        for l in range(4):
            a_local = l
            b_local = l % 2
            c_local = l // 2
            u = {map_a(a_local): 1}
            v = {map_b(b_local): 1}
            w = {map_c(c_local): 1}
            add_rank1_term(u, v, w)
    
    # Block 1: A1 (rows 0,1; cols 0,1), B1 (rows 0,1; cols 0,4)
    add_strassen_block(0, 0, 0, 0, 0, 0)
    add_strassen_block(0, 0, 0, 2, 0, 2)
    add_schoolbook_2x2_1(0, 0, 0, 4, 0, 4)
    
    # Block 2: A2 (rows 0,1; cols 2,3), B2 (rows 2,3; cols 0,4)
    add_strassen_block(0, 2, 2, 0, 0, 0)
    add_strassen_block(0, 2, 2, 2, 0, 2)
    add_schoolbook_2x2_1(0, 2, 2, 4, 0, 4)
    
    # Verify the decomposition
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: T[a,b,c] should equal sum_r U[a,r] * V[b,r] * W[c,r]
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    if not np.allclose(T_recon, T, atol=1e-10):
        # Fallback: use schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][i * m * p + j * p + k] = 1
                    V[b][i * m * p + j * p + k] = 1
                    W[c][i * m * p + j * p + k] = 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
