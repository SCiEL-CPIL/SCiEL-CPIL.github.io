import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    
    # We will construct a rank-56 decomposition.
    # Rank 28 for the first 4 columns of C, Rank 28 for the last 4 columns.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Helper to add a term to the first block (cols 0-3)
    def add_term_1(u_idx, u_val, v_idx, v_val, w_idx, w_val, col_offset):
        U[u_idx, col_offset] += u_val
        V[v_idx, col_offset] += v_val
        W[w_idx, col_offset] += w_val

    # Helper to add a term to the second block (cols 4-7)
    # Note: V index for second block is j*p + (k+4)
    # W index for second block is (k+4)*n + i
    def add_term_2(u_idx, u_val, v_idx, v_val, w_idx, w_val, col_offset):
        U[u_idx, col_offset] += u_val
        V[v_idx, col_offset] += v_val
        W[w_idx, col_offset] += w_val

    # Define the 28 terms for 4x4x4
    # Indices for 4x4 matrices:
    # A[i, j] -> idx = i*4 + j
    # B[j, k] -> idx = j*4 + k
    # C[i, k] -> idx = k*4 + i (Trace convention: W[k*n+i])
    
    # Strassen forms for 2x2 blocks
    # Block (bi, bj) of A covers rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # A[2*bi+u, 2*bj+v]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M7
    # C11 = M1 - M3 + M6 - M5
    
    # For each M_s, we compute the 4 scalar products of the 2x2 matrices.
    # Let U_s and V_s be the 2x2 matrices of linear combos.
    # U_s[a, t] = sum of coeff * A[2*bi+a, 2*bj+t]
    # V_s[t, b] = sum of coeff * B[2*bi+t, 2*bj+b]
    
    # We define the blocks for U and V for each Strassen form.
    # U_blocks[s] is a list of (bi, bj, coeff)
    # V_blocks[s] is a list of (bi, bj, coeff)
    
    U_blocks = [
        [(0,0,1), (1,1,1)],      # M1
        [(1,0,1), (1,1,1)],      # M2
        [(0,0,1)],                # M3
        [(1,1,1)],                # M4
        [(0,0,1), (0,1,1)],      # M5
        [(1,0,1), (0,0,-1)],     # M6
        [(0,1,1), (1,1,-1)],     # M7
    ]
    
    V_blocks = [
        [(0,0,1), (1,1,1)],      # M1
        [(0,0,1)],                # M2
        [(1,0,1), (1,1,-1)],     # M3
        [(0,1,1), (1,0,-1)],     # M4
        [(1,1,1)],                # M5
        [(0,0,1), (1,0,1)],      # M6
        [(0,1,1), (1,1,1)],      # M7
    ]
    
    # C contributions
    # C_block (ci, cj) gets contribution from M_s with coeff c_s
    C_contrib = {
        (0,0): [(0,1), (3,-1), (4,1), (6,-1)],
        (0,1): [(2,1), (3,1)],
        (1,0): [(1,1), (6,1)],
        (1,1): [(0,1), (2,-1), (5,1), (4,-1)],
    }
    
    col = 0
    for s in range(7):
        # For each of the 4 scalar multiplications in the 2x2 product M_s
        # M_s[a, b] = sum_t U_s[a, t] * V_s[t, b]
        # We treat each term U_s[a, t] * V_s[t, b] as a separate rank-1 term?
        # No, the rank of the 2x2 matrix multiplication tensor is 4 (schoolbook).
        # So we add 4 terms for each M_s.
        # Term index within M_s: t (0..1) and b (0..1)?
        # Actually, the schoolbook decomposition of 2x2x2 has 4 terms:
        # (0,0), (0,1), (1,0), (1,1) corresponding to (a, t, b) where t is the summation index?
        # No, the 4 terms are:
        # 1. A[0,0]B[0,0] -> C[0,0]
        # 2. A[0,1]B[1,0] -> C[0,0]
        # 3. A[0,0]B[0,1] -> C[0,1]
        # 4. A[0,1]B[1,1] -> C[0,1]
        # 5. A[1,0]B[0,0] -> C[1,0]
        # 6. A[1,1]B[1,0] -> C[1,0]
        # 7. A[1,0]B[0,1] -> C[1,1]
        # 8. A[1,1]B[1,1] -> C[1,1]
        # Wait, that's 8 terms.
        # The rank of 2x2x2 is 7 (Strassen).
        # But we are using schoolbook for the inner 2x2 products?
        # If we use schoolbook for the inner 2x2 products, we have 4 scalar mults per M_s.
        # Total 28.
        
        # Let's stick to the 4 scalar mults per M_s.
        # The 4 terms correspond to the 4 entries of the product matrix M_s.
        # M_s[0,0] = U_s[0,0]V_s[0,0] + U_s[0,1]V_s[1,0]
        # This is a sum of 2 products.
        # So we need 2 rank-1 terms for M_s[0,0].
        # Total 8 terms per M_s? 7*8=56.
        # That gives rank 56 for 4x4x4.
        # Then 112 for 4x4x8.
        
        # I want rank 28 for 4x4x4.
        # This implies using Strassen for the inner 2x2 products as well?
        # No, Strassen is for 2x2.
        # If we use Strassen for the inner 2x2, we have 7 terms per M_s.
        # 7*7 = 49.
        
        # Let's just use the rank-28 decomposition which is known to exist.
        # The rank-28 decomposition uses 28 scalar multiplications.
        # It is constructed by applying Strassen to the 4x4 blocks.
        # The 4x4 matrix is viewed as 2x2 blocks of 2x2.
        # The 7 Strassen products are of 2x2 matrices.
        # Each 2x2 matrix product is done with 4 scalar multiplications (schoolbook).
        # So 7 * 4 = 28.
        
        # So for each M_s, we have 4 terms.
        # The 4 terms are the 4 scalar multiplications in the schoolbook 2x2 product.
        # Let's identify them.
        # M_s = U_s V_s.
        # M_s[0,0] uses U_s[0,0]V_s[0,0] and U_s[0,1]V_s[1,0].
        # These are 2 distinct products.
        # So M_s[0,0] contributes 2 rank-1 terms.
        # M_s[0,1] contributes 2 rank-1 terms.
        # M_s[1,0] contributes 2 rank-1 terms.
        # M_s[1,1] contributes 2 rank-1 terms.
        # Total 8 terms per M_s.
        # 7 * 8 = 56 terms.
        
        # This gives rank 56 for 4x4x4.
        # Total rank 112 for 4x4x8.
        # 128/112 = 1.14. Still an improvement.
        
        # I will implement this rank-56 decomposition for 4x4x4.
        
        for a in range(2):
            for b in range(2):
                # M_s[a, b] = sum_t U_s[a, t] V_s[t, b]
                for t in range(2):
                    # Term: U_s[a, t] * V_s[t, b]
                    # This is one rank-1 term.
                    
                    # Construct U vector
                    u_vec = {}
                    for (bi, bj, coeff) in U_blocks[s]:
                        # U_s[a, t] involves A[2*bi+a, 2*bj+t]
                        idx_u = (2*bi+a)*4 + (2*bj+t)
                        u_vec[idx_u] = u_vec.get(idx_u, 0) + coeff
                    
                    # Construct V vector
                    v_vec = {}
                    for (bi, bj, coeff) in V_blocks[s]:
                        # V_s[t, b] involves B[2*bi+t, 2*bj+b]
                        idx_v = (2*bi+t)*4 + (2*bj+b)
                        v_vec[idx_v] = v_vec.get(idx_v, 0) + coeff
                    
                    # Construct W vector
                    # Contribution to C[a, b] from M_s[a, b]
                    # C_contrib[(a, b)] gives (s, coeff) pairs
                    w_vec = {}
                    for (cs, coeff_c) in C_contrib[(a, b)]:
                        if cs == s:
                            # C[a, b] is a 2x2 block.
                            # C_block(a, b) covers rows 2*a..2*a+1, cols 2*b..2*b+1
                            # The term U_s[a, t] V_s[t, b] contributes to C[2*a+u, 2*b+v]
                            # Wait, M_s is a 2x2 matrix.
                            # M_s[a, b] is a scalar.
                            # It contributes to C[2*a, 2*b]?
                            # No, M_s is the product of two 2x2 matrices.
                            # M_s[u, v] is a scalar.
                            # M_s[u, v] contributes to C[2*ci+u, 2*cj+v].
                            # Here u=a, v=b.
                            # So it contributes to C[2*a, 2*b], C[2*a, 2*b+1], etc?
                            # No, M_s[u, v] is a single scalar.
                            # It adds to C[2*ci+u, 2*cj+v] for the specific block (ci, cj).
                            # The block (ci, cj) is determined by the Strassen combination.
                            # C00 = M1 - M4 + M5 - M7.
                            # So M1[0,0] contributes to C00[0,0] which is C[0,0].
                            # M1[0,1] contributes to C00[0,1] which is C[0,1].
                            # M1[1,0] contributes to C00[1,0] which is C[1,0].
                            # M1[1,1] contributes to C00[1,1] which is C[1,1].
                            
                            # So for term (s, a, b, t):
                            # It contributes to C[2*a, 2*b] with coeff coeff_c.
                            # Wait, M_s[a, b] is the element at (a, b) of the 2x2 matrix M_s.
                            # So it contributes to C[2*a, 2*b] of the block (ci, cj).
                            # But which block?
                            # The block (ci, cj) is the one where M_s appears in C_contrib.
                            # C_contrib[(ci, cj)] contains (s, coeff).
                            # So for a given (a, b), we look at C_contrib[(a, b)]?
                            # No, C_contrib is indexed by the C-block (ci, cj).
                            # M_s is a 2x2 matrix.
                            # M_s contributes to C_block(ci, cj) with coeff coeff_c.
                            # So M_s[u, v] contributes to C[2*ci+u, 2*cj+v].
                            
                            # So for term (s, a, b, t):
                            # It is part of M_s[a, b].
                            # It contributes to C[2*ci+a, 2*cj+b] for each (ci, cj) where M_s appears.
                            
                            for (ci, cj) in C_contrib:
                                if (s, coeff_c) in C_contrib[(ci, cj)]:
                                    # Find coeff_c for this s
                                    for (ss, cc) in C_contrib[(ci, cj)]:
                                        if ss == s:
                                            idx_w = (2*cj+b)*4 + (2*ci+a) # W[k*n+i] -> k=2*cj+b, i=2*ci+a
                                            w_vec[idx_w] = w_vec.get(idx_w, 0) + cc
                                    break # Only one coeff per s per block? Yes.
                    
                    # Add term to U, V, W
                    # First block (cols 0-3)
                    for idx, val in u_vec.items():
                        U[idx, col] = val
                    for idx, val in v_vec.items():
                        V[idx, col] = val
                    for idx, val in w_vec.items():
                        W[idx, col] = val
                    
                    # Second block (cols 4-7)
                    # U is same
                    # V index: j*8 + k. Here k is 0..3 in the 4x4 block. In 4x8, k is 4..7.
                    # So v_idx_2 = idx_v + 4? No.
                    # V[idx_v] corresponds to B[j, k].
                    # In the second block, B is B[:, 4:8].
                    # So B[j, k+4].
                    # Index in V for 4x8 is j*8 + (k+4).
                    # idx_v is j*4 + k.
                    # j = idx_v // 4, k = idx_v % 4.
                    # new_idx_v = j*8 + k + 4.
                    
                    for idx, val in u_vec.items():
                        U[idx, col + 28] = val
                    for idx, val in v_vec.items():
                        j = idx // 4
                        k = idx % 4
                        new_idx_v = j * 8 + k + 4
                        V[new_idx_v, col + 28] = val
                    for idx, val in w_vec.items():
                        # W index: k*n + i.
                        # In second block, k is 4..7.
                        # idx_w is k*4 + i.
                        # k = idx_w // 4, i = idx_w % 4.
                        # new_idx_w = (k+4)*4 + i.
                        k = idx // 4
                        i = idx % 4
                        new_idx_w = (k + 4) * 4 + i
                        W[new_idx_w, col + 28] = val
                    
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
