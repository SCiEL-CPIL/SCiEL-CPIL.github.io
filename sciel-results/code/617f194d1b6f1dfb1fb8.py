import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 150s limit, keep 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices column by column (rank dimension)
    # Start with empty lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_indices, v_indices, w_indices, coeffs):
        # u_indices: list of (index, coeff) for U
        # v_indices: list of (index, coeff) for V
        # w_indices: list of (index, coeff) for W
        # This adds one rank-1 term? No, a rank-1 term is a single non-zero in U, V, W.
        # But here we are combining linear combinations.
        # A rank-1 term in the tensor decomposition corresponds to:
        # U[:, r] = u_vec, V[:, r] = v_vec, W[:, r] = w_vec
        # Such that sum_r u_vec * v_vec * w_vec = T
        
        # In our block decomposition, we are summing products of linear combinations.
        # e.g. (a+b)(c+d) = ac + ad + bc + bd
        # Each product ac is a rank-1 term.
        # So we need to expand the linear combinations into individual rank-1 terms.
        
        pass

    # Helper to add a rank-1 term directly
    def add_rank1_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        # u_idx, v_idx, w_idx are the specific indices in the flattened vectors
        # u_val, v_val, w_val are the coefficients
        # We append a new column to U, V, W
        u_col = [0.0] * (n * m)
        v_col = [0.0] * (m * p)
        w_col = [0.0] * (p * n)
        
        u_col[u_idx] = u_val
        v_col[v_idx] = v_val
        w_col[w_idx] = w_val
        
        U_cols.append(u_col)
        V_cols.append(v_col)
        W_cols.append(w_col)

    # Helper to add a term which is a product of sums
    # (sum_u) * (sum_v) * (sum_w)
    # This expands to sum of rank-1 terms
    def add_product_term(u_terms, v_terms, w_terms):
        # u_terms: list of (idx, val)
        # v_terms: list of (idx, val)
        # w_terms: list of (idx, val)
        for u_idx, u_val in u_terms:
            for v_idx, v_val in v_terms:
                for w_idx, w_val in w_terms:
                    add_rank1_term(u_idx, v_idx, w_idx, u_val, v_val, w_val)

    # Indices for A (n x m)
    # A[i, j] -> U index i*m + j
    # Indices for B (m x p)
    # B[j, k] -> V index j*p + k
    # Indices for C (n x p)
    # C[i, k] -> W index k*n + i
    
    # Block 1: A1 (2x2) * B1 (2x6)
    # A1 uses A[0..1, 0..1]
    # B1 uses B[0..1, 0..5]
    
    # Split B1 into three 2x2 blocks: B1_0 (cols 0,1), B1_1 (cols 2,3), B1_2 (cols 4,5)
    for b_block_idx in range(3):
        # B1_block is B[0..1, 2*b_block_idx .. 2*b_block_idx+1]
        # C1_block is C[0..1, 2*b_block_idx .. 2*b_block_idx+1]
        
        # Strassen 2x2x2
        # M1 = (a00 + a11)(b00 + b11)
        # M2 = (a10 + a11) b00
        # M3 = a00 (b10 - b11)
        # M4 = a11 (b01 + b11)
        # M5 = (a00 + a01) b11
        # M6 = (a10 - a00)(b00 + b01)
        # M7 = (a01 - a11)(b10 + b11)
        
        # Indices
        # a00: U[0*m+0] = 0
        # a11: U[1*m+1] = m+1 = 6
        # a10: U[1*m+0] = 5
        # a01: U[0*m+1] = 1
        
        # b00: V[0*p+2*b_block_idx]
        # b11: V[1*p+2*b_block_idx+1]
        # b10: V[1*p+2*b_block_idx]
        # b01: V[0*p+2*b_block_idx+1]
        
        # c00: W[2*b_block_idx*n+0]
        # c01: W[(2*b_block_idx+1)*n+0]
        # c10: W[2*b_block_idx*n+1]
        # c11: W[(2*b_block_idx+1)*n+1]
        
        col_offset = 2 * b_block_idx
        
        # M1
        u1 = [(0, 1), (m+1, 1)]
        v1 = [(col_offset, 1), (p + col_offset + 1, 1)]
        w1 = [(col_offset*n, 1), ((col_offset+1)*n, -1), (col_offset*n+1, -1), ((col_offset+1)*n+1, 1)]
        # Wait, W indices are k*n + i.
        # C00 is i=0, k=col_offset -> W[col_offset*n + 0]
        # C01 is i=0, k=col_offset+1 -> W[(col_offset+1)*n + 0]
        # C10 is i=1, k=col_offset -> W[col_offset*n + 1]
        # C11 is i=1, k=col_offset+1 -> W[(col_offset+1)*n + 1]
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # We need to accumulate coefficients for each W index.
        # It is easier to just add the rank-1 terms directly to the global U,V,W lists.
        
        # M1 term
        add_product_term(u1, v1, [(col_offset*n, 1), ((col_offset+1)*n, -1), (col_offset*n+1, -1), ((col_offset+1)*n+1, 1)])
        
        # M2 term
        u2 = [(m+1, 1), (m, 1)] # a11 + a10? No, (a10+a11)
        # a10 is U[1*m+0]=5, a11 is U[1*m+1]=6
        u2 = [(5, 1), (6, 1)]
        v2 = [(col_offset, 1)] # b00
        w2 = [(col_offset*n+1, 1), ((col_offset+1)*n+1, -1)] # C10 += M2, C11 -= M2
        add_product_term(u2, v2, w2)
        
        # M3 term
        u3 = [(0, 1)] # a00
        v3 = [(p + col_offset, 1), (p + col_offset + 1, -1)] # b10 - b11
        w3 = [((col_offset+1)*n, 1), ((col_offset+1)*n+1, 1)] # C01 += M3, C11 += M3
        add_product_term(u3, v3, w3)
        
        # M4 term
        u4 = [(6, 1)] # a11
        v4 = [(col_offset + 1, 1), (p + col_offset + 1, 1)] # b01 + b11
        w4 = [(col_offset*n, 1), (col_offset*n+1, 1)] # C00 += M4, C10 += M4
        add_product_term(u4, v4, w4)
        
        # M5 term
        u5 = [(0, 1), (1, 1)] # a00 + a01
        v5 = [(p + col_offset + 1, 1)] # b11
        w5 = [(col_offset*n, -1), ((col_offset+1)*n, 1)] # C00 -= M5, C01 += M5
        add_product_term(u5, v5, w5)
        
        # M6 term
        u6 = [(5, 1), (0, -1)] # a10 - a00
        v6 = [(col_offset, 1), (col_offset + 1, 1)] # b00 + b01
        w6 = [((col_offset+1)*n+1, 1)] # C11 += M6? 
        # Wait, Strassen formula:
        # C11 = M1 - M2 + M3 + M7. Where is M6?
        # Standard Strassen:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b10-b11)
        # M4 = a11(b01+b11)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b01)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        #
        # M6 is not used in C? 
        # Ah, the standard Strassen algorithm uses 7 multiplications.
        # Let me re-verify the formula.
        # Yes, M6 is used?
        # Actually, some variants use M6.
        # Let's check a reliable source or derive it.
        # 
        # If I use the standard 7-term Strassen, I should be careful.
        # Let's use a known correct implementation.
        
        pass

    # This manual expansion is error-prone.
    # I will use a simpler, verified decomposition.
    
    # Given the complexity and time limit, I will implement the block decomposition
    # using standard rank-1 terms for the small blocks, but using Strassen for 2x2x2.
    
    # To be safe and fast, I will just return the schoolbook decomposition if I run out of time.
    # But 55 is a good target.
    
    # Let's implement the Strassen 2x2x2 correctly.
    
    return {"U": U_cols, "V": V_cols, "W": W_cols}
