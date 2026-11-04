"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    rank = 22
    U = [[0]*rank for _ in range(n*m)]  # 6 x 22
    V = [[0]*rank for _ in range(m*p)]  # 18 x 22
    W = [[0]*rank for _ in range(p*n)]  # 12 x 22
    
    col = 0
    
    # Helper: add a rank-1 component
    def add_rank1(u_vals, v_vals, w_vals):
        nonlocal col
        for idx, val in u_vals.items():
            U[idx][col] += val
        for idx, val in v_vals.items():
            V[idx][col] += val
        for idx, val in w_vals.items():
            W[idx][col] += val
        col += 1
    
    # Strassen for [2,2,2]: A is 2x2, B is 2x2, C is 2x2
    # A indices: (i,j) -> U[i*2+j]
    # B indices: (j,k) -> V[j*2+k]
    # C indices: (k,i) -> W[k*2+i]
    def strassen_222(u_base, v_base, w_base):
        # u_base: (i,j) -> offset in U
        # v_base: (j,k) -> offset in V
        # w_base: (k,i) -> offset in W
        # M1 = (A00+A11)*(B00+B11)
        add_rank1(
            {u_base(0,0): 1, u_base(1,1): 1},
            {v_base(0,0): 1, v_base(1,1): 1},
            {}
        )
        # Wait, I need to be more careful. The W values come from the combination.
        # Actually, for Strassen, each M is a product, and then C is a combination of M's.
        # So I need to handle this differently.
        
        # Let me restructure. For each Strassen product, I create a rank component
        # where U and V have the appropriate values, and W is zero.
        # Then at the end, I need to add the W contributions.
        # 
        # Actually, the standard way: each rank-1 component has U, V, W all non-zero.
        # For Strassen, the 7 products are intermediate, and the final C is a linear
        # combination. So I can't directly map Strassen to rank-1 components of the
        # original tensor.
        #
        # Let me think differently. The Strassen decomposition says:
        # C = sum_{r=1}^{7} U_r * V_r * W_r
        # where each (U_r, V_r, W_r) is a rank-1 tensor.
        #
        # For [2,2,2], the 7 rank-1 components are:
        # r=1: U = [1,0,0,1] (A00+A11), V = [1,0,0,1] (B00+B11), W = [1,1,1,1] (C00+C01+C10+C11)... 
        # No, that's not right either.
        #
        # Let me look up the exact Strassen rank-1 decomposition.
        # 
        # Actually, I think the way to think about it is:
        # Each Strassen "multiplication" M_r = X_r * Y_r where X_r is a linear combo of A entries
        # and Y_r is a linear combo of B entries. Then C entries are linear combos of M_r's.
        # 
        # To express this as a rank decomposition of the tensor, I need:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[i,k,r]
        #
        # For Strassen [2,2,2]:
        # The 7 rank-1 components can be written as:
        # 
        # Let me use the standard Strassen formulas and derive the rank-1 decomposition.
        #
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a01)b11
        # M6 = (a10-a00)(b00+b10)
        # M7 = (a01-a11)(b10+b11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Now, each M_r = sum_{i,j} X_r[i,j] * A[i,j] * sum_{j',k} Y_r[j',k] * B[j',k]
        # where X_r and Y_r are the coefficient matrices.
        #
        # For the tensor decomposition, I need:
        # sum_r U[i,j,r] * V[j,k,r] * W[i',k',r] = [i=i'] * [j=j'] * [k=k']
        #
        # This is getting complex. Let me just directly construct the rank-1 components.
        
        pass
    
    # Let me take a different approach and directly construct the rank-1 components
    # for each sub-problem.
    
    # For [2,2,2] Strassen, the 7 rank-1 components are:
    # I'll use the known decomposition.
    
    # Actually, let me just use the schoolbook for [2,2,1] and [2,1,3] and
    # a direct construction for [2,2,2] Strassen.
    
    # Let me restart with a cleaner approach.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
