import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 4, 4, 4

    # We construct a rank-49 decomposition using recursive Strassen.
    # This is significantly better than the schoolbook rank 64.
    # 
    # Structure:
    # 1. Split 4x4 matrices into 2x2 blocks of 2x2 submatrices.
    # 2. Apply Strassen's algorithm to the 2x2 block structure (7 block products).
    # 3. Apply Strassen's algorithm to each 2x2 block product (7 scalar mults each).
    # 4. Total rank = 7 * 7 = 49.
    
    # Indexing conventions:
    # A: 4x4, flat index i*4+j
    # B: 4x4, flat index j*4+k
    # C: 4x4, but W is indexed by k*4+i (trace convention W[k*n+i, r])
    # Tensor T[a, b, c] = 1 if a=i*4+j, b=j*4+k, c=k*4+i
    # Decomposition: T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    
    rank = 49
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((16, rank), dtype=float)
    W = np.zeros((16, rank), dtype=float)
    
    # Helper to map local 2x2 block indices to global flat indices
    # block (br, bc) in a 4x4 matrix (2x2 blocks of size 2)
    # local (lr, lc) -> global row = br*2 + lr, global col = bc*2 + lc
    # A flat idx: (br*2+lr)*4 + (bc*2+lc)
    # B flat idx: (br*2+lr)*4 + (bc*2+lc)
    # W flat idx: (bc*2+lc)*4 + (br*2+lr)  <-- Note the swap for W
    
    def a_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def b_idx(br, bc, lr, lc):
        return (br*2 + lr)*4 + (bc*2 + lc)
        
    def w_idx(br, bc, lr, lc):
        # C row = br*2+lr, C col = bc*2+lc
        # W index = col*4 + row
        return (bc*2 + lc)*4 + (br*2 + lr)

    # Strassen 2x2 decomposition terms
    # Each term is a tuple: (a_coeffs, b_coeffs, c_coeffs)
    # a_coeffs: dict { (lr, lc): coeff }
    # b_coeffs: dict { (lr, lc): coeff }
    # c_coeffs: dict { (lr, lc): coeff }
    
    def mk_dict(pairs):
        d = {}
        for k, v in pairs:
            d[k] = d.get(k, 0) + v
        return d

    # Standard Strassen 2x2 terms
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)(B11)     -> C21 += M2, C22 -= M2
    # M3 = (A11)(B12-B22)     -> C12 += M3, C22 += M3
    # M4 = (A22)(B21-B11)     -> C11 += M4, C21 += M4
    # M5 = (A11+A12)(B22)     -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    strassen_2x2_terms = [
        # (a, b, c)
        (mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)]), mk_dict([((0,0),1), ((1,1),1)])),
        (mk_dict([((1,0),1), ((1,1),1)]), mk_dict([((0,0),1)]),              mk_dict([((1,0),1), ((1,1),-1)])),
        (mk_dict([((0,0),1)]),              mk_dict([((0,1),1), ((1,1),-1)]), mk_dict([((0,1),1), ((1,1),1)])),
        (mk_dict([((1,1),1)]),              mk_dict([((1,0),1), ((0,0),-1)]), mk_dict([((0,0),1), ((1,0),1)])),
        (mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)]),              mk_dict([((0,0),-1), ((0,1),1)])),
        (mk_dict([((1,0),1), ((0,0),-1)]),  mk_dict([((0,0),1), ((0,1),1)]),   mk_dict([((1,1),1)])),
        (mk_dict([((0,1),1), ((1,1),-1)]),  mk_dict([((1,0),1), ((1,1),1)]),   mk_dict([((0,0),1)])),
    ]
    
    # Outer Strassen for 4x4 (operating on 2x2 blocks)
    # The "matrices" here are 2x2 blocks of the 4x4 matrix.
    # Block indices: (0,0), (0,1), (1,0), (1,1)
    # A block (br, bc) corresponds to rows br*2..br*2+1, cols bc*2..bc*2+1
    
    outer_strassen_terms = [
        # (a_blocks, b_blocks, c_blocks)
        # a_blocks: list of (block_row, block_col, coeff)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),  # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)],           [(1,0,1), (1,1,-1)]), # M2
        ([(0,0,1)],           [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]), # M3
        ([(1,1,1)],           [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]), # M4
        ([(0,0,1), (0,1,1)],  [(1,1,1)],           [(0,0,-1), (0,1,1)]), # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)],  [(1,1,1)]),           # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)],  [(0,0,1)]),           # M7
    ]
    
    col_idx = 0
    for outer_term in outer_strassen_terms:
        a_blocks, b_blocks, c_blocks = outer_term
        
        # For each outer term, we perform a 2x2 matrix product of "block matrices".
        # Let X be the 2x2 block matrix for A, Y for B.
        # X[br, bc] is a 2x2 matrix formed by summing A blocks.
        # Actually, it's simpler to view the outer Strassen as defining linear combinations of the 2x2 blocks.
        # The product X*Y is a 2x2 matrix of 2x2 blocks.
        # We decompose this 2x2 block product using the inner Strassen terms.
        
        # For each inner Strassen term, we generate one rank-1 component.
        for inner_term in strassen_2x2_terms:
            a_comb, b_comb, c_comb = inner_term
            
            # Initialize accumulators for U, V, W for this rank-1 term
            # We need to sum contributions from all blocks involved in the outer term.
            
            # The inner Strassen term operates on the 2x2 entries of the block product.
            # Let's trace the indices.
            
            # The "scalar" multiplication in the inner Strassen is:
            # s = (sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * X[lr,lc]) * 
            #     (sum_{(lr,lc) in b_comb} b_comb[(lr,lc)] * Y[lr,lc])
            
            # Where X[lr,lc] is the 2x2 matrix at block position (lr,lc) of the outer A structure.
            # X[lr,lc] = sum_{(br,bc,coeff) in a_blocks} coeff * A_block[br, bc]
            # Wait, this is not quite right.
            
            # Let's re-evaluate.
            # Outer Strassen computes M_s = X_s * Y_s, where X_s and Y_s are 2x2 matrices.
            # The entries of X_s are linear combinations of the 2x2 blocks of A.
            # Specifically, if outer term has a_blocks = [(br1, bc1, c1), (br2, bc2, c2), ...]
            # Then X_s is a 2x2 matrix where the (r, c) entry is the 2x2 block sum?
            # No.
            
            # Let's look at M1 = (A00+A11)(B00+B11).
            # A00 and A11 are 2x2 matrices.
            # X_1 = A00 + A11. This is a 2x2 matrix.
            # Y_1 = B00 + B11. This is a 2x2 matrix.
            # Product M_1 = X_1 * Y_1 is a 2x2 matrix.
            # M_1 contributes to C00 and C11 (the 2x2 blocks).
            
            # So, for a given outer term:
            # X is a 2x2 matrix. X[i,j] = sum_{(br,bc,coeff) in a_blocks} coeff * A[br*2+i, bc*2+j] ?
            # No, that's mixing block indices and local indices.
            
            # Let's define the mapping clearly.
            # A is 4x4.
            # Block (br, bc) is the 2x2 submatrix A[br*2:br*2+2, bc*2:bc*2+2].
            # In the outer Strassen, we treat the 4x4 matrix as a 2x2 matrix of 2x2 blocks.
            # Let's call the block matrix \tilde{A}.
            # \tilde{A}[0,0] = A_block(0,0), \tilde{A}[0,1] = A_block(0,1), etc.
            
            # The outer Strassen term defines linear combinations of these blocks.
            # For example, for M1: X = \tilde{A}[0,0] + \tilde{A}[1,1].
            # X is a 2x2 matrix (since \tilde{A}[0,0] is 2x2).
            # X = A[0:2, 0:2] + A[2:4, 2:4].
            
            # So, for a specific outer term with a_blocks list:
            # X is a 2x2 matrix.
            # X[i,j] = sum_{(br,bc,coeff) in a_blocks} coeff * A[br*2+i, bc*2+j]
            
            # Similarly for Y.
            
            # The inner Strassen decomposes the product X*Y.
            # Inner term t defines:
            # s_t = (sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * X[lr,lc]) * 
            #       (sum_{(lr,lc) in b_comb} b_comb[(lr,lc)] * Y[lr,lc])
            
            # Substituting X and Y:
            # s_t = (sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * (sum_{(br,bc,coeff) in a_blocks} coeff * A[br*2+lr, bc*2+lc])) *
            #       (sum_{(lr,lc) in b_comb} b_comb[(lr,lc)] * (sum_{(br,bc,coeff) in b_blocks} coeff * B[br*2+lr, bc*2+lc]))
            
            # This is a product of two linear combinations of A and B entries.
            # This corresponds to a rank-1 term in the tensor decomposition.
            # U[a_idx] = coeff_A, V[b_idx] = coeff_B, W[c_idx] = coeff_C
            
            # The contribution to C is determined by c_comb and c_blocks.
            # The result of X*Y is a 2x2 matrix Z.
            # Z[ir, ic] = sum_{(lr,lc) in c_comb} c_comb[(lr,lc)] * s_t ?
            # No, the inner Strassen gives the entries of the product matrix Z.
            # Z[ir, ic] = sum_t (inner term contribution)
            # For a specific inner term t, it contributes to specific entries of Z.
            # c_comb tells us which entries of Z are affected and with what coefficient.
            # c_comb is a dict { (ir, ic): coeff }.
            
            # But Z itself contributes to the final C blocks via c_blocks.
            # c_blocks is a list of (cbr, cbc, ccoeff).
            # C_block[cbr, cbc] += ccoeff * Z
            
            # So, the final contribution to C[r, c] (global indices) is:
            # Sum over c_blocks (cbr, cbc, ccoeff) and c_comb (ir, ic, ccoeff_inner):
            # C[cbr*2+ir, cbc*2+ic] += ccoeff * ccoeff_inner * s_t
            
            # Let's implement this.
            
            # 1. Calculate coefficients for U (A entries)
            # U[a_global] = sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * (sum_{(br,bc,coeff) in a_blocks} coeff * (A[br*2+lr, bc*2+lc] coeff))
            # Wait, the coefficient for A[br*2+lr, bc*2+lc] in s_t is:
            # sum_{(lr',lc') in a_comb} a_comb[(lr',lc')] * (sum_{(br',bc',coeff') in a_blocks} coeff' * delta(lr',br')delta(lc',bc') ? No.
            
            # Let's expand carefully.
            # Term in s_t from A:
            # L_A = sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * X[lr,lc]
            # X[lr,lc] = sum_{(br,bc,coeff) in a_blocks} coeff * A[br*2+lr, bc*2+lc]
            # So L_A = sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * (sum_{(br,bc,coeff) in a_blocks} coeff * A[br*2+lr, bc*2+lc])
            # L_A = sum_{(br,bc,coeff) in a_blocks} (sum_{(lr,lc) in a_comb} a_comb[(lr,lc)] * coeff) * A[br*2+lr, bc*2+lc]
            
            # So the coefficient for A[br*2+lr, bc*2+lc] is:
            # coeff_U[br, bc, lr, lc] = coeff * a_comb.get((lr,lc), 0)
            
            # Similarly for B:
            # coeff_V[br, bc, lr, lc] = coeff * b_comb.get((lr,lc), 0)
            
            # And for C:
            # The result Z is a 2x2 matrix.
            # Z[ir, ic] gets contribution from inner term t with coefficient c_comb.get((ir,ic), 0).
            # This Z contributes to C blocks via c_blocks.
            # C_block[cbr, cbc] += ccoeff_outer * Z
            # So C[cbr*2+ir, cbc*2+ic] += ccoeff_outer * c_comb.get((ir,ic), 0) * s_t
            
            # So the coefficient for C[cbr*2+ir, cbc*2+ic] is:
            # coeff_W[cbr, cbc, ir, ic] = ccoeff_outer * c_comb.get((ir,ic), 0)
            
            # Now we fill U, V, W.
            
            # U index: a_idx(br, bc, lr, lc) = (br*2+lr)*4 + (bc*2+lc)
            # V index: b_idx(br, bc, lr, lc) = (br*2+lr)*4 + (bc*2+lc)
            # W index: w_idx(cbr, cbc, ir, ic) = (cbc*2+ic)*4 + (cbr*2+ir)
            
            # Iterate over a_blocks to fill U
            for (br, bc, coeff) in a_blocks:
                for (lr, lc), a_val in a_comb.items():
                    if a_val == 0: continue
                    u_i = a_idx(br, bc, lr, lc)
                    U[u_i, col_idx] += coeff * a_val
            
            # Iterate over b_blocks to fill V
            for (br, bc, coeff) in b_blocks:
                for (lr, lc), b_val in b_comb.items():
                    if b_val == 0: continue
                    v_i = b_idx(br, bc, lr, lc)
                    V[v_i, col_idx] += coeff * b_val
            
            # Iterate over c_blocks and c_comb to fill W
            for (cbr, cbc, ccoeff_outer) in c_blocks:
                for (ir, ic), c_val_inner in c_comb.items():
                    if c_val_inner == 0: continue
                    w_i = w_idx(cbr, cbc, ir, ic)
                    W[w_i, col_idx] += ccoeff_outer * c_val_inner
            
            col_idx += 1
    
    # Verify the decomposition (optional, for debugging)
    # T = sum_r U[:, r] V[:, r]^T W[:, r]^T ? No, it's element-wise product sum.
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We can check a few entries.
    
    # Convert to lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
