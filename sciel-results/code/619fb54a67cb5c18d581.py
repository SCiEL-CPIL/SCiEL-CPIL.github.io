import time
import sys

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 189
    rank = 189
    
    # Initialize U, V, W
    # U: (36, 189), V: (36, 189), W: (36, 189)
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    # Schoolbook decomposition for 3x3
    # Indices i, j, k in 0..2
    # u[i,j] = 1, v[j,k] = 1, w[k,i] = 1
    # We store these as lists of tuples (i, j, k)
    schoolbook_terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                schoolbook_terms.append((i, j, k))
                
    # Strassen definitions
    # alpha: {(b_i, b_j): coeff}
    # beta: {(b_j, b_k): coeff}
    # delta: {(b_i, b_k): coeff}
    
    strassen_terms = [
        {
            'alpha': {(0,0): 1, (1,1): 1},
            'beta':  {(0,0): 1, (1,1): 1},
            'delta': {(0,0): 1, (1,0): 1, (1,1): -1} # M1: C00+, C10+, C11- (Check)
        },
        {
            'alpha': {(1,0): 1, (1,1): 1},
            'beta':  {(0,0): 1},
            'delta': {(0,1): 1, (1,1): -1} # M2: C01+, C11- (Check)
        },
        {
            'alpha': {(0,0): 1},
            'beta':  {(0,1): 1, (1,1): -1},
            'delta': {(0,0): -1, (0,1): -1, (1,0): 1} # M3: C00-, C01-, C10+ (Check)
        },
        {
            'alpha': {(1,1): 1},
            'beta':  {(1,0): 1, (0,0): -1},
            'delta': {(0,0): -1, (1,1): 1} # M4: C00-, C11+ (Check)
        },
        {
            'alpha': {(0,0): 1, (0,1): 1},
            'beta':  {(1,1): 1},
            'delta': {(0,1): -1, (1,0): -1} # M5: C01-, C10- (Check)
        },
        {
            'alpha': {(1,0): 1, (0,0): -1},
            'beta':  {(0,0): 1, (0,1): 1},
            'delta': {(0,1): -1} # M6: C01- (Check)
        },
        {
            'alpha': {(0,1): 1, (1,1): -1},
            'beta':  {(1,0): 1, (1,1): 1},
            'delta': {(0,0): 1, (1,0): -1} # M7: C00+, C10- (Check)
        }
    ]
    
    col = 0
    for t in strassen_terms:
        alpha = t['alpha']
        beta = t['beta']
        delta = t['delta']
        
        for (i, j, k) in schoolbook_terms:
            # i, j, k are local indices 0..2
            for (b_i, b_j), a_val in alpha.items():
                for (b_j2, b_k), b_val in beta.items():
                    # b_j must match for valid multiplication? 
                    # In Strassen, the inner dimension j matches.
                    # The blocks are A_{b_i b_j} and B_{b_j' b_k}.
                    # The virtual multiplication is X Y.
                    # X is composed of A blocks. Y is composed of B blocks.
                    # The term X_{ij} Y_{jk} sums over j.
                    # Here i, j, k are the local indices of the 3x3 multiplication.
                    # So the A block is A_{b_i b_j} and B block is B_{b_j b_k}?
                    # No, the Strassen terms define linear combinations of blocks.
                    # The "virtual" matrix X has entries X_{ij} = sum A_{b_i b_j, ij}.
                    # The "virtual" matrix Y has entries Y_{jk} = sum B_{b_j' b_k, jk}.
                    # The product is sum X_{ij} Y_{jk}.
                    # This sums over the *local* index j.
                    # But the block indices b_j and b_j' must correspond?
                    # Actually, the Strassen algorithm treats the blocks as scalar variables.
                    # So A_{00} and B_{00} are multiplied together.
                    # The local index j is summed over.
                    # So for a fixed local j, we pick A block (b_i, b_j) and B block (b_j, b_k).
                    # So b_j in alpha must match b_j' in beta.
                    
                    if b_j != b_j2: continue
                    
                    w_val = a_val * b_val
                    if w_val == 0: continue
                    
                    # Now determine the W coefficient
                    # The result Z_{ik} is added to C blocks according to delta.
                    # Z_{ik} corresponds to local indices i, k.
                    # So we need to sum delta[(b_i, b_k)]?
                    # No, the Strassen term M_t produces a single 3x3 matrix Z_t.
                    # Z_t is then added to C_{b_i b_k} with coefficient delta_{b_i b_k}.
                    # But wait, Z_t is the result of the multiplication.
                    # The decomposition of the multiplication gives terms contributing to Z_{ik}.
                    # So for each (i, k), the coefficient in W is delta[(b_i, b_k)]?
                    # No, b_i and b_k are the block indices of the *result*?
                    # In Strassen, M_t is a matrix product of two "virtual" matrices.
                    # The result is a single 3x3 matrix.
                    # This matrix is added to specific C blocks.
                    # The mapping from Z_t to C blocks is linear.
                    # C_{b_i b_k} += delta_{b_i b_k} * Z_t.
                    # So for a term contributing to Z_t[i, k], the global W coefficient
                    # for block (b_i, b_k) is delta_{b_i b_k} * w_val?
                    # No, the w_val is for the A and B parts.
                    # The W part depends on which C block the result goes to.
                    # But the result Z_t[i, k] contributes to ALL C blocks that M_t adds to.
                    # So for each (b_i_res, b_k_res) in delta, we have a contribution.
                    
                    for (b_i_res, b_k_res), d_val in delta.items():
                         if d_val == 0: continue
                         
                         # Wait, does b_i_res have to match b_i from alpha?
                         # No. M_t is a full matrix.
                         # But in Strassen, the block structure is preserved?
                         # M_1 = (A00+A11)(B00+B11).
                         # Result is a 3x3 matrix.
                         # This result is added to C00, C10, C11.
                         # The entry (i, k) of M_1 is added to C00[i, k], C10[i, k], etc.
                         # So yes, b_i_res and b_k_res are independent of the input block indices?
                         # No, the Strassen algorithm outputs 4 blocks C00, C01, C10, C11.
                         # Each is a linear combination of the M_t matrices.
                         # So for a fixed M_t, the result Z_t is added to multiple C blocks.
                         # The local indices i, k are the same for all of them.
                         
                         # So we need to create a rank-1 term for EACH (b_i_res, b_k_res)?
                         # No, the rank-1 term is for the scalar product.
                         # The scalar product result is a scalar.
                         # This scalar is added to C[b_i_res, b_k_res][i, k].
                         # So for each (b_i_res, b_k_res), we add to W.
                         # But U and V are fixed for this (i, j, k, b_i, b_j, b_j2, b_k).
                         # So we would be adding the SAME U and V vectors to multiple columns?
                         # No, each (b_i_res, b_k_res) defines a different output location.
                         # So we need a separate rank-1 term for each output block?
                         # That would increase the rank!
                         
                         # Let's re-evaluate.
                         # Rank of M_t decomposition is 27.
                         # If we need to add Z_t to 3 C blocks, do we need 3 * 27 = 81 terms?
                         # No. The linearity is in the output.
                         # The term is U V W.
                         # If W has non-zero entries in multiple places, it's still one rank-1 term.
                         # W is a vector. It can have multiple non-zero entries.
                         # So yes, W_{row, col} can be non-zero for multiple rows.
                         # The row index is 6*k + i (using k for col, i for row? No, W[k*n+i]).
                         # Global index for C[b_i_res, b_k_res][i, k] is:
                         # Row I = 3 b_i_res + i.
                         # Col K = 3 b_k_res + k.
                         # W index = 6 K + I = 6 (3 b_k_res + k) + (3 b_i_res + i).
                         
                         # So for a fixed scalar term (fixed i, j, k, b_i, b_j, b_k, w_val),
                         # we have one column c.
                         # U is filled based on b_i, b_j, i, j, a_val.
                         # V is filled based on b_j, b_k, j, k, b_val.
                         # W is filled based on b_i_res, b_k_res, i, k, d_val.
                         # We iterate over b_i_res, b_k_res in delta.
                         # And add d_val to W[global_w_index, c].
                         
                         global_w_index = 6 * (3 * b_k_res + k) + (3 * b_i_res + i)
                         W[global_w_index][col] += d_val
                        
                # End for b_j2, b_k
            # End for b_i, b_j
            col += 1
            
    return {"U": U, "V": W, "W": W} # Wait, V is V.

# Re-verify the logic for W accumulation.
# For a specific scalar term (i, j, k, b_i, b_j, b_k), we have one column c.
# U is set to a_val * (1 at global_u_index).
# V is set to b_val * (1 at global_v_index).
# W is set to sum of d_val * (1 at global_w_index) for each (b_i_res, b_k_res) in delta.
# This is correct.

# Let's refine the code structure to be safe.

def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(36)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]
    
    schoolbook_terms = [(i, j, k) for i in range(3) for j in range(3) for k in range(3)]
    
    strassen_terms = [
        {
            'alpha': {(0,0): 1, (1,1): 1},
            'beta':  {(0,0): 1, (1,1): 1},
            'delta': {(0,0): 1, (1,0): 1, (1,1): -1}
        },
        {
            'alpha': {(1,0): 1, (1,1): 1},
            'beta':  {(0,0): 1},
            'delta': {(0,1): 1, (1,1): -1}
        },
        {
            'alpha': {(0,0): 1},
            'beta':  {(0,1): 1, (1,1): -1},
            'delta': {(0,0): -1, (0,1): -1, (1,0): 1}
        },
        {
            'alpha': {(1,1): 1},
            'beta':  {(1,0): 1, (0,0): -1},
            'delta': {(0,0): -1, (1,1): 1}
        },
        {
            'alpha': {(0,0): 1, (0,1): 1},
            'beta':  {(1,1): 1},
            'delta': {(0,1): -1, (1,0): -1}
        },
        {
            'alpha': {(1,0): 1, (0,0): -1},
            'beta':  {(0,0): 1, (0,1): 1},
            'delta': {(0,1): -1}
        },
        {
            'alpha': {(0,1): 1, (1,1): -1},
            'beta':  {(1,0): 1, (1,1): 1},
            'delta': {(0,0): 1, (1,0): -1}
        }
    ]
    
    col = 0
    for t in strassen_terms:
        alpha = t['alpha']
        beta = t['beta']
        delta = t['delta']
        
        for (i, j, k) in schoolbook_terms:
            for (b_i, b_j), a_val in alpha.items():
                for (b_j2, b_k), b_val in beta.items():
                    if b_j != b_j2:
                        continue
                    
                    u_index = 18 * b_i + 6 * i + 3 * b_j + j
                    v_index = 18 * b_j2 + 6 * j + 3 * b_k + k
                    
                    U[u_index][col] = a_val
                    V[v_index][col] = b_val
                    
                    for (b_i_res, b_k_res), d_val in delta.items():
                        w_index = 6 * (3 * b_k_res + k) + (3 * b_i_res + i)
                        W[w_index][col] = d_val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
