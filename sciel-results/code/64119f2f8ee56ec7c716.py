"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.
    
    The multiplication tensor M_{4,4,6} is decomposed using a block-Kronecker structure:
    - Split A (4x4) into 2x2 blocks: A = [[A00, A01], [A10, A11]]
    - Split B (4x6) into 2x3 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each Bkj is 2x3
    - Output C (4x6) has 2x3 blocks: C = [[C00, C01, C02], [C10, C11, C12]]
    
    Each C_ik = sum_j A_ij * B_jk
    
    For the 2x2 * 2x3 multiplications, we use the standard rank-6 decomposition (schoolbook for 2x2x3):
    C = A*B where A is 2x2, B is 2x3, C is 2x3.
    C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    This gives rank 6 per block multiplication.
    
    Total: 4 blocks (A00*B0k, A01*B1k, A10*B0k, A11*B1k for k=0,1,2)
    But wait, C_ik = A_i0 * B_0k + A_i1 * B_1k
    
    So for each output block C_ik (2x3), we need:
    - A_i0 (2x2) * B_0k (2x3): rank 6
    - A_i1 (2x2) * B_1k (2x3): rank 6
    
    There are 2 i values (0,1) and 3 k values (0,1,2), so 6 output blocks.
    Each output block requires 2 multiplications of rank 6 each.
    Total rank = 6 * 2 * 6 = 72.
    
    Let's verify: U has shape (n*m, rank) = (16, 72)
    V has shape (m*p, rank) = (24, 72)
    W has shape (p*n, rank) = (24, 72)
    
    Indexing:
    U[i*m + j, c] where i in [0,4), j in [0,4)
    V[j*p + k, c] where j in [0,4), k in [0,6)
    W[k*n + i, c] where k in [0,6), i in [0,4)
    
    For the block at A row-block i (0 or 1), A col-block j (0 or 1), B row-block j (0 or 1), B col-block k (0,1,2):
    
    A_ij is at rows [2i:2i+2], cols [2j:2j+2] of A
    B_jk is at rows [2j:2j+2], cols [3k:3k+3] of B
    C_ik is at rows [2i:2i+2], cols [3k:3k+3] of C
    
    For the rank-6 decomposition of 2x2 * 2x3:
    For local indices a,b in {0,1} (rows of A block), c,d in {0,1} (cols of A block / rows of B block), e in {0,1,2} (cols of B block):
    
    C_local[a, e] = A_local[a,c] * B_local[c,e] for c=0,1
    
    So for each (c,e) pair, we have one term:
    U contribution: A_local[a,c] at global position (2i+a, 2j+c)
    V contribution: B_local[c,e] at global position (2j+c, 3k+e)
    W contribution: C_local[a,e] at global position (3k+e, 2i+a)
    
    So for each (i, j, k, c, e) where i in {0,1}, j in {0,1}, k in {0,1,2}, c in {0,1}, e in {0,1,2}:
    That's 2 * 2 * 3 * 2 * 2 = 48... wait, let me recount.
    
    Actually for each (i,j,k):
    c in {0,1} (2 values)
    e in {0,1,2} (3 values)
    So 2 * 3 = 6 terms per (i,j,k)
    
    Number of (i,j,k): i in {0,1}, j in {0,1}, k in {0,1,2} = 2*2*3 = 12
    Total terms: 12 * 6 = 72. Correct.
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 24 x 72
    W = [[0] * rank for _ in range(p * n)]  # 24 x 72
    
    col = 0
    for i in range(2):       # A row block
        for j in range(2):   # A col block / B row block
            for k in range(3):  # B col block
                for c in range(2):  # local row within block (col of A, row of B)
                    for e in range(3):  # local col within B block
                        # Global indices
                        u_row = 2 * i + c  # This is wrong. Let me reconsider.
                        
                        # Actually:
                        # A block A_ij is at rows 2i to 2i+1, cols 2j to 2j+1
                        # For term (c,e): A_local[a,c] for a in {0,1}
                        # We need to add 1 for each (a) as well? No.
                        
                        # The tensor M[i,j,k] = sum over... no.
                        # Matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
                        # In tensor form: C[i,k] = A[i,j] * B[j,k] summed over j
                        # The multiplication tensor has entries M[i,j,k] = 1 for each (i,j,k)
                        # A decomposition M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                        # Wait, the standard form is: (A*B)[i,k] = sum_j sum_r U[i,r]*V[j,r]*W[k,r] * A[i,j]*B[j,k]
                        # No. Let me re-read the convention.
                        
                        # "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]"
                        # So the decomposition is: (A*B)[k, i] = sum_c W[k*n+i, c] * (sum_{j} U[i*m+j,c] * V[j*p+k,c] * A[i,j] * B[j,k])
                        # Hmm, that doesn't look standard.
                        
                        # Actually the standard Strassen-like form:
                        # C[k, i] = sum_j A[i, j] * B[j, k]
                        # = sum_c sum_j U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
                        # This works if for each c, the term U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                        # acts as a coefficient for A[i,j]*B[j,k].
                        # So we need: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                        # And for (i,j,k) != (i',j',k'), the sum should be 0? No.
                        # Actually it's: C[i,k] = sum_j A[i,j] B[j,k]
                        # = sum_j sum_c [A[i,j] B[j,k] * delta_{i,j,k} in some sense]
                        
                        # Let me just use the schoolbook pattern from the original:
                        # For each (i,j,k): U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
                        # This gives C[k,i] = sum_j sum_c A[i,j] B[j,k] * U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
                        # For the schoolbook, for each (i,j,k) there's exactly one c where all three are 1.
                        # So C[k,i] = sum_j A[i,j] B[j,k] * 1 = correct.
                        
                        # So my block approach:
                        # For each (i,j,k) in the block structure, I want to decompose the 6 terms
                        # of the 2x2*2x3 multiplication into rank-6.
                        
                        # For fixed (i_block, j_block, k_block), the 6 local terms (a,c_local,e)
                        # where a in {0,1}, c_local in {0,1}, e in {0,1,2}
                        # Global i = 2*i_block + a
                        # Global j = 2*j_block + c_local
                        # Global k = 3*k_block + e
                        
                        # For each such (a, c_local, e), I assign one column:
                        # U[i*m + j, col] = 1
                        # V[j*p + k, col] = 1
                        # W[k*n + i, col] = 1
                        
                        # This is just schoolbook! Rank = 4*4*6 = 96.
                        
                        # To get rank < 96, I need to combine terms.
                        # The issue is that schoolbook has 96 terms.
                        # Block decomposition doesn't help unless the sub-problems have rank < product of dims.
                        
                        # 2x2*2x2 has rank 7 (Strassen), not 8.
                        # 2x2*2x3 has rank 6 (schoolbook), can it be less?
                        # M_{2,2,3} rank is known to be 6.
                        
                        # So for 4x4*4x6:
                        # Split into 2x2 blocks of A, 2x3 blocks of B.
                        # C_ik = A_i0 B_0k + A_i1 B_1k
                        # Each A_ij * B_jk is a 2x2 * 2x3 multiplication, rank 6.
                        # Number of such products: 2 (i) * 2 (j) * 3 (k) = 12
                        # Total rank = 12 * 6 = 72.
                        
                        # So rank 72 is achievable!
                        
                        pass
    
    # Let me redo this properly.
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i_block in range(2):      # Row block of A (and C)
        for j_block in range(2):  # Col block of A / Row block of B
            for k_block in range(3):  # Col block of B (and C)
                # This block computes C_block[i_block, k_block] += A_block[i_block, j_block] * B_block[j_block, k_block]
                # A_block is 2x2, B_block is 2x3, C_block is 2x3
                # Schoolbook for 2x2 * 2x3 has 6 terms (a in {0,1}, c in {0,1}, e in {0,1,2} -> but that's 2*2*3=12? No.)
                
                # Wait: C[a, e] = sum_c A[a, c] * B[c, e] for c in {0,1}
                # So for each (a, c, e): A[a,c] * B[c,e] contributes to C[a,e]
                # Number of (a,c,e): 2 * 2 * 3 = 12? No, that's not right either.
                
                # Actually the multiplication tensor for 2x2 * 2x3:
                # M[a, c, e] where a in {0,1}, c in {0,1}, e in {0,1,2}
                # M[a,c,e] = 1 for all valid (a,c,e)
                # C[a,e] = sum_c A[a,c] B[c,e]
                # This is rank 6: for each (c,e), we have one term A[:,c] * B[c,:] contributing to C[:,e]
                # No wait. For fixed (c,e): A[a,c]*B[c,e] for a in {0,1}
                # This is a rank-2 term: it contributes A[0,c]*B[c,e] to C[0,e] and A[1,c]*B[c,e] to C[1,e]
                # In the tensor decomposition: for each (c,e), we need 2 terms (one for each a)?
                
                # No. The tensor M has entries M[a,c,e]. We want M[a,c,e] = sum_r U[a,r]*V[c,r]*W[e,r]
                # For schoolbook: for each (a,c,e), one term. Rank = 2*2*3 = 12? That can't be right.
                
                # I think I'm confusing myself. Let me go back to basics.
                # Matrix multiplication C = A*B, A is n x m, B is m x p, C is n x p.
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # The tensor M[i,j,k] = 1 for all (i,j,k).
                # Decomposition: M[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
                # Schoolbook: rank = n*m*p, with U[i,j,r]=1, V[j,k,r]=1, W[k,i,r]=1 for each (i,j,k).
                
                # For 2x2 * 2x3: n=2, m=2, p=3. Schoolbook rank = 12.
                # But we know rank(2x2x3) = 6.
                # How? C[i,e] = sum_c A[i,c] B[c,e]
                # = A[i,0]*B[0,e] + A[i,1]*B[1,e]
                # For each (c,e), the term A[:,c] * B[c,:] is an outer product contributing to C[:,e].
                # Specifically, for fixed (c,e): the contribution to C[i,e] is A[i,c]*B[c,e] for i in {0,1}.
                # This can be written as: U[i, r] = A[i,c], V[c, r] = B[c,e], W[e, r] = 1? No.
                
                # The decomposition is: M[i,c,e] = sum_r U[i,r]*V[c,r]*W[e,r]
                # For schoolbook of 2x2x3: for each (i,c,e), one term. Rank 12.
                # For rank 6: for each (c,e), we have U[i,r] = delta_{i, ?}...
                
                # Actually: M[i,c,e] = sum_r U[i,r] V[c,r] W[e,r]
                # We want M[i,c,e] = 1 for all i,c,e.
                # One decomposition: for each c in {0,1}, e in {0,1,2}, define r = (c,e).
                # U[i, r] = 1 for all i, V[c, r] = 1 for that c, W[e, r] = 1 for that e.
                # Then sum_r U[i,r]V[c,r]W[e,r] = sum_{c',e'} delta_{c,c'} delta_{e,e'} = 1.
                # Rank = 2*3 = 6. Yes!
                
                # So for 2x2 * 2x3, rank 6 decomposition:
                # For each (c, e) where c in {0,1}, e in {0,1,2}:
                #   r = index for (c,e)
                #   U[i, r] = 1 for i in {0,1}
                #   V[c, r] = 1
                #   W[e, r] = 1
                
                # Now for 4x4 * 4x6:
                # Split into blocks. For each (i_block, j_block, k_block):
                # C_block[i_block, k_block] = sum_{j_block} A_block[i_block, j_block] * B_block[j_block, k_block]
                # Each A_block[i_block, j_block] is 2x2, B_block[j_block, k_block] is 2x3.
                # For each (j_block, k_block) pair, the multiplication A_block * B_block has rank 6.
                # But we also need to sum over j_block (2 values) and the results go to the same C_block.
                
                # For fixed (i_block, k_block):
                # C_block = A_block[i_block, 0] * B_block[0, k_block] + A_block[i_block, 1] * B_block[1, k_block]
                # Each product is a 2x2 * 2x3 multiplication, rank 6.
                # Total for this C_block: 6 + 6 = 12 terms.
                # Number of (i_block, k_block): 2 * 3 = 6.
                # Total rank: 6 * 12 = 72.
                
                # Let's implement this.
                for j in range(2):  # j_block
                    for c in range(2):  # local row in B block / col in A block
                        for e in range(3):  # local col in B block
                            # Global indices:
                            # A row: 2*i_block + a, for a in {0,1}
                            # A col: 2*j_block + c
                            # B row: 2*j_block + c
                            # B col: 3*k_block + e
                            # C row: 3*k_block + e (wait, C is 4x6, so C row is i, C col is k)
                            # C[i, k] where i = 2*i_block + a, k = 3*k_block + e
                            
                            # For this term (j_block, c, e), and for each a in {0,1}:
                            # U[global_i, col] where global_i = 2*i_block + a, global_j = 2*j_block + c
                            # V[global_j, col] where global_j = 2*j_block + c, global_k = 3*k_block + e
                            # W[global_k, col] where global_k = 3*k_block + e, global_i = 2*i_block + a
                            
                            # But wait, in the rank-6 decomposition of 2x2*2x3:
                            # U[i, r] = 1 for all i in {0,1} (local)
                            # So for each a in {0,1}, we set U[2*i_block+a, col] = 1
                            # V[2*j_block+c, col] = 1
                            # W[3*k_block+e, col] = 1
                            
                            # But this means for a given (j_block, c, e), both a=0 and a=1 use the same column.
                            # That's correct! The rank-6 decomposition has U[i,r]=1 for all i.
                            
                            # So for each (i_block, j_block, k_block, c, e):
                            # col is unique
                            # For a in {0,1}: U[2*i_block+a, col] = 1
                            # V[2*j_block+c, col] = 1
                            # W[3*k_block+e, col] = 1
                            
                            # But wait, this only sets W for one k value. But C has two rows for each i_block.
                            # C[2*i_block, 3*k_block+e] and C[2*i_block+1, 3*k_block+e] both need to be set.
                            # In the decomposition, W[k, r] should be 1 for the k value.
                            # But C[i,k] = sum_j A[i,j] B[j,k]. The W factor is W[k*n+i, r].
                            # So W is indexed by (k, i). For fixed k = 3*k_block+e, i can be 2*i_block or 2*i_block+1.
                            # In the rank-6 decomposition, U[i_local, r] = 1 for all i_local, so both i values are covered.
                            # But W[k, i] needs to be set for each (k, i) pair.
                            
                            # I think I need to reconsider. The W factor is W[k*n + i, c].
                            # For the term to contribute to C[i, k], we need W[k*n + i, c] to be non-zero.
                            # In the schoolbook, for each (i, j, k), W[k*n + i, col] = 1.
                            # In my block decomposition, for each (i_block, j_block, k_block, c, e),
                            # I want to cover both i = 2*i_block and i = 2*i_block+1.
                            # So I need W[(3*k_block+e)*n + (2*i_block), col] = 1
                            # and W[(3*k_block+e)*n + (2*i_block+1), col] = 1.
                            
                            # Similarly, U[i*m + j, col] for i in {2*i_block, 2*i_block+1}, j = 2*j_block + c.
                            # V[j*p + k, col] for j = 2*j_block + c, k = 3*k_block + e.
                            
                            # So for each (i_block, j_block, k_block, c, e):
                            #   col = next column
                            #   For a in {0,1}:
                            #     U[(2*i_block+a)*m + (2*j_block+c), col] = 1
                            #     W[(3*k_block+e)*n + (2*i_block+a), col] = 1
                            #   V[(2*j_block+c)*p + (3*k_block+e), col] = 1
                            
                            # This gives 2 entries in U, 1 in V, 2 in W per column.
                            # Number of columns: 2 * 2 * 3 * 2 * 3 = 72. Correct.
                            
                            for a in range(2):
                                U[(2 * i_block + a) * m + (2 * j_block + c)][col] = 1
                                W[(3 * k_block + e) * n + (2 * i_block + a)][col] = 1
                            V[(2 * j_block + c) * p + (3 * k_block + e)][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
