"""Matrix multiplication 2x2 x 2x11 via Kronecker product decomposition."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    
    # Decompose the 2x2 matrix multiplication into rank-2 using (x+y) and (x-y)
    # Then Kronecker with schoolbook 2x11 multiplication
    # Total rank = 2 * 11 = 22
    
    rank = 2 * p  # 22
    
    U = [[0] * rank for _ in range(n * m)]  # (4, 22)
    V = [[0] * rank for _ in range(m * p)]  # (22, 22)
    W = [[0] * rank for _ in range(p * n)]  # (22, 22)
    
    # For each "sign" s in {+1, -1} (2 values)
    # And each column k in 0..10 (11 values)
    # We create a rank-1 component
    
    for s_idx, s in enumerate([1, -1]):
        for k in range(p):
            col = s_idx * p + k
            
            # U components: A_{i,j} with coefficient from (x+y) or (x-y)
            # For s=+1: U uses (1,1,1,1) -> A_{0,0}+A_{0,1}+A_{1,0}+A_{1,1}? No...
            # Actually, we need to be careful. Let me think again.
            
            # The idea: C = A * B where A is 2x2, B is 2x11
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            
            # Using the identity: we can write A*B as a combination of 
            # rank-1 tensors using the fact that for 2x2, we have:
            # (a,b,c,d) * (e,f,g,h)^T style decomposition
            
            # Simpler: use the standard 2x2 rank-2 decomposition
            # A * B where both are 2x2:
            # C_{0,0} = a*e + b*g
            # C_{0,1} = a*f + b*h
            # C_{1,0} = c*e + d*g
            # C_{1,1} = c*f + d*h
            
            # Using (x+y, x-y, y+z, y-z) style... actually for 2x2*2x2, 
            # the minimal rank is 7 (Strassen), but we're doing 2x2 * 2x11
            
            # Let's use a different approach: treat A as two column vectors
            # A_col0 = [A_{0,0}, A_{1,0}], A_col1 = [A_{0,1}, A_{1,1}]
            # Then C_col_k = A_col0 * B_{0,k} + A_col1 * B_{1,k}
            # This is just schoolbook: rank = 4*11 = 44
            
            # To reduce: use the fact that we can combine A_{0,0} and A_{0,1} etc.
            # Let u1 = A_{0,0} + A_{0,1}, u2 = A_{0,0} - A_{0,1}
            # Then A_{0,0} = (u1+u2)/2, A_{0,1} = (u1-u2)/2
            # Similarly for row 1: v1 = A_{1,0} + A_{1,1}, v2 = A_{1,0} - A_{1,1}
            
            # C_{i,k} = A_{i,0}*B_{0,k} + A_{i,1}*B_{1,k}
            # For i=0: C_{0,k} = (u1+u2)/2 * B_{0,k} + (u1-u2)/2 * B_{1,k}
            #           = u1*(B_{0,k}+B_{1,k})/2 + u2*(B_{0,k}-B_{1,k})/2
            # For i=1: C_{1,k} = (v1+v2)/2 * B_{0,k} + (v1-v2)/2 * B_{1,k}
            #           = v1*(B_{0,k}+B_{1,k})/2 + v2*(B_{0,k}-B_{1,k})/2
            
            # So we can define:
            # w1_k = (B_{0,k} + B_{1,k})/2
            # w2_k = (B_{0,k} - B_{1,k})/2
            # Then: C_{0,k} = u1*w1_k + u2*w2_k
            #       C_{1,k} = v1*w1_k + v2*w2_k
            
            # This gives us rank = 4 * 11 = 44 still... no reduction.
            
            # Wait, the issue is that u1, u2, v1, v2 are all independent.
            # We need to share structure between rows.
            
            # Alternative: use the fact that for 2x2 matrices, we can use:
            # m1 = (A_{0,0}+A_{1,1})*(B_{0,0}+B_{1,1}) - (A_{0,1}+A_{1,0})*(B_{0,1}+B_{1,0})
            # This is getting complicated.
            
            # Let's just use the simple Kronecker approach with rank-2 for the 2x2 part
            # by treating it as a bilinear form with 2 terms per output.
            
            # Actually, the simplest correct approach: 
            # C = A @ B, where A is (2,2) and B is (2,11)
            # View as: for each j in {0,1}, we have A[:,j] @ B[j,:]
            # That's 2 * (2*11) = 44 rank-1 terms... no, each A[:,j]@B[j,:] is a 2x11 matrix
            # which has rank 1, so 2 such terms give rank 2 for the whole product.
            # But we need to express this as sum of rank-1 tensors in the matmul tensor.
            
            # The matmul tensor T_{i,j,k} = 1 if the output (i,k) depends on input (i,j) and (j,k)
            # We want to write T = sum_r U_r ⊗ V_r ⊗ W_r
            
            # For our case: T_{i,j,k} = 1 for all i,j,k (since every A_{i,j} multiplies every B_{j,k} to contribute to C_{i,k})
            # Wait no, T is a 3-tensor where T_{(i,j),(j,k),(i,k)} = 1
            # Using the indexing: U is (n*m, rank) = (4, rank), V is (m*p, rank) = (22, rank), W is (p*n, rank) = (22, rank)
            # U_{i*m+j, r}, V_{j*p+k, r}, W_{k*n+i, r}
            
            # The tensor element is sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1 if the product contributes
            # For matmul: C_{i,k} = sum_j A_{i,j} B_{j,k}, so the coefficient of A_{i,j} * B_{j,k} in C_{i,k} is 1
            # So we need: sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1 for all i,j,k
            
            # This is the identity tensor for the bilinear map.
            
            # Simple decomposition: for each (i,j,k), set U_{i*m+j, r}=1, V_{j*p+k, r}=1, W_{k*n+i, r}=1
            # That's rank 44 (schoolbook).
            
            # To reduce: notice that for fixed j, we have 2*11 = 22 pairs of (i,k)
            # For j=0: (i,k) pairs are (0,0),(0,1),...,(0,10),(1,0),(1,1),...,(1,10)
            # For j=1: same set of (i,k) pairs
            
            # We can factor: for each k, the contribution to C_{0,k} from A_{0,0}*B_{0,k} and A_{0,1}*B_{1,k}
            # And to C_{1,k} from A_{1,0}*B_{0,k} and A_{1,1}*B_{1,k}
            
            # Using the substitution: let's define new variables
            # A'_{0,0} = A_{0,0} + A_{1,1}, A'_{0,1} = A_{0,0} - A_{1,1}
            # A'_{1,0} = A_{1,0} + A_{0,1}, A'_{1,1} = A_{1,0} - A_{0,1}
            # This is getting too complicated. Let's just use rank 22 with a simple construction.
            
            # For each k in 0..10, and for each "mode" in {0, 1}:
            # mode 0: handles the "sum" part
            # mode 1: handles the "difference" part
            
            # Let me use a cleaner approach:
            # U has 4 rows (i*m+j for i,j in {0,1})
            # For s in {0, 1} (the two "eigenmodes") and k in 0..10:
            
            if s_idx == 0:
                # "Plus" mode: U_{0,0}=1, U_{0,1}=1, U_{1,0}=1, U_{1,1}=1
                # But we need this to work with V and W
                U[0][col] = 1  # i=0,j=0
                U[1][col] = 1  # i=0,j=1
                U[2][col] = 1  # i=1,j=0
                U[3][col] = 1  # i=1,j=1
            else:
                # "Minus" mode: U_{0,0}=1, U_{0,1}=-1, U_{1,0}=-1, U_{1,1}=1
                U[0][col] = 1   # i=0,j=0
                U[1][col] = -1  # i=0,j=1
                U[2][col] = -1  # i=1,j=0
                U[3][col] = 1   # i=1,j=1
            
            # V has 22 rows (j*p+k for j in {0,1}, k in 0..10)
            # For the "plus" mode, V should extract (B_{0,k} + B_{1,k})
            # For the "minus" mode, V should extract (B_{0,k} - B_{1,k})
            # But V is indexed by (j,k), so V_{j,k} is the coefficient for B_{j,k}
            # We want: sum_j V_{j,k} * B_{j,k} = B_{0,k} + B_{1,k} for plus mode
            # So V_{0,k} = 1, V_{1,k} = 1 for plus mode
            # And V_{0,k} = 1, V_{1,k} = -1 for minus mode
            
            V[0 * p + k][col] = 1 if s_idx == 0 else 1
            V[1 * p + k][col] = 1 if s_idx == 0 else -1
            
            # W has 22 rows (k*n+i for k in 0..10, i in {0,1})
            # W_{k,i} is the coefficient for C_{i,k}
            # We want: C_{i,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
            # For plus mode: C_{i,k} should get contribution (A_{i,0}+A_{i,1})*(B_{0,k}+B_{1,k}) * W
            # But we need the actual C_{i,k} = A_{i,0}*B_{0,k} + A_{i,1}*B_{1,k}
            # (A_{i,0}+A_{i,1})*(B_{0,k}+B_{1,k}) = A_{i,0}B_{0,k} + A_{i,0}B_{1,k} + A_{i,1}B_{0,k} + A_{i,1}B_{1,k}
            # This has extra terms! So this approach is wrong.
            
            # Let me reconsider. The Kronecker product approach doesn't directly work here
            # because the dimensions don't factor nicely.
            
            # New plan: use rank 22 by observing that for each k, we have 4 products:
            # A_{0,0}B_{0,k}, A_{0,1}B_{1,k}, A_{1,0}B_{0,k}, A_{1,1}B_{1,k}
            # We can combine: (A_{0,0}+A_{1,1})B_{0,k} + (A_{0,1}+A_{1,0})B_{1,k} gives some terms
            # but still has cross terms.
            
            # I think the safest approach is to just return the schoolbook decomposition
            # but with a slightly different structure to ensure it's valid.
            
            # Let me restart with a clean, correct approach.
            pass
    
    # The above approach is getting messy. Let me just construct the schoolbook decomposition
    # which is guaranteed correct, and that will be the fallback.
    
    # Actually, let me try a different valid decomposition with rank 22.
    # Key insight: for 2x2 @ 2x11, we can write:
    # C_{i,k} = A_{i,0}*B_{0,k} + A_{i,1}*B_{1,k}
    # = (A_{i,0} + A_{i,1}) * ((B_{0,k} + B_{1,k})/2) + (A_{i,0} - A_{i,1}) * ((B_{0,k} - B_{1,k})/2)
    
    # Wait, that's not right either. Let me verify:
    # (A_{i,0}+A_{i,1})*(B_{0,k}+B_{1,k})/2 + (A_{i,0}-A_{i,1})*(B_{0,k}-B_{1,k})/2
    # = [A_{i,0}B_{0,k} + A_{i,0}B_{1,k} + A_{i,1}B_{0,k} + A_{i,1}B_{1,k} 
    #    + A_{i,0}B_{0,k} - A_{i,0}B_{1,k} - A_{i,1}B_{0,k} + A_{i,1}B_{1,k}]/2
    # = [2*A_{i,0}B_{0,k} + 2*A_{i,1}B_{1,k}]/2
    # = A_{i,0}B_{0,k} + A_{i,1}B_{1,k} ✓
    
    # So this works! For each i and k, we have two terms.
    # Total rank = 2 (for i) * 2 (for the +/-) * 11 (for k) = 44. No reduction!
    
    # Hmm, the i and k are coupled in the output, so we can't separate them that way.
    
    # Let me just go with the schoolbook decomposition which is rank 44.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
