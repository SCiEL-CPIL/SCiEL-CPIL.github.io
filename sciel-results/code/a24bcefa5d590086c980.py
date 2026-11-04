def construct():
    n, m, p = 3, 3, 6
    # Rank 23 decomposition for 3x3x3 (Laderman)
    # Each term is a list of (index, value) for U, V, W
    # U index: i*m + j
    # V index: j*p + k (for 3x3x3, p=3, so j*3+k, but we map to j*6+k)
    # W index: k*n + i
    
    terms_333 = [
        # 1
        ([(0, 1)], [(0, 1)], [(0, 1)]), # A00 B00 C00
        # 2
        ([(1, 1)], [(3, 1)], [(0, 1)]), # A01 B10 C00 (j=1, k=0 -> V index 1*3+0=3)
        # 3
        ([(2, 1)], [(6, 1)], [(0, 1)]), # A02 B20 C00 (j=2, k=0 -> V index 2*3+0=6)
        # 4
        ([(3, 1)], [(1, 1)], [(3, 1)]), # A10 B01 C10 (i=1, j=0, k=1 -> W index 1*3+1=4? No, W index k*n+i. k=1, i=1 -> 1*3+1=4)
        # Wait, C10 is row 1, col 0. In W, index is k*n + i.
        # If C is stored as C[i][k], then W index for C[i][k] is?
        # Task says W[k*n + i]. So C[i][k] is at W[k*n + i].
        # So for C10 (i=1, k=0), W index is 0*3 + 1 = 1.
        # Let's re-verify the seed code.
        # Seed: W[k * n + i][column] = 1.
        # Loop: for i in range(n): for j in range(m): for k in range(p):
        # So U[i*m+j], V[j*p+k], W[k*n+i].
        # This implies C[i][k] is formed by summing over j.
        # And W[k*n+i] is the coefficient for C[i][k].
        # So C[i][k] corresponds to W index k*n + i.
        
        # Let's redefine the terms with explicit indices.
    ]
    
    # I will generate the 23 terms programmatically based on the patterns.
    # Pattern 1-9: Single terms.
    # A[i][j], B[j][k], C[i][k].
    # U index: i*3 + j.
    # V index: j*6 + k (for first block k=0,1,2).
    # W index: k*3 + i.
    
    def make_term(a_idx, b_idx, w_idx):
        return [a_idx, b_idx, w_idx]
        
    terms = []
    
    # 1-9: Diagonal-like
    # 1. A00 B00 C00
    terms.append(make_term(0*3+0, 0*6+0, 0*3+0))
    # 2. A01 B10 C00 -> A[0,1] B[1,0] C[0,0]
    terms.append(make_term(0*3+1, 1*6+0, 0*3+0))
    # 3. A02 B20 C00 -> A[0,2] B[2,0] C[0,0]
    terms.append(make_term(0*3+2, 2*6+0, 0*3+0))
    # 4. A10 B01 C10 -> A[1,0] B[0,1] C[1,0]
    terms.append(make_term(1*3+0, 0*6+1, 1*3+0))
    # 5. A11 B11 C10 -> A[1,1] B[1,1] C[1,0]
    terms.append(make_term(1*3+1, 1*6+1, 1*3+0))
    # 6. A12 B21 C10 -> A[1,2] B[2,1] C[1,0]
    terms.append(make_term(1*3+2, 2*6+1, 1*3+0))
    # 7. A20 B02 C20 -> A[2,0] B[0,2] C[2,0]
    terms.append(make_term(2*3+0, 0*6+2, 2*3+0))
    # 8. A21 B12 C20 -> A[2,1] B[1,2] C[2,0]
    terms.append(make_term(2*3+1, 1*6+2, 2*3+0))
    # 9. A22 B22 C20 -> A[2,2] B[2,2] C[2,0]
    terms.append(make_term(2*3+2, 2*6+2, 2*3+0))
    
    # 10-12: Diagonal sums
    # 10. (A00+A11+A22)(B00+B11+B22) -> C00+C11+C22
    # This term contributes to multiple C entries.
    # In the tensor decomposition, this is ONE term with multiple non-zero entries in U, V, W.
    # U: 1 at A00, 1 at A11, 1 at A22.
    # V: 1 at B00, 1 at B11, 1 at B22.
    # W: 1 at C00, 1 at C11, 1 at C22.
    
    def make_multi_term(a_indices, v_indices, w_indices, val=1):
        return a_indices, v_indices, w_indices, val
        
    terms_multi = []
    # 10
    terms_multi.append(make_multi_term([0*3+0, 1*3+1, 2*3+2], [0*6+0, 1*6+1, 2*6+2], [0*3+0, 1*3+1, 2*3+2], 1))
    # 11
    terms_multi.append(make_multi_term([0*3+0, 1*3+1, 2*3+2], [0*6+0, 1*6+1, 2*6+2], [0*3+0, 1*3+1, 2*3+2], 1)) # Signs: A00-A11+A22, B00-B11+B22, C00-C11+C22
    # Need to handle signs.
    
    # I will build the U, V, W matrices directly.
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(a_list, b_list, w_list, val):
        nonlocal col
        for a_idx in a_list:
            U[a_idx][col] += val
        for b_idx in b_list:
            V[b_idx][col] += val
        for w_idx in w_list:
            W[w_idx][col] += val
        col += 1
        
    # Terms 1-9
    add_term([0*3+0], [0*6+0], [0*3+0], 1)
    add_term([0*3+1], [1*6+0], [0*3+0], 1)
    add_term([0*3+2], [2*6+0], [0*3+0], 1)
    add_term([1*3+0], [0*6+1], [1*3+0], 1)
    add_term([1*3+1], [1*6+1], [1*3+0], 1)
    add_term([1*3+2], [2*6+1], [1*3+0], 1)
    add_term([2*3+0], [0*6+2], [2*3+0], 1)
    add_term([2*3+1], [1*6+2], [2*3+0], 1)
    add_term([2*3+2], [2*6+2], [2*3+0], 1)
    
    # Terms 10-12
    add_term([0*3+0, 1*3+1, 2*3+2], [0*6+0, 1*6+1, 2*6+2], [0*3+0, 1*3+1, 2*3+2], 1)
    add_term([0*3+0, 1*3+1, 2*3+2], [0*6+0, 1*6+1, 2*6+2], [0*3+0, 1*3+1, 2*3+2], 1) # Signs: A: 1,-1,1. B: 1,-1,1. C: 1,-1,1
    # Correction for signs:
    # I passed a single value. I need to pass a list of values or handle it.
    
    # Let's restart the construction with a helper that takes lists of (index, value).
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_term_2(a_pairs, b_pairs, w_pairs):
        nonlocal col
        for idx, val in a_pairs:
            U[idx][col] += val
        for idx, val in b_pairs:
            V[idx][col] += val
        for idx, val in w_pairs:
            W[idx][col] += val
        col += 1
        
    # 1-9
    add_term_2([(0*3+0, 1)], [(0*6+0, 1)], [(0*3+0, 1)])
    add_term_2([(0*3+1, 1)], [(1*6+0, 1)], [(0*3+0, 1)])
    add_term_2([(0*3+2, 1)], [(2*6+0, 1)], [(0*3+0, 1)])
    add_term_2([(1*3+0, 1)], [(0*6+1, 1)], [(1*3+0, 1)])
    add_term_2([(1*3+1, 1)], [(1*6+1, 1)], [(1*3+0, 1)])
    add_term_2([(1*3+2, 1)], [(2*6+1, 1)], [(1*3+0, 1)])
    add_term_2([(2*3+0, 1)], [(0*6+2, 1)], [(2*3+0, 1)])
    add_term_2([(2*3+1, 1)], [(1*6+2, 1)], [(2*3+0, 1)])
    add_term_2([(2*3+2, 1)], [(2*6+2, 1)], [(2*3+0, 1)])
    
    # 10: A00+A11+A22, B00+B11+B22, C00+C11+C22
    add_term_2([(0*3+0, 1), (1*3+1, 1), (2*3+2, 1)], 
               [(0*6+0, 1), (1*6+1, 1), (2*6+2, 1)], 
               [(0*3+0, 1), (1*3+1, 1), (2*3+2, 1)])
               
    # 11: A00-A11+A22, B00-B11+B22, C00-C11+C22
    add_term_2([(0*3+0, 1), (1*3+1, -1), (2*3+2, 1)], 
               [(0*6+0, 1), (1*6+1, -1), (2*6+2, 1)], 
               [(0*3+0, 1), (1*3+1, -1), (2*3+2, 1)])
               
    # 12: A00+A11-A22, B00+B11-B22, C00+C11-C22
    add_term_2([(0*3+0, 1), (1*3+1, 1), (2*3+2, -1)], 
               [(0*6+0, 1), (1*6+1, 1), (2*6+2, -1)], 
               [(0*3+0, 1), (1*3+1, 1), (2*3+2, -1)])
               
    # 13: A01+A12+A20, B10+B21+B02, C01+C12+C20
    # A[0,1]=1, A[1,2]=5, A[2,0]=6
    # B[1,0]=6, B[2,1]=13, B[0,2]=12
    # C[0,1]=1, C[1,2]=7, C[2,0]=6
    add_term_2([(0*3+1, 1), (1*3+2, 1), (2*3+0, 1)], 
               [(1*6+0, 1), (2*6+1, 1), (0*6+2, 1)], 
               [(0*3+1, 1), (1*3+2, 1), (2*3+0, 1)])
               
    # 14: A01-A12+A20, B10-B21+B02, C01-C12+C20
    add_term_2([(0*3+1, 1), (1*3+2, -1), (2*3+0, 1)], 
               [(1*6+0, 1), (2*6+1, -1), (0*6+2, 1)], 
               [(0*3+1, 1), (1*3+2, -1), (2*3+0, 1)])
               
    # 15: A01+A12-A20, B10+B21-B02, C01+C12-C20
    add_term_2([(0*3+1, 1), (1*3+2, 1), (2*3+0, -1)], 
               [(1*6+0, 1), (2*6+1, 1), (0*6+2, -1)], 
               [(0*3+1, 1), (1*3+2, 1), (2*3+0, -1)])
               
    # 16: A10+A21+A02, B01+B12+B20, C10+C21+C02
    # A[1,0]=3, A[2,1]=7, A[0,2]=2
    # B[0,1]=1, B[1,2]=7, B[2,0]=6
    # C[1,0]=3, C[2,1]=7, C[0,2]=2
    add_term_2([(1*3+0, 1), (2*3+1, 1), (0*3+2, 1)], 
               [(0*6+1, 1), (1*6+2, 1), (2*6+0, 1)], 
               [(1*3+0, 1), (2*3+1, 1), (0*3+2, 1)])
               
    # 17: A10-A21+A02, B01-B12+B20, C10-C21+C02
    add_term_2([(1*3+0, 1), (2*3+1, -1), (0*3+2, 1)], 
               [(0*6+1, 1), (1*6+2, -1), (2*6+0, 1)], 
               [(1*3+0, 1), (2*3+1, -1), (0*3+2, 1)])
               
    # 18: A10+A21-A02, B01+B12-B20, C10+C21-C02
    add_term_2([(1*3+0, 1), (2*3+1, 1), (0*3+2, -1)], 
               [(0*6+1, 1), (1*6+2, 1), (2*6+0, -1)], 
               [(1*3+0, 1), (2*3+1, 1), (0*3+2, -1)])
               
    # 19: A00+A01+A02, B00+B10+B20, C00+C01+C02
    # A[0,0]=0, A[0,1]=1, A[0,2]=2
    # B[0,0]=0, B[1,0]=6, B[2,0]=12
    # C[0,0]=0, C[0,1]=1, C[0,2]=2
    add_term_2([(0*3+0, 1), (0*3+1, 1), (0*3+2, 1)], 
               [(0*6+0, 1), (1*6+0, 1), (2*6+0, 1)], 
               [(0*3+0, 1), (0*3+1, 1), (0*3+2, 1)])
               
    # 20: A10+A11+A12, B01+B11+B21, C10+C11+C12
    add_term_2([(1*3+0, 1), (1*3+1, 1), (1*3+2, 1)], 
               [(0*6+1, 1), (1*6+1, 1), (2*6+1, 1)], 
               [(1*3+0, 1), (1*3+1, 1), (1*3+2, 1)])
               
    # 21: A20+A21+A22, B02+B12+B22, C20+C21+C22
    add_term_2([(2*3+0, 1), (2*3+1, 1), (2*3+2, 1)], 
               [(0*6+2, 1), (1*6+2, 1), (2*6+2, 1)], 
               [(2*3+0, 1), (2*3+1, 1), (2*3+2, 1)])
               
    # 22: A00+A10+A20, B00+B01+B02, C00+C10+C20
    add_term_2([(0*3+0, 1), (1*3+0, 1), (2*3+0, 1)], 
               [(0*6+0, 1), (0*6+1, 1), (0*6+2, 1)], 
               [(0*3+0, 1), (1*3+0, 1), (2*3+0, 1)])
               
    # 23: A01+A11+A21, B10+B11+B12, C01+C11+C21
    add_term_2([(0*3+1, 1), (1*3+1, 1), (2*3+1, 1)], 
               [(1*6+0, 1), (1*6+1, 1), (1*6+2, 1)], 
               [(0*3+1, 1), (1*3+1, 1), (2*3+1, 1)])
    
    # Now duplicate for k=3,4,5
    # Map k=0->3, k=1->4, k=2->5 in V and W.
    # V index j*6+k -> j*6+(k+3)
    # W index k*3+i -> (k+3)*3+i
    
    for r in range(23):
        # Copy U part (same)
        for i in range(n*m):
            U[i][r + 23] = U[i][r]
            
        # Copy V part (shift k)
        for j in range(m):
            for k in range(3):
                # Original V[j*6+k][r]
                # New V[j*6+(k+3)][r+23]
                V[j*6 + (k+3)][r + 23] = V[j*6 + k][r]
                
        # Copy W part (shift k)
        for k in range(3):
            for i in range(n):
                # Original W[k*3+i][r]
                # New W[(k+3)*3+i][r+23]
                W[(k+3)*3 + i][r + 23] = W[k*3 + i][r]
                
    return {"U": U, "V": V, "W": W}
