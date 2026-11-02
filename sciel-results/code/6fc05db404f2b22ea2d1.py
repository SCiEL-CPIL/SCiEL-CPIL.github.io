"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank calculation: 7 pairs * 7 Strassen ops + 1 single * 4 Schoolbook ops = 53
    rank = 7 * 7 + 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions for 2x2 * 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (b-d)(e+g)
    
    # Indices for A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # Indices for B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # Indices for C: (0,0), (0,1), (1,0), (1,1) -> W index (k,i)
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val=1):
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val

    # Process 7 pairs of columns
    for pair_idx in range(7):
        k_start = pair_idx * 2
        k_end = k_start + 1
        
        # M1: (a+d)(e+h) -> C00 += M1, C11 += M1
        # A: (0,0), (1,1). B: (0, k_start), (1, k_end).
        # Note: B indices are (j, k).
        # e = B[0, k_start], h = B[1, k_end]
        
        # M1
        add_term((0,0), (0, k_start), (k_start, 0)) # C00
        add_term((1,1), (0, k_start), (k_start, 0))
        add_term((0,0), (1, k_end), (k_start, 0))
        add_term((1,1), (1, k_end), (k_start, 0))
        
        add_term((0,0), (0, k_start), (k_end, 1)) # C11
        add_term((1,1), (0, k_start), (k_end, 1))
        add_term((0,0), (1, k_end), (k_end, 1))
        add_term((1,1), (1, k_end), (k_end, 1))
        col += 1
        
        # M2: (c+d)e -> C10 += M2, C11 -= M2
        # A: (1,0), (1,1). B: (0, k_start).
        # c = A[1,0], d = A[1,1], e = B[0, k_start]
        
        # M2
        add_term((1,0), (0, k_start), (k_start, 1)) # C10
        add_term((1,1), (0, k_start), (k_start, 1))
        
        add_term((1,0), (0, k_start), (k_end, 1), -1) # C11
        add_term((1,1), (0, k_start), (k_end, 1), -1)
        col += 1
        
        # M3: a(f-h) -> C01 += M3, C11 += M3
        # A: (0,0). B: (0, k_end), (1, k_end).
        # a = A[0,0], f = B[0, k_end], h = B[1, k_end]
        
        # M3
        add_term((0,0), (0, k_end), (k_end, 0)) # C01
        add_term((0,0), (1, k_end), (k_end, 0), -1)
        
        add_term((0,0), (0, k_end), (k_end, 1)) # C11
        add_term((0,0), (1, k_end), (k_end, 1), -1)
        col += 1
        
        # M4: d(g-e) -> C00 += M4, C10 += M4
        # A: (1,1). B: (1, k_start), (0, k_start).
        # d = A[1,1], g = B[1, k_start], e = B[0, k_start]
        
        # M4
        add_term((1,1), (1, k_start), (k_start, 0)) # C00
        add_term((1,1), (0, k_start), (k_start, 0), -1)
        
        add_term((1,1), (1, k_start), (k_start, 1)) # C10
        add_term((1,1), (0, k_start), (k_start, 1), -1)
        col += 1
        
        # M5: (a+b)h -> C00 -= M5, C01 += M5
        # A: (0,0), (0,1). B: (1, k_end).
        # a = A[0,0], b = A[0,1], h = B[1, k_end]
        
        # M5
        add_term((0,0), (1, k_end), (k_start, 0), -1) # C00
        add_term((0,1), (1, k_end), (k_start, 0), -1)
        
        add_term((0,0), (1, k_end), (k_end, 0)) # C01
        add_term((0,1), (1, k_end), (k_end, 0))
        col += 1
        
        # M6: (c-d)(f+h) -> C00 += M6
        # A: (1,0), (1,1). B: (0, k_end), (1, k_end).
        # c = A[1,0], d = A[1,1], f = B[0, k_end], h = B[1, k_end]
        
        # M6
        add_term((1,0), (0, k_end), (k_start, 0)) # C00
        add_term((1,0), (1, k_end), (k_start, 0))
        add_term((1,1), (0, k_end), (k_start, 0), -1)
        add_term((1,1), (1, k_end), (k_start, 0), -1)
        col += 1
        
        # M7: (b-d)(e+g) -> C11 += M7
        # A: (0,1), (1,1). B: (0, k_start), (1, k_start).
        # b = A[0,1], d = A[1,1], e = B[0, k_start], g = B[1, k_start]
        
        # M7
        add_term((0,1), (0, k_start), (k_end, 1)) # C11
        add_term((0,1), (1, k_start), (k_end, 1))
        add_term((1,1), (0, k_start), (k_end, 1), -1)
        add_term((1,1), (1, k_start), (k_end, 1), -1)
        col += 1

    # Process last column (index 14) with schoolbook
    k_single = 14
    
    # C00 += A00 * B0k
    add_term((0,0), (0, k_single), (k_single, 0))
    col += 1
    
    # C01 += A01 * B1k  (Wait, C is 2x15, so C[0,14] and C[1,14])
    # C[i,k] = sum_j A[i,j] B[j,k]
    # C[0,14] = A[0,0]B[0,14] + A[0,1]B[1,14]
    # C[1,14] = A[1,0]B[0,14] + A[1,1]B[1,14]
    
    # Term 2: A01 * B1k -> C00 (which is C[0,14])
    add_term((0,1), (1, k_single), (k_single, 0))
    col += 1
    
    # Term 3: A10 * B0k -> C10 (which is C[1,14])
    add_term((1,0), (0, k_single), (k_single, 1))
    col += 1
    
    # Term 4: A11 * B1k -> C10 (which is C[1,14])
    add_term((1,1), (1, k_single), (k_single, 1))
    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
