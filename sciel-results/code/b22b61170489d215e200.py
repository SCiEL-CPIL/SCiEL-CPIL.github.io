"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll build the decomposition as:
    # n=3 split into n=1 (1 term block) and n=2 (Strassen blocks)
    # m=4 split into two blocks of m=2
    # p=6 split into three blocks of p=2
    
    # Part 1: n=1, m=4, p=6: schoolbook, rank 24
    # Part 2: n=2, m=4, p=6: 6 Strassen blocks of 2x2x2, rank 6*7=42
    # Total rank: 66
    
    rank = 66
    U = [[0]*rank for _ in range(n*m)]  # 12 x 66
    V = [[0]*rank for _ in range(m*p)]  # 24 x 66
    W = [[0]*rank for _ in range(p*n)]  # 18 x 66
    
    col = 0
    
    # Part 1: n=1 (i=0), m=4, p=6
    # C[0,k] = sum_j A[0,j] B[j,k]
    # W index: k*n + i = k*3 + 0 = 3k
    for j in range(4):
        for k in range(6):
            U[0*m + j][col] = 1  # A[0,j], index = 0*4+j = j
            V[j*p + k][col] = 1  # B[j,k], index = j*6+k
            W[k*n + 0][col] = 1  # C[0,k] -> W[3k]
            col += 1
    
    # Part 2: n=2 (i=1,2), m=4, p=6
    # Split m=4 into j-blocks: {0,1} and {2,3}
    # Split p=6 into k-blocks: {0,1}, {2,3}, {4,5}
    # For each (j-block, k-block), use Strassen 2x2x2
    
    # Strassen 2x2x2 decomposition (for local indices a,b,c in {0,1})
    # Local A: a*2+b (a=row, b=col), local B: b*2+c (b=row, c=col), local C: c*2+a (c=col, a=row)
    # 7 terms:
    # Term 0: M1=(A00+A11)(B00+B11), W: C00+1, C11+1
    # Term 1: M2=(A10+A11)B00, W: C10+1, C11-1
    # Term 2: M3=A00(B10-B11), W: C01+1, C11+1
    # Term 3: M4=A11(B01-B00), W: C00+1, C10+1
    # Term 4: M5=(A00+A01)B11, W: C00-1, C01+1
    # Term 5: M6=(A10-A00)(B00+B01), W: C11+1
    # Term 6: M7=(A01-A11)(B10+B11), W: C00+1, C11+1
    
    # For global indices:
    # i_global = 1 + a (since n=2 part starts at i=1)
    # j_global = j_offset + b (j_offset is 0 or 2)
    # k_global = k_offset + c (k_offset is 0, 2, or 4)
    
    # U index: i_global * m + j_global = (1+a)*4 + (j_offset+b)
    # V index: j_global * p + k_global = (j_offset+b)*6 + (k_offset+c)
    # W index: k_global * n + i_global = (k_offset+c)*3 + (1+a)
    
    strassen_U = [
        [1,0,0,1],  # M1: A00, A11
        [0,0,1,1],  # M2: A10, A11
        [1,0,0,0],  # M3: A00
        [0,0,0,1],  # M4: A11
        [1,1,0,0],  # M5: A00, A01
        [-1,0,1,0], # M6: A00, A10
        [0,1,0,-1], # M7: A01, A11
    ]
    strassen_V = [
        [1,0,0,1],  # M1: B00, B11
        [1,0,0,0],  # M2: B00
        [0,0,1,-1], # M3: B10, B11
        [-1,1,0,0], # M4: B00, B01
        [0,0,0,1],  # M5: B11
        [1,1,0,0],  # M6: B00, B01
        [0,0,1,1],  # M7: B10, B11
    ]
    strassen_W = [
        [1,0,0,1],  # M1: C00, C11
        [0,1,0,-1], # M2: C10, C11
        [0,0,1,1],  # M3: C01, C11
        [1,1,0,0],  # M4: C00, C10
        [-1,0,1,0], # M5: C00, C01
        [0,0,0,1],  # M6: C11
        [1,0,0,1],  # M7: C00, C11
    ]
    
    for j_offset in [0, 2]:
        for k_offset in [0, 2, 4]:
            for t in range(7):
                for a in range(2):
                    for b in range(2):
                        if strassen_U[t][a*2+b] != 0:
                            i_global = 1 + a
                            j_global = j_offset + b
                            U[i_global*m + j_global][col] += strassen_U[t][a*2+b]
                    for c in range(2):
                        if strassen_V[t][b*2+c] != 0:
                            # Wait, V index is b*2+c where b is the row of B (which is j local) and c is col of B (k local)
                            # But b is already determined by the U loop. This is wrong.
                            pass
                # Let me redo this more carefully.
                # Each term t has:
                # U entries at (a,b) for a,b in {0,1}
                # V entries at (b,c) for b,c in {0,1}  -- NOTE: the b here is the same b as in U!
                # W entries at (c,a) for c,a in {0,1}
                
                # So I need to iterate over all (a,b,c) and check if U[a*2+b] * V[b*2+c] * W[c*2+a] contributes.
                
                # Actually, the rank-1 term is: for each (a,b,c), the contribution is U_t[a*2+b] * V_t[b*2+c] * W_t[c*2+a]
                # But that's not how it works. The rank-1 term is a single scalar product:
                # sum_{a,b,c} U_t[a*2+b] * A[a*2+b] * V_t[b*2+c] * B[b*2+c] * W_t[c*2+a]
                # = sum_{a,b,c} (U_t[a*2+b] * A[a*2+b]) * (V_t[b*2+c] * B[b*2+c]) * W_t[c*2+a]
                
                # Hmm, I think the decomposition is:
                # C_local[c*2+a] = sum_t (sum_b U_t[a*2+b] * A_local[a*2+b]) * (sum_b V_t[b*2+c] * B_local[b*2+c]) * W_t[c*2+a]
                
                # No, that's not right either. Let me think again.
                
                # The tensor T[a*2+b, b*2+c, c*2+a] = 1 for all a,b,c in {0,1}.
                # A rank-r decomposition is: T = sum_t U_t ⊗ V_t ⊗ W_t
                # where U_t is a 4-vector, V_t is a 4-vector, W_t is a 4-vector.
                # T[i,j,k] = sum_t U_t[i] * V_t[j] * W_t[k]
                
                # So for each term t, I add U_t[i] to U[i][col], V_t[j] to V[j][col], W_t[k] to W[k][col].
                
                # For the Strassen decomposition, the 7 terms are:
                # Term t has U_t, V_t, W_t as 4-vectors.
                
                # Let me redefine:
                pass
                
                # I'll just directly set the values.
                for a in range(2):
                    for b in range(2):
                        u_val = strassen_U[t][a*2+b]
                        if u_val != 0:
                            i_global = 1 + a
                            j_global = j_offset + b
                            U[i_global*m + j_global][col] = u_val  # Not +=, since each (a,b) appears in exactly one term's U for this block
                
                # Wait, that's wrong. Each term t has a full 4-vector U_t, so I need to set all 4 entries.
                # Let me redo.
                
                # Clear and redo
                for a in range(2):
                    for b in range(2):
                        U[(1+a)*m + (j_offset+b)][col] = strassen_U[t][a*2+b]
                for b in range(2):
                    for c in range(2):
                    # V index: local b*2+c, global (j_offset+b)*p + (k_offset+c)
                        V[(j_offset+b)*p + (k_offset+c)][col] = strassen_V[t][b*2+c]
                for c in range(2):
                    for a in range(2):
                    # W index: local c*2+a, global (k_offset+c)*n + (1+a)
                        W[(k_offset+c)*n + (1+a)][col] = strassen_W[t][c*2+a]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
