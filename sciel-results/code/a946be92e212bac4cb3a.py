import time
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = 2, 4, 5
    
    def verify(U, V, W, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if s != 1:
                        return False
        return True
    
    # Schoolbook decomposition: rank = 40
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j][col] = 1
                V_school[j*p+k][col] = 1
                W_school[k*n+i][col] = 1
                col += 1
    
    best = {"U": U_school, "V": V_school, "W": W_school}
    
    # Strassen-based decomposition: rank = 36
    rank_strassen = 36
    U = [[0] * rank_strassen for _ in range(n * m)]
    V = [[0] * rank_strassen for _ in range(m * p)]
    W = [[0] * rank_strassen for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2x2 terms: (U_coeffs, V_coeffs, W_coeffs)
    # Local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    
    # Standard 2x2x1 terms
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                # 2x2x2 Strassen block
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W[gk*n + gi][col] += val_w
                    col += 1
            else:
                # 2x2x1 standard block
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank_strassen and verify(U, V, W, rank_strassen):
        best = {"U": U, "V": V, "W": W}
    
    # 1+3 partition: split inner dim 4 into 1+3
    # For each of the 3 row-pairs (q=0,1,2):
    #   2x2x3 block: rank 9 (3-term Strassen-like)
    #   2x2x1 block: rank 4 (schoolbook)
    # Total rank = 3 * (9 + 4) = 39 -> but let's try 2x2x3 with rank 9
    # Actually: n=2, m=4, p=5
    # Split m=4 into 1+3: 
    #   Part A: 2x1x5 (rank 10 schoolbook)
    #   Part B: 2x3x5 (split into 2x2x3 + 2x1x3)
    #     2x2x3: rank 9
    #     2x1x3: rank 6
    # Total: 10 + 9 + 6 = 25
    
    # Alternative: split m=4 into 2+2
    #   Part A: 2x2x5 (two 2x2x2 blocks + one 2x2x1)
    #   Part B: 2x2x5 (same)
    # This is what we already have (rank 36)
    
    # Let's try: split p=5 into 2+3
    #   Part A: 2x4x2 (two 2x2x2 Strassen blocks) = 2*7 = 14
    #   Part B: 2x4x3 (two 2x2x3 blocks) = 2*9 = 18
    # Total: 14 + 18 = 32
    
    # Even better: use the 1+3 split on m
    # m=4 = 1 + 3
    # 2x1x5: rank 10
    # 2x3x5: split 3 into 2+1
    #   2x2x5: two 2x2x2 (rank 7 each) + one 2x2x1 (rank 4) = 18
    #   2x1x5: rank 10
    # Total: 10 + 18 + 10 = 38 (worse)
    
    # Best: split p=5 into 2+3
    # 2x4x2: split m=4 into 2+2, each 2x2x2 uses Strassen rank 7
    #   Total for 2x4x2: 2*7 = 14
    # 2x4x3: split m=4 into 2+2, each 2x2x3 uses rank 9
    #   Total for 2x4x3: 2*9 = 18
    # Grand total: 14 + 18 = 32
    
    rank_new = 32
    U2 = [[0] * rank_new for _ in range(n * m)]
    V2 = [[0] * rank_new for _ in range(m * p)]
    W2 = [[0] * rank_new for _ in range(p * n)]
    
    col2 = 0
    
    # 2x2x3 rank-9 decomposition (3-term Strassen-like)
    # For matrices A(2x2), B(2x2), C(2x2) where C = A*B with B being 2x3
    # Actually 2x2 times 2x3 gives 2x3
    # Standard 2x2x3: rank is at most 9 (3x3 schoolbook)
    # Can we do better? The tensor rank of 2x2x3 is known to be 9.
    
    # 2x2x3 schoolbook: 9 terms
    # For each of 2 output rows, 3 output cols, 2 sum terms = 12
    # But we can use Strassen-like: 
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]]
    # C = A*B = [[ae+bh, af+bi, ag+bj], [ce+dh, cf+di, cg+dj]]
    # 9 terms schoolbook:
    # t1=ae, t2=bh, t3=af, t4=bi, t5=ag, t6=bj, t7=ce, t8=dh, t9=cf, t10=di, t11=cg, t12=dj
    # That's 12, not 9.
    
    # Actually 2x2x3 rank: The multiplication tensor [2,2,3] has rank 9.
    # Let me construct it explicitly.
    
    # For 2x2x3: A is 2x2, B is 2x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Schoolbook: 2*2*3 = 12 terms
    # Strassen-like reduction:
    # Define:
    # m1 = A[0,0] * (B[1,0] - B[0,0])  -- no, let me think differently
    
    # Known: rank([2,2,3]) = 9
    # Construction:
    # m1 = a*e, m2 = b*h, m3 = c*e, m4 = d*h, m5 = (a+b)*(f-i), m6 = (c+d)*(f-i), 
    # m7 = a*g, m8 = b*j, m9 = c*g, m10 = d*j
    # That's 10. Need 9.
    
    # Let me look up: for 2x2 times 2x3:
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i  
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    
    # 9-term decomposition:
    # m1 = a*e -> C[0,0] += m1, C[1,0] += c/m1... no
    
    # Let me just use 12-term schoolbook for 2x2x3 and see if total is better:
    # 2x4x2: 2*2x2x2 = 2*7 = 14
    # 2x4x3: 2*2x2x3 = 2*12 = 24
    # Total: 38 (worse than 36)
    
    # So the 1+3 split on p doesn't help with schoolbook 2x2x3.
    # We need a true rank-9 2x2x3 decomposition.
    
    # Known rank-9 for [2,2,3]:
    # m1 = a*e
    # m2 = b*h  
    # m3 = c*e
    # m4 = d*h
    # m5 = (a+b)*(i-f)
    # m6 = (c+d)*(i-f)
    # m7 = a*g
    # m8 = b*j
    # m9 = (c+d)*(g)  -- no this doesn't work cleanly
    
    # Let me try a different approach. Use the fact that [2,2,3] can be seen as
    # embedding [2,2,2] with an extra column.
    
    # For 2x2x2 with Strassen (7 terms) + 2x2x1 (4 terms) = 11, not 9.
    
    # I'll use the 12-term schoolbook for 2x2x3 and accept rank 38, 
    # OR find a better overall strategy.
    
    # Alternative: keep the current 36-rank solution as best.
    # Try to find if there's a way to get below 36.
    
    # Let me try: 2x4x5 split as 2x2x5 + 2x2x5
    # Each 2x2x5: split 5 into 2+2+1
    #   2x2x2: rank 7
    #   2x2x2: rank 7  
    #   2x2x1: rank 4
    # Total per 2x2x5: 18
    # Grand total: 36 (same as before)
    
    # What about 2x4x5 as 2x3x5 + 2x1x5?
    # 2x1x5: rank 10
    # 2x3x5: split 3 into 2+1
    #   2x2x5: 7+7+4 = 18
    #   2x1x5: 10
    # Total: 10 + 18 + 10 = 38
    
    # Or 2x4x5 as 2x4x2 + 2x4x3
    # 2x4x2: 7+7 = 14
    # 2x4x3: 9+9 = 18 (if rank-9 exists)
    # Total: 32
    
    # Let me carefully construct a rank-9 2x2x3.
    # Reference: the border rank of [2,2,3] is 9, and rank is 9.
    
    # Construction from literature:
    # For A(2x2) * B(2x3) = C(2x3):
    # Let A = [[a,b],[c,d]], B = [[e,f,g],[h,i,j]], C = [[c00,c01,c02],[c10,c11,c12]]
    
    # m1 = a*e
    # m2 = b*h
    # m3 = c*e  
    # m4 = d*h
    # m5 = (a+b)*(f)
    # m6 = (c+d)*(f)
    # m7 = (a+b)*(i)
    # m8 = (c+d)*(i)
    # m9 = (a+b)*(g-j)  -- this won't work cleanly
    
    # Let me try yet another approach: just verify if rank 32 is achievable
    # with a known 2x2x3 rank-9 scheme.
    
    # I'll use a verified 9-term scheme:
    # m1 = a*e
    # m2 = b*(i-f)  
    # m3 = c*e
    # m4 = d*(i-f)
    # m5 = (a+b)*f
    # m6 = (c+d)*f
    # m7 = a*g
    # m8 = b*j
    # m9 = (c+d)*(g) + d*j - ... this is getting complicated
    
    # Given the complexity and risk of error, I'll keep the current best (rank 36)
    # and return it.
    
    return best
