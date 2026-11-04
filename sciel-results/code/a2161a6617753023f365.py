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
    
    # Rank-36 Strassen-based solution (fallback)
    rank36 = 36
    U36 = [[0] * rank36 for _ in range(n * m)]
    V36 = [[0] * rank36 for _ in range(m * p)]
    W36 = [[0] * rank36 for _ in range(p * n)]
    
    col = 0
    strassen_terms = [
        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),
        ({2:1, 3:1}, {0:1}, {2:1, 3:-1}),
        ({0:1}, {1:1, 3:-1}, {1:1, 3:1}),
        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),
        ({0:1, 1:1}, {3:1}, {0:-1, 1:1}),
        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),
        ({1:1, 3:-1}, {2:1, 3:1}, {0:1})
    ]
    standard_terms = [
        ({0:1}, {0:1}, {0:1}),
        ({1:1}, {1:1}, {0:1}),
        ({2:1}, {0:1}, {1:1}),
        ({3:1}, {1:1}, {1:1})
    ]
    
    for q in range(2):
        for s in range(3):
            if s < 2:
                for u_t, v_t, w_t in strassen_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v // 2
                        lk = idx_v % 2
                        gj = 2*q + lj
                        gk = 2*s + lk
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w // 2
                        lk = idx_w % 2
                        gi = li
                        gk = 2*s + lk
                        W36[gk*n + gi][col] += val_w
                    col += 1
            else:
                for u_t, v_t, w_t in standard_terms:
                    for idx_u, val_u in u_t.items():
                        li = idx_u // 2
                        lj = idx_u % 2
                        gi = li
                        gj = 2*q + lj
                        U36[gi*m + gj][col] += val_u
                    for idx_v, val_v in v_t.items():
                        lj = idx_v
                        gj = 2*q + lj
                        gk = 4
                        V36[gj*p + gk][col] += val_v
                    for idx_w, val_w in w_t.items():
                        li = idx_w
                        gi = li
                        gk = 4
                        W36[gk*n + gi][col] += val_w
                    col += 1
    
    if col == rank36 and verify(U36, V36, W36, rank36):
        best = {"U": U36, "V": V36, "W": W36}
    else:
        # Schoolbook fallback
        rank_school = n * m * p
        best = {
            "U": [[0]*rank_school for _ in range(n*m)],
            "V": [[0]*rank_school for _ in range(m*p)],
            "W": [[0]*rank_school for _ in range(p*n)]
        }
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    best["U"][i*m+j][c] = 1
                    best["V"][j*p+k][c] = 1
                    best["W"][k*n+i][c] = 1
                    c += 1
        return best
    
    # Rank-32 solution: split p=5 into 2+3
    # 2x4x2: two 2x2x2 Strassen blocks (rank 7 each) = 14
    # 2x4x3: two 2x2x3 rank-9 blocks = 18
    # Total: 32
    
    rank32 = 32
    U32 = [[0] * rank32 for _ in range(n * m)]
    V32 = [[0] * rank32 for _ in range(m * p)]
    W32 = [[0] * rank32 for _ in range(p * n)]
    
    col2 = 0
    
    # 2x2x3 rank-9 decomposition
    # A = [[a,b],[c,d]] (2x2), B = [[e,f,g],[h,i,j]] (2x3), C = A*B (2x3)
    # C[0,0] = a*e + b*h
    # C[0,1] = a*f + b*i
    # C[0,2] = a*g + b*j
    # C[1,0] = c*e + d*h
    # C[1,1] = c*f + d*i
    # C[1,2] = c*g + d*j
    #
    # 9-term construction using 3x3 Strassen on the output columns:
    # The key idea: group the 3 output columns and apply a 3x3 scheme
    # where the "rows" of the 3x3 are the input columns of B,
    # and the "columns" are the output columns of C.
    #
    # m1 = a*(e+i) -> contributes to C[0,0] via e, C[0,1] via i
    # m2 = b*(h+j) -> contributes to C[0,0] via h, C[0,2] via j
    # m3 = c*(e+i) -> contributes to C[1,0] via e, C[1,1] via i
    # m4 = d*(h+j) -> contributes to C[1,0] via h, C[1,2] via j
    # m5 = (a+b)*f -> C[0,1]
    # m6 = (c+d)*f -> C[1,1]
    # m7 = a*(g - (e+i)) -> C[0,2] correction
    # m8 = b*(j - (h+j)) -> C[0,0] correction (negative)
    # m9 = c*(g - (e+i)) -> C[1,2] correction
    #
    # Let me verify this more carefully with a different approach.
    # Actually, let me use a cleaner construction:
    #
    # For 2x2x3, think of it as: for each of the 2 input rows of A,
    # we need to compute 2x3 dot products. The trick is to share
    # computations across the 3 output columns.
    #
    # Alternative clean rank-9:
    # m1 = a*e, m2 = b*h, m3 = c*e, m4 = d*h
    # m5 = (a+b)*f, m6 = (c+d)*f
    # m7 = (a+b)*(g-j), m8 = b*j, m9 = (c+d)*(g) + d*j - ... 
    #
    # Let me just try to verify a specific 9-term scheme:
    #
    # m1 = a*(e + i - f)
    # m2 = b*(h + j - f)  
    # m3 = c*(e + i - f)
    # m4 = d*(h + j - f)
    # m5 = (a + b)*f
    # m6 = (c + d)*f
    # m7 = a*(g - e)
    # m8 = b*(j - h)
    # m9 = c*(g - e) + d*(j - h)  -- no, this mixes rows
    
    # Let me use a verified approach. The [2,2,3] tensor rank is 9.
    # Here's a construction:
    #
    # m1 = a*e       -> C[0,0] += a*e, C[1,0] += 0
    # m2 = b*h       -> C[0,0] += b*h
    # m3 = c*e       -> C[1,0] += c*e
    # m4 = d*h       -> C[1,0] += d*h
    # m5 = (a+b)*f   -> C[0,1] += a*f, C[1,1] += c*f (no, this gives (a+b)*f for both)
    #
    # Actually (a+b)*f contributes a*f to C[0,1] and b*f to C[1,1]... no.
    # In the tensor decomposition, W coefficients determine which C entries get contributions.
    #
    # Let me be very explicit. For each rank-1 term r:
    # U_r[i,j] * V_r[j,k] * W_r[k,i] should sum to 1 for the (i,j,k) tensor entry.
    #
    # For [2,2,3]: i in {0,1}, j in {0,1}, k in {0,1,2}
    # Tensor entry T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[k,i] = 1 for all (i,j,k)
    #
    # 9-term scheme:
    # r=1: U[0,0]=1, V[0,0]=1, W[0,0]=1  -> T[0,0,0] += 1
    # r=2: U[0,1]=1, V[1,0]=1, W[0,0]=1  -> T[0,1,0] += 1
    # r=3: U[1,0]=1, V[0,0]=1, W[0,0]=1  -> T[1,0,0] += 1
    # r=4: U[1,1]=1, V[1,0]=1, W[0,0]=1  -> T[1,1,0] += 1
    # r=5: U[0,0]=1, V[0,1]=1, W[1,0]=1  -> T[0,0,1] += 1
    # r=6: U[0,1]=1, V[1,1]=1, W[1,0]=1  -> T[0,1,1] += 1
    # r=7: U[1,0]=1, V[0,1]=1, W[1,1]=1  -> T[1,0,1] += 1
    # r=8: U[1,1]=1, V[1,1]=1, W[1,1]=1  -> T[1,1,1] += 1
    # r=9: ??? -> need T[0,0,2], T[0,1,2], T[1,0,2], T[1,1,2]
    #
    # One term can't cover 4 independent entries unless the U,V,W vectors have
    # the right structure. Let's see:
    # For k=2: need sum_r U_r[i,j]*V_r[j,2]*W_r[2,i] = 1 for all i,j
    # With only 1 term left, we need U_r[i,j]*V_r[j,2]*W_r[2,i] = 1 for all 4 (i,j)
    # This requires U_r[0,0]*V_r[0,2]*W_r[2,0] = 1
    #              U_r[0,1]*V_r[1,2]*W_r[2,0] = 1
    #              U_r[1,0]*V_r[0,2]*W_r[2,1] = 1
    #              U_r[1,1]*V_r[1,2]*W_r[2,1] = 1
    # From first two: U_r[0,0]*V_r[0,2] = U_r[0,1]*V_r[1,2]
    # From last two: U_r[1,0]*V_r[0,2] = U_r[1,1]*V_r[1,2]
    # Let V_r[0,2] = 1, V_r[1,2] = 1, W_r[2,0] = 1, W_r[2,1] = 1
    # Then U_r[0,0] = 1, U_r[0,1] = 1, U_r[1,0] = 1, U_r[1,1] = 1
    # But this term also contributes to k=0 and k=1!
    # U_r[0,0]*V_r[0,0]*W_r[0,0] = 1*?*1 -- V_r[0,0] is already used in r=1
    #
    # The issue is that in a rank-1 term, the same U,V,W vectors are used for ALL k.
    # So we can't have separate terms for each k.
    #
    # This means the 9-term decomposition must have terms that span multiple k values.
    # Let me reconsider.
    #
    # Actually, I realize the 8 terms above (schoolbook for k=0 and k=1) already use
    # specific U,V,W patterns. The 9th term would need to handle k=2 without
    # disturbing k=0 and k=1, which means V_r[j,0]=0 and V_r[j,1]=0 for all j.
    # But then it can't contribute to k=0 or k=1, which is fine.
    # However, we showed one term can't cover all 4 entries for k=2 with the
    # constraint that it doesn't interfere with k=0,1.
    #
    # Wait, I made an error. Let me recount. The schoolbook for 2x2x3 is 2*2*3=12 terms.
    # To get 9, we need to save 3 terms by sharing.
    #
    # Let me try a different grouping. Use Strassen-like sharing on the j dimension:
    # For each k, C[:,k] = A * B[:,k]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    #
    # For k=0: need a*e, b*h, c*e, d*h (4 terms)
    # For k=1: need a*f, b*i, c*f, d*i (4 terms)
    # For k=2: need a*g, b*j, c*g, d*j (4 terms)
    # Total: 12
    #
    # Sharing idea: (a+b) appears in C[0,1] and C[1,1] if we use f.
    # (a+b)*f gives a*f for C[0,1] and b*f for C[1,1] -- but C[1,1] needs c*f + d*i, not b*f.
    #
    # OK here's the correct rank-9 construction for [2,2,3]:
    #
    # m1 = a*e        U=(1,0,0,0) V=(1,0,0,0,0,0) W=(1,0,0,0,0,0)  [k=0,i=0]
    # m2 = b*h        U=(0,1,0,0) V=(0,0,1,0,0,0) W=(1,0,0,0,0,0)  [k=0,i=0]
    # m3 = c*e        U=(0,0,1,0) V=(1,0,0,0,0,0) W=(0,1,0,0,0,0)  [k=0,i=1]
    # m4 = d*h        U=(0,0,0,1) V=(0,0,1,0,0,0) W=(0,1,0,0,0,0)  [k=0,i=1]
    # m5 = (a+b)*f    U=(1,1,0,0) V=(0,1,0,0,0,0) W=(0,0,1,0,0,0)  [k=1,i=0]
    # m6 = (c+d)*f    U=(0,0,1,1) V=(0,1,0,0,0,0) W=(0,0,0,1,0,0)  [k=1,i=1]
    # m7 = a*(g)      U=(1,0,0,0) V=(0,0,0,1,0,0) W=(0,0,0,0,1,0)  [k=2,i=0]
    # m8 = b*(j)      U=(0,1,0,0) V=(0,0,0,0,1,0) W=(0,0,0,0,1,0)  [k=2,i=0]
    # m9 = c*g + d*j  U=(0,0,1,1) V=(0,0,0,1,1,0) W=(0,0,0,0,0,1)  [k=2,i=1]
    #
    # Check m9: U=(0,0,1,1), V=(0,0,0,1,1,0), W=(0,0,0,0,0,1)
    # T[1,0,2] += U[1,0]*V[0,2]*W[2,1] = 1*1*1 = 1 ✓
    # T[1,1,2] += U[1,1]*V[1,2]*W[2,1] = 1*1*1 = 1 ✓
    # T[0,0,2] += U[0,0]*V[0,2]*W[2,0] = 0*1*0 = 0 ✓ (no interference)
    # T[0,1,2] += U[0,1]*V[1,2]*W[2,0] = 0*1*0 = 0 ✓
    #
    # But wait, m9's V vector is (0,0,0,1,1,0) meaning V[0,2]=1, V[1,2]=1
    # And U=(0,0,1,1) meaning U[1,0]=1, U[1,1]=1
    # So for k=2, i=1: sum over j of U[1,j]*V[j,2] = U[1,0]*V[0,2] + U[1,1]*V[1,2] = 1*1+1*1 = 2
    # Then times W[2,1]=1 gives 2, not 1!
    #
    # That's wrong. Let me fix: use V=(0,0,0,1,0,0) for m9 with U=(0,0,1,0)
    # m9a: c*g: U=(0,0,1,0), V=(0,0,0,1,0,0), W=(0,0,0,0,0,1)
    # m9b: d*j: U=(0,0,0,1), V=(0,0,0,0,1,0), W=(0,0,0,0,0,1)
    # That's 10 terms total. Not 9.
    #
    # I think the rank-9 for [2,2,3] requires a more subtle construction that I'm
    # not getting right from first principles. Let me just use the rank-36 solution
    # which is verified to work, and also try the rank-32 with schoolbook 2x2x3=12
    # to see if maybe I miscounted.
    #
    # 2x4x2: 2*7=14, 2x4x3: 2*12=24, total=38. Worse than 36.
    #
    # So the best I can reliably do is rank 36. Let me return that.
    
    if time.time() > deadline - 10:
        return best
    
    return best
