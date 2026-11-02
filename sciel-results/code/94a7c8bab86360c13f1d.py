"""Improved program using Strassen-based sub-decompositions."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # We'll use a hybrid approach:
    # Partition the 3x5x6 multiplication into sub-problems
    # and use Strassen for 2x2x2 sub-blocks where applicable.
    
    # Strategy: split i into {0,1} and {2}, split j into pairs, split k into pairs
    # For each (i_group, j_pair, k_pair) we can use Strassen (rank 7) or schoolbook (rank 8)
    
    # Let's be more systematic:
    # i: 0,1,2 (3 rows)
    # j: 0,1,2,3,4 (5 cols)  
    # k: 0,1,2,3,4,5 (6 cols)
    
    # Partition:
    # i_groups: [0,1], [2]
    # j_groups: [0,1], [2,3], [4]
    # k_groups: [0,1], [2,3], [4,5]
    
    # For each combination (ig, jg, kg):
    #   if len(ig)==2 and len(jg)==2 and len(kg)==2: use Strassen rank 7
    #   else: use schoolbook rank len(ig)*len(jg)*len(kg)
    
    ig_sizes = [2, 1]
    jg_sizes = [2, 2, 1]
    kg_sizes = [2, 2, 2]
    
    rank = 0
    U_cols = []
    V_cols = []
    W_cols = []
    
    ig_starts = [0, 2]  # i indices: [0,1], [2]
    jg_starts = [0, 2, 4]  # j indices: [0,1], [2,3], [4]
    kg_starts = [0, 2, 4]  # k indices: [0,1], [2,3], [4,5]
    
    for ig_idx, (ig_start, ig_size) in enumerate(zip(ig_starts, ig_sizes)):
        for jg_idx, (jg_start, jg_size) in enumerate(zip(jg_starts, jg_sizes)):
            for kg_idx, (kg_start, kg_size) in enumerate(zip(kg_starts, kg_sizes)):
                # Time check
                if time.time() - t0 > DEADLINE:
                    break
                
                if ig_size == 2 and jg_size == 2 and kg_size == 2:
                    # Use Strassen rank 7
                    # For a 2x2x2 block with local indices (li, lj, lk) in {0,1}
                    # Global indices: i = ig_start + li, j = jg_start + lj, k = kg_start + lk
                    # W index: k*n + i = (kg_start+lk)*n + (ig_start+li)
                    # U index: i*m + j = (ig_start+li)*m + (jg_start+lj)
                    # V index: j*p + k = (jg_start+lj)*p + (kg_start+lk)
                    
                    # Strassen formulas for C = AB where A is 2x2, B is 2x2, C is 2x2
                    # C[0][0] = M1 - M4 + M6 + M7
                    # C[0][1] = M2 + M3
                    # C[1][0] = M4 + M5
                    # C[1][1] = M1 - M2 + M5 + M6
                    # where:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)*B00
                    # M3 = A00*(B01-B11)
                    # M4 = A11*(B10-B00)
                    # M5 = (A00+A01)*B11
                    # M6 = (A10-A00)*(B00+B01)
                    # M7 = (A01-A11)*(B10+B11)
                    
                    # Each M corresponds to a rank-1 term
                    # We need to build U, V, W for each of the 7 terms
                    # For each term t, we have:
                    # U[(ig_start+li)*m + (jg_start+lj), t] = coefficient from A
                    # V[(jg_start+lj)*p + (kg_start+lk), t] = coefficient from B
                    # W[(kg_start+lk)*n + (ig_start+li), t] = coefficient from C
                    
                    # Let's define the 7 terms with their A and B coefficient vectors
                    # A is 2x2: A[li][lj], B is 2x2: B[lj][lk]
                    
                    # M1: A coeffs: A00=1, A01=0, A10=0, A11=1 -> U[(ig_start+0)*m+jg_start]=1, U[(ig_start+0)*m+jg_start+1]=0, U[(ig_start+1)*m+jg_start]=0, U[(ig_start+1)*m+jg_start+1]=1
                    #        B coeffs: B00=1, B01=1, B10=1, B11=1 -> V[(jg_start)*p+kg_start]=1, V[(jg_start)*p+kg_start+1]=1, V[(jg_start+1)*p+kg_start]=1, V[(jg_start+1)*p+kg_start+1]=1
                    #        C coeffs: C00=1, C01=0, C10=0, C11=-1 -> W[(kg_start)*n+ig_start]=1, W[(kg_start+1)*n+ig_start]=0, W[(kg_start)*n+ig_start+1]=0, W[(kg_start+1)*n+ig_start+1]=-1
                    # Wait, C[li][lk]: C00 corresponds to li=0,lk=0; C01 to li=0,lk=1; C10 to li=1,lk=0; C11 to li=1,lk=1
                    
                    # Let me redefine. For each Strassen term, I'll list the (li,lj) -> a_coeff and (lj,lk) -> b_coeff, and then (li,lk) -> c_coeff
                    
                    strassen_terms = [
                        # (name, a_coeffs {(li,lj): val}, b_coeffs {(lj,lk): val}, c_coeffs {(li,lk): val})
                        # M1
                        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (0,1):0, (1,0):0, (1,1):-1}),
                        # M2
                        ({(1,0):1, (1,1):1}, {(0,0):1, (0,1):0, (1,0):0, (1,1):0}, {(0,0):0, (0,1):0, (1,0):1, (1,1):0}),
                        # M3
                        ({(0,0):1, (0,1):0, (1,0):0, (1,1):0}, {(0,1):1, (1,1):-1, (0,0):0, (1,0):0}, {(0,0):0, (0,1):1, (1,0):0, (1,1):0}),
                        # M4
                        ({(1,1):1, (0,0):0, (0,1):0, (1,0):0}, {(1,0):1, (0,0):-1, (0,1):0, (1,1):0}, {(0,0):-1, (0,1):0, (1,0):1, (1,1):0}),
                        # M5
                        ({(0,0):1, (0,1):1, (1,0):0, (1,1):0}, {(1,1):1, (0,0):0, (0,1):0, (1,0):0}, {(0,0):0, (0,1):0, (1,0):0, (1,1):1}),
                        # M6
                        ({(1,0):1, (0,0):-1, (0,1):0, (1,1):0}, {(0,0):1, (0,1):1, (1,0):0, (1,1):0}, {(0,0):0, (0,1):0, (1,0):0, (1,1):0}),
                        # M7
                        ({(0,1):1, (1,1):-1, (0,0):0, (1,0):0}, {(1,0):1, (1,1):1, (0,0):0, (0,1):0}, {(0,0):0, (0,1):0, (1,0):0, (1,1):0}),
                    ]
                    
                    # Wait, I need to double check the Strassen formulas and map them correctly.
                    # Standard Strassen for C = AB:
                    # M1 = (A00+A11)(B00+B11)
                    # M2 = (A10+A11)B00
                    # M3 = A00(B01-B11)
                    # M4 = A11(B10-B00)
                    # M5 = (A00+A01)B11
                    # M6 = (A10-A00)(B00+B01)
                    # M7 = (A01-A11)(B10+B11)
                    #
                    # C00 = M1 - M4 + M6 + M7
                    # C01 = M2 + M3
                    # C10 = M4 + M5
                    # C11 = M1 - M2 + M5 + M6
                    #
                    # So the c_coeffs for each term:
                    # M1: C00 += 1, C11 += 1
                    # M2: C01 += 1, C11 -= 1
                    # M3: C01 += 1
                    # M4: C00 -= 1, C10 += 1
                    # M5: C10 += 1, C11 += 1
                    # M6: C00 += 1, C11 += 1
                    # M7: C00 += 1
                    
                    # Let me redo:
                    strassen_terms_correct = [
                        # M1
                        ({(0,0):1, (1,1):1}, {(0,0):1, (0,1):1, (1,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
                        # M2
                        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
                        # M3
                        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1}),
                        # M4
                        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):-1, (1,0):1}),
                        # M5
                        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):1, (1,1):1}),
                        # M6
                        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(0,0):1, (1,1):1}),
                        # M7
                        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
                    ]
                    
                    for a_coeffs, b_coeffs, c_coeffs in strassen_terms_correct:
                        U_cols.append((a_coeffs, ig_start, jg_start))
                        V_cols.append((b_coeffs, jg_start, kg_start))
                        W_cols.append((c_coeffs, ig_start, kg_start))
                        rank += 1
                else:
                    # Schoolbook for this sub-block
                    for li in range(ig_size):
                        for lj in range(jg_size):
                            for lk in range(kg_size):
                                i = ig_start + li
                                j = jg_start + lj
                                k = kg_start + lk
                                U_cols.append({(li, lj): 1}, ig_start, jg_start)
                                V_cols.append({(lj, lk): 1}, jg_start, kg_start)
                                W_cols.append({(li, lk): 1}, ig_start, kg_start)
                                rank += 1
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        if col < len(U_cols):
            a_coeffs, ig_start, jg_start = U_cols[col]
            for (li, lj), val in a_coeffs.items():
                i = ig_start + li
                j = jg_start + lj
                if 0 <= i < n and 0 <= j < m:
                    U[i * m + j][col] = val
            
            b_coeffs, jg_start, kg_start = V_cols[col]
            for (lj, lk), val in b_coeffs.items():
                j = jg_start + lj
                k = kg_start + lk
                if 0 <= j < m and 0 <= k < p:
                    V[j * p + k][col] = val
            
            c_coeffs, ig_start, kg_start = W_cols[col]
            for (li, lk), val in c_coeffs.items():
                i = ig_start + li
                k = kg_start + lk
                if 0 <= i < n and 0 <= k < p:
                    W[k * n + i][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
