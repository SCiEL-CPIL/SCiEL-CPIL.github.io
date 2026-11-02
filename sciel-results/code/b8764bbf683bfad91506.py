"""
Optimal rank-14 decomposition for 2x2x4 matrix multiplication.
The tensor decomposes into two independent 2x2x2 blocks.
Each block uses Strassen's rank-7 algorithm.
Total rank = 7 + 7 = 14.
"""

def construct():
    n, m, p = 2, 2, 4
    
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    
    # Initialize matrices
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2
    # U indices: (i, j) -> i*m + j
    # V indices: (j, k) -> j*p + k  (k is local 0 or 1)
    # W indices: (k, i) -> k*n + i  (k is local 0 or 1)
    
    # Coefficients for the 7 Strassen products
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # Output C = P1 + P4 - P5 + P7 (for C00)
    #          = P3 + P5         (for C01)
    #          = P2 + P4         (for C10)
    #          = P1 - P2 + P3 + P6 (for C11)
    
    # Define terms as (u_dict, v_dict, w_dict)
    # Keys are (index_in_factor, value)
    # We use explicit indices for clarity
    
    # Local indices for 2x2:
    # A: a(0,0)=0, b(0,1)=1, c(1,0)=2, d(1,1)=3
    # B: e(0,0)=0, f(0,1)=1, g(1,0)=2, h(1,1)=3 (local k)
    # C: c00(0,0)=0, c01(1,0)=1, c10(0,1)=2, c11(1,1)=3 (local k, i)
    
    # Term 1: (a+d)(e+h)
    t1_u = {0: 1, 3: 1}
    t1_v = {0: 1, 3: 1}
    t1_w = {0: 1, 1: 1, 2: 1, 3: 1} # Wait, W is output.
    # C00 gets +1, C01 gets 0, C10 gets 0, C11 gets +1?
    # C00 = P1... C11 = P1...
    # So W for P1 should have 1 at (k=0, i=0) and (k=1, i=1).
    # Local indices: (0,0)->0, (1,1)->3.
    t1_w = {0: 1, 3: 1}
    
    # Term 2: (c+d)e
    t2_u = {2: 1, 3: 1}
    t2_v = {0: 1}
    t2_w = {2: 1, 3: -1} # C10 gets +1, C11 gets -1
    
    # Term 3: a(f-h)
    t3_u = {0: 1}
    t3_v = {1: 1, 3: -1}
    t3_w = {1: 1, 3: 1} # C01 gets +1, C11 gets +1
    
    # Term 4: d(g-e)
    t4_u = {3: 1}
    t4_v = {2: 1, 0: -1}
    t4_w = {0: 1, 2: 1} # C00 gets +1, C10 gets +1
    
    # Term 5: (a+b)h
    t5_u = {0: 1, 1: 1}
    t5_v = {3: 1}
    t5_w = {1: 1, 0: -1} # C01 gets +1, C00 gets -1
    
    # Term 6: (c-a)(e+f)
    t6_u = {2: 1, 0: -1}
    t6_v = {0: 1, 1: 1}
    t6_w = {3: 1} # C11 gets +1
    
    # Term 7: (b-d)(g+h)
    t7_u = {1: 1, 3: -1}
    t7_v = {2: 1, 3: 1}
    t7_w = {0: 1} # C00 gets +1

    terms = [
        (t1_u, t1_v, t1_w),
        (t2_u, t2_v, t2_w),
        (t3_u, t3_v, t3_w),
        (t4_u, t4_v, t4_w),
        (t5_u, t5_v, t5_w),
        (t6_u, t6_v, t6_w),
        (t7_u, t7_v, t7_w),
    ]
    
    # Block 1: k_local in {0, 1}
    # Global k = k_local
    for r in range(7):
        u_dict, v_dict, w_dict = terms[r]
        col = r
        
        for idx, val in u_dict.items():
            U[idx][col] += val
            
        for idx, val in v_dict.items():
            # idx is local j*p + k? No, idx in dict is local index in 2x2 B.
            # My dict keys above are local indices 0..3.
            # Local index 0: j=0, k=0. Global V index: 0*4 + 0 = 0.
            # Local index 1: j=0, k=1. Global V index: 0*4 + 1 = 1.
            # Local index 2: j=1, k=0. Global V index: 1*4 + 0 = 4.
            # Local index 3: j=1, k=1. Global V index: 1*4 + 1 = 5.
            j_local = idx // 2
            k_local = idx % 2
            v_global = j_local * p + k_local
            V[v_global][col] += val
            
        for idx, val in w_dict.items():
            # idx is local index in 2x2 C.
            # Local index 0: k=0, i=0. Global W index: 0*2 + 0 = 0.
            # Local index 1: k=1, i=0. Global W index: 1*2 + 0 = 2.
            # Wait, trace convention W[k*n+i].
            # n=2.
            # k=0, i=0 -> 0
            # k=0, i=1 -> 1
            # k=1, i=0 -> 2
            # k=1, i=1 -> 3
            # My dict keys:
            # 0: C00 (k=0, i=0)
            # 1: C01 (k=0, i=1)? No, standard matrix C is C[row][col].
            # C00 is row 0, col 0.
            # C01 is row 0, col 1.
            # C10 is row 1, col 0.
            # C11 is row 1, col 1.
            # Trace convention W[k*n+i] where k is col of C, i is row of C.
            # So W[0*2+0]=C00, W[1*2+0]=C10, W[0*2+1]=C01, W[1*2+1]=C11?
            # Let's check the task description: "W[k*n+i,r]".
            # Usually C[i,k] is row i, col k.
            # If W index is k*n+i, then:
            # k=0, i=0 -> 0 (C00)
            # k=0, i=1 -> 1 (C10)
            # k=1, i=0 -> 2 (C01)
            # k=1, i=1 -> 3 (C11)
            
            # Let's re-verify my W assignments in terms.
            # C00 = P1 + P4 - P5 + P7.
            # C10 = P2 + P4.
            # C01 = P3 + P5.
            # C11 = P1 - P2 + P3 + P6.
            
            # Term 1 (P1): C00 (+1), C11 (+1).
            # Indices: C00->0, C11->3.
            # My t1_w was {0: 1, 3: 1}. Correct.
            
            # Term 2 (P2): C10 (+1), C11 (-1).
            # Indices: C10->1, C11->3.
            # My t2_w was {2: 1, 3: -1}. 
            # WAIT. C10 index is 1 (k=0, i=1). C11 index is 3 (k=1, i=1).
            # My dict key 2 corresponds to k=1, i=0 which is C01.
            # So t2_w should be {1: 1, 3: -1}.
            
            # Let's re-map W indices carefully.
            # W_idx = k * n + i
            # C00: k=0, i=0 -> 0
            # C10: k=0, i=1 -> 1
            # C01: k=1, i=0 -> 2
            # C11: k=1, i=1 -> 3
            
            # Re-evaluating terms with correct W indices:
            
            # T1 (P1): C00(+1), C11(+1). W: {0: 1, 3: 1}
            # T2 (P2): C10(+1), C11(-1). W: {1: 1, 3: -1}
            # T3 (P3): C01(+1), C11(+1). W: {2: 1, 3: 1}
            # T4 (P4): C00(+1), C10(+1). W: {0: 1, 1: 1}
            # T5 (P5): C01(+1), C00(-1). W: {2: 1, 0: -1}
            # T6 (P6): C11(+1). W: {3: 1}
            # T7 (P7): C00(+1). W: {0: 1}
            
            # Re-evaluating V indices:
            # V_idx = j * p + k
            # p=4.
            # Block 1 (k_local 0,1):
            # e (j=0, k=0) -> 0
            # f (j=0, k=1) -> 1
            # g (j=1, k=0) -> 4
            # h (j=1, k=1) -> 5
            
            # T1 (P1): (a+d)(e+h). V: e(+1), h(+1). V: {0: 1, 5: 1}
            # T2 (P2): (c+d)e. V: e(+1). V: {0: 1}
            # T3 (P3): a(f-h). V: f(+1), h(-1). V: {1: 1, 5: -1}
            # T4 (P4): d(g-e). V: g(+1), e(-1). V: {4: 1, 0: -1}
            # T5 (P5): (a+b)h. V: h(+1). V: {5: 1}
            # T6 (P6): (c-a)(e+f). V: e(+1), f(+1). V: {0: 1, 1: 1}
            # T7 (P7): (b-d)(g+h). V: g(+1), h(+1). V: {4: 1, 5: 1}
            
            # Re-evaluating U indices:
            # U_idx = i * m + j. m=2.
            # a (0,0) -> 0
            # b (0,1) -> 1
            # c (1,0) -> 2
            # d (1,1) -> 3
            
            # T1 (P1): a+d. U: {0: 1, 3: 1}
            # T2 (P2): c+d. U: {2: 1, 3: 1}
            # T3 (P3): a. U: {0: 1}
            # T4 (P4): d. U: {3: 1}
            # T5 (P5): a+b. U: {0: 1, 1: 1}
            # T6 (P6): c-a. U: {2: 1, 0: -1}
            # T7 (P7): b-d. U: {1: 1, 3: -1}
            
            pass # The loop below will use the corrected dicts

    # Corrected Terms
    terms = [
        # (U, V, W)
        ({0: 1, 3: 1}, {0: 1, 5: 1}, {0: 1, 3: 1}), # P1
        ({2: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}),      # P2
        ({0: 1}, {1: 1, 5: -1}, {2: 1, 3: 1}),      # P3
        ({3: 1}, {4: 1, 0: -1}, {0: 1, 1: 1}),      # P4
        ({0: 1, 1: 1}, {5: 1}, {2: 1, 0: -1}),      # P5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),      # P6
        ({1: 1, 3: -1}, {4: 1, 5: 1}, {0: 1}),      # P7
    ]
    
    # Fill Block 1 (k=0,1)
    for r in range(7):
        u_dict, v_dict, w_dict = terms[r]
        col = r
        
        for idx, val in u_dict.items():
            U[idx][col] += val
            
        for idx, val in v_dict.items():
            V[idx][col] += val
            
        for idx, val in w_dict.items():
            W[idx][col] += val
            
    # Fill Block 2 (k=2,3)
    # V indices shift: k_local 0->2, 1->3.
    # e' (j=0, k=2) -> 0*4 + 2 = 2
    # f' (j=0, k=3) -> 0*4 + 3 = 3
    # g' (j=1, k=2) -> 1*4 + 2 = 6
    # h' (j=1, k=3) -> 1*4 + 3 = 7
    
    # W indices shift: k_local 0->2, 1->3.
    # C'00 (k=2, i=0) -> 2*2 + 0 = 4
    # C'10 (k=2, i=1) -> 2*2 + 1 = 5
    # C'01 (k=3, i=0) -> 3*2 + 0 = 6
    # C'11 (k=3, i=1) -> 3*2 + 1 = 7
    
    # U indices are the same.
    
    # Mapping V: 0->2, 1->3, 4->6, 5->7
    # Mapping W: 0->4, 1->5, 2->6, 3->7
    
    v_map = {0: 2, 1: 3, 4: 6, 5: 7}
    w_map = {0: 4, 1: 5, 2: 6, 3: 7}
    
    for r in range(7):
        u_dict, v_dict, w_dict = terms[r]
        col = 7 + r
        
        for idx, val in u_dict.items():
            U[idx][col] += val
            
        for idx, val in v_dict.items():
            V[v_map[idx]][col] += val
            
        for idx, val in w_dict.items():
            W[w_map[idx]][col] += val
            
    return {"U": U, "V": V, "W": W}
