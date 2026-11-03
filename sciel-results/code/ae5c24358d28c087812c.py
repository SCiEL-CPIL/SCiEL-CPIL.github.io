    def add_2x2x3(A_block, B_block, C_block, start_rank):
        # A_block: (2,2) indices in A
        # B_block: (2,3) indices in B (cols 4-6)
        # C_block: (2,3) indices in C (cols 4-6)
        
        # A_block rows: ar0, ar1
        # A_block cols: ac0, ac1
        # B_block rows: br0, br1
        # B_block cols: bc0, bc1, bc2 (global cols 4,5,6)
        # C_block rows: cr0, cr1
        # C_block cols: cc0, cc1, cc2 (global cols 4,5,6)
        
        # Strassen for A_block * B_block[:, 0:2]
        # A: [[a, b], [c, d]]
        # B1: [[e, f], [h, i]]
        
        a_idx = A_block[0][0] # global idx
        b_idx = A_block[0][1]
        c_idx = A_block[1][0]
        d_idx = A_block[1][1]
        
        e_idx = B_block[0][0]
        f_idx = B_block[0][1]
        h_idx = B_block[1][0]
        i_idx = B_block[1][1]
        
        # C1 (2x2)
        c00_idx = C_block[0][0]
        c01_idx = C_block[0][1]
        c10_idx = C_block[1][0]
        c11_idx = C_block[1][1]
        
        # m1..m7
        # m1: (a+d)(e+i)
        U[a_idx] += 1; U[d_idx] += 1
        V[e_idx] += 1; V[i_idx] += 1
        W[c00_idx] += 1; W[c11_idx] += 1
        start_rank += 1
        
        # m2: (c+d)e
        U[c_idx] += 1; U[d_idx] += 1
        V[e_idx] += 1
        W[c10_idx] += 1; W[c11_idx] += -1
        start_rank += 1
        
        # m3: a(f-i)
        U[a_idx] += 1
        V[f_idx] += 1; V[i_idx] += -1
        W[c01_idx] += 1; W[c11_idx] += 1
        start_rank += 1
        
        # m4: d(h-e)
        U[d_idx] += 1
        V[h_idx] += 1; V[e_idx] += -1
        W[c00_idx] += 1; W[c10_idx] += 1
        start_rank += 1
        
        # m5: (a+b)i
        U[a_idx] += 1; U[b_idx] += 1
        V[i_idx] += 1
        W[c00_idx] += -1; W[c01_idx] += 1
        start_rank += 1
        
        # m6: (c-a)(e+f)
        U[c_idx] += 1; U[a_idx] += -1
        V[e_idx] += 1; V[f_idx] += 1
        W[c11_idx] += 1
        start_rank += 1
        
        # m7: (b-d)(h+i)
        U[b_idx] += 1; U[d_idx] += -1
        V[h_idx] += 1; V[i_idx] += 1
        W[c00_idx] += 1
        start_rank += 1
        
        # Schoolbook for A_block * B_block[:, 2]
        # B2: [[g], [j]]
        g_idx = B_block[0][2]
        j_idx = B_block[1][2]
        
        # C2 (2x1)
        c02_idx = C_block[0][2]
        c12_idx = C_block[1][2]
        
        # p1: ag
        U[a_idx] += 1
        V[g_idx] += 1
        W[c02_idx] += 1
        start_rank += 1
        
        # p2: bj
        U[b_idx] += 1
        V[j_idx] += 1
        W[c02_idx] += 1
        start_rank += 1
        
        # p3: cg
        U[c_idx] += 1
        V[g_idx] += 1
        W[c12_idx] += 1
        start_rank += 1
        
        # p4: dj
        U[d_idx] += 1
        V[j_idx] += 1
        W[c12_idx] += 1
        start_rank += 1
        
        return start_rank
    