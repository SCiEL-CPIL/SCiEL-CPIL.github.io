import time

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 5, 7
    
    # Partition
    row_parts = [2, 1]
    mid_parts = [2, 2, 1]
    col_parts = [2, 2, 2, 1]
    
    row_offsets = [0]
    for rp in row_parts[:-1]:
        row_offsets.append(row_offsets[-1] + rp)
    mid_offsets = [0]
    for mp in mid_parts[:-1]:
        mid_offsets.append(mid_offsets[-1] + mp)
    col_offsets = [0]
    for cp in col_parts[:-1]:
        col_offsets.append(col_offsets[-1] + cp)
    
    # Use dictionaries to accumulate entries
    U_dict = {}  # (row, col) -> val
    V_dict = {}
    W_dict = {}
    
    col_idx = 0
    
    for ri, rn in enumerate(row_parts):
        for mi, mn in enumerate(mid_parts):
            for pi, pn in enumerate(col_parts):
                r_off = row_offsets[ri]
                m_off = mid_offsets[mi]
                p_off = col_offsets[pi]
                
                if rn == 2 and mn == 2 and pn == 2:
                    # Strassen 2x2x2, rank 7
                    # Local indices: A[i,j] -> global (r_off+i, m_off+j)
                    #                B[j,k] -> global (m_off+j, p_off+k)
                    #                C[i,k] -> W index (p_off+k)*n + (r_off+i)
                    
                    # Strassen terms with (U_coeffs, V_coeffs, W_coeffs)
                    # U_coeffs: {local_A_index: coeff}
                    # V_coeffs: {local_B_index: coeff}
                    # W_coeffs: {local_C_index: coeff}
                    
                    # Local A index: i*2+j (for 2x2)
                    # Local B index: j*2+k (for 2x2)
                    # Local C index: k*2+i (for 2x2, but we use k*n+i globally)
                    
                    strassen = [
                        # m1 = (A00+A11)(B00+B11)
                        ({0:1, 3:1}, {0:1, 3:1}, {0:1, 3:1}),  # C00+=m1, C11+=m1
                        # m2 = (A10+A11)B00
                        ({2:1, 3:1}, {0:1}, {2:1}),  # C10+=m2
                        # m3 = A00(B01-B11)
                        ({0:1}, {1:1, 3:-1}, {1:1}),  # C01+=m3
                        # m4 = A11(B10-B00)
                        ({3:1}, {2:1, 0:-1}, {0:1, 2:1}),  # C00+=m4, C10+=m4
                        # m5 = (A00+A01)B11
                        ({0:1, 1:1}, {3:1}, {1:1, 0:-1}),  # C01+=m5, C00-=m5
                        # m6 = (A10-A00)(B00+B01)
                        ({2:1, 0:-1}, {0:1, 1:1}, {3:1}),  # C11+=m6
                        # m7 = (A01-A11)(B10+B11)
                        ({1:1, 3:-1}, {2:1, 3:1}, {0:1}),  # C00+=m7
                    ]
                    
                    for u_c, v_c, w_c in strassen:
                        for la, val in u_c.items():
                            li, lj = divmod(la, 2)  # la = i*2+j
                            gr = r_off + li
                            gm = m_off + lj
                            U_dict[(gr, col_idx)] = U_dict.get((gr, col_idx), 0) + val
                        for lb, val in v_c.items():
                            lj, lk = divmod(lb, 2)  # lb = j*2+k
                            gm = m_off + lj
                            gp = p_off + lk
                            V_dict[(gm, col_idx)] = V_dict.get((gm, col_idx), 0) + val
                        for lc, val in w_c.items():
                            # lc is local C index in 0..3, representing k*2+i
                            lk, li = divmod(lc, 2)  # lc = k*2+i
                            w_row = (p_off + lk) * n + (r_off + li)
                            W_dict[(w_row, col_idx)] = W_dict.get((w_row, col_idx), 0) + val
                        col_idx += 1
                else:
                    # Schoolbook
                    for li in range(rn):
                        for lj in range(mn):
                            for lk in range(pn):
                                gr = r_off + li
                                gm = m_off + lj
                                gp = p_off + lk
                                U_dict[(gr, col_idx)] = U_dict.get((gr, col_idx), 0) + 1
                                V_dict[(gm, col_idx)] = V_dict.get((gm, col_idx), 0) + 1
                                w_row = gp * n + gr
                                W_dict[(w_row, col_idx)] = W_dict.get((w_row, col_idx), 0) + 1
                                col_idx += 1
    
    rank = col_idx
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for (row, col), val in U_dict.items():
        U[row][col] = val
    for (row, col), val in V_dict.items():
        V[row][col] = val
    for (row, col), val in W_dict.items():
        W[row][col] = val
    
    return {"U": U, "V": V, "W": W}
