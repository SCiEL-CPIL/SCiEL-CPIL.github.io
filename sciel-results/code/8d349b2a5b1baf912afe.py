    def add_strassen(i_base, j_base, k_base):
        nonlocal rank_offset
        
        # M1: (u11+u22)(v11+v22) -> w11, w22
        U[(i_base+0)*m + (j_base+0), rank_offset] = 1
        U[(i_base+1)*m + (j_base+1), rank_offset] = 1
        V[(j_base+0)*p + (k_base+0), rank_offset] = 1
        V[(j_base+0)*p + (k_base+1), rank_offset] = 1
        V[(j_base+1)*p + (k_base+0), rank_offset] = 1
        V[(j_base+1)*p + (k_base+1), rank_offset] = 1
        W[(k_base+0)*n + (i_base+0), rank_offset] = 1 # w11
        W[(k_base+1)*n + (i_base+1), rank_offset] = 1 # w22
        rank_offset += 1
        
        # M2: (u21+u22)v11 -> w21
        U[(i_base+1)*m + (j_base+0), rank_offset] = 1
        U[(i_base+1)*m + (j_base+1), rank_offset] = 1
        V[(j_base+0)*p + (k_base+0), rank_offset] = 1
        W[(k_base+0)*n + (i_base+1), rank_offset] = 1 # w21
        rank_offset += 1
        
        # M3: u11(v12-v22) -> w12
        U[(i_base+0)*m + (j_base+0), rank_offset] = 1
        V[(j_base+0)*p + (k_base+1), rank_offset] = 1
        V[(j_base+1)*p + (k_base+1), rank_offset] = -1
        W[(k_base+1)*n + (i_base+0), rank_offset] = 1 # w12
        rank_offset += 1
        
        # M4: u22(v21-v11) -> w11, w21
        U[(i_base+1)*m + (j_base+1), rank_offset] = 1
        V[(j_base+1)*p + (k_base+0), rank_offset] = 1
        V[(j_base+0)*p + (k_base+0), rank_offset] = -1
        W[(k_base+0)*n + (i_base+0), rank_offset] = 1 # w11
        W[(k_base+0)*n + (i_base+1), rank_offset] = 1 # w21
        rank_offset += 1
        
        # M5: (u11+u12)v22 -> w12, w22
        U[(i_base+0)*m + (j_base+0), rank_offset] = 1
        U[(i_base+0)*m + (j_base+1), rank_offset] = 1
        V[(j_base+1)*p + (k_base+1), rank_offset] = 1
        W[(k_base+1)*n + (i_base+0), rank_offset] = 1 # w12
        W[(k_base+1)*n + (i_base+1), rank_offset] = -1 # w22 (Wait, w22 = M1 - M2 + M3 + M6. M5 is not in w22?)
        # Let's re-check w22 formula: w22 = M1 - M2 + M3 + M6.
        # w12 = M3 + M5.
        # So M5 contributes to w12 only?
        # No, standard Strassen:
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        # c11 = M1 + M4 - M5 + M7
        # c12 = M3 + M5
        # c21 = M2 + M4
        # c22 = M1 - M2 + M3 + M6
        
        # So M5 contributes to c11 and c12.
        # c11 coeff: -1
        # c12 coeff: 1
        
        W[(k_base+0)*n + (i_base+0), rank_offset] = -1 # w11 (c11)
        W[(k_base+1)*n + (i_base+0), rank_offset] = 1 # w12 (c12)
        rank_offset += 1
        
        # M6: (u21-u11)(v11+v12) -> w22
        U[(i_base+1)*m + (j_base+0), rank_offset] = 1
        U[(i_base+0)*m + (j_base+0), rank_offset] = -1
        V[(j_base+0)*p + (k_base+0), rank_offset] = 1
        V[(j_base+0)*p + (k_base+1), rank_offset] = 1
        W[(k_base+1)*n + (i_base+1), rank_offset] = 1 # w22 (c22)
        rank_offset += 1
        
        # M7: (u12-u22)(v21+v22) -> w11
        U[(i_base+0)*m + (j_base+1), rank_offset] = 1
        U[(i_base+1)*m + (j_base+1), rank_offset] = -1
        V[(j_base+1)*p + (k_base+0), rank_offset] = 1
        V[(j_base+1)*p + (k_base+1), rank_offset] = 1
        W[(k_base+0)*n + (i_base+0), rank_offset] = 1 # w11 (c11)
        rank_offset += 1
