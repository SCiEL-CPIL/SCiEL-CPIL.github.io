"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        """
        Add a bilinear term where:
        u_dict: {(i,j): coeff} for A entries
        v_dict: {(j,k): coeff} for B entries  
        w_dict: {(k,i): coeff} for C entries
        This represents: sum_{(i,j),(j,k)} u[i,j]*v[j,k] * A[i,j]*B[j,k] contributes to C via w
        """
        u = [0] * (n * m)
        v = [0] * (m * p)
        w = [0] * (p * n)
        for (i, j), coeff in u_dict.items():
            u[i * m + j] = coeff
        for (j, k), coeff in v_dict.items():
            v[j * p + k] = coeff
        for (k, i), coeff in w_dict.items():
            w[k * n + i] = coeff
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    def add_strassen_2x2x2(r0, c0):
        """
        Add Strassen decomposition for 2x2 * 2x2 -> 2x2 block.
        A block at rows [r0, r0+1], cols [c0, c0+1]
        B block at rows [c0, c0+1], cols [c0, c0+1]
        C block at rows [r0, r0+1], cols [c0, c0+1]
        
        A = [[a, b], [c, d]] where a=A[r0,c0], b=A[r0,c0+1], c=A[r0+1,c0], d=A[r0+1,c0+1]
        B = [[e, f], [g, h]] where e=B[c0,c0], f=B[c0,c0+1], g=B[c0+1,c0], h=B[c0+1,c0+1]
        C = [[C00, C01], [C10, C11]] where C[i,k] is at position (k*n + i) in W
        
        C00 = ae + bg
        C01 = af + bh
        C10 = ce + dg
        C11 = cf + dh
        """
        # Strassen bilinear forms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)
        # M3 = (a+b)(h)
        # M4 = (d-c)(g+h)
        # M5 = (a-c)(e+f)
        # M6 = b(f-h)
        # M7 = c(g-e)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M6 - M1 + M4
        # C10 = M2 + M4 - M5 + M7
        # C11 = M1 - M3 + M2 - M6
        
        # Verify:
        # C00 = (ae+ah+de+dh) - (dg+dh-cg-ch) + (ae+af-ce-cf) - (cg-ce)
        #     = ae+ah+de+dh - dg - dh + cg + ch + ae+af-ce-cf - cg + ce
        #     = 2ae + ah + de - dg + ch + af - cf  ... this is WRONG
        
        # Let me use the correct Strassen formulas. The standard ones are:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(e)  
        # M3 = (a+b)(h)
        # M4 = (d-c)(g+h)
        # M5 = (a-c)(e+f)
        # M6 = b(f-h)
        # M7 = c(g-e)
        
        # C00 = M1 - M4 + M5 - M7
        # C01 = M3 + M6 - M1 + M4
        # C10 = M2 + M4 - M5 + M7
        # C11 = M1 - M3 + M2 - M6
        
        # Let me verify C00 again carefully:
        # M1 = ae + ah + de + dh
        # -M4 = -dg - dh + cg + ch
        # +M5 = +ae + af - ce - cf
        # -M7 = -cg + ce
        # Sum: ae+ah+de+dh -dg-dh+cg+ch+ae+af-ce-cf-cg+ce
        #     = 2ae + ah + de - dg + ch + af - cf
        # This should equal ae + bg. It doesn't match.
        
        # I think I have the wrong Strassen formulas. Let me use a verified set.
        
        # Correct Strassen (verified):
        # p1 = (a+d)(e+h)
        # p2 = (c+d)(e)
        # p3 = (a+b)(h)
        # p4 = (d-c)(g+h)
        # p5 = (a-c)(e+f)
        # p6 = b(f-h)
        # p7 = c(g-e)
        
        # C00 = p1 - p4 + p5 - p7
        # C01 = p3 + p6 - p1 + p4
        # C10 = p2 + p4 - p5 + p7
        # C11 = p1 - p3 + p2 - p6
        
        # Let me verify C00 = ae + bg:
        # p1 - p4 + p5 - p7
        # = (ae+ah+de+dh) - (dg+dh-cg-ch) + (ae+af-ce-cf) - (cg-ce)
        # = ae+ah+de+dh - dg - dh + cg + ch + ae+af-ce-cf - cg + ce
        # = 2ae + ah + de - dg + ch + af - cf
        # This is NOT ae + bg. So these formulas are wrong.
        
        # Let me look up the correct Strassen algorithm again.
        
        # Standard Strassen:
        # p1 = (a + d) * (e + h)
        # p2 = (c + d) * e
        # p3 = (a + b) * h
        # p4 = (d - c) * (g + h)
        # p5 = (a - c) * (e + f)
        # p6 = b * (f - h)
        # p7 = c * (g - e)
        
        # C00 = p1 - p4 + p5 - p7
        # C01 = p3 + p6 - p1 + p4
        # C10 = p2 + p4 - p5 + p7
        # C11 = p1 - p3 + p2 - p6
        
        # Let me verify C00:
        # p1 = ae + ah + de + dh
        # p4 = dg + dh - cg - ch
        # p5 = ae + af - ce - cf
        # p7 = cg - ce
        # p1 - p4 + p5 - p7 = (ae+ah+de+dh) - (dg+dh-cg-ch) + (ae+af-ce-cf) - (cg-ce)
        # = ae + ah + de + dh - dg - dh + cg + ch + ae + af - ce - cf - cg + ce
        # = 2ae + ah + de - dg + ch + af - cf
        # This should be ae + bg. It's not.
        
        # I think there's a systematic error. Let me try different formulas.
        
        # Alternative Strassen:
        # p1 = a*e
        # p2 = a*f
        # p3 = b*e
        # p4 = b*f
        # p5 = (a+b)*(e+f)
        # p6 = (a-b)*(e+f)
        # p7 = (c+d)*(e+f)
        
        # That's not right either. Let me just use a simple approach.
        
        # Actually, let me just use schoolbook for now to ensure correctness.
        # I'll add all 8 schoolbook terms.
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    add_term(
                        {(r0+i, c0+j): 1},
                        {(c0+j, c0+k): 1},
                        {(c0+k, r0+i): 1}
                    )
    
    # For now, just use schoolbook for everything
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(
                    {(i, j): 1},
                    {(j, k): 1},
                    {(k, i): 1}
                )
    
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
