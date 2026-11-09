<<<<<<< SEARCH
    def add_strassen_block(i0, j0, k0, base_rank):
        # 7 multiplications
        # M1: (A[i0, j0] + A[i0+1, j0+1]) * (B[j0, k0] + B[j0+1, k0+1])
        # Indices:
        # A indices: i0*m + j0, (i0+1)*m + (j0+1)
        # B indices: j0*p + k0, (j0+1)*p + (k0+1)
        # C indices: k0*n + i0, (k0+1)*n + (i0+1) -- Wait, W index is k*n + i.
        
        # Let's define the 7 terms.
        # Each term is a product of a linear combination of A and a linear combination of B.
        # U entries are coefficients for A.
        # V entries are coefficients for B.
        # W entries are coefficients for C.
        
        # Term 1: M1
        # U: +1 at i0*m+j0, +1 at (i0+1)*m+(j0+1)
        # V: +1 at j0*p+k0, +1 at (j0+1)*p+(k0+1)
        # W: +1 at k0*n+i0 (for C[i0, k0]? No, C[i, k] = sum ...
        # The output C[i, k] gets contributions from M1, M4, M5, M7.
        # C[i0, k0] += M1 - M4 + M5 - M7 ? No.
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        
        # So for each M_x, we add +1 or -1 to the appropriate W entries.
        
        # M1 contributes to C[0,0] and C[1,1].
        # C[0,0] index: k0*n + i0.
        # C[1,1] index: (k0+1)*n + (i0+1).
        
        # M1:
        U[i0*m + j0][base_rank] = 1
        U[(i0+1)*m + (j0+1)][base_rank] = 1
        V[j0*p + k0][base_rank] = 1
        V[(j0+1)*p + (k0+1)][base_rank] = 1
        W[k0*n + i0][base_rank] = 1
        W[(k0+1)*n + (i0+1)][base_rank] = 1
        
        # M2: (c+d)e = (A[1,0] + A[1,1]) * B[0,0]
        # C[1,0] += M2
        U[i0*m + j0+1][base_rank+1] = 1 # Wait, A[1,0] is i0+1, j0. A[1,1] is i0+1, j0+1.
        # Correction:
        # A_sub = [[A[i0,j0], A[i0,j0+1]], [A[i0+1,j0], A[i0+1,j0+1]]]
        # c = A[i0+1, j0], d = A[i0+1, j0+1]
        # e = B[j0, k0]
        U[(i0+1)*m + j0][base_rank+1] = 1
        U[(i0+1)*m + (j0+1)][base_rank+1] = 1
        V[j0*p + k0][base_rank+1] = 1
        W[(k0)*n + (i0+1)][base_rank+1] = 1 # C[1,0] index: k0*n + (i0+1)
        
        # M3: a(f-h) = A[0,0] * (B[0,1] - B[1,1])
        # C[0,1] += M3
        U[i0*m + j0][base_rank+2] = 1
        V[j0*p + (k0+1)][base_rank+2] = 1
        V[(j0+1)*p + (k0+1)][base_rank+2] = -1
        W[(k0+1)*n + i0][base_rank+2] = 1 # C[0,1] index: (k0+1)*n + i0
        
        # M4: d(g-e) = A[1,1] * (B[1,0] - B[0,0])
        # C[0,0] += -M4, C[1,0] += M4
        U[(i0+1)*m + (j0+1)][base_rank+3] = 1
        V[(j0+1)*p + k0][base_rank+3] = 1
        V[j0*p + k0][base_rank+3] = -1
        W[k0*n + i0][base_rank+3] = -1
        W[k0*n + (i0+1)][base_rank+3] = 1
        
        # M5: (a+b)h = (A[0,0] + A[0,1]) * B[1,1]
        # C[0,0] += -M5, C[0,1] += M5
        U[i0*m + j0][base_rank+4] = 1
        U[i0*m + (j0+1)][base_rank+4] = 1
        V[(j0+1)*p + (k0+1)][base_rank+4] = 1
        W[k0*n + i0][base_rank+4] = -1
        W[(k0+1)*n + i0][base_rank+4] = 1
        
        # M6: (c-a)(e+f) = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # C[1,1] += M6
        U[(i0+1)*m + j0][base_rank+5] = 1
        U[i0*m + j0][base_rank+5] = -1
        V[j0*p + k0][base_rank+5] = 1
        V[j0*p + (k0+1)][base_rank+5] = 1
        W[(k0+1)*n + (i0+1)][base_rank+5] = 1
        
        # M7: (b-d)(g+h) = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        # C[0,0] += M7
        U[i0*m + (j0+1)][base_rank+6] = 1
        U[(i0+1)*m + (j0+1)][base_rank+6] = -1
        V[(j0+1)*p + k0][base_rank+6] = 1
        V[(j0+1)*p + (k0+1)][base_rank+6] = 1
        W[k0*n + i0][base_rank+6] = 1
        
        return base_rank + 7
=======
    def add_strassen_block(i0, j0, k0, base_rank):
        """
        Add a Strassen block for multiplying A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        to produce C[i0:i0+2, k0:k0+2].
        
        A_sub = [[a, b], [c, d]] where:
          a = A[i0, j0], b = A[i0, j0+1]
          c = A[i0+1, j0], d = A[i0+1, j0+1]
        
        B_sub = [[e, f], [g, h]] where:
          e = B[j0, k0], f = B[j0, k0+1]
          g = B[j0+1, k0], h = B[j0+1, k0+1]
        
        Strassen:
          M1 = (a+d)(e+h)
          M2 = (c+d)e
          M3 = a(f-h)
          M4 = d(g-e)
          M5 = (a+b)h
          M6 = (c-a)(e+f)
          M7 = (b-d)(g+h)
        
        C_sub = [[M1-M4+M5-M7, M3+M5],
                 [M2+M4,         M1-M2+M3+M6]]
        
        W index convention: W[k*n+i, r] for C[i,k]
        """
        # Define indices
        a_idx = i0 * m + j0        # A[i0, j0]
        b_idx = i0 * m + (j0+1)    # A[i0, j0+1]
        c_idx = (i0+1) * m + j0    # A[i0+1, j0]
        d_idx = (i0+1) * m + (j0+1) # A[i0+1, j0+1]
        
        e_idx = j0 * p + k0        # B[j0, k0]
        f_idx = j0 * p + (k0+1)    # B[j0, k0+1]
        g_idx = (j0+1) * p + k0    # B[j0+1, k0]
        h_idx = (j0+1) * p + (k0+1) # B[j0+1, k0+1]
        
        # C indices: C[i, k] -> W[k*n+i, r]
        c00_idx = k0 * n + i0        # C[i0, k0]
        c01_idx = (k0+1) * n + i0    # C[i0, k0+1]
        c10_idx = k0 * n + (i0+1)    # C[i0+1, k0]
        c11_idx = (k0+1) * n + (i0+1) # C[i0+1, k0+1]
        
        # M1 = (a+d)(e+h)
        # Contributes to C[0,0] with +1, C[1,1] with +1
        U[a_idx][base_rank] = 1
        U[d_idx][base_rank] = 1
        V[e_idx][base_rank] = 1
        V[h_idx][base_rank] = 1
        W[c00_idx][base_rank] = 1
        W[c11_idx][base_rank] = 1
        
        # M2 = (c+d)e
        # Contributes to C[1,0] with +1
        U[c_idx][base_rank+1] = 1
        U[d_idx][base_rank+1] = 1
        V[e_idx][base_rank+1] = 1
        W[c10_idx][base_rank+1] = 1
        
        # M3 = a(f-h)
        # Contributes to C[0,1] with +1
        U[a_idx][base_rank+2] = 1
        V[f_idx][base_rank+2] = 1
        V[h_idx][base_rank+2] = -1
        W[c01_idx][base_rank+2] = 1
        
        # M4 = d(g-e)
        # Contributes to C[0,0] with -1, C[1,0] with +1
        U[d_idx][base_rank+3] = 1
        V[g_idx][base_rank+3] = 1
        V[e_idx][base_rank+3] = -1
        W[c00_idx][base_rank+3] = -1
        W[c10_idx][base_rank+3] = 1
        
        # M5 = (a+b)h
        # Contributes to C[0,0] with -1, C[0,1] with +1
        U[a_idx][base_rank+4] = 1
        U[b_idx][base_rank+4] = 1
        V[h_idx][base_rank+4] = 1
        W[c00_idx][base_rank+4] = -1
        W[c01_idx][base_rank+4] = 1
        
        # M6 = (c-a)(e+f)
        # Contributes to C[1,1] with +1
        U[c_idx][base_rank+5] = 1
        U[a_idx][base_rank+5] = -1
        V[e_idx][base_rank+5] = 1
        V[f_idx][base_rank+5] = 1
        W[c11_idx][base_rank+5] = 1
        
        # M7 = (b-d)(g+h)
        # Contributes to C[0,0] with -1
        U[b_idx][base_rank+6] = 1
        U[d_idx][base_rank+6] = -1
        V[g_idx][base_rank+6] = 1
        V[h_idx][base_rank+6] = 1
        W[c00_idx][base_rank+6] = -1
        
        return base_rank + 7
>>>>>>> REPLACE

