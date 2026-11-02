   def add_strassen_block(U, V, W, base_idx, i_offset, j_offset, k_offset, n, m, p):
       # i_offset, j_offset, k_offset are the starting global indices for this 2x2x2 block
       # i_offset in [0, 1]
       # j_offset in [0, 2] (since m=4, j is 0..3)
       # k_offset in [0, 2, 4] (since p=6, k is 0..5)
       
       # Strassen factors for 2x2x2
       # M1
       # A part: (x00 + x11) -> coeffs: (0,0):1, (1,1):1
       # B part: (y00 + y11) -> coeffs: (0,0):1, (1,1):1
       # C part: z00: 1, z11: 1
       
       # We need to iterate through the 7 Strassen multiplications.
       
       # Define the 7 terms as tuples: (U_coeffs, V_coeffs, W_coeffs)
       # U_coeffs: dict mapping (local_i, local_j) to coeff
       # V_coeffs: dict mapping (local_j, local_k) to coeff
       # W_coeffs: dict mapping (local_k, local_i) to coeff
       
       # Note: local_i, local_j, local_k are 0 or 1.
       
       # M1
       u1 = {(0,0): 1, (1,1): 1}
       v1 = {(0,0): 1, (1,1): 1}
       w1 = {(0,0): 1, (1,1): 1} # z00 and z11 get +1
       
       # M2
       u2 = {(1,0): 1, (1,1): 1}
       v2 = {(0,0): 1}
       w2 = {(0,0): 1, (1,1): 1} # z10: +1, z11: -1? No.
       
       # Let's re-verify Strassen formulas carefully.
       # M1 = (x00 + x11)(y00 + y11)
       # M2 = (x10 + x11)y00
       # M3 = x00(y10 - y11)
       # M4 = x11(y01 - y00)
       # M5 = (x00 + x01)y11
       # M6 = (x10 - x00)(y00 + y01)
       # M7 = (x01 - x11)(y10 + y11)
       
       # z00 = M1 + M4 - M5 + M7
       # z01 = M3 + M5
       # z10 = M2 + M4
       # z11 = M1 - M2 + M3 + M6
       
       # So W coefficients for each M:
       # M1: z00:1, z11:1
       # M2: z10:1, z11:-1
       # M3: z01:1, z11:1
       # M4: z00:1, z10:1
       # M5: z00:-1, z01:1
       # M6: z11:1
       # M7: z00:1
       
       # Now define U and V coeffs for each M.
       
       terms = []
       
       # M1
       terms.append((
           {(0,0): 1, (1,1): 1},
           {(0,0): 1, (1,1): 1},
           {(0,0): 1, (1,1): 1}
       ))
       # M2
       terms.append((
           {(1,0): 1, (1,1): 1},
           {(0,0): 1},
           {(0,1): 1, (1,1): -1} # z10 is (1,0) in local? No.
           # local z indices: z00=(0,0), z01=(0,1), z10=(1,0), z11=(1,1).
           # So z10 is key (1,0) in W?
           # W maps (local_k, local_i) to coeff.
           # z10 corresponds to i=1, k=0. So local_i=1, local_k=0.
           # W key is (local_k, local_i) = (0, 1).
           # z11 corresponds to i=1, k=1. W key (1, 1).
           # So M2 W: {(0,1): 1, (1,1): -1}
       ))
       # Correction on W keys:
       # W[k*n+i, r].
       # Local k, i.
       # z00: i=0, k=0 -> key (0,0)
       # z01: i=0, k=1 -> key (1,0)
       # z10: i=1, k=0 -> key (0,1)
       # z11: i=1, k=1 -> key (1,1)
       
       # Let's redo W coeffs with this mapping.
       # M1: z00(0,0):1, z11(1,1):1
       # M2: z10(0,1):1, z11(1,1):-1
       # M3: z01(1,0):1, z11(1,1):1
       # M4: z00(0,0):1, z10(0,1):1
       # M5: z00(0,0):-1, z01(1,0):1
       # M6: z11(1,1):1
       # M7: z00(0,0):1
       
       # M3
       terms.append((
           {(0,0): 1},
           {(1,0): 1, (1,1): -1},
           {(1,0): 1, (1,1): 1}
       ))
       # M4
       terms.append((
           {(1,1): 1},
           {(0,1): 1, (0,0): -1},
           {(0,0): 1, (0,1): 1}
       ))
       # M5
       terms.append((
           {(0,0): 1, (0,1): 1},
           {(1,1): 1},
           {(0,0): -1, (1,0): 1}
       ))
       # M6
       terms.append((
           {(1,0): 1, (0,0): -1},
           {(0,0): 1, (0,1): 1},
           {(1,1): 1}
       ))
       # M7
       terms.append((
           {(0,1): 1, (1,1): -1},
           {(1,0): 1, (1,1): 1},
           {(0,0): 1}
       ))
       
       # Now populate U, V, W
       for u_coeff, v_coeff, w_coeff in terms:
           # Get current column index
           col = base_idx
           
           for (li, lj), val in u_coeff.items():
               gi = i_offset + li
               gj = j_offset + lj
               U[gi * m + gj][col] = val
           
           for (lj, lk), val in v_coeff.items():
               gj = j_offset + lj
               gk = k_offset + lk
               V[gj * p + gk][col] = val
           
           for (lk, li), val in w_coeff.items():
               gk = k_offset + lk
               gi = i_offset + li
               W[gk * n + gi][col] = val
           
           base_idx += 1
           
       return base_idx
   