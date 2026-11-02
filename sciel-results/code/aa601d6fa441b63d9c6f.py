"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Use Strassen's algorithm for pairs of columns (2x2 matrix mult)
    # and schoolbook for the last single column.
    # n=2, m=2, p=13
    # We have 6 pairs of columns (24 columns total) + 1 single column = 25 columns? 
    # Wait, 2x13 means p=13. So 13 columns.
    # 13 columns = 6 pairs (12 columns) + 1 single column.
    # For each 2x2 block (A is 2x2, B_block is 2x2), Strassen uses 7 multiplications.
    # For the single column (2x1), we use 4 multiplications (schoolbook).
    # Total rank = 6 * 7 + 4 = 46.
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (4, 46)
    # V: (m*p, rank) = (26, 46)
    # W: (p*n, rank) = (26, 46)
    
    rank = 6 * 7 + 4
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # A indices: A[i, j] where i in [0,1], j in [0,1]
    # U row index: i * m + j = i * 2 + j
    # B indices: B[j, k] where j in [0,1], k in [0,12]
    # V row index: j * p + k = j * 13 + k
    # C indices: C[k, i] where k in [0,12], i in [0,1]
    # W row index: k * n + i = k * 2 + i
    
    col = 0
    
    # Strassen's algorithm for 2x2 * 2x2
    # A = [[a, b], [c, d]] -> U indices: a=0, b=1, c=2, d=3
    # B = [[e, f], [g, h]] -> V indices for specific columns
    # C = [[C00, C01], [C10, C11]] -> W indices for specific rows
    
    # M1 = (a+d)(e+h)
    # M2 = (c+d)(e)
    # M3 = (a)(f-h)
    # M4 = (d)(g-e)
    # M5 = (a+b)(h)
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_block(k_start):
        """Add 7 rank-1 tensors for columns k_start and k_start+1."""
        nonlocal col
        # V indices for B columns k_start and k_start+1
        # e = B[0, k_start] -> V[13 + k_start]
        # f = B[0, k_start+1] -> V[13 + k_start+1]
        # g = B[1, k_start] -> V[26 + k_start]  Wait, V row is j*p+k. j=0 -> k, j=1 -> 13+k.
        # Correction:
        # B[j, k]. V row index = j * p + k.
        # j=0, k=k_start -> 0*13 + k_start = k_start
        # j=0, k=k_start+1 -> k_start+1
        # j=1, k=k_start -> 1*13 + k_start = 13 + k_start
        # j=1, k=k_start+1 -> 13 + k_start+1
        
        # W indices for C rows k_start and k_start+1
        # C[k, i]. W row index = k * n + i = k * 2 + i.
        # C[k_start, 0] -> 2*k_start
        # C[k_start, 1] -> 2*k_start + 1
        # C[k_start+1, 0] -> 2*(k_start+1)
        # C[k_start+1, 1] -> 2*(k_start+1) + 1
        
        # Define the 7 tensors
        # U vectors (4 elements: a,b,c,d)
        # V vectors (2 elements: coeff for B[0,ks], B[0,ks+1], B[1,ks], B[1,ks+1] -> actually 4 elements in V subspace, but we only fill specific rows)
        # W vectors (4 elements: C[ks,0], C[ks,1], C[ks+1,0], C[ks+1,1])
        
        # M1: (a+d) * (e+h) -> C00 + C11
        U_vec = [1, 0, 0, 1]
        V_rows = [k_start, 13 + k_start]
        V_vals = [1, 1]
        W_rows = [2*k_start, 2*k_start+2]
        W_vals = [1, 1]
        for idx in range(4):
            U[idx][col] += U_vec[idx]
        for v_r, v_v in zip(V_rows, V_vals):
            V[v_r][col] += v_v
        for w_r, w_v in zip(W_rows, W_vals):
            W[w_r][col] += w_v
        col += 1
        
        # M2: (c+d) * e -> C10 + C11 (with sign - for C11? No, C11 = M1 - M2 + M3 + M6)
        U_vec = [0, 0, 1, 1]
        V_rows = [k_start]
        V_vals = [1]
        W_rows = [2*k_start+1, 2*k_start+2+1] # C[ks,1] is 2ks+1? No.
        # C[ks, 0] = 2ks, C[ks, 1] = 2ks+1
        # C[ks+1, 0] = 2ks+2, C[ks+1, 1] = 2ks+3
        # C10 is C[ks, 1]? No.
        # A is 2x2, B is 2x2.
        # C = A B.
        # C[i, k].
        # i=0,1. k=ks, ks+1.
        # C[0, ks] = a e + b g.
        # C[0, ks+1] = a f + b h.
        # C[1, ks] = c e + d g.
        # C[1, ks+1] = c f + d h.
        
        # Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C[0, ks] = M1 + M4 - M5 + M7 ?
        # Let's check:
        # M1 + M4 - M5 + M7
        # = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
        # = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
        # = ae + bg. Correct.
        
        # C[0, ks+1] = M3 + M5 ?
        # M3 + M5 = a(f-h) + (a+b)h = af - ah + ah + bh = af + bh. Correct.
        
        # C[1, ks] = M2 + M4 ?
        # M2 + M4 = (c+d)e + d(g-e) = ce + de + dg - de = ce + dg. Correct.
        
        # C[1, ks+1] = M1 - M2 + M3 + M6 ?
        # M1 - M2 + M3 + M6
        # = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
        # = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
        # = dh + cf. Correct.
        
        # So:
        # W for M1: C[0,ks] (2ks) +1, C[1,ks+1] (2ks+3) +1
        # W for M2: C[1,ks] (2ks+1) +1, C[1,ks+1] (2ks+3) -1
        # W for M3: C[0,ks+1] (2ks+2) +1, C[1,ks+1] (2ks+3) +1
        # W for M4: C[0,ks] (2ks) +1, C[1,ks] (2ks+1) +1
        # W for M5: C[0,ks] (2ks) -1, C[0,ks+1] (2ks+2) +1
        # W for M6: C[1,ks+1] (2ks+3) +1
        # W for M7: C[0,ks] (2ks) +1
        
        # Let's re-fill M1
        # Already did M1 above?
        # Previous M1: W_rows [2ks, 2ks+2] vals [1, 1].
        # Correct M1 W: C[0,ks] (2ks) and C[1,ks+1] (2ks+3).
        # I made a mistake in the first M1 block. I need to reset or correct.
        # Since I'm building this from scratch in the code, I will just write the correct loops.
        # The `col` has been incremented. I should probably clear the previous work or just not do the first M1 block separately.
        # I will rewrite the function body cleanly.
        return

    # Reset and implement correctly
    col = 0
    for pair in range(6):
        ks = pair * 2
        # M1: (a+d)(e+h)
        U[:, col] = [1, 0, 0, 1]
        V[ks, col] = 1
        V[13+ks, col] = 1
        W[2*ks, col] = 1      # C[0,ks]
        W[2*ks+3, col] = 1    # C[1,ks+1]
        col += 1
        
        # M2: (c+d)e
        U[:, col] = [0, 0, 1, 1]
        V[ks, col] = 1
        W[2*ks+1, col] = 1    # C[1,ks]
        W[2*ks+3, col] = -1   # C[1,ks+1]
        col += 1
        
        # M3: a(f-h)
        U[:, col] = [1, 0, 0, 0]
        V[ks+1, col] = 1
        V[13+ks+1, col] = -1
        W[2*ks+2, col] = 1    # C[0,ks+1]
        W[2*ks+3, col] = 1    # C[1,ks+1]
        col += 1
        
        # M4: d(g-e)
        U[:, col] = [0, 0, 0, 1]
        V[13+ks, col] = 1
        V[ks, col] = -1
        W[2*ks, col] = 1      # C[0,ks]
        W[2*ks+1, col] = 1    # C[1,ks]
        col += 1
        
        # M5: (a+b)h
        U[:, col] = [1, 1, 0, 0]
        V[13+ks+1, col] = 1
        W[2*ks, col] = -1     # C[0,ks]
        W[2*ks+2, col] = 1    # C[0,ks+1]
        col += 1
        
        # M6: (c-a)(e+f)
        U[:, col] = [-1, 0, 1, 0]
        V[ks, col] = 1
        V[ks+1, col] = 1
        W[2*ks+3, col] = 1    # C[1,ks+1]
        col += 1
        
        # M7: (b-d)(g+h)
        U[:, col] = [0, 1, 0, -1]
        V[13+ks, col] = 1
        V[13+ks+1, col] = 1
        W[2*ks, col] = 1      # C[0,ks]
        col += 1

    # Last column: k=12
    ks = 12
    # Schoolbook for 2x2 * 2x1
    # C[0,12] = a e + b g
    # C[1,12] = c e + d g
    
    # T1: a e
    U[:, col] = [1, 0, 0, 0]
    V[ks, col] = 1
    W[2*ks, col] = 1
    col += 1
    
    # T2: b g
    U[:, col] = [0, 1, 0, 0]
    V[13+ks, col] = 1
    W[2*ks, col] = 1
    col += 1
    
    # T3: c e
    U[:, col] = [0, 0, 1, 0]
    V[ks, col] = 1
    W[2*ks+1, col] = 1
    col += 1
    
    # T4: d g
    U[:, col] = [0, 0, 0, 1]
    V[13+ks, col] = 1
    W[2*ks+1, col] = 1
    col += 1
    
    # Convert to lists as required by the interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
