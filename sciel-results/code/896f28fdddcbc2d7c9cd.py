import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each 2x2 * 2x2 block uses the rank-7 Strassen decomposition.
    Total rank = 5 * 7 = 35.
    
    The B matrix is 2x10, which can be viewed as 5 blocks of 2x2 (columns 0-1, 2-3, ..., 8-9).
    The output C is 2x10, with each 2x2 block corresponding to one 2x2 submatrix of B.
    
    Indexing conventions:
    - A is n x m = 2 x 2, indexed as A[i, j]
    - B is m x p = 2 x 10, indexed as B[j, k]
    - C is n x p = 2 x 10, indexed as C[i, k]
    
    For the r-th block (r = 0..4), B's columns are 2*r and 2*r+1.
    The r-th block computes: C_block_r = A * B_block_r where B_block_r is 2x2.
    
    U has shape (n*m, rank) = (4, 35)
    V has shape (m*p, rank) = (20, 35)
    W has shape (p*n, rank) = (20, 35) with W[k*n + i, c]
    
    For each block r and each Strassen term t (0..6):
      U[i*m + j, r*7 + t] = U_strassen[i, j, t]
      V[j*p + (2*r + k), r*7 + t] = V_strassen[j, k, t]
      W[(2*r + k)*n + i, r*7 + t] = W_strassen[k, i, t]
    """
    n, m, p = DIMENSIONS
    # Number of 2x2 blocks in the p dimension
    num_blocks = p // 2  # 5 blocks
    
    # Strassen rank-7 decomposition for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = A * B
    #
    # Strassen's 7 products:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)e
    # M3 = a(f - h)
    # M4 = d(g - e)
    # M5 = (a + b)h
    # M6 = (c - a)(e + f)
    # M7 = (b - d)(g + h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # We need to express each product as a tensor term.
    # Each term t has U, V, W vectors such that:
    #   sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = C[i,k]
    #
    # For 2x2, indices: i,j in {0,1}, k in {0,1}
    # A[i,j], B[j,k], C[i,k]
    
    # Let's define the 7 Strassen terms.
    # For each term, we have factors that multiply to give the intermediate product,
    # and then W combines them.
    
    # Term 0: M1 = (a+d)(e+h)
    #   U factors: a+d means U[0,0]=1, U[1,1]=1
    #   V factors: e+h means V[0,0]=1, V[1,1]=1
    #   W: contributes to C[0,0] with +1, C[1,1] with +1
    #   W[k*n+i, t]: W[0*2+0]=W[0]=+1 (C[0,0]), W[1*2+1]=W[3]=+1 (C[1,1])
    
    # Term 1: M2 = (c+d)e
    #   U: c+d means U[1,0]=1, U[1,1]=1
    #   V: e means V[0,0]=1
    #   W: contributes to C[1,0] with +1, C[1,1] with -1
    #   W[0*2+1]=W[1]=+1 (C[1,0]), W[1*2+1]=W[3]=-1 (C[1,1])
    
    # Term 2: M3 = a(f-h)
    #   U: a means U[0,0]=1
    #   V: f-h means V[0,1]=1, V[1,1]=-1
    #   W: contributes to C[0,1] with +1, C[1,1] with +1
    #   W[1*2+0]=W[2]=+1 (C[0,1]), W[1*2+1]=W[3]=+1 (C[1,1])
    
    # Term 3: M4 = d(g-e)
    #   U: d means U[1,1]=1
    #   V: g-e means V[1,0]=1, V[0,0]=-1
    #   W: contributes to C[0,0] with +1, C[1,0] with +1
    #   W[0*2+0]=W[0]=+1 (C[0,0]), W[0*2+1]=W[1]=+1 (C[1,0])
    
    # Term 4: M5 = (a+b)h
    #   U: a+b means U[0,0]=1, U[0,1]=1
    #   V: h means V[1,1]=1
    #   W: contributes to C[0,0] with -1, C[0,1] with +1
    #   W[0*2+0]=W[0]=-1 (C[0,0]), W[1*2+0]=W[2]=+1 (C[0,1])
    
    # Term 5: M6 = (c-a)(e+f)
    #   U: c-a means U[1,0]=1, U[0,0]=-1
    #   V: e+f means V[0,0]=1, V[0,1]=1
    #   W: contributes to C[1,1] with +1
    #   W[1*2+1]=W[3]=+1 (C[1,1])
    
    # Term 6: M7 = (b-d)(g+h)
    #   U: b-d means U[0,1]=1, U[1,1]=-1
    #   V: g+h means V[1,0]=1, V[1,1]=1
    #   W: contributes to C[0,0] with +1
    #   W[0*2+0]=W[0]=+1 (C[0,0])
    
    # Let me verify:
    # C[0,0] = M1 + M4 - M5 + M7
    #   = (a+d)(e+h) + d(g-e) - (a+b)h + (b-d)(g+h)
    #   = ae + ah + de + dh + dg - de - ah - bh + bg + bh - dg - dh
    #   = ae + bg
    #   Correct!
    
    # C[0,1] = M3 + M5
    #   = a(f-h) + (a+b)h
    #   = af - ah + ah + bh
    #   = af + bh
    #   Correct!
    
    # C[1,0] = M2 + M4
    #   = (c+d)e + d(g-e)
    #   = ce + de + dg - de
    #   = ce + dg
    #   Correct!
    
    # C[1,1] = M1 - M2 + M3 + M6
    #   = (a+d)(e+h) - (c+d)e + a(f-h) + (c-a)(e+f)
    #   = ae + ah + de + dh - ce - de + af - ah + ce + cf - ae - af
    #   = dh + cf
    #   Correct!
    
    # Now define the Strassen terms as (U_entries, V_entries, W_entries)
    # U_entries: dict from (i,j) to value
    # V_entries: dict from (j,k) to value
    # W_entries: dict from (k,i) to value
    
    strassen_terms = [
        # Term 0: M1 = (a+d)(e+h)
        (
            {(0,0): 1, (1,1): 1},   # U: a+d
            {(0,0): 1, (1,1): 1},   # V: e+h
            {(0,0): 1, (1,1): 1}    # W: C[0,0]+=1, C[1,1]+=1
        ),
        # Term 1: M2 = (c+d)e
        (
            {(1,0): 1, (1,1): 1},   # U: c+d
            {(0,0): 1},             # V: e
            {(0,1): 1, (1,1): -1}   # W: C[1,0]+=1, C[1,1]-=1
        ),
        # Term 2: M3 = a(f-h)
        (
            {(0,0): 1},             # U: a
            {(0,1): 1, (1,1): -1},  # V: f-h
            {(1,0): 1, (1,1): 1}    # W: C[0,1]+=1, C[1,1]+=1
        ),
        # Term 3: M4 = d(g-e)
        (
            {(1,1): 1},             # U: d
            {(1,0): 1, (0,0): -1},  # V: g-e
            {(0,0): 1, (0,1): 1}    # W: C[0,0]+=1, C[1,0]+=1
        ),
        # Term 4: M5 = (a+b)h
        (
            {(0,0): 1, (0,1): 1},   # U: a+b
            {(1,1): 1},             # V: h
            {(0,0): -1, (1,0): 1}   # W: C[0,0]-=1, C[0,1]+=1
        ),
        # Term 5: M6 = (c-a)(e+f)
        (
            {(1,0): 1, (0,0): -1},  # U: c-a
            {(0,0): 1, (0,1): 1},   # V: e+f
            {(1,1): 1}              # W: C[1,1]+=1
        ),
        # Term 6: M7 = (b-d)(g+h)
        (
            {(0,1): 1, (1,1): -1},  # U: b-d
            {(1,0): 1, (1,1): 1},   # V: g+h
            {(0,0): 1}              # W: C[0,0]+=1
        ),
    ]
    
    rank = num_blocks * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    for r in range(num_blocks):
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = strassen_terms[t]
            
            for (i, j), val in u_entries.items():
                U[i * m + j][col] = val
            
            for (j, k), val in v_entries.items():
                # B's column for this block is 2*r + k
                V[j * p + (2 * r + k)][col] = val
            
            for (k, i), val in w_entries.items():
                # C's column for this block is 2*r + k
                W[(2 * r + k) * n + i][col] = val
    
    # Optional: Verify the solution to be safe
    # Convert to numpy for quick check
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check a few random entries
    # The tensor T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # Non-zero entries are at a=i*m+j, b=j*p+k, c=k*n+i
    # We expect T[a,b,c] = 1 for these, and 0 otherwise.
    
    # We can just trust the Strassen construction, but a quick check on one block is cheap.
    # Block 0:
    # A = [[1, 2], [3, 4]] -> U indices 0,1,2,3
    # B = [[5, 6], [7, 8]] -> V indices 0,1,10,11 (since p=10, j=0->0,1; j=1->10,11)
    # C = [[19, 22], [43, 50]] -> W indices 0,1,2,3 (k=0->0,1; k=1->2,3)
    
    # Let's just return the lists as required.
    return {"U": U, "V": V, "W": W}
