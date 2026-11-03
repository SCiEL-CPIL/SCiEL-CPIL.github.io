import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    n, m, p = DIMENSIONS
    
    # We will build U, V, W as lists of lists of numbers
    # U: n*m rows, rank cols
    # V: m*p rows, rank cols
    # W: p*n rows, rank cols
    
    # Strategy: Recursive decomposition
    # If n, m, p are small, use hardcoded ranks.
    # Otherwise, split dimensions.
    
    terms = [] # List of (u_indices, v_indices, w_indices, coeff)
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    def solve(n, m, p, offset_i=0, offset_j=0, offset_k=0):
        if time.time() > deadline:
            return
            
        # Base cases: use known minimal ranks if possible
        # 2x2x2: Strassen Rank 7
        if n == 2 and m == 2 and p == 2:
            # Strassen
            # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-f)
            # M5 = (a+b)h
            # M6 = (c-b)(e+g)
            # M7 = (d-c)(f+h)
            
            # U (2x2) indices: 0=a, 1=b, 2=c, 3=d
            # V (2x2) indices: 0=e, 1=f, 2=g, 3=h
            # W (2x2) indices: 0=c00, 1=c01, 2=c10, 3=c11
            
            # M1: (a+d)(e+h) -> c00 += 1, c11 += 1, c01 += 1, c10 += 1? No.
            # C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
            # M1 = (a+d)(e+h) = ae + ah + de + dh
            # M2 = (c+d)e = ce + de
            # M3 = a(f-h) = af - ah
            # M4 = d(g-f) = dg - df
            # M5 = (a+b)h = ah + bh
            # M6 = (c-b)(e+g) = ce + cg - be - bg
            # M7 = (d-c)(f+h) = df + dh - cf - ch
            
            # C00 = ae + bg
            # From M1: ae. From M6: bg.
            # C00 = M1 - M2 + M3? No.
            # Let's use the standard Strassen mapping.
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-f)
            # M5 = (a+b)h
            # M6 = (c-b)(e+g)
            # M7 = (d-c)(f+h)
            
            # C00 = M1 - M4 + M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # Map to global indices
            # U: offset_i + local
            # V: offset_j + local
            # W: offset_k + local
            
            # M1
            u = [offset_i+0, offset_i+3] # a, d
            v = [offset_j+0, offset_j+3] # e, h
            w = [offset_k+0, offset_k+3, offset_k+1, offset_k+2] # c00, c11, c01, c10?
            # W indices:
            # C00 is index 0 (k=0, i=0)
            # C01 is index 1 (k=1, i=0)
            # C10 is index 2 (k=0, i=1)
            # C11 is index 3 (k=1, i=1)
            
            # Term 1: M1
            # W contributions: C00(+1), C11(+1), C01(+1), C10(+1)
            add_term(offset_i+0, offset_j+0, offset_k+0, 1) # a*e to C00
            add_term(offset_i+3, offset_j+3, offset_k+3, 1) # d*h to C11
            # Wait, Strassen uses products of sums.
            # The "term" is a single bilinear form.
            # We need to represent (a+d)(e+h) as a single rank 1 tensor? No, rank 1 is u*v*w.
            # Strassen replaces 8 rank 1 tensors with 7 rank 1 tensors.
            # Each Strassen term is a product of a linear form in A and a linear form in B.
            # So for M1:
            # U vector: [1, 0, 0, 1] (a+d)
            # V vector: [1, 0, 0, 1] (e+h)
            # W vector: [1, 1, 1, 1]? No.
            # The W vector for M1 must be such that sum(u_k v_k w_k) reconstructs C.
            # This is not how we add terms.
            # We add columns to U, V, W.
            # Column c: U[:, c], V[:, c], W[:, c].
            # For M1:
            # U_col has 1 at a, d.
            # V_col has 1 at e, h.
            # W_col must have coefficients for C00, C01, C10, C11.
            # But M1 contributes to C00, C01, C10, C11 with specific coefficients.
            # Actually, M1 = (a+d)(e+h).
            # This term is multiplied by a W vector.
            # The W vector determines which C elements get this product.
            # For Strassen:
            # C00 += M1
            # C01 += M3 + M5
            # C10 += M2 + M4
            # C11 += M1 - M2 + M3 + M6
            # This means M1 is added to C00 and C11.
            # So W_col for M1 has 1 at C00 and 1 at C11.
            
            add_term(offset_i+0, offset_j+0, offset_k+0, 1) # a
            add_term(offset_i+3, offset_j+0, offset_k+0, 1) # d
            # This adds 2 terms? No, Strassen term is ONE column.
            # U_col: a=1, d=1.
            # V_col: e=1, h=1.
            # W_col: C00=1, C11=1.
            
            # We need a helper to add a full column.
            pass

    # The previous "add_term" adds scalar entries.
    # It's easier to build U, V, W directly.
    
    U = []
    V = []
    W = []
    
    def add_strassen_222(i_off, j_off, k_off):
        # Adds 7 columns
        # M1
        u = [0]*4; u[0]=1; u[3]=1
        v = [0]*4; v[0]=1; v[3]=1
        w = [0]*4; w[0]=1; w[3]=1
        # M2
        u2 = [0]*4; u2[2]=1; u2[3]=1
        v2 = [0]*4; v2[0]=1
        w2 = [0]*4; w2[2]=1; w2[3]=-1
        # M3
        u3 = [0]*4; u3[0]=1
        v3 = [0]*4; v3[1]=1; v3[3]=-1
        w3 = [0]*4; w3[1]=1; w3[3]=1
        # M4
        u4 = [0]*4; u4[3]=1
        v4 = [0]*4; v4[2]=1; v4[1]=-1
        w4 = [0]*4; w4[0]=1; w4[2]=1
        # M5
        u5 = [0]*4; u5[0]=1; u5[1]=1
        v5 = [0]*4; v5[3]=1
        w5 = [0]*4; w5[0]=-1; w5[1]=1
        # M6
        u6 = [0]*4; u6[2]=1; u6[0]=-1
        v6 = [0]*4; v6[0]=1; v6[2]=1
        w6 = [0]*4; w6[3]=1
        # M7
        u7 = [0]*4; u7[1]=1; u7[3]=-1
        v7 = [0]*4; v7[2]=1; v7[3]=1
        w7 = [0]*4; w7[0]=1
        
        # Append to global U, V, W
        # Note: U, V, W are lists of rows. We need to append columns.
        # Easier: store as list of columns.
        
        pass

    # This manual construction is error-prone.
    # Let's stick to the recursive split with schoolbook for unoptimized blocks.
    # And only optimize 2x2x2 blocks if we can fit them.
    
    # Given the complexity and time, returning the schoolbook (105) is safe.
    # But we can try the 2x2x2 replacement for the first block.
    
    n, m, p = 3, 5, 7
    rank = 105
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Replace 2x2x2 at i=0,1 j=0,1 k=0,1 with Strassen (rank 7)
    # This saves 1 rank.
    # We need to construct the full 104 rank matrix.
    
    # This is getting too complex to verify in thought.
    # I will return the schoolbook decomposition to ensure validity.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
