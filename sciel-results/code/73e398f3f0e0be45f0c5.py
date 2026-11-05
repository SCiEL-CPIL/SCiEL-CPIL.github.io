import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Start with schoolbook rank 27
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Apply Strassen to the 2x2x2 sub-block (i,j,k in {0,1})
    # The 8 terms are at indices (i*3+j)*3+k for i,j,k in {0,1}
    # We replace these 8 columns with 7 Strassen columns.
    # New rank = 27 - 8 + 7 = 26
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4+M5-M7, M3+M5], [M2+M4, M1+M3-M2+M6]]
    
    # We need to map these to the tensor decomposition.
    # The 7 new terms will be added to the matrices.
    # We will remove the 8 old terms and add 7 new ones.
    
    # Indices of the 8 terms to remove
    remove_indices = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                idx = i * 3 + j * 3 + k # This is not the column index in U,V,W
                # Column index in schoolbook is i*9 + j*3 + k
                col_idx = i * 9 + j * 3 + k
                remove_indices.append(col_idx)
                
    # We will create new U, V, W with rank 26
    new_rank = 26
    new_U = np.zeros((n * m, new_rank))
    new_V = np.zeros((m * p, new_rank))
    new_W = np.zeros((p * n, new_rank))
    
    # Copy the 19 terms that are NOT in the 2x2x2 block
    # The 2x2x2 block corresponds to i,j,k in {0,1}
    # The other terms are those where i=2 or j=2 or k=2
    
    new_col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue # Skip the 8 terms
                new_U[i * m + j, new_col] = 1
                new_V[j * p + k, new_col] = 1
                new_W[k * n + i, new_col] = 1
                new_col += 1
                
    # Now add the 7 Strassen terms
    # These terms operate on i,j,k in {0,1}
    # U is indexed by (i, j) where i,j in {0,1}
    # V is indexed by (j, k) where j,k in {0,1}
    # W is indexed by (k, i) where k,i in {0,1}
    
    # Strassen M1: (a+d)(e+h)
    # a=A00, d=A11, e=B00, h=B11
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1? No, W is C_ki.
    # C00 gets M1. C11 gets M1.
    # So W[0,0]=1, W[1,1]=1.
    
    # Let's define the 7 terms explicitly
    # Term 1: M1
    new_U[0, new_col] = 1 # A00
    new_U[3, new_col] = 1 # A11 (index 1*3+1=4? No, i*m+j. i=1,j=1 -> 1*3+1=4)
    # Correction: U index is i*m+j.
    # i=0, j=0 -> 0
    # i=1, j=1 -> 1*3+1=4
    new_U[0, new_col] = 1
    new_U[4, new_col] = 1
    
    new_V[0, new_col] = 1 # B00 (j=0,k=0 -> 0*3+0=0)
    new_V[4, new_col] = 1 # B11 (j=1,k=1 -> 1*3+1=4)
    
    new_W[0, new_col] = 1 # C00 (k=0,i=0 -> 0*3+0=0)
    new_W[4, new_col] = 1 # C11 (k=1,i=1 -> 1*3+1=4)
    new_col += 1
    
    # Term 2: M2 = (c+d)e = (A10+A11)B00
    new_U[3, new_col] = 1 # A10 (i=1,j=0 -> 3)
    new_U[4, new_col] = 1 # A11
    new_V[0, new_col] = 1 # B00
    new_W[3, new_col] = 1 # C10 (k=0,i=1 -> 0*3+1=1? No, k*n+i. k=0,i=1 -> 1)
    # W index: k*n+i.
    # C10 is k=0, i=1. Index 0*3+1=1.
    new_W[1, new_col] = 1
    new_col += 1
    
    # Term 3: M3 = a(f-h) = A00(B01-B11)
    new_U[0, new_col] = 1 # A00
    new_V[1, new_col] = 1 # B01 (j=0,k=1 -> 1)
    new_V[4, new_col] = -1 # B11
    new_W[0, new_col] = 1 # C00? No, C01.
    # C01 is k=1, i=0. Index 1*3+0=3.
    new_W[3, new_col] = 1
    new_col += 1
    
    # Term 4: M4 = d(g-e) = A11(B10-B00)
    new_U[4, new_col] = 1 # A11
    new_V[3, new_col] = 1 # B10 (j=1,k=0 -> 3)
    new_V[0, new_col] = -1 # B00
    new_W[1, new_col] = 1 # C10 (k=0,i=1 -> 1)
    new_W[4, new_col] = 1 # C11 (k=1,i=1 -> 4)
    new_col += 1
    
    # Term 5: M5 = (a+b)h = (A00+A01)B11
    new_U[0, new_col] = 1 # A00
    new_U[1, new_col] = 1 # A01 (i=0,j=1 -> 1)
    new_V[4, new_col] = 1 # B11
    new_W[3, new_col] = 1 # C01 (k=1,i=0 -> 3)
    new_W[4, new_col] = 1 # C11 (k=1,i=1 -> 4)
    new_col += 1
    
    # Term 6: M6 = (c-a)(e+f) = (A10-A00)(B00+B01)
    new_U[3, new_col] = 1 # A10
    new_U[0, new_col] = -1 # A00
    new_V[0, new_col] = 1 # B00
    new_V[1, new_col] = 1 # B01
    new_W[1, new_col] = 1 # C10 (k=0,i=1 -> 1)
    new_col += 1
    
    # Term 7: M7 = (b-d)(g+h) = (A01-A11)(B10+B11)
    new_U[1, new_col] = 1 # A01
    new_U[4, new_col] = -1 # A11
    new_V[3, new_col] = 1 # B10
    new_V[4, new_col] = 1 # B11
    new_W[0, new_col] = -1 # C00 (k=0,i=0 -> 0)
    new_W[4, new_col] = 1 # C11 (k=1,i=1 -> 4)
    new_col += 1
    
    # Verify new_col is 26
    if new_col != 26:
        # Fallback to schoolbook if logic error
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": new_U.tolist(), "V": new_V.tolist(), "W": new_W.tolist()}
# EVOLVE-BLOCK-END
