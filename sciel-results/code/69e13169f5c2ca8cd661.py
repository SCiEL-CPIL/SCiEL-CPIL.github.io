def add_2x2x5(i0, j0, k0, col_start):
    """
    Decompose C[i0:i0+2, k0:k0+5] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+5]
    using rank-10 decomposition.
    """
    nonlocal col
    c = col_start
    
    # Strassen for first 2 columns (k0, k0+1)
    # M1 = (A00+A11)(B00+B11)
    U[i0*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    c += 1
    
    # M2 = (A10+A11)B00
    U[(i0+1)*m + j0][c] += 1
    U[(i0+1)*m + (j0+1)][c] += 1
    V[j0*p + k0][c] += 1
    W[k0*n + (i0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] -= 1
    c += 1
    
    # M3 = A00(B01-B11)
    U[i0*m + j0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] -= 1
    W[(k0+1)*n + i0][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    c += 1
    
    # M4 = A11(B10-B00)
    U[(i0+1)*m + (j0+1)][c] += 1
    V[(j0+1)*p + k0][c] += 1
    V[j0*p + k0][c] -= 1
    W[k0*n + i0][c] += 1
    W[k0*n + (i0+1)][c] += 1
    c += 1
    
    # M5 = (A00+A01)B11
    U[i0*m + j0][c] += 1
    U[i0*m + (j0+1)][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] -= 1
    W[(k0+1)*n + i0][c] += 1
    c += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[(i0+1)*m + j0][c] += 1
    U[i0*m + j0][c] -= 1
    V[j0*p + k0][c] += 1
    V[j0*p + (k0+1)][c] += 1
    W[(k0+1)*n + (i0+1)][c] += 1
    c += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[i0*m + (j0+1)][c] += 1
    U[(i0+1)*m + (j0+1)][c] -= 1
    V[(j0+1)*p + k0][c] += 1
    V[(j0+1)*p + (k0+1)][c] += 1
    W[k0*n + i0][c] += 1
    c += 1
    
    # Now for columns k0+2, k0+3, k0+4:
    # C00 += A00*B02 + A01*B12... wait, no.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # For k=k0+2: C0,k0+2 = A00*B0,k0+2 + A01*B1,k0+2
    # For k=k0+3: C0,k0+3 = A00*B0,k0+3 + A01*B1,k0+3
    # For k=k0+4: C0,k0+4 = A00*B0,k0+4 + A01*B1,k0+4
    # Similarly for row 1.
    # That's 6 more terms, total 13. Not 10.
    
    # Hmm, I need a smarter decomposition for the remaining 3 columns.
    # Let me use: 
    # M8 = (A00+A01) * (B02+B12)  -- no, this doesn't work directly
    # 
    # Actually for the 3 remaining columns, I need:
    # C0,k = A00*B0k + A01*B1k for k=k0+2,k0+3,k0+4
    # C1,k = A10*B0k + A11*B1k for k=k0+2,k0+3,k0+4
    # 
    # I can use: M8 = (A00+A01) * something, but I need to separate the B columns.
    # 
    # Alternative: use 3 rank-1 terms for each column pair... 
    # Actually, the minimum rank for 2×2×3 is 5 (I think), so 7+5=12.
    # Or maybe I can do better.
    
    # Let me just use schoolbook for the remaining 3 columns: 2*2*3 = 12 terms
    # Total: 7 + 12 = 19 per 2×2×5, total 2*19 = 38. Worse than 36.
    
    # This approach doesn't help. Let me abandon it.
    
    return c
