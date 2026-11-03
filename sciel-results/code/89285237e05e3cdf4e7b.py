import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # We will use 6 Strassen blocks of size 2x2x2 and schoolbook for the rest.
    # Strassen blocks cover i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Specifically:
    # j_blocks = [(0,1), (2,3)]
    # k_blocks = [(0,1), (2,3), (4,5)]
    # i_block = (0,1)
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end, is_strassen):
        nonlocal rank
        if is_strassen:
            # Strassen 2x2x2
            # Indices: i in {0,1}, j in {0,1}, k in {0,1} relative to block
            # 7 multiplications
            # M1 = (A00 + A11) (B00 + B11)
            # M2 = (A10 + A11) B00
            # M3 = A00 (B10 - B11)
            # M4 = A11 B11
            # M5 = (A00 + A01) (B01 - B11)
            # M6 = (A10 - A00) B00
            # M7 = (A01 - A11) (B01 + B11)
            
            # We need to map these to the global U, V, W matrices.
            # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
            
            # Helper to add a rank-1 term
            def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
                nonlocal rank
                col = rank
                # Pad U, V, W if necessary (we will build them as lists of lists)
                # Actually, we will build them as numpy arrays at the end or pre-allocate.
                # Let's pre-allocate with 0s and fill.
                pass
            
            # It's easier to build the matrices directly.
            # Let's collect the terms first.
            terms = []
            
            # M1
            # U: A00 (i=0, j=j_start), A11 (i=1, j=j_start+1)
            # V: B00 (j=j_start, k=k_start), B11 (j=j_start+1, k=k_start+1)
            # W: C00 (k=k_start, i=0), C11 (k=k_start+1, i=1)
            # C00 = M1 + M3 - M5 - M7
            # C11 = M1 - M2 + M4 + M7
            # C01 = M5 + M7
            # C10 = M2 + M4
            
            # Term 1: M1
            # U: A00=1, A11=1
            # V: B00=1, B11=1
            # W: C00=1, C11=1
            terms.append((
                [(i_start + 0, j_start + 0, 1), (i_start + 1, j_start + 1, 1)],
                [(j_start + 0, k_start + 0, 1), (j_start + 1, k_start + 1, 1)],
                [(k_start + 0, i_start + 0, 1), (k_start + 1, i_start + 1, 1)]
            ))
            
            # Term 2: M2
            # U: A10=1, A11=1
            # V: B00=1
            # W: C10=1, C11=-1
            terms.append((
                [(i_start + 1, j_start + 0, 1), (i_start + 1, j_start + 1, 1)],
                [(j_start + 0, k_start + 0, 1)],
                [(k_start + 0, i_start + 1, 1), (k_start + 1, i_start + 1, -1)]
            ))
            
            # Term 3: M3
            # U: A00=1
            # V: B10=1, B11=-1
            # W: C00=1
            terms.append((
                [(i_start + 0, j_start + 0, 1)],
                [(j_start + 1, k_start + 0, 1), (j_start + 1, k_start + 1, -1)],
                [(k_start + 0, i_start + 0, 1)]
            ))
            
            # Term 4: M4
            # U: A11=1
            # V: B11=1
            # W: C10=1, C11=1
            terms.append((
                [(i_start + 1, j_start + 1, 1)],
                [(j_start + 1, k_start + 1, 1)],
                [(k_start + 0, i_start + 1, 1), (k_start + 1, i_start + 1, 1)]
            ))
            
            # Term 5: M5
            # U: A00=1, A01=1
            # V: B01=1, B11=-1
            # W: C01=1
            terms.append((
                [(i_start + 0, j_start + 0, 1), (i_start + 0, j_start + 1, 1)],
                [(j_start + 0, k_start + 1, 1), (j_start + 1, k_start + 1, -1)],
                [(k_start + 1, i_start + 0, 1)]
            ))
            
            # Term 6: M6
            # U: A10=1, A00=-1
            # V: B00=1
            # W: C10=1
            terms.append((
                [(i_start + 1, j_start + 0, 1), (i_start + 0, j_start + 0, -1)],
                [(j_start + 0, k_start + 0, 1)],
                [(k_start + 0, i_start + 1, 1)]
            ))
            
            # Term 7: M7
            # U: A01=1, A11=-1
            # V: B01=1, B11=1
            # W: C01=1, C11=1
            terms.append((
                [(i_start + 0, j_start + 1, 1), (i_start + 1, j_start + 1, -1)],
                [(j_start + 0, k_start + 1, 1), (j_start + 1, k_start + 1, 1)],
                [(k_start + 1, i_start + 0, 1), (k_start + 1, i_start + 1, 1)]
            ))
            
            for u_terms, v_terms, w_terms in terms:
                col = rank
                for idx, val in u_terms:
                    # U index: i * m + j
                    U[i * m + j].append(val) # Wait, U is not initialized.
                rank += 1
                
        else:
            # Schoolbook 1x1x1
            # U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
            pass

    # Re-initialization strategy:
    # Pre-allocate U, V, W with 0s of size (n*m, 90), (m*p, 90), (p*n, 90)
    # Then fill them.
    
    max_rank = n * m * p
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # Add 6 Strassen blocks
    j_blocks = [(0, 1), (2, 3)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for jb in j_blocks:
        for kb in k_blocks:
            i_start, i_end = 0, 1
            j_start, j_end = jb
            k_start, k_end = kb
            
            # Strassen terms
            # 1
            U[i_start * m + j_start][col] += 1
            U[(i_start+1) * m + (j_start+1)][col] += 1
            V[j_start * p + k_start][col] += 1
            V[(j_start+1) * p + (k_start+1)][col] += 1
            W[k_start * n + i_start][col] += 1
            W[(k_start+1) * n + (i_start+1)][col] += 1
            col += 1
            
            # 2
            U[(i_start+1) * m + j_start][col] += 1
            U[(i_start+1) * m + (j_start+1)][col] += 1
            V[j_start * p + k_start][col] += 1
            W[k_start * n + (i_start+1)][col] += 1
            W[(k_start+1) * n + (i_start+1)][col] -= 1
            col += 1
            
            # 3
            U[i_start * m + j_start][col] += 1
            V[(j_start+1) * p + k_start][col] += 1
            V[(j_start+1) * p + (k_start+1)][col] -= 1
            W[k_start * n + i_start][col] += 1
            col += 1
            
            # 4
            U[(i_start+1) * m + (j_start+1)][col] += 1
            V[(j_start+1) * p + (k_start+1)][col] += 1
            W[k_start * n + (i_start+1)][col] += 1
            W[(k_start+1) * n + (i_start+1)][col] += 1
            col += 1
            
            # 5
            U[i_start * m + j_start][col] += 1
            U[i_start * m + (j_start+1)][col] += 1
            V[j_start * p + (k_start+1)][col] += 1
            V[(j_start+1) * p + (k_start+1)][col] -= 1
            W[(k_start+1) * n + i_start][col] += 1
            col += 1
            
            # 6
            U[(i_start+1) * m + j_start][col] += 1
            U[i_start * m + j_start][col] -= 1
            V[j_start * p + k_start][col] += 1
            W[k_start * n + (i_start+1)][col] += 1
            col += 1
            
            # 7
            U[i_start * m + (j_start+1)][col] += 1
            U[(i_start+1) * m + (j_start+1)][col] -= 1
            V[j_start * p + (k_start+1)][col] += 1
            V[(j_start+1) * p + (k_start+1)][col] += 1
            W[(k_start+1) * n + i_start][col] += 1
            W[(k_start+1) * n + (i_start+1)][col] += 1
            col += 1
            
    # Add schoolbook for remaining
    # Remaining: i=2, j=0..4, k=0..5
    for i in range(2, 3):
        for j in range(5):
            for k in range(6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Remaining: i=0,1, j=4, k=0..5
    for i in range(0, 2):
        for j in range(4, 5):
            for k in range(6):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Trim U, V, W to col
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
