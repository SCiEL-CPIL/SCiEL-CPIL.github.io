"""Strassen-based tensor decomposition for 3x3 @ 3x8 matrix multiplication."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    # We'll build the factor matrices as lists of (row, coeff) pairs per column,
    # then convert to dense at the end.
    
    # Each term is a tuple (u_idx, v_idx, w_idx, u_coeff, v_coeff, w_coeff)
    # where u_idx is index into U rows (n*m = 9),
    # v_idx is index into V rows (m*p = 24),
    # w_idx is index into W rows (p*n = 24).
    
    terms = []
    
    # Strassen for 2x2 blocks:
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B01-B11)
    # m4 = A11(B10-B00)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # Part 1: For i in {0,1}, j in {0,1}, k in {2q, 2q+1} for q in {0,1,2,3}
    # That's 4 k-blocks, each 2x2x2 Strassen
    for i_block in range(2):  # i=0 or i=1 (local row in the 2x2 A block)
        for j_block in range(2):  # j=0 or j=1 (local col in the 2x2 B block)
            for q in range(4):  # k-block index
                k_base = 2 * q  # starting k index
                
                # Local indices within the 2x2x2 sub-problem:
                # A rows: 0,1 -> global i = i_block
                # B cols: 0,1 -> global j = j_block  
                # C cols: 0,1 -> global k = k_base, k_base+1
                
                # Strassen terms for this 2x2x2:
                # Each term has specific (A_row, B_col) combinations with coefficients
                
                # m1 = (A00+A11)(B00+B11): 
                # U entries: (i_block, 0, 1), (i_block, 1, 1)
                # V entries: (0, k_base, 1), (1, k_base+1, 1)
                # W entry: (k_base, i_block, 1), (k_base+1, i_block, 1)
                
                # Actually, let me be more careful. The Strassen decomposition 
                # of C = A*B where A is 2x2, B is 2x2, C is 2x2:
                # C[a,c] = sum over b of A[a,b] * B[b,c]
                
                # In our global notation:
                # A has rows indexed by i (here i_block and i_block+1... wait no)
                
                # Let me re-think the indexing.
                # The full matrix mult is C[i,k] = sum_j A[i,j] * B[j,k]
                # where A is 3x3, B is 3x8, C is 3x8.
                
                # U rows are indexed by (i,j) for A[i,j], shape (n*m, rank) = (9, rank)
                # V rows are indexed by (j,k) for B[j,k], shape (m*p, rank) = (24, rank)  
                # W rows are indexed by (k,i) for C[k*n+i] per the trace convention, shape (p*n, rank) = (24, rank)
                
                # Wait, the problem says W[k*n+i,r]. So W row index is k*n+i = k*3+i.
                
                # For a Strassen term on sub-block with:
                # i in {i0, i0+1} (two rows of A)
                # j in {j0, j0+1} (two cols of B)
                # k in {k0, k0+1} (two cols of C / rows of B)
                
                # The 7 Strassen products m1..m7 each involve:
                # - A linear combination of A[i,j] entries
                # - A linear combination of B[j,k] entries  
                # - And contribute to C[i,k] entries with certain coefficients
                
                # Let me enumerate the 7 terms for a 2x2x2 block:
                # Using local indices a,b,c in {0,1} for A rows, B cols (which are A cols), C cols:
                
                # m1 = (A[a=0,b=0] + A[a=1,b=1]) * (B[b=0,c=0] + B[b=1,c=1])
                # m2 = (A[a=1,b=0] + A[a=1,b=1]) * B[b=0,c=0]
                # m3 = A[a=0,b=0] * (B[b=0,c=1] - B[b=1,c=1])
                # m4 = A[a=1,b=1] * (B[b=1,c=0] - B[b=0,c=0])
                # m5 = (A[a=0,b=0] + A[a=0,b=1]) * B[b=1,c=1]
                # m6 = (A[a=1,b=0] - A[a=0,b=0]) * (B[b=0,c=0] + B[b=0,c=1])
                # m7 = (A[a=0,b=1] - A[a=1,b=1]) * (B[b=1,c=0] + B[b=1,c=1])
                
                # C[a,c] reconstruction:
                # C[0,0] = m1 + m4 - m5 + m7
                # C[0,1] = m3 + m5
                # C[1,0] = m2 + m4
                # C[1,1] = m1 - m2 + m3 + m6
                
                # For our global indices:
                # a -> i = i0 + a
                # b -> j = j0 + b
                # c -> k = k0 + c
                
                # U row for A[i,j] is i*m + j = i*3 + j
                # V row for B[j,k] is j*p + k = j*8 + k
                # W row for C at position (k,i) is k*n + i = k*3 + i
                
                i0 = i_block
                j0 = j_block
                k0 = k_base
                
                # Term m1:
                # A part: A[i0,j0] + A[i0+1,j0+1] -> U rows: i0*3+j0 (coeff 1), (i0+1)*3+(j0+1) (coeff 1)
                # B part: B[j0,k0] + B[j0+1,k0+1] -> V rows: j0*8+k0 (coeff 1), (j0+1)*8+(k0+1) (coeff 1)
                # C part: contributes to C[i0,k0] with +1, C[i0,k0+1] with 0... 
                # Wait, no. Each m_r is a scalar product. The C contribution is:
                # C[i0+k0_offset, k0+c_offset] gets coefficient from the reconstruction.
                
                # Actually, each term m_r produces a rank-1 contribution to the tensor.
                # m_r = (sum_a u_a * A[ia, jb]) * (sum_b v_b * B[jb, kc])
                # and it contributes to C[i,k] with coefficient w_ik.
                
                # So for term m1:
                # u coefficients: {(i0, j0): 1, (i0+1, j0+1): 1}
                # v coefficients: {(j0, k0): 1, (j0+1, k0+1): 1}
                # w coefficients: {(k0, i0): 1, (k0+1, i0+1): 1}  [from C[0,0]=m1+... and C[1,1]=m1-...]
                # Wait: C[a=0,c=0] = m1 + m4 - m5 + m7, so m1 contributes +1 to C[0,0] i.e. C[i0, k0]
                # C[a=1,c=1] = m1 - m2 + m3 + m6, so m1 contributes +1 to C[1,1] i.e. C[i0+1, k0+1]
                
                # Hmm, but that's not a single rank-1 tensor. The Strassen trick is that each m_r 
                # is ONE scalar, and it appears in multiple C entries. So the tensor term is:
                # For term r, the contribution to T[i,j,k] (where T is the matmul tensor with T[i,j,k]=1 if...)
                
                # Actually, the matmul tensor T has T[i,j,k] = 1 for all valid (i,j,k), meaning
                # C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
                
                # A decomposition is: sum_r U[:,r] * V[:,r] * W[:,r] where
                # (U*V*W)[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i,j,k]
                
                # For Strassen on a 2x2x2 subblock, each m_r gives one rank-1 term.
                # m1 = (A00+A11)(B00+B11) contributes to:
                #   C[0,0] += m1 (coeff +1)
                #   C[1,1] += m1 (coeff +1)
                # So W coefficients for m1: W[k0*3+i0, r] = 1, W[(k0+1)*3+(i0+1), r] = 1
                # U coefficients: U[i0*3+j0, r] = 1, U[(i0+1)*3+(j0+1), r] = 1
                # V coefficients: V[j0*8+k0, r] = 1, V[(j0+1)*8+(k0+1), r] = 1
                
                terms.append({
                    'U': [(i0*3 + j0, 1), ((i0+1)*3 + (j0+1), 1)],
                    'V': [(j0*8 + k0, 1), ((j0+1)*8 + (k0+1), 1)],
                    'W': [(k0*3 + i0, 1), ((k0+1)*3 + (i0+1), 1)]
                })
                
                # Term m2 = (A10+A11)*B00
                # U: {(i0+1, j0): 1, (i0+1, j0+1): 1}
                # V: {(j0, k0): 1}
                # C contributions: C[1,0] += m2 (coeff +1)
                # W: {(k0, i0+1): 1}
                terms.append({
                    'U': [((i0+1)*3 + j0, 1), ((i0+1)*3 + (j0+1), 1)],
                    'V': [(j0*8 + k0, 1)],
                    'W': [(k0*3 + (i0+1), 1)]
                })
                
                # Term m3 = A00*(B01-B11)
                # U: {(i0, j0): 1}
                # V: {(j0, k0+1): 1, (j0+1, k0+1): -1}
                # C contributions: C[0,1] += m3 (coeff +1)
                # W: {(k0+1, i0): 1}
                terms.append({
                    'U': [(i0*3 + j0, 1)],
                    'V': [(j0*8 + (k0+1), 1), ((j0+1)*8 + (k0+1), -1)],
                    'W': [((k0+1)*3 + i0, 1)]
                })
                
                # Term m4 = A11*(B10-B00)
                # U: {(i0+1, j0+1): 1}
                # V: {(j0+1, k0): 1, (j0, k0): -1}
                # C contributions: C[0,0] += m4 (coeff +1), C[1,0] += m4 (coeff +1)
                # W: {(k0, i0): 1, (k0, i0+1): 1}
                terms.append({
                    'U': [((i0+1)*3 + (j0+1), 1)],
                    'V': [((j0+1)*8 + k0, 1), (j0*8 + k0, -1)],
                    'W': [(k0*3 + i0, 1), (k0*3 + (i0+1), 1)]
                })
                
                # Term m5 = (A00+A01)*B11
                # U: {(i0, j0): 1, (i0, j0+1): 1}
                # V: {(j0+1, k0+1): 1}
                # C contributions: C[0,0] -= m5 (coeff -1), C[0,1] += m5 (coeff +1)
                # W: {(k0, i0): -1, (k0+1, i0): 1}
                terms.append({
                    'U': [(i0*3 + j0, 1), (i0*3 + (j0+1), 1)],
                    'V': [((j0+1)*8 + (k0+1), 1)],
                    'W': [(k0*3 + i0, -1), ((k0+1)*3 + i0, 1)]
                })
                
                # Term m6 = (A10-A00)*(B00+B01)
                # U: {(i0+1, j0): 1, (i0, j0): -1}
                # V: {(j0, k0): 1, (j0, k0+1): 1}
                # C contributions: C[1,1] += m6 (coeff +1)
                # W: {(k0+1, i0+1): 1}
                terms.append({
                    'U': [((i0+1)*3 + j0, 1), (i0*3 + j0, -1)],
                    'V': [(j0*8 + k0, 1), (j0*8 + (k0+1), 1)],
                    'W': [((k0+1)*3 + (i0+1), 1)]
                })
                
                # Term m7 = (A01-A11)*(B10+B11)
                # U: {(i0, j0+1): 1, (i0+1, j0+1): -1}
                # V: {(j0+1, k0): 1, (j0+1, k0+1): 1}
                # C contributions: C[0,0] += m7 (coeff +1)
                # W: {(k0, i0): 1}
                terms.append({
                    'U': [(i0*3 + (j0+1), 1), ((i0+1)*3 + (j0+1), -1)],
                    'V': [((j0+1)*8 + k0, 1), ((j0+1)*8 + (k0+1), 1)],
                    'W': [(k0*3 + i0, 1)]
                })
    
    # Part 2: For i in {0,1}, j=2, k in {0..7}
    # Schoolbook: C[i,k] = A[i,2] * B[2,k]
    # Each (i,k) pair is one term.
    for i in range(2):
        for k in range(8):
            terms.append({
                'U': [(i*3 + 2, 1)],
                'V': [(2*8 + k, 1)],
                'W': [(k*3 + i, 1)]
            })
    
    # Part 3: For i=2, j in {0,1,2}, k in {0..7}
    # Schoolbook: C[2,k] = sum_j A[2,j] * B[j,k]
    for j in range(3):
        for k in range(8):
            terms.append({
                'U': [(2*3 + j, 1)],
                'V': [(j*8 + k, 1)],
                'W': [(k*3 + 2, 1)]
            })
    
    rank = len(terms)
    
    # Build U, V, W as lists of lists
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col, term in enumerate(terms):
        for (row, coeff) in term['U']:
            U[row][col] = coeff
        for (row, coeff) in term['V']:
            V[row][col] = coeff
        for (row, coeff) in term['W']:
            W[row][col] = coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
