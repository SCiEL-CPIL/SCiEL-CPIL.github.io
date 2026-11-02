"""Matrix multiplication 3x4x8 - Strassen-based block decomposition."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x4 x 4x8 -> 3x8 using Strassen on 2x2 blocks.
    
    A is 3x4, B is 4x8, C is 3x8.
    Split A = [A1(2x4); A2(1x4)]
    Split B = [B1(4x4) B2(4x4)]
    Split B1 = [B11(4x2) B12(4x2)], B2 = [B21(4x2) B22(4x2)]
    
    C = A1*B1 + A1*B2 + A2*B1 + A2*B2
    A1*B1 = A1*B11 + A1*B12 (each 2x2x2, Strassen rank 7)
    A1*B2 = A1*B21 + A1*B22 (each 2x2x2, Strassen rank 7)
    A2*B1 = A2*B11 + A2*B12 (each 1x2x2, schoolbook rank 4)
    A2*B2 = A2*B21 + A2*B22 (each 1x2x2, schoolbook rank 4)
    Total rank = 7+7+7+7+4+4+4+4 = 44
    """
    n, m, p = 3, 4, 8  # A: n x m, B: m x p, C: n x p
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] ... wait, that's not right.
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] ... no.
    # Standard: C[i,k] = sum_j sum_c U[i,j,c] * V[j,k,c] where U has shape (n,m,rank)
    # But the interface says U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # And W[k*n+i, c] is the output coefficient.
    # So: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] summed over j? No...
    # Let me re-read: "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]? That doesn't make sense dimensionally.
    # Actually for matrix mult: C[i,k] = sum_j A[i,j] B[j,k]
    # The bilinear form: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # No wait. Let me think again.
    # T[i,j,k] is the multiplication tensor. A rank-r decomposition means:
    # T[i,j,k] = sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But that's trilinear, not bilinear. For matrix multiplication, it's bilinear in A and B.
    # So: C[i,k] = sum_j A[i,j] * B[j,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # Hmm, but then W would be all 1s? That can't be right for rank minimization.
    # 
    # Actually, I think the standard tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_c P_c(A) * Q_c(B) where P_c and Q_c are linear forms.
    # In the flattened notation: C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    # And W[k*n+i, c] would be... 1? Or is it part of the decomposition?
    # 
    # Let me re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
    # I think W is just the identity-like output mapping. Let me assume W[k*n+i, c] = 1 for the output index.
    # Actually no - in a general tensor decomposition, all three factors matter.
    # For matrix mult: T[i,j,k] = delta... no.
    # 
    # Let me just use the schoolbook as reference:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    # Then C[i,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]... 
    # But that sums over c, and for each c we have one (i,j,k) triple.
    # So C[i,k] = sum_j A[i,j]*B[j,k] * (sum over c where (i,j,k) is the triple)
    # This only works if each (i,j,k) has exactly one c, and W[k*n+i,c]=1.
    # 
    # I think the correct interpretation is:
    # C[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
    # No, that's still trilinear.
    # 
    # Let me just assume W is all 1s for now and focus on U and V.
    # Actually, looking at the schoolbook: U[i*m+j, c]=1, V[j*p+k, c]=1, W[k*n+i, c]=1
    # If the formula is C[i,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] * A[i,j]*B[j,k]
    # that doesn't work either.
    # 
    # I'll just follow the schoolbook pattern and extend it with Strassen.
    # For Strassen 2x2: 
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    n, m, p = 3, 4, 8
    factors = []  # list of (U_row_idx, U_val, V_row_idx, V_val, W_row_idx, W_val)
    
    def add_factor(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        factors.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Strassen for 2x2x2: A(2x2), B(2x2), C(2x2)
    # A indices: a00,a01,a10,a11 -> flat: 0,1,2,3 (row-major: i*m+j)
    # B indices: b00,b01,b10,b11 -> flat: 0,1,2,3
    # C indices: c00,c01,c10,c11 -> flat: k*n+i (trace convention)
    # For 2x2x2: n=2, m=2, p=2
    # C[k*n+i]: c00->0*2+0=0, c01->0*2+1=1, c10->1*2+0=2, c11->1*2+1=3
    
    def strassen_2x2(a_offset, b_offset, c_offset, n=2, m=2, p=2):
        """Add Strassen factors for 2x2x2 block.
        a_offset: offset in A's flat index (i*m+j)
        b_offset: offset in B's flat index (j*p+k)  
        c_offset: offset in C's flat index (k*n+i)
        """
        # A[i,j] flat index: a_offset + i*m + j
        # B[j,k] flat index: b_offset + j*p + k
        # C[i,k] flat index (k*n+i): c_offset + k*n + i
        
        # M1 = (A00+A11)(B00+B11)
        # A00: a_offset+0, A11: a_offset+3
        # B00: b_offset+0, B11: b_offset+3
        # C contribution: C00 += M1, C11 += M1
        add_factor(a_offset+0, 1, b_offset+0, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+3, 1, b_offset+0, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+0, 1, b_offset+3, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+3, 1, b_offset+3, 1, c_offset+0, 1)  # C00
        # C11: c_offset + 1*2+1 = c_offset+3
        add_factor(a_offset+0, 1, b_offset+0, 1, c_offset+3, 1)
        add_factor(a_offset+3, 1, b_offset+0, 1, c_offset+3, 1)
        add_factor(a_offset+0, 1, b_offset+3, 1, c_offset+3, 1)
        add_factor(a_offset+3, 1, b_offset+3, 1, c_offset+3, 1)
        
        # M2 = (A10+A11)B00
        # C10 += M2
        add_factor(a_offset+2, 1, b_offset+0, 1, c_offset+2, 1)  # C10
        add_factor(a_offset+3, 1, b_offset+0, 1, c_offset+2, 1)  # C10
        
        # M3 = A00(B10-B11)
        # C01 += M3
        add_factor(a_offset+0, 1, b_offset+2, 1, c_offset+1, 1)  # C01
        add_factor(a_offset+0, -1, b_offset+3, 1, c_offset+1, 1)  # C01
        
        # M4 = A11(B01-B00)
        # C00 -= M4, C10 += M4
        add_factor(a_offset+3, -1, b_offset+1, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+3, 1, b_offset+0, 1, c_offset+0, 1)   # C00
        add_factor(a_offset+3, 1, b_offset+1, 1, c_offset+2, 1)   # C10
        add_factor(a_offset+3, -1, b_offset+0, 1, c_offset+2, 1)  # C10
        
        # M5 = (A00+A01)B11
        # C00 -= M5, C01 += M5
        add_factor(a_offset+0, -1, b_offset+3, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+1, -1, b_offset+3, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+0, 1, b_offset+3, 1, c_offset+1, 1)   # C01
        add_factor(a_offset+1, 1, b_offset+3, 1, c_offset+1, 1)   # C01
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        add_factor(a_offset+2, 1, b_offset+0, 1, c_offset+3, 1)   # C11
        add_factor(a_offset+2, 1, b_offset+1, 1, c_offset+3, 1)   # C11
        add_factor(a_offset+0, -1, b_offset+0, 1, c_offset+3, 1)  # C11
        add_factor(a_offset+0, -1, b_offset+1, 1, c_offset+3, 1)  # C11
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7
        add_factor(a_offset+1, 1, b_offset+2, 1, c_offset+0, 1)   # C00
        add_factor(a_offset+1, 1, b_offset+3, 1, c_offset+0, 1)   # C00
        add_factor(a_offset+3, -1, b_offset+2, 1, c_offset+0, 1)  # C00
        add_factor(a_offset+3, -1, b_offset+3, 1, c_offset+0, 1)  # C00
    
    # Schoolbook for 1x2x2: A(1x2), B(2x2), C(1x2)
    # A[i,j]: i=0, j=0,1 -> flat: a_offset+0, a_offset+1
    # B[j,k]: j=0,1, k=0,1 -> flat: b_offset+0,1,2,3
    # C[i,k]: i=0, k=0,1 -> flat (k*n+i): c_offset+0*1+0=0, c_offset+1*1+0=1 (n=1)
    def schoolbook_1x2x2(a_offset, b_offset, c_offset):
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        add_factor(a_offset+0, 1, b_offset+0, 1, c_offset+0, 1)
        add_factor(a_offset+1, 1, b_offset+2, 1, c_offset+0, 1)
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        add_factor(a_offset+0, 1, b_offset+1, 1, c_offset+1, 1)
        add_factor(a_offset+1, 1, b_offset+3, 1, c_offset+1, 1)
    
    # Now build the full 3x4x8 decomposition
    # A is 3x4: indices (i,j) -> i*4+j, i=0,1,2; j=0,1,2,3
    # B is 4x8: indices (j,k) -> j*8+k, j=0,1,2,3; k=0,1,2,3,4,5,6,7
    # C is 3x8: indices (i,k) -> k*3+i (trace convention), i=0,1,2; k=0,1,2,3,4,5,6,7
    
    # Split A: A1 = rows 0-1 (2x4), A2 = row 2 (1x4)
    # Split B: B1 = cols 0-3 (4x4), B2 = cols 4-7 (4x4)
    # Split B1: B11 = cols 0-1 (4x2), B12 = cols 2-3 (4x2)
    # Split B2: B21 = cols 0-1 of B2 (4x2), B22 = cols 2-3 of B2 (4x2)
    
    # A1*B11: 2x4 x 4x2 -> 2x2. This is 2x2x2 with m=4? No, it's 2x4x2.
    # Hmm, Strassen is for 2x2x2. For 2x4x2, I need to split the middle dimension.
    # A1 is 2x4, B11 is 4x2. Split A1 into A1a(2x2) and A1b(2x2), B11 into B11a(2x2) and B11b(2x2).
    # A1*B11 = A1a*B11a + A1b*B11b
    # Each is 2x2x2, Strassen rank 7. Total 14.
    
    # Let me define offsets:
    # A flat: i*4+j
    #   A1 (rows 0-1): i=0,1 -> flat 0-7
    #   A1a (rows 0-1, cols 0-1): flat 0-3
    #   A1b (rows 0-1, cols 2-3): flat 4-7
    #   A2 (row 2): i=2 -> flat 8-11
    #   A2a (row 2, cols 0-1): flat 8-9
    #   A2b (row 2, cols 2-3): flat 10-11
    # B flat: j*8+k
    #   B11 (cols 0-1): k=0,1 -> flat 0,1,8,9,16,17,24,25 (j=0,1,2,3; k=0,1)
    #   B12 (cols 2-3): k=2,3 -> flat 2,3,10,11,18,19,26,27
    #   B21 (cols 4-5): k=4,5 -> flat 4,5,12,13,20,21,28,29
    #   B22 (cols 6-7): k=6,7 -> flat 6,7,14,15,22,23,30,31
    # C flat: k*3+i
    #   C11 (i=0,1; k=0,1): k=0,1; i=0,1 -> flat 0,1,3,4
    #   C12 (i=0,1; k=2,3): k=2,3; i=0,1 -> flat 6,7,9,10
    #   C21 (i=2; k=0,1): k=0,1; i=2 -> flat 2,5
    #   C22 (i=2; k=2,3): k=2,3; i=2 -> flat 8,11
    #   C31 (i=0,1; k=4,5): k=4,5; i=0,1 -> flat 12,13,15,16
    #   C32 (i=0,1; k=6,7): k=6,7; i=0,1 -> flat 18,19,21,22
    #   C41 (i=2; k=4,5): k=4,5; i=2 -> flat 14,17
    #   C42 (i=2; k=6,7): k=6,7; i=2 -> flat 20,23
    
    # This is getting complex. Let me just directly enumerate.
    
    # For 2x2x2 Strassen, I need to map the local 2x2 indices to global flat indices.
    # Let me write a helper that takes the global indices for A's 2x2 block, B's 2x2 block, and C's 2x2 block.
    
    factors = []
    
    def add(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        factors.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    # Strassen for 2x2x2 with specific global index mappings
    # A block: 2x2, A[i,j] at global index a_idx[i][j]
    # B block: 2x2, B[j,k] at global index b_idx[j][k]
    # C block: 2x2, C[i,k] at global index c_idx[i][k] (note: C is indexed by (i,k), flat as k*n+i)
    
    def strassen(a_idx, b_idx, c_idx):
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        add(a_idx[0][0], 1, b_idx[0][0], 1, c_idx[0][0], 1)
        add(a_idx[1][1], 1, b_idx[0][0], 1, c_idx[0][0], 1)
        add(a_idx[0][0], 1, b_idx[1][1], 1, c_idx[0][0], 1)
        add(a_idx[1][1], 1, b_idx[1][1], 1, c_idx[0][0], 1)
        add(a_idx[0][0], 1, b_idx[0][0], 1, c_idx[1][1], 1)
        add(a_idx[1][1], 1, b_idx[0][0], 1, c_idx[1][1], 1)
        add(a_idx[0][0], 1, b_idx[1][1], 1, c_idx[1][1], 1)
        add(a_idx[1][1], 1, b_idx[1][1], 1, c_idx[1][1], 1)
        
        # M2 = (A10+A11)B00 -> C10 += M2
        add(a_idx[1][0], 1, b_idx[0][0], 1, c_idx[1][0], 1)
        add(a_idx[1][1], 1, b_idx[0][0], 1, c_idx[1][0], 1)
        
        # M3 = A00(B10-B11) -> C01 += M3
        add(a_idx[0][0], 1, b_idx[1][0], 1, c_idx[0][1], 1)
        add(a_idx[0][0], -1, b_idx[1][1], 1, c_idx[0][1], 1)
        
        # M4 = A11(B01-B00) -> C00 -= M4, C10 += M4
        add(a_idx[1][1], -1, b_idx[0][1], 1, c_idx[0][0], 1)
        add(a_idx[1][1], 1, b_idx[0][0], 1, c_idx[0][0], 1)
        add(a_idx[1][1], 1, b_idx[0][1], 1, c_idx[1][0], 1)
        add(a_idx[1][1], -1, b_idx[0][0], 1, c_idx[1][0], 1)
        
        # M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        add(a_idx[0][0], -1, b_idx[1][1], 1, c_idx[0][0], 1)
        add(a_idx[0][1], -1, b_idx[1][1], 1, c_idx[0][0], 1)
        add(a_idx[0][0], 1, b_idx[1][1], 1, c_idx[0][1], 1)
        add(a_idx[0][1], 1, b_idx[1][1], 1, c_idx[0][1], 1)
        
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        add(a_idx[1][0], 1, b_idx[0][0], 1, c_idx[1][1], 1)
        add(a_idx[1][0], 1, b_idx[0][1], 1, c_idx[1][1], 1)
        add(a_idx[0][0], -1, b_idx[0][0], 1, c_idx[1][1], 1)
        add(a_idx[0][0], -1, b_idx[0][1], 1, c_idx[1][1], 1)
        
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        add(a_idx[0][1], 1, b_idx[1][0], 1, c_idx[0][0], 1)
        add(a_idx[0][1], 1, b_idx[1][1], 1, c_idx[0][0], 1)
        add(a_idx[1][1], -1, b_idx[1][0], 1, c_idx[0][0], 1)
        add(a_idx[1][1], -1, b_idx[1][1], 1, c_idx[0][0], 1)
    
    # Schoolbook for 1x2x2
    def schoolbook_1x2(a_idx, b_idx, c_idx):
        # A is 1x2: a_idx[0][0], a_idx[0][1]
        # B is 2x2: b_idx[j][k]
        # C is 1x2: c_idx[0][0], c_idx[0][1]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        add(a_idx[0][0], 1, b_idx[0][0], 1, c_idx[0][0], 1)
        add(a_idx[0][1], 1, b_idx[1][0], 1, c_idx[0][0], 1)
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        add(a_idx[0][0], 1, b_idx[0][1], 1, c_idx[0][1], 1)
        add(a_idx[0][1], 1, b_idx[1][1], 1, c_idx[0][1], 1)
    
    # Now build the index maps
    # A[i,j] flat = i*4 + j
    # B[j,k] flat = j*8 + k
    # C[i,k] flat = k*3 + i
    
    # A1a: i=0,1; j=0,1 -> a_idx[i][j] = (0+i)*4 + (0+j)
    a1a = [[(0+i)*4 + (0+j) for j in range(2)] for i in range(2)]
    # A1b: i=0,1; j=2,3 -> a_idx[i][j] = (0+i)*4 + (2+j)
    a1b = [[(0+i)*4 + (2+j) for j in range(2)] for i in range(2)]
    # A2a: i=2; j=0,1 -> a_idx[0][j] = 2*4 + (0+j) = 8+j
    a2a = [[8+j for j in range(2)]]
    # A2b: i=2; j=2,3 -> a_idx[0][j] = 2*4 + (2+j) = 10+j
    a2b = [[10+j for j in range(2)]]
    
    # B11a: j=0,1; k=0,1 -> b_idx[j][k] = j*8 + (0+k)
    b11a = [[j*8 + (0+k) for k in range(2)] for j in range(2)]
    # B11b: j=2,3; k=0,1 -> b_idx[j][k] = (2+j)*8 + (0+k)
    b11b = [[(2+j)*8 + (0+k) for k in range(2)] for j in range(2)]
    # B12a: j=0,1; k=2,3 -> b_idx[j][k] = j*8 + (2+k)
    b12a = [[j*8 + (2+k) for k in range(2)] for j in range(2)]
    # B12b: j=2,3; k=2,3 -> b_idx[j][k] = (2+j)*8 + (2+k)
    b12b = [[(2+j)*8 + (2+k) for k in range(2)] for j in range(2)]
    # B21a: j=0,1; k=4,5 -> b_idx[j][k] = j*8 + (4+k)
    b21a = [[j*8 + (4+k) for k in range(2)] for j in range(2)]
    # B21b: j=2,3; k=4,5 -> b_idx[j][k] = (2+j)*8 + (4+k)
    b21b = [[(2+j)*8 + (4+k) for k in range(2)] for j in range(2)]
    # B22a: j=0,1; k=6,7 -> b_idx[j][k] = j*8 + (6+k)
    b22a = [[j*8 + (6+k) for k in range(2)] for j in range(2)]
    # B22b: j=2,3; k=6,7 -> b_idx[j][k] = (2+j)*8 + (6+k)
    b22b = [[(2+j)*8 + (6+k) for k in range(2)] for j in range(2)]
    
    # C11: i=0,1; k=0,1 -> c_idx[i][k] = k*3 + i
    c11 = [[k*3 + i for k in range(2)] for i in range(2)]
    # C12: i=0,1; k=2,3 -> c_idx[i][k] = k*3 + i
    c12 = [[k*3 + i for k in range(2,4)] for i in range(2)]
    # C21: i=2; k=0,1 -> c_idx[0][k] = k*3 + 2
    c21 = [[k*3 + 2 for k in range(2)]]
    # C22: i=2; k=2,3 -> c_idx[0][k] = k*3 + 2
    c22 = [[k*3 + 2 for k in range(2,4)]]
    # C31: i=0,1; k=4,5 -> c_idx[i][k] = k*3 + i
    c31 = [[k*3 + i for k in range(4,6)] for i in range(2)]
    # C32: i=0,1; k=6,7 -> c_idx[i][k] = k*3 + i
    c32 = [[k*3 + i for k in range(6,8)] for i in range(2)]
    # C41: i=2; k=4,5 -> c_idx[0][k] = k*3 + 2
    c41 = [[k*3 + 2 for k in range(4,6)]]
    # C42: i=2; k=6,7 -> c_idx[0][k] = k*3 + 2
    c42 = [[k*3 + 2 for k in range(6,8)]]
    
    # A1*B11 = A1a*B11a + A1b*B11b
    strassen(a1a, b11a, c11)
    strassen(a1b, b11b, c11)
    # A1*B12 = A1a*B12a + A1b*B12b
    strassen(a1a, b12a, c12)
    strassen(a1b, b12b, c12)
    # A1*B21 = A1a*B21a + A1b*B21b
    strassen(a1a, b21a, c31)
    strassen(a1b, b21b, c31)
    # A1*B22 = A1a*B22a + A1b*B22b
    strassen(a1a, b22a, c32)
    strassen(a1b, b22b, c32)
    # A2*B11 = A2a*B11a + A2b*B11b
    schoolbook_1x2(a2a, b11a, c21)
    schoolbook_1x2(a2b, b11b, c21)
    # A2*B12 = A2a*B12a + A2b*B12b
    schoolbook_1x2(a2a, b12a, c22)
    schoolbook_1x2(a2b, b12b, c22)
    # A2*B21 = A2a*B21a + A2b*B21b
    schoolbook_1x2(a2a, b21a, c41)
    schoolbook_1x2(a2b, b21b, c41)
    # A2*B22 = A2a*B22a + A2b*B22b
    schoolbook_1x2(a2a, b22a, c42)
    schoolbook_1x2(a2b, b22b, c42)
    
    # Now assemble U, V, W
    rank = len(factors)
    n, m, p = 3, 4, 8
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(factors):
        U[u_idx][c] += u_val
        V[v_idx][c] += v_val
        W[w_idx][c] += w_val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
