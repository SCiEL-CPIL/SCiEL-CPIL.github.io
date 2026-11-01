"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 32

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a tensor decomposition for 2x4x4 matrix multiplication.
    We try to find a rank-28 decomposition by combining Strassen-style
    rank-7 decompositions for the 2x2x2 subproblems.
    
    The 2x4 matrix A has rows indexed by i (0,1), columns by j (0,1,2,3).
    The 4x4 matrix B has rows j (0,1,2,3), columns k (0,1,2,3).
    Output C[i,k] = sum_j A[i,j] * B[j,k].
    
    We can view this as two independent 2x2x2 problems if we pair up
    columns of A: (j=0,1) and (j=2,3), or use a different pairing.
    
    Actually, the natural decomposition is:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    
    This is the sum of 4 terms, each of which is a 2x1x4 outer product.
    Each term has "rank" 1 in the tensor sense (it's a single product of
    2 scalars from A, 1 from B... wait, no).
    
    Let me reconsider. The matrix multiplication tensor T is in
    C^{2x4} ⊗ C^{4x4} ⊗ C^{2x4} (with appropriate indices).
    
    Actually, the standard way: T[i,j,k] = 1 if this term appears in C[i,k].
    For 2x4 * 4x4, T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}.
    The "rank" of the multiplication tensor is the minimum number of rank-1
    tensors that sum to T.
    
    A rank-1 tensor has the form u ⊗ v ⊗ w, which corresponds to
    (sum over some linear combination of A entries) * (sum over some linear
    combination of B entries) * (sum over some linear combination of C entries).
    
    For the schoolbook decomposition, each (i,j,k) triple gives one rank-1 term:
    A[i,j] * B[j,k] * C[i,k].
    So rank <= 2*4*4 = 32.
    
    Can we do better? For 2x2*2x2, Strassen gives rank 7 < 8.
    For 2x4*4x4, we can think of it as:
    C = [A_left, A_right] * B where A_left, A_right are 2x2.
    Or C[i,k] = sum_j A[i,j] B[j,k].
    
    We can group the j indices: {0,1} and {2,3}.
    Let A1 = A[:,0:2], A2 = A[:,2:4], B1 = B[0:2,:], B2 = B[2:4,:].
    Then C = A1 * B1 + A2 * B2.
    
    Each A1*B1 is a 2x2*2x4 multiplication, and A2*B2 is also 2x2*2x4.
    
    For 2x2*2x4: C[i,k] = sum_{j=0}^{1} A1[i,j] * B1[j,k].
    This can be viewed as two 2x1*2x2 products... or we can use a Strassen-like
    approach.
    
    Actually, let's think about 2x2*2x2 first. Strassen's algorithm:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A10)B11
    M6 = (A10-A00)(B00+B10)
    M7 = (A11-A10)(B01+B11)
    
    C00 = M1 - M4 + M5
    C01 = M3 + M4
    C10 = M2 + M7
    C11 = M1 - M2 + M3
    
    For 2x2*2x4, we have two 2x2*2x2 subproblems (for columns 0,1 and 2,3 of B).
    So rank would be 7 + 7 = 14 for 2x2*2x4.
    
    Then for 2x4*4x4, we split into two 2x2*2x4 problems, each of rank 14,
    giving total rank 28.
    
    Let me verify this works. We need to construct U, V, W matrices.
    
    U has shape (n*m, rank) = (8, 28).
    V has shape (m*p, rank) = (16, 28).
    W has shape (p*n, rank) = (8, 28).
    
    The index conventions:
    U: row i*m+j (i in 0..1, j in 0..3), column c in 0..27
    V: row j*p+k (j in 0..3, k in 0..3), column c in 0..27
    W: row k*n+i (k in 0..3, i in 0..1), column c in 0..27
    
    For the first 2x2*2x4 block (j in {0,1}):
    We have A indices: (i,0) and (i,1) for i in {0,1}.
    B indices: (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3).
    C indices: (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3).
    
    Using Strassen on the "left" 2x2 part of B (columns 0,1):
    M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    M2 = (A10+A11)B00 -> contributes to C10, C11
    M3 = A00(B10-B11) -> contributes to C01, C11
    M4 = A11(B01-B00) -> contributes to C00, C01
    M5 = (A00+A10)B11 -> contributes to C00, C11
    M6 = (A10-A00)(B00+B10) -> contributes to C00
    M7 = (A11-A10)(B01+B11) -> contributes to C01
    
    Wait, I need to be more careful. Let me use standard Strassen formulas.
    
    For 2x2*2x2:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+cg, af+dh], [ce+dg, cf+gh]]
    
    M1 = (a+d)(e+h)
    M2 = (c+d)e
    M3 = a(g-h)
    M4 = d(f-e)
    M5 = (a+c)h
    M6 = (c-a)(e+g)
    M7 = (d-c)(f+h)
    
    C11 = M1 - M4 + M5
    C12 = M3 + M4
    C21 = M2 + M7
    C22 = M1 - M2 + M3
    
    Now for 2x2*2x4, let A be 2x2, B be 2x4 = [[b00,b01,b02,b03],[b10,b11,b12,b13]].
    C = [[a*b00+c*b10, a*b01+c*b11, a*b02+c*b12, a*b03+c*b13],
         [c*b00+d*b10, c*b01+d*b11, c*b02+d*b12, c*b03+d*b13]]
    
    We can apply Strassen to each pair of columns. For columns 0,1:
    M1 = (a+d)(b00+b11)
    M2 = (c+d)b00
    M3 = a(b10-b11)
    M4 = d(b01-b00)
    M5 = (a+c)b11
    M6 = (c-a)(b00+b10)
    M7 = (d-c)(b01+b11)
    
    C00 = M1 - M4 + M5
    C01 = M3 + M4
    C10 = M2 + M7
    C11 = M1 - M2 + M3
    
    For columns 2,3, similarly:
    M1' = (a+d)(b02+b13)
    M2' = (c+d)b02
    M3' = a(b12-b13)
    M4' = d(b03-b02)
    M5' = (a+c)b13
    M6' = (c-a)(b02+b12)
    M7' = (d-c)(b03+b13)
    
    C02 = M1' - M4' + M5'
    C03 = M3' + M4'
    C12 = M2' + M7'
    C13 = M1' - M2' + M3'
    
    So for one 2x2*2x4 block, we have 14 rank-1 terms.
    
    For the full 2x4*4x4, we split B into B_left (rows 0,1) and B_right (rows 2,3).
    A_left = A[:,0:2], A_right = A[:,2:4].
    C = A_left * B_left + A_right * B_right.
    
    Each is a 2x2*2x4 product with rank 14, so total rank = 28.
    
    Let me now construct the matrices.
    
    Block 1: A_left = A[:,0:2] (indices (i,0), (i,1) for i in 0,1), B_left = B[0:2,:]
    A indices in U: row i*4+j where i in {0,1}, j in {0,1} -> rows 0,1,4,5
    Wait, U has shape (n*m, rank) = (8, 28). Row index is i*m+j = i*4+j.
    For i=0: rows 0,1,2,3 (j=0,1,2,3)
    For i=1: rows 4,5,6,7 (j=0,1,2,3)
    
    For A_left, we use j=0,1: rows 0,1 (i=0) and 4,5 (i=1).
    For A_right, we use j=2,3: rows 2,3 (i=0) and 6,7 (i=1).
    
    V has shape (m*p, rank) = (16, 28). Row index is j*p+k = j*4+k.
    For B_left (j=0,1): rows 0-7 (j=0: 0-3, j=1: 4-7)
    For B_right (j=2,3): rows 8-15 (j=2: 8-11, j=3: 12-15)
    
    W has shape (p*n, rank) = (8, 28). Row index is k*n+i = k*2+i.
    For k=0: rows 0,1 (i=0,1)
    For k=1: rows 2,3
    For k=2: rows 4,5
    For k=3: rows 6,7
    
    Now let's build the 28 rank-1 terms.
    
    Block 1 (A_left, B_left): 14 terms
    Term 1 (M1 for cols 0,1):
    U: (a00+a11) -> U[0,0]=1, U[5,0]=1 (wait, A_left indices: (0,0)->row0, (0,1)->row1, (1,0)->row4, (1,1)->row5)
    Actually, A[i,j] for i in {0,1}, j in {0,1}:
    (0,0) -> row 0*4+0 = 0
    (0,1) -> row 0*4+1 = 1
    (1,0) -> row 1*4+0 = 4
    (1,1) -> row 1*4+1 = 5
    
    M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    U: A[0,0]=1, A[1,1]=1 -> U[0,0]=1, U[5,0]=1
    V: B[0,0]=1, B[1,1]=1 -> V[0,0]=1, V[5,0]=1 (B[0,0]: row 0*4+0=0, B[1,1]: row 1*4+1=5)
    W: C[0,0]=1, C[1,1]=1? No, C[i,k] where i is row of C, k is column.
    C[0,0] = k=0,i=0 -> W row 0*2+0=0
    C[0,1] = k=1,i=0 -> W row 1*2+0=2
    C[1,0] = k=0,i=1 -> W row 0*2+1=1
    C[1,1] = k=1,i=1 -> W row 1*2+1=3
    
    C00 = M1 - M4 + M5, C01 = M3 + M4, C10 = M2 + M7, C11 = M1 - M2 + M3
    
    So M1 contributes to C00 (+1) and C11 (+1).
    W: W[0,0]=1, W[3,0]=1
    
    Let me code this up systematically.
    """
    U = np.zeros((8, 28), dtype=int)
    V = np.zeros((16, 28), dtype=int)
    W = np.zeros((8, 28), dtype=int)
    
    # Helper to add a rank-1 term
    # u_idx: dict {u_row: coeff}
    # v_idx: dict {v_row: coeff}
    # w_idx: dict {w_row: coeff}
    def add_term(term_idx, u_coeffs, v_coeffs, w_coeffs):
        for row, coeff in u_coeffs.items():
            U[row, term_idx] += coeff
        for row, coeff in v_coeffs.items():
            V[row, term_idx] += coeff
        for row, coeff in w_coeffs.items():
            W[row, term_idx] += coeff
    
    # Block 1: A_left (j=0,1), B_left (j=0,1)
    # A rows: (0,0)=0, (0,1)=1, (1,0)=4, (1,1)=5
    # B rows: (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, (1,2)=6, (1,3)=7
    # W rows: C(i,k) -> k*2+i: C(0,0)=0, C(1,0)=1, C(0,1)=2, C(1,1)=3, C(0,2)=4, C(1,2)=5, C(0,3)=6, C(1,3)=7
    
    t = 0
    # M1 = (A00+A11)(B00+B11), C00+=M1, C11+=M1
    add_term(t, {0:1, 5:1}, {0:1, 5:1}, {0:1, 3:1})
    t += 1
    # M2 = (A10+A11)B00, C10+=M2, C11-=M2
    add_term(t, {4:1, 5:1}, {0:1}, {1:1, 3:-1})
    t += 1
    # M3 = A00(B10-B11), C01+=M3, C11+=M3
    add_term(t, {0:1}, {4:1, 5:-1}, {2:1, 3:1})
    t += 1
    # M4 = A11(B01-B00), C00-=M4, C01+=M4
    add_term(t, {5:1}, {1:1, 0:-1}, {0:-1, 2:1})
    t += 1
    # M5 = (A00+A10)B11, C00+=M5, C11+=M5
    add_term(t, {0:1, 4:1}, {5:1}, {0:1, 3:1})
    t += 1
    # M6 = (A10-A00)(B00+B10), C00+=M6
    add_term(t, {4:1, 0:-1}, {0:1, 4:1}, {0:1})
    t += 1
    # M7 = (A11-A10)(B01+B11), C01+=M7
    add_term(t, {5:1, 4:-1}, {1:1, 5:1}, {2:1})
    t += 1
    
    # M1' = (A00+A11)(B02+B13), C02+=M1', C13+=M1'
    add_term(t, {0:1, 5:1}, {2:1, 7:1}, {4:1, 7:1})
    t += 1
    # M2' = (A10+A11)B02, C12+=M2', C13-=M2'
    add_term(t, {4:1, 5:1}, {2:1}, {5:1, 7:-1})
    t += 1
    # M3' = A00(B12-B13), C03+=M3', C13+=M3'
    add_term(t, {0:1}, {6:1, 7:-1}, {6:1, 7:1})
    t += 1
    # M4' = A11(B03-B02), C02-=M4', C03+=M4'
    add_term(t, {5:1}, {3:1, 2:-1}, {4:-1, 6:1})
    t += 1
    # M5' = (A00+A10)B13, C02+=M5', C13+=M5'
    add_term(t, {0:1, 4:1}, {7:1}, {4:1, 7:1})
    t += 1
    # M6' = (A10-A00)(B02+B12), C02+=M6'
    add_term(t, {4:1, 0:-1}, {2:1, 6:1}, {4:1})
    t += 1
    # M7' = (A11-A10)(B03+B13), C03+=M7'
    add_term(t, {5:1, 4:-1}, {3:1, 7:1}, {6:1})
    t += 1
    
    # Block 2: A_right (j=2,3), B_right (j=2,3)
    # A rows: (0,2)=2, (0,3)=3, (1,2)=6, (1,3)=7
    # B rows: (2,0)=8, (2,1)=9, (2,2)=10, (2,3)=11, (3,0)=12, (3,1)=13, (3,2)=14, (3,3)=15
    
    # M1'' = (A02+A13)(B20+B31), C00+=M1'', C11+=M1''
    add_term(t, {2:1, 7:1}, {8:1, 13:1}, {0:1, 3:1})
    t += 1
    # M2'' = (A12+A13)B20, C10+=M2'', C11-=M2''
    add_term(t, {6:1, 7:1}, {8:1}, {1:1, 3:-1})
    t += 1
    # M3'' = A02(B30-B31), C01+=M3'', C11+=M3''
    add_term(t, {2:1}, {12:1, 13:-1}, {2:1, 3:1})
    t += 1
    # M4'' = A13(B21-B20), C00-=M4'', C01+=M4''
    add_term(t, {7:1}, {9:1, 8:-1}, {0:-1, 2:1})
    t += 1
    # M5'' = (A02+A12)B31, C00+=M5'', C11+=M5''
    add_term(t, {2:1, 6:1}, {13:1}, {0:1, 3:1})
    t += 1
    # M6'' = (A12-A02)(B20+B30), C00+=M6''
    add_term(t, {6:1, 2:-1}, {8:1, 12:1}, {0:1})
    t += 1
    # M7'' = (A13-A12)(B21+B31), C01+=M7''
    add_term(t, {7:1, 6:-1}, {9:1, 13:1}, {2:1})
    t += 1
    
    # M1''' = (A02+A13)(B22+B33), C02+=M1''', C13+=M1'''
    add_term(t, {2:1, 7:1}, {10:1, 15:1}, {4:1, 7:1})
    t += 1
    # M2''' = (A12+A13)B22, C12+=M2''', C13-=M2'''
    add_term(t, {6:1, 7:1}, {10:1}, {5:1, 7:-1})
    t += 1
    # M3''' = A02(B32-B33), C03+=M3''', C13+=M3'''
    add_term(t, {2:1}, {14:1, 15:-1}, {6:1, 7:1})
    t += 1
    # M4''' = A13(B23-B22), C02-=M4''', C03+=M4'''
    add_term(t, {7:1}, {11:1, 10:-1}, {4:-1, 6:1})
    t += 1
    # M5''' = (A02+A12)B33, C02+=M5''', C13+=M5'''
    add_term(t, {2:1, 6:1}, {15:1}, {4:1, 7:1})
    t += 1
    # M6''' = (A12-A02)(B22+B32), C02+=M6'''
    add_term(t, {6:1, 2:-1}, {10:1, 14:1}, {4:1})
    t += 1
    # M7''' = (A13-A12)(B23+B33), C03+=M7'''
    add_term(t, {7:1, 6:-1}, {11:1, 15:1}, {6:1})
    t += 1
    
    # Verify: t should be 28
    assert t == 28, f"Expected 28 terms, got {t}"
    
    # Verify the decomposition
    # C[i,k] should equal sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c] summed over j
    # Actually, the tensor decomposition means:
    # T[i,j,k] = sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*2+i, c]
    # But T[i,j,k] = 1 for all i,j,k in the multiplication tensor.
    # Wait, no. The matrix multiplication tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*4+j,c] V[j*4+k,c] W[k*2+i,c] A[i,j] B[j,k]... 
    # 
    # Actually, the decomposition means:
    # The bilinear map (A,B) -> C = A*B is represented as:
    # C[i,k] = sum_c (sum_j U[i*4+j,c] A[i,j]) * (sum_k V[j*4+k,c] B[j,k]) * W[k*2+i,c]
    # Wait, that's not right either.
    #
    # The standard tensor decomposition for matrix multiplication:
    # T is a 3-tensor with T[i,j,k] = 1 (for the multiplication, each term A[i,j]*B[j,k] contributes to C[i,k])
    # Rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in C^{n*m}, v_c in C^{m*p}, w_c in C^{p*n}
    #
    # The multiplication formula: C[i,k] = sum_j A[i,j] B[j,k]
    # = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * (A[i,j] B[j,k] ... no)
    #
    # Actually: if T = sum_c u_c ⊗ v_c ⊗ w_c, then
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_c (sum_{i,j} u_c[i*m+j] A[i,j]) (sum_{j,k} v_c[j*p+k] B[j,k]) (sum_{i,k} w_c[k*n+i] C[i,k])
    #
    # But we want C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T satisfies: <T, A ⊗ B ⊗ C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[i,k]
    # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k, and
    # sum_{i,j,k} A[i,j] B[j,k] C[i,k] = sum_{i,k} C[i,k] (sum_j A[i,j] B[j,k])
    # = sum_{i,k} C[i,k] C_computed[i,k]
    #
    # So the decomposition works if:
    # sum_c u_c[i*m+j] v_c[j*p+k] w_c[k*n+i] = 1 for all i,j,k
    # i.e., the "flattened" tensor product equals the all-ones tensor.
    #
    # Let me verify this numerically.
    
    # Build the tensor from U, V, W
    tensor_reconstructed = np.zeros((2, 4, 4))
    for c in range(28):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    tensor_reconstructed[i, j, k] += uc[i*4+j] * vc[j*4+k] * wc[k*2+i]
    
    # The target tensor is all 1s
    target_tensor = np.ones((2, 4, 4))
    
    if not np.allclose(tensor_reconstructed, target_tensor):
        # Fall back to schoolbook
        rank = 32
        U_fb = np.zeros((8, 32), dtype=int)
        V_fb = np.zeros((16, 32), dtype=int)
        W_fb = np.zeros((8, 32), dtype=int)
        col = 0
        for i in range(2):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*4+k, col] = 1
                    W_fb[k*2+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
