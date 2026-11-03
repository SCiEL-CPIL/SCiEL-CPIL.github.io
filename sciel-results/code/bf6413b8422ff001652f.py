import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 124
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(i, j, k, coef_u, coef_v, coef_w):
        nonlocal col
        if col >= rank:
            return
        U[i * m + j][col] += coef_u
        V[j * p + k][col] += coef_v
        W[k * n + i][col] += coef_w
        col += 1

    # Helper for Strassen 2x2
    # i0, i1 are global rows of A
    # j0, j1 are global cols of A (rows of B)
    # k0, k1 are global cols of B
    def add_strassen_2x2(i0, i1, j0, j1, k0, k1):
        nonlocal col
        # M1 = (a00 + a11)(b00 + b11)
        add_term(i0, j0, k0, 1, 1, 0) # a00 b00 -> c00, c11
        add_term(i0, j0, k1, 1, 1, 0) # a00 b11 -> c00, c11
        add_term(i1, j1, k0, 1, 1, 0) # a11 b00 -> c00, c11
        add_term(i1, j1, k1, 1, 1, 0) # a11 b11 -> c00, c11
        # Wait, this is schoolbook expansion. Strassen uses 7 "multiplications".
        # The "multiplication" is of the sum.
        # In tensor decomposition, a "multiplication" of (u1+u2) and (v1+v2) is NOT 4 terms.
        # It is 1 term where U has u1, u2 and V has v1, v2.
        # But the output W is the sum of contributions.
        
        # Let's re-evaluate.
        # A rank-1 term is u * v * w.
        # If we use Strassen, we define M1 = (a00+a11)(b00+b11).
        # This M1 contributes to c00 and c11.
        # The term is:
        # U has 1 at (i0, j0) and 1 at (i1, j1).
        # V has 1 at (j0, k0) and 1 at (j1, k1).
        # W has 1 at (k0, i0) [c00] and 1 at (k1, i1) [c11].
        # This is ONE rank-1 term?
        # No. The product of sums is a bilinear form.
        # (a00+a11)(b00+b11) = a00b00 + a00b11 + a11b00 + a11b11.
        # This bilinear form has rank 4 if we treat it as a sum of products.
        # BUT, Strassen's insight is that we don't need to compute these 4 separately.
        # We compute M1, M2...M7.
        # Each Mk is a "scalar" (linear combination of inputs) times a "scalar" (linear combination of inputs).
        # In the tensor decomposition of the whole matrix multiplication,
        # each Mk corresponds to a rank-1 term in the decomposition of the SUB-tensor?
        # No. The whole tensor T is decomposed into sum of rank-1 tensors.
        # If we replace a 2x2x2 block (rank 8) with 7 rank-1 tensors,
        # we are effectively saying that the sub-tensor has rank 7.
        # This is possible because the sub-tensor is not independent of the rest?
        # No, Strassen works for the whole matrix.
        
        # Correct implementation of Strassen in Tensor Decomposition:
        # We add 7 terms to the global U, V, W.
        # Term 1 (M1):
        # U: 1 at (i0, j0), 1 at (i1, j1)
        # V: 1 at (j0, k0), 1 at (j1, k1)
        # W: 1 at (k0, i0), 1 at (k1, i1)
        # This single term represents the product (a00+a11)(b00+b11) contributing to c00 and c11?
        # Let's check:
        # Term value = (a00+a11)(b00+b11) * (1 at c00 + 1 at c11)?
        # No, W is a vector. The term contributes to W[k*n+i].
        # If W has 1 at c00 and 1 at c11, the term adds M1 to c00 and M1 to c11.
        # This is correct for M1's contribution to c00 and c11.
        
        # So, for each Strassen product Mk, we add ONE term to U, V, W.
        # U has the coefficients of the A-linear-combination.
        # V has the coefficients of the B-linear-combination.
        # W has the coefficients of the C-linear-combination (how Mk contributes to C).
        
        # M1 = (a00+a11)(b00+b11)
        # U: 1 at (i0,j0), 1 at (i1,j1)
        # V: 1 at (j0,k0), 1 at (j1,k1)
        # W: 1 at (k0,i0) [c00], 1 at (k1,i1) [c11]
        
        # M2 = (a10+a11)b00
        # U: 1 at (i1,j0), 1 at (i1,j1)
        # V: 1 at (j0,k0)
        # W: 1 at (k0,i1) [c10], -1 at (k1,i1) [c11]
        
        # M3 = a00(b10-b11)
        # U: 1 at (i0,j0)
        # V: 1 at (j1,k0), -1 at (j1,k1)
        # W: 1 at (k0,i0) [c00], 1 at (k0,i1) [c01]
        
        # M4 = a11(b00-b10)
        # U: 1 at (i1,j1)
        # V: 1 at (j0,k0), -1 at (j1,k0)
        # W: 1 at (k0,i0) [c00], 1 at (k0,i1) [c10]
        
        # M5 = (a00+a10)b11
        # U: 1 at (i0,j0), 1 at (i1,j0)
        # V: 1 at (j1,k1)
        # W: -1 at (k0,i0) [c00], 1 at (k0,i1) [c01]
        
        # M6 = (a10-a00)(b00+b01)
        # U: 1 at (i1,j0), -1 at (i0,j0)
        # V: 1 at (j0,k0), 1 at (j0,k1)
        # W: 1 at (k1,i1) [c11]
        
        # M7 = (a01-a11)(b10+b11)
        # U: 1 at (i0,j1), -1 at (i1,j1)
        # V: 1 at (j1,k0), 1 at (j1,k1)
        # W: 1 at (k0,i0) [c00], 1 at (k1,i1) [c11]

        # Now, add these 7 terms.
        
        # M1
        U[i0*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M2
        U[i1*m+j0][col] += 1
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        W[k0*n+i1][col] += 1
        W[k1*n+i1][col] -= 1
        col += 1
        
        # M3
        U[i0*m+j0][col] += 1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] -= 1
        W[k0*n+i0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # M4
        U[i1*m+j1][col] += 1
        V[j0*p+k0][col] += 1
        V[j1*p+k0][col] -= 1
        W[k0*n+i0][col] += 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # M5
        U[i0*m+j0][col] += 1
        U[i1*m+j0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] -= 1
        W[k0*n+i1][col] += 1
        col += 1
        
        # M6
        U[i1*m+j0][col] += 1
        U[i0*m+j0][col] -= 1
        V[j0*p+k0][col] += 1
        V[j0*p+k1][col] += 1
        W[k1*n+i1][col] += 1
        col += 1
        
        # M7
        U[i0*m+j1][col] += 1
        U[i1*m+j1][col] -= 1
        V[j1*p+k0][col] += 1
        V[j1*p+k1][col] += 1
        W[k0*n+i0][col] += 1
        W[k1*n+i1][col] += 1
        col += 1

    # Helper for Schoolbook Rectangular
    def add_schoolbook_rect(i_list, j_list, k_list):
        nonlocal col
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1

    # Blocks
    # A11 (2x2), A12 (2x3), A21 (3x2), A22 (3x3)
    # B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    
    # C11 = A11 B11 + A12 B21
    # A11 B11 is 2x2x2. Use Strassen.
    add_strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # A12 B21 is 2x3 x 3x2 -> 2x2.
    # i in {0,1}, j in {2,3,4}, k in {0,1}
    add_schoolbook_rect([0, 1], [2, 3, 4], [0, 1])
    
    # C12 = A11 B12 + A12 B22
    # A11 B12 is 2x2 x 2x3 -> 2x3.
    # i in {0,1}, j in {0,1}, k in {2,3,4}
    add_schoolbook_rect([0, 1], [0, 1], [2, 3, 4])
    
    # A12 B22 is 2x3 x 3x3 -> 2x3.
    # i in {0,1}, j in {2,3,4}, k in {2,3,4}
    add_schoolbook_rect([0, 1], [2, 3, 4], [2, 3, 4])
    
    # C21 = A21 B11 + A22 B21
    # A21 B11 is 3x2 x 2x2 -> 3x2.
    # i in {2,3,4}, j in {0,1}, k in {0,1}
    add_schoolbook_rect([2, 3, 4], [0, 1], [0, 1])
    
    # A22 B21 is 3x3 x 3x2 -> 3x2.
    # i in {2,3,4}, j in {2,3,4}, k in {0,1}
    add_schoolbook_rect([2, 3, 4], [2, 3, 4], [0, 1])
    
    # C22 = A21 B12 + A22 B22
    # A21 B12 is 3x2 x 2x3 -> 3x3.
    # i in {2,3,4}, j in {0,1}, k in {2,3,4}
    add_schoolbook_rect([2, 3, 4], [0, 1], [2, 3, 4])
    
    # A22 B22 is 3x3 x 3x3 -> 3x3.
    # i in {2,3,4}, j in {2,3,4}, k in {2,3,4}
    add_schoolbook_rect([2, 3, 4], [2, 3, 4], [2, 3, 4])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
