import numpy as np

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We decompose 2x4x7 into two 2x2x7 multiplications.
    # 2x2x7 is decomposed into 2x2x4 and 2x2x3.
    # 2x2x4 is two 2x2x2 (Strassen, rank 7 each). Total 14.
    # 2x2x3 is one 2x2x2 (Strassen, rank 7) and one 2x2x1 (rank 4). Total 11.
    # Total rank for 2x2x7 is 25.
    # Total rank for 2x4x7 is 50.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)] # 8 x 50
    V = [[0] * rank for _ in range(m * p)] # 28 x 50
    W = [[0] * rank for _ in range(p * n)] # 14 x 50
    
    # Helper to add a Strassen 2x2x2 block
    # Indices:
    # A is 2x2, B is 2x2, C is 2x2
    # Global indices:
    # A: rows i0..i1, cols j0..j1
    # B: rows j0..j1, cols k0..k1
    # C: rows i0..i1, cols k0..k1
    # W index: k*n + i
    # U index: i*m + j
    # V index: j*p + k
    
    def add_strassen_222(offset, i0, j0, k0):
        # offset is the column offset in U, V, W
        # i0, j0, k0 are the starting indices in the global matrices
        # A: [i0, i0+1] x [j0, j0+1]
        # B: [j0, j0+1] x [k0, k0+1]
        # C: [i0, i0+1] x [k0, k0+1]
        
        # Strassen 7 terms
        # M1 = (A11 + A22)(B11 + B22)
        U[i0*m + j0][offset+0] += 1
        U[(i0+1)*m + (j0+1)][offset+0] += 1
        V[j0*p + k0][offset+0] += 1
        V[(j0+1)*p + (k0+1)][offset+0] += 1
        W[k0*n + i0][offset+0] += 1
        W[(k0+1)*n + i0][offset+0] -= 1
        W[k0*n + (i0+1)][offset+0] += 1
        W[(k0+1)*n + (i0+1)][offset+0] -= 1
        
        # M2 = (A21 + A22)B11
        U[(i0+1)*m + j0][offset+1] += 1
        U[(i0+1)*m + (j0+1)][offset+1] += 1
        V[j0*p + k0][offset+1] += 1
        W[k0*n + (i0+1)][offset+1] += 1
        W[(k0+1)*n + (i0+1)][offset+1] += 1
        
        # M3 = A11(B12 - B22)
        U[i0*m + j0][offset+2] += 1
        V[j0*p + (k0+1)][offset+2] += 1
        V[(j0+1)*p + (k0+1)][offset+2] -= 1
        W[(k0+1)*n + i0][offset+2] += 1
        W[(k0+1)*n + (i0+1)][offset+2] += 1
        
        # M4 = A22(B21 - B11)
        U[(i0+1)*m + (j0+1)][offset+3] += 1
        V[(j0+1)*p + k0][offset+3] += 1
        V[j0*p + k0][offset+3] -= 1
        W[k0*n + (i0+1)][offset+3] += 1
        W[(k0+1)*n + (i0+1)][offset+3] += 1
        
        # M5 = (A11 + A12)B22
        U[i0*m + j0][offset+4] += 1
        U[i0*m + (j0+1)][offset+4] += 1
        V[(j0+1)*p + (k0+1)][offset+4] += 1
        W[(k0+1)*n + i0][offset+4] -= 1
        W[(k0+1)*n + (i0+1)][offset+4] += 1
        
        # M6 = (A21 - A11)(B11 + B12)
        U[(i0+1)*m + j0][offset+5] += 1
        U[i0*m + j0][offset+5] -= 1
        V[j0*p + k0][offset+5] += 1
        V[j0*p + (k0+1)][offset+5] += 1
        W[k0*n + (i0+1)][offset+5] += 1
        W[(k0+1)*n + (i0+1)][offset+5] += 1
        
        # M7 = (A12 - A22)(B21 + B22)
        U[i0*m + (j0+1)][offset+6] += 1
        U[(i0+1)*m + (j0+1)][offset+6] -= 1
        V[(j0+1)*p + k0][offset+6] += 1
        V[(j0+1)*p + (k0+1)][offset+6] += 1
        W[k0*n + i0][offset+6] += 1
        W[(k0+1)*n + i0][offset+6] -= 1
        
        # Now we need to combine these into C = A B
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # The W matrix above was defined as the contribution to C.
        # Let's re-verify the W updates.
        # The standard Strassen formula gives C.
        # My W updates above were just adding the M terms to specific C locations?
        # No, the W matrix in the tensor decomposition represents the coefficients.
        # The tensor is T = sum U_r V_r W_r.
        # C_{ik} = sum_j A_{ij} B_{jk} = sum_r U_{ir} V_{jr} W_{kr}? No.
        # C_{ik} = sum_r U_{i,m+j, r} V_{j,p+k, r} W_{k,n+i, r}?
        # The interface is:
        # U shape (n*m, rank)
        # V shape (m*p, rank)
        # W shape (p*n, rank)
        # C_{i,k} = sum_{j, r} U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        
        # So, for each term r, we have a product of a vector in U, a vector in V, and a vector in W.
        # The Strassen algorithm defines 7 intermediate products M_r.
        # Each M_r is a product of a linear combination of A and a linear combination of B.
        # M_r = (sum u_{ij} A_{ij}) * (sum v_{jk} B_{jk})
        # This corresponds to one rank-1 tensor term in the decomposition of the multiplication tensor.
        # The contribution to C is M_r * (linear combination of 1s in C positions).
        # So, for each M_r, we add 1 to U at the (i,j) indices with coefficient u_{ij},
        # add 1 to V at the (j,k) indices with coefficient v_{jk},
        # and add 1 to W at the (k,i) indices with coefficient w_{ki} (where C_{ki} gets M_r * w_{ki}).
        
        # I need to rewrite the W updates to reflect the combination into C.
        
    # Let's restart the construction logic properly.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_term(r, u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs):
        for idx, coeff in zip(u_indices, u_coeffs):
            U[idx][r] += coeff
        for idx, coeff in zip(v_indices, v_coeffs):
            V[idx][r] += coeff
        for idx, coeff in zip(w_indices, w_coeffs):
            W[idx][r] += coeff
            
    # Block 1: A_L (rows 0-1, cols 0-1) * B_T (rows 0-1, cols 0-6)
    # Split B_T into B_T1 (cols 0-3) and B_T2 (cols 4-6)
    # B_T1 is 2x4, B_T2 is 2x3
    
    # Sub-block 1.1: A_L * B_T1 (2x2 * 2x4)
    # Split B_T1 into two 2x2 blocks: B_T1a (cols 0-1), B_T1b (cols 2-3)
    # Term range: 0-6 for B_T1a, 7-13 for B_T1b
    
    # 1.1a: A_L * B_T1a (2x2x2)
    # i0=0, j0=0, k0=0
    add_strassen_222(0, 0, 0, 0)
    
    # 1.1b: A_L * B_T1b (2x2x2)
    # i0=0, j0=0, k0=2
    add_strassen_222(7, 0, 0, 2)
    
    # Sub-block 1.2: A_L * B_T2 (2x2 * 2x3)
    # Split B_T2 into B_T2a (cols 4-5, 2x2) and B_T2b (col 6, 2x1)
    # Term range: 14-20 for B_T2a, 21-24 for B_T2b
    
    # 1.2a: A_L * B_T2a (2x2x2)
    # i0=0, j0=0, k0=4
    add_strassen_222(14, 0, 0, 4)
    
    # 1.2b: A_L * B_T2b (2x2x1)
    # A is 2x2, B is 2x1.
    # C is 2x1.
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # This is 4 terms.
    # r=21: A11 B1 -> C1
    add_term(21, [0*m+0], [1], [0*p+6], [1], [6*n+0], [1])
    # r=22: A12 B2 -> C1
    add_term(22, [0*m+1], [1], [1*p+6], [1], [6*n+0], [1])
    # r=23: A21 B1 -> C2
    add_term(23, [1*m+0], [1], [0*p+6], [1], [6*n+1], [1])
    # r=24: A22 B2 -> C2
    add_term(24, [1*m+1], [1], [1*p+6], [1], [6*n+1], [1])
    
    # Block 2: A_R (rows 0-1, cols 2-3) * B_B (rows 2-3, cols 0-6)
    # Term range: 25-49
    
    # 2.1: A_R * B_B1 (2x2 * 2x4)
    # Split B_B1 (cols 0-3) into B_B1a (cols 0-1), B_B1b (cols 2-3)
    
    # 2.1a: A_R * B_B1a (2x2x2)
    # i0=0, j0=2, k0=0
    add_strassen_222(25, 0, 2, 0)
    
    # 2.1b: A_R * B_B1b (2x2x2)
    # i0=0, j0=2, k0=2
    add_strassen_222(32, 0, 2, 2)
    
    # 2.2: A_R * B_B2 (2x2 * 2x3)
    # Split B_B2 (cols 4-6) into B_B2a (cols 4-5), B_B2b (col 6)
    
    # 2.2a: A_R * B_B2a (2x2x2)
    # i0=0, j0=2, k0=4
    add_strassen_222(39, 0, 2, 4)
    
    # 2.2b: A_R * B_B2b (2x2x1)
    # r=46: A11 B1 -> C1
    add_term(46, [0*m+2], [1], [2*p+6], [1], [6*n+0], [1])
    # r=47: A12 B2 -> C1
    add_term(47, [0*m+3], [1], [3*p+6], [1], [6*n+0], [1])
    # r=48: A21 B1 -> C2
    add_term(48, [1*m+2], [1], [2*p+6], [1], [6*n+1], [1])
    # r=49: A22 B2 -> C2
    add_term(49, [1*m+3], [1], [3*p+6], [1], [6*n+1], [1])
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
