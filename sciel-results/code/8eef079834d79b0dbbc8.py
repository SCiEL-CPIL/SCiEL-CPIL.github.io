import numpy as np

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank 22 decomposition
    rank = 22
    
    # U is (n*m, rank) = (6, 22)
    # V is (m*p, rank) = (12, 22)
    # W is (p*n, rank) = (8, 22)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a term
    # u_idx is index in U (i*m + j)
    # v_idx is index in V (j*p + k)
    # w_idx is index in W (k*n + i)
    # coeff_u, coeff_v, coeff_w are the coefficients for this rank-1 term
    # Note: The decomposition is sum_t U[:, t] * V[:, t] * W[:, t] = Tensor
    # So U[i, t] * V[j, t] * W[k, t] should equal 1 if (i,j,k) is in the tensor, 0 else.
    # Actually, the tensor M_{n,m,p} has entries 1 if i,j,k are valid indices for multiplication.
    # T[i, j, k] = 1 if i in [0,n), j in [0,m), k in [0,p).
    # We want sum_t U[i, t] V[j, t] W[k, t] = 1 for all valid i,j,k.
    
    def add_term(col, i, j, k, cu, cv, cw):
        U[i * m + j, col] += cu
        V[j * p + k, col] += cv
        W[k * n + i, col] += cw

    # Strassen for 2x2 block (i=0,1; j=0,1; k=0,1)
    # M1 = (A00+A11)(B00+B11)
    # U: A00 (i=0,j=0) coeff 1, A11 (i=1,j=1) coeff 1
    # V: B00 (j=0,k=0) coeff 1, B11 (j=1,k=1) coeff 1
    # W: C00 (k=0,i=0) coeff 1, C11 (k=1,i=1) coeff 1?
    # Wait, W index is k*n + i.
    # Term 1:
    # U[0, 0] = 1, U[3, 0] = 1  (i=0,j=0 and i=1,j=1)
    # V[0, 0] = 1, V[5, 0] = 1  (j=0,k=0 and j=1,k=1)
    # W[0, 0] = 1, W[3, 0] = 1  (k=0,i=0 and k=1,i=1)
    
    # Let's manually construct the 7 Strassen terms for the first 2x2x2 block.
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    
    # T1: (A00+A11)(B00+B11)
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    add_term(0, 0, 0, 0, 1, 1, 1)
    add_term(0, 1, 1, 0, 1, 1, 1) # Wait, add_term adds to specific indices.
    # My add_term function above is for a single index.
    # I need a function that adds a product of sums.
    
    # Let's redefine add_term to take lists of (index, coeff)
    def add_rank1_term(col, u_indices, v_indices, w_indices):
        for (idx, coeff) in u_indices:
            U[idx, col] += coeff
        for (idx, coeff) in v_indices:
            V[idx, col] += coeff
        for (idx, coeff) in w_indices:
            W[idx, col] += coeff

    # Reset U, V, W
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen 7 terms
    
    # 1: (A00+A11)(B00+B11) -> C00, C11 contributions
    # U: A00 (0,0), A11 (1,1)
    # V: B00 (0,0), B11 (1,1)
    # W: C00 (0,0), C11 (1,1)
    add_rank1_term(0, [(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)])
    
    # 2: (A10+A11)B00 -> C10, C11
    # U: A10 (1,0), A11 (1,1)
    # V: B00 (0,0)
    # W: C10 (0,1), C11 (1,1)
    add_rank1_term(1, [(4, 1), (5, 1)], [(0, 1)], [(1, 1), (3, 1)])
    
    # 3: A00(B01-B11) -> C01, C11
    # U: A00 (0,0)
    # V: B01 (0,1), B11 (1,1)
    # W: C01 (1,0), C11 (1,1)
    add_rank1_term(2, [(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)])
    
    # 4: A11(B10-B00) -> C10, C11
    # U: A11 (1,1)
    # V: B10 (1,0), B00 (0,0)
    # W: C10 (0,1), C11 (1,1)
    add_rank1_term(3, [(5, 1)], [(4, 1), (0, -1)], [(1, 1), (3, 1)])
    
    # 5: (A00+A01)B11 -> C01, C11
    # U: A00 (0,0), A01 (0,1)
    # V: B11 (1,1)
    # W: C01 (1,0), C11 (1,1)
    add_rank1_term(4, [(0, 1), (1, 1)], [(5, 1)], [(2, 1), (3, 1)])
    
    # 6: (A10-A00)(B00+B01) -> C10, C11
    # U: A10 (1,0), A00 (0,0)
    # V: B00 (0,0), B01 (0,1)
    # W: C10 (0,1), C11 (1,1)
    add_rank1_term(5, [(4, 1), (0, -1)], [(0, 1), (1, 1)], [(1, 1), (3, 1)])
    
    # 7: (A01-A11)(B10+B11) -> C01, C11
    # U: A01 (0,1), A11 (1,1)
    # V: B10 (1,0), B11 (1,1)
    # W: C01 (1,0), C11 (1,1)
    add_rank1_term(6, [(1, 1), (5, -1)], [(4, 1), (5, 1)], [(2, 1), (3, 1)])

    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Same structure, but k indices are 2,3 instead of 0,1.
    # V indices: j*p + k.
    # j=0, k=2 -> 2. j=0, k=3 -> 3.
    # j=1, k=2 -> 6. j=1, k=3 -> 7.
    # W indices: k*n + i.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    # 8: (A00+A11)(B02+B12)
    add_rank1_term(7, [(0, 1), (3, 1)], [(2, 1), (6, 1)], [(4, 1), (5, 1)])
    
    # 9: (A10+A11)B02
    add_rank1_term(8, [(4, 1), (5, 1)], [(2, 1)], [(5, 1), (7, 1)])
    
    # 10: A00(B03-B12)
    add_rank1_term(9, [(0, 1)], [(3, 1), (6, -1)], [(6, 1), (7, 1)])
    
    # 11: A11(B12-B02)
    add_rank1_term(10, [(5, 1)], [(6, 1), (2, -1)], [(5, 1), (7, 1)])
    
    # 12: (A00+A01)B12
    add_rank1_term(11, [(0, 1), (1, 1)], [(6, 1)], [(6, 1), (7, 1)])
    
    # 13: (A10-A00)(B02+B03)
    add_rank1_term(12, [(4, 1), (0, -1)], [(2, 1), (3, 1)], [(5, 1), (7, 1)])
    
    # 14: (A01-A11)(B12+B13)
    add_rank1_term(13, [(1, 1), (5, -1)], [(6, 1), (7, 1)], [(6, 1), (7, 1)])

    # Block 3: i in {0,1}, j in {2}, k in {0,1,2,3}
    # Schoolbook for A_col2 and B_row2.
    # U indices: i*m + 2. i=0 -> 2, i=1 -> 5.
    # V indices: 2*p + k. k=0 -> 8, k=1 -> 9, k=2 -> 10, k=3 -> 11.
    # W indices: k*n + i.
    # k=0, i=0 -> 0. k=0, i=1 -> 1.
    # k=1, i=0 -> 2. k=1, i=1 -> 3.
    # k=2, i=0 -> 4. k=2, i=1 -> 5.
    # k=3, i=0 -> 6. k=3, i=1 -> 7.
    
    # 15: A02 B20
    add_rank1_term(14, [(2, 1)], [(8, 1)], [(0, 1)])
    
    # 16: A02 B21
    add_rank1_term(15, [(2, 1)], [(9, 1)], [(2, 1)])
    
    # 17: A02 B22
    add_rank1_term(16, [(2, 1)], [(10, 1)], [(4, 1)])
    
    # 18: A02 B23
    add_rank1_term(17, [(2, 1)], [(11, 1)], [(6, 1)])
    
    # 19: A12 B20
    add_rank1_term(18, [(5, 1)], [(8, 1)], [(1, 1)])
    
    # 20: A12 B21
    add_rank1_term(19, [(5, 1)], [(9, 1)], [(3, 1)])
    
    # 21: A12 B22
    add_rank1_term(20, [(5, 1)], [(10, 1)], [(5, 1)])
    
    # 22: A12 B23
    add_rank1_term(21, [(5, 1)], [(11, 1)], [(7, 1)])

    return {"U": U, "V": V, "W": W}
