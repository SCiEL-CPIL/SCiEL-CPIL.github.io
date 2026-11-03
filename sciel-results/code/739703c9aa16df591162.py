import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 5
    
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, val=1):
        # u_idx: (i, j)
        # v_idx: (j, k)
        # w_idx: (k, i)
        i, j = u_idx
        j2, k = v_idx
        k2, i2 = w_idx
        # Indices must match for the term to be valid in the tensor product
        # Actually, the decomposition is sum over r of U[:,r] V[:,r] W[:,r]
        # U[i,j,r] * V[j,k,r] * W[k,i,r] contributes to C[i,k]
        # So for a specific term, we pick specific i, j, k.
        # The "val" is the coefficient.
        
        # We need to append to the end of the lists
        # U is a list of rows. We are building column by column.
        # This approach is inefficient if we append to rows.
        # Better to accumulate in a dictionary or list of tuples and convert at the end.
        # Or just keep U, V, W as lists of lists and append 0s.
        
        pass
        
    # Let's use a list of tuples (u_row, v_row, w_row, coeff)
    terms = []
    
    def add_term_simple(i, j, k, coeff=1):
        # i in 0..n-1, j in 0..m-1, k in 0..p-1
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        terms.append((u_row, v_row, w_row, coeff))

    # Strassen for 2x2x2
    # A11 is A[0:2, 0:2], B11 is B[0:2, 0:2]
    # Indices:
    # A11: i in 0,1; j in 0,1
    # B11: j in 0,1; k in 0,1
    # Result C11: i in 0,1; k in 0,1
    
    # Strassen terms for A11 * B11
    # M1 = (A11 + A22) * (B11 + B22)
    # Here A11, A22 are 1x1 blocks of the 2x2 matrix A[0:2,0:2]
    # Let's denote the 1x1 blocks of A[0:2,0:2] as a11, a12, a21, a22
    # a11 is A[0,0], a12 is A[0,1], a21 is A[1,0], a22 is A[1,1]
    # Similarly for B[0:2,0:2]: b11=B[0,0], b12=B[0,1], b21=B[1,0], b22=B[1,1]
    
    # M1 = (a11 + a22) * (b11 + b22)
    # This term contributes to C11.
    # C11[0,0] = a11*b11 + a12*b21 + ...
    # Strassen reconstruction:
    # C11[0,0] = M1 + M4 - M5 + M7
    # C11[0,1] = M3 + M5
    # C11[1,0] = M2 + M4
    # C11[1,1] = M1 - M2 + M3 + M6
    
    # Let's define the 7 multiplications as terms in our decomposition.
    # Each multiplication is a product of a sum of A elements and a sum of B elements.
    # We need to expand these into elementary terms (i, j, k) with coefficients.
    
    # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # Terms:
    # (0,0,0) coeff 1
    # (0,0,1) coeff 1
    # (1,1,0) coeff 1
    # (1,1,1) coeff 1
    
    # M2: (A[1,0] + A[1,1]) * B[0,0]
    # = A[1,0]B[0,0] + A[1,1]B[0,0]
    # Terms:
    # (1,0,0) coeff 1
    # (1,1,0) coeff 1
    
    # M3: A[0,0] * (B[0,1] - B[1,1])
    # = A[0,0]B[0,1] - A[0,0]B[1,1]
    # Terms:
    # (0,0,1) coeff 1
    # (0,1,1) coeff -1
    
    # M4: A[1,1] * (B[1,0] - B[0,0])
    # = A[1,1]B[1,0] - A[1,1]B[0,0]
    # Terms:
    # (1,1,0) coeff 1
    # (1,0,0) coeff -1
    
    # M5: (A[0,0] + A[0,1]) * B[1,1]
    # = A[0,0]B[1,1] + A[0,1]B[1,1]
    # Terms:
    # (0,1,1) coeff 1
    # (0,1,1) coeff 1? No.
    # A[0,0] is i=0, j=0. B[1,1] is j=1, k=1. Term (0,0,1)?? No.
    # The term is A[i,j] * B[j,k].
    # M5 involves A[0,0] (i=0,j=0) and B[1,1] (j=1,k=1). These don't match in j.
    # Strassen formula: M5 = (A11 + A12) * B22.
    # A11 is top-left 1x1 (i=0,j=0). A12 is top-right 1x1 (i=0,j=1).
    # B22 is bottom-right 1x1 (j=1,k=1).
    # Product (A11 + A12) * B22 is not a valid matrix multiplication term because indices don't match for the sum.
    # Wait, Strassen computes C = AB.
    # M5 = (A11 + A12) B22.
    # A11 is 1x1, A12 is 1x1. Sum is 1x1.
    # B22 is 1x1.
    # The result is 1x1.
    # How does it contribute to C?
    # C12 = A11 B12 + A12 B22.
    # C22 = A21 B12 + A22 B22.
    # Strassen:
    # M5 = (A11 + A12) B22.
    # This term contributes to C12?
    # Let's check the standard Strassen reconstruction.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's verify M5 contribution to C12.
    # C12 = A11 B12 + A12 B22.
    # M5 = A11 B22 + A12 B22.
    # M3 = A11 B12 - A11 B22.
    # M3 + M5 = A11 B12 - A11 B22 + A11 B22 + A12 B22 = A11 B12 + A12 B22. Correct.
    
    # So we need to expand each M term into elementary (i,j,k) terms.
    # M1: (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This is a product of two scalars (sums).
    # It generates terms for C11.
    # The "multiplication" is just 1 scalar multiplication.
    # But in the tensor decomposition, we decompose the scalar multiplication into elementary terms?
    # No, the rank is the number of scalar multiplications.
    # So M1 is 1 rank unit.
    # But M1 itself is a sum of products of A and B elements.
    # To implement this in the U, V, W format, we need to represent M1 as a sum of elementary terms?
    # No. The format is:
    # C[i,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] ... wait.
    # The standard decomposition is:
    # C[i,k] = sum_r sum_j U[i,j,r] V[j,k,r] ... No.
    # The tensor is T[i,j,k].
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]? No.
    # The definition is:
    # T = sum_r u_r \otimes v_r \otimes w_r
    # where u_r is a vector of size n*m, v_r is size m*p, w_r is size p*n.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # This allows U, V, W to have arbitrary values.
    # If we use Strassen, we are effectively changing the basis of the multiplications.
    # The "rank" is the number of terms r.
    # Each term r corresponds to a multiplication of a linear combination of A entries and a linear combination of B entries.
    # Let $L_A^{(r)} = \sum_{i,j} U_{i,j}^{(r)} A_{i,j}$
    # Let $L_B^{(r)} = \sum_{j,k} V_{j,k}^{(r)} B_{j,k}$
    # Then the term contributes $L_A^{(r)} L_B^{(r)}$ to the product.
    # The contribution to $C_{i,k}$ is $\sum_r W_{k,i}^{(r)} L_A^{(r)} L_B^{(r)}$.
    # We need $\sum_r W_{k,i}^{(r)} L_A^{(r)} L_B^{(r)} = \sum_j A_{i,j} B_{j,k}$.
    
    # So for Strassen, we define the linear forms.
    # M1: $L_A = A_{0,0} + A_{1,1}$, $L_B = B_{0,0} + B_{1,1}$.
    # This term contributes to C11.
    # We need to distribute this contribution to C[0,0], C[0,1], C[1,0], C[1,1] using W coefficients.
    # Actually, the W matrix determines how the scalar product $L_A L_B$ is added to C.
    # If we have a term r with $L_A, L_B$, we can set $W_{k,i}^{(r)}$ to add it to $C_{i,k}$.
    # But one term r can only add to one specific $C_{i,k}$?
    # No, $W$ is a vector of size $p \times n$.
    # So $W_{k,i}^{(r)}$ can be non-zero for multiple $(k,i)$.
    # However, the product $L_A L_B$ is a single scalar.
    # So term r adds $L_A^{(r)} L_B^{(r)} W_{k,i}^{(r)}$ to $C_{i,k}$.
    # This means we can use the same multiplication result to update multiple entries of C.
    
    # So for M1:
    # $L_A = A_{0,0} + A_{1,1}$
    # $L_B = B_{0,0} + B_{1,1}$
    # We need to add this to C11 components.
    # From Strassen:
    # C[0,0] gets +M1
    # C[1,1] gets +M1
    # C[0,1] gets 0
    # C[1,0] gets 0
    # Wait, let's check the reconstruction again.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So M1 contributes to C[0,0] (coeff 1) and C[1,1] (coeff 1).
    # So for term r1 corresponding to M1:
    # U: U[0,0]=1, U[1,1]=1, others 0.
    # V: V[0,0]=1, V[1,1]=1, others 0.
    # W: W[0,0]=1 (for C[0,0]), W[1,1]=1 (for C[1,1]), others 0.
    # Note: W index is k*n+i.
    # C[0,0] -> k=0, i=0 -> W[0,0]
    # C[1,1] -> k=1, i=1 -> W[1,1]
    
    # This works!
    
    # We will implement this logic.
    
    def add_strassen_term(L_A_coeffs, L_B_coeffs, W_coeffs):
        # L_A_coeffs: dict (i,j) -> coeff
        # L_B_coeffs: dict (j,k) -> coeff
        # W_coeffs: dict (k,i) -> coeff
        
        u_vec = [0] * (n*m)
        v_vec = [0] * (m*p)
        w_vec = [0] * (p*n)
        
        for (i, j), c in L_A_coeffs.items():
            u_vec[i*m + j] = c
            
        for (j, k), c in L_B_coeffs.items():
            v_vec[j*p + k] = c
            
        for (k, i), c in W_coeffs.items():
            w_vec[k*n + i] = c
            
        terms.append((u_vec, v_vec, w_vec))

    # Strassen Terms for A[0:2,0:2] * B[0:2,0:2]
    
    # M1
    add_strassen_term(
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    
    # M2: (A[1,0] + A[1,1]) * B[0,0]
    # Contributes to C[1,0] (coeff 1) and C[1,1] (coeff -1)
    # C21 = M2 + M4 -> C[1,0] gets M2
    # C22 = M1 - M2 + M3 + M6 -> C[1,1] gets -M2
    add_strassen_term(
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1}
    )
    
    # M3: A[0,0] * (B[0,1] - B[1,1])
    # Contributes to C[0,1] (coeff 1) and C[1,1] (coeff 1)
    # C12 = M3 + M5 -> C[0,1] gets M3
    # C22 = M1 - M2 + M3 + M6 -> C[1,1] gets M3
    add_strassen_term(
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1}
    )
    
    # M4: A[1,1] * (B[1,0] - B[0,0])
    # Contributes to C[0,0] (coeff 1) and C[1,0] (coeff 1)
    # C11 = M1 + M4 - M5 + M7 -> C[0,0] gets M4
    # C21 = M2 + M4 -> C[1,0] gets M4
    add_strassen_term(
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1}
    )
    
    # M5: (A[0,0] + A[0,1]) * B[1,1]
    # Contributes to C[0,1] (coeff 1) and C[0,0] (coeff -1)
    # C12 = M3 + M5 -> C[0,1] gets M5
    # C11 = M1 + M4 - M5 + M7 -> C[0,0] gets -M5
    add_strassen_term(
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(1,0): 1, (0,0): -1}
    )
    
    # M6: (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
    # Contributes to C[1,1] (coeff 1)
    # C22 = M1 - M2 + M3 + M6 -> C[1,1] gets M6
    add_strassen_term(
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    )
    
    # M7: (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
    # Contributes to C[0,0] (coeff 1)
    # C11 = M1 + M4 - M5 + M7 -> C[0,0] gets M7
    add_strassen_term(
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    )
    
    # Now add the remaining terms using schoolbook
    # We need to add terms for:
    # A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 2:5] -> C[0:2, 2:5]
    # A[0:2, 2] * B[2, 2:5] -> C[0:2, 2:5]
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # A[2, 2] * B[2, 0:2] -> C[2, 0:2]
    # A[2, 0:2] * B[0:2, 2:5] -> C[2, 2:5]
    # A[2, 2] * B[2, 2:5] -> C[2, 2:5]
    
    def add_schoolbook_block(i_range, j_range, k_range):
        for i in range(i_range[0], i_range[1]):
            for j in range(j_range[0], j_range[1]):
                for k in range(k_range[0], k_range[1]):
                    add_term_simple(i, j, k, 1)
                    
    # Note: The Strassen part already covered A[0:2,0:2] * B[0:2,0:2] contributing to C[0:2,0:2].
    # So we must NOT add schoolbook terms for this block.
    
    # 1. A[0:2, 2] * B[2, 0:2]
    # i in 0,1; j=2; k in 0,1
    add_schoolbook_block((0,2), (2,3), (0,2))
    
    # 2. A[0:2, 0:2] * B[0:2, 2:5]
    # i in 0,1; j in 0,1; k in 2,3,4
    add_schoolbook_block((0,2), (0,2), (2,5))
    
    # 3. A[0:2, 2] * B[2, 2:5]
    # i in 0,1; j=2; k in 2,3,4
    add_schoolbook_block((0,2), (2,3), (2,5))
    
    # 4. A[2, 0:2] * B[0:2, 0:2]
    # i=2; j in 0,1; k in 0,1
    add_schoolbook_block((2,3), (0,2), (0,2))
    
    # 5. A[2, 2] * B[2, 0:2]
    # i=2; j=2; k in 0,1
    add_schoolbook_block((2,3), (2,3), (0,2))
    
    # 6. A[2, 0:2] * B[0:2, 2:5]
    # i=2; j in 0,1; k in 2,3,4
    add_schoolbook_block((2,3), (0,2), (2,5))
    
    # 7. A[2, 2] * B[2, 2:5]
    # i=2; j=2; k in 2,3,4
    add_schoolbook_block((2,3), (2,3), (2,5))
    
    # Convert terms to U, V, W
    rank = len(terms)
    U = [[0] * rank for _ in range(n*m)]
    V = [[0] * rank for _ in range(m*p)]
    W = [[0] * rank for _ in range(p*n)]
    
    for r, (u_vec, v_vec, w_vec) in enumerate(terms):
        for i in range(n*m):
            if u_vec[i] != 0:
                U[i][r] = u_vec[i]
        for j in range(m*p):
            if v_vec[j] != 0:
                V[j][r] = v_vec[j]
        for k in range(p*n):
            if w_vec[k] != 0:
                W[k][r] = w_vec[k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
