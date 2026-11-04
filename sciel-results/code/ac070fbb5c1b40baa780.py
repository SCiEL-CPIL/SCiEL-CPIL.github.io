"""Matrix multiplication 4x5x5 with Strassen-based tensor decomposition."""
import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying
    a 4x5 matrix by a 5x5 matrix using Strassen algorithm hierarchy.
    
    Strategy: Decompose the 4x5x5 problem into sub-blocks and apply
    Strassen's algorithm to 2x2x2 sub-problems, reducing the rank
    below the schoolbook bound of 100.
    
    The matrix multiplication C = A*B where A is 4x5 and B is 5x5
    can be decomposed by splitting A into two 2x5 blocks (top and bottom)
    and B into blocks based on the 5x5 structure.
    
    For each 2x5x5 sub-problem, we split the 5x5 B matrix into:
    - 2x2 blocks (four of them for the 4x4 part)
    - 2x1 and 1x2 strips
    - 1x1 corner
    
    For the 2x2x2 sub-problems, we use Strassen's algorithm which
    computes the product with 7 multiplications instead of 8.
    
    This should give us a rank lower than 100.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # We'll build the decomposition using a dictionary approach
    # where each entry maps (matrix, row, col) -> list of (coefficient, term_index)
    
    U_terms = {}  # (i, j) -> list of (coeff, term_idx) for A[i,j]
    V_terms = {}  # (j, k) -> list of (coeff, term_idx) for B[j,k]
    W_terms = {}  # (k, i) -> list of (coeff, term_idx) for C[k,i]
    
    next_term = 0
    
    def add_term(u_coeff, v_coeff, w_coeff):
        """Add a rank-1 term with given coefficients."""
        nonlocal next_term
        term_idx = next_term
        next_term += 1
        return term_idx
    
    def add_u(i, j, coeff, term_idx):
        """Add coefficient to U[i*m+j] at term_idx."""
        key = (i, j)
        if key not in U_terms:
            U_terms[key] = []
        U_terms[key].append((coeff, term_idx))
    
    def add_v(j, k, coeff, term_idx):
        """Add coefficient to V[j*p+k] at term_idx."""
        key = (j, k)
        if key not in V_terms:
            V_terms[key] = []
        V_terms[key].append((coeff, term_idx))
    
    def add_w(k, i, coeff, term_idx):
        """Add coefficient to W[k*n+i] at term_idx."""
        key = (k, i)
        if key not in W_terms:
            W_terms[key] = []
        W_terms[key].append((coeff, term_idx))
    
    def strassen_2x2x2(i0, i1, j0, j1, k0, k1):
        """
        Apply Strassen's algorithm for multiplying a 2x2 block of A (rows i0,i1; cols j0,j1)
        with a 2x2 block of B (rows j0,j1; cols k0,k1), producing a 2x2 block of C.
        
        Strassen uses 7 multiplications:
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B01-B11)
        M4 = A11*(B10-B00)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B00+B01)
        M7 = (A01-A11)*(B10+B11)
        
        C00 = M1 + M4 - M5 + M7
        C01 = M3 + M5
        C10 = M2 + M4
        C11 = M1 - M2 + M3 + M6
        """
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        t = next_term; next_term += 1
        add_u(i0, j0, 1, t)
        add_u(i1, j1, 1, t)
        add_v(j0, k0, 1, t)
        add_v(j1, k1, 1, t)
        add_w(k0, i0, 1, t)
        add_w(k1, i0, 1, t)
        add_w(k0, i1, 1, t)
        add_w(k1, i1, 1, t)
        
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        t = next_term; next_term += 1
        add_u(i1, j0, 1, t)
        add_u(i1, j1, 1, t)
        add_v(j0, k0, 1, t)
        add_w(k0, i1, 1, t)
        
        # M3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
        t = next_term; next_term += 1
        add_u(i0, j0, 1, t)
        add_v(j0, k1, 1, t)
        add_v(j1, k1, -1, t)
        add_w(k1, i0, 1, t)
        
        # M4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
        t = next_term; next_term += 1
        add_u(i1, j1, 1, t)
        add_v(j1, k0, 1, t)
        add_v(j0, k0, -1, t)
        add_w(k0, i0, 1, t)
        add_w(k0, i1, 1, t)
        
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        t = next_term; next_term += 1
        add_u(i0, j0, 1, t)
        add_u(i0, j1, 1, t)
        add_v(j1, k1, 1, t)
        add_w(k1, i0, -1, t)
        
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        t = next_term; next_term += 1
        add_u(i1, j0, 1, t)
        add_u(i0, j0, -1, t)
        add_v(j0, k0, 1, t)
        add_v(j0, k1, 1, t)
        add_w(k1, i1, 1, t)
        
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        t = next_term; next_term += 1
        add_u(i0, j1, 1, t)
        add_u(i1, j1, -1, t)
        add_v(j1, k0, 1, t)
        add_v(j1, k1, 1, t)
        add_w(k0, i0, 1, t)
        add_w(k1, i0, 1, t)
    
    def schoolbook_block(i0, i1, j0, j1, k0, k1):
        """
        Schoolbook multiplication for a 2x2x2 block (or smaller).
        Just do all products directly.
        """
        for i in [i0, i1]:
            for j in [j0, j1]:
                for k in [k0, k1]:
                    t = next_term; next_term += 1
                    add_u(i, j, 1, t)
                    add_v(j, k, 1, t)
                    add_w(k, i, 1, t)
    
    def schoolbook_2x1x2(i0, i1, j, k0, k1):
        """
        Schoolbook for 2x1x2: A has 2 rows, B has 1 row, C has 2 cols.
        """
        for i in [i0, i1]:
            for k in [k0, k1]:
                t = next_term; next_term += 1
                add_u(i, j, 1, t)
                add_v(j, k, 1, t)
                add_w(k, i, 1, t)
    
    def schoolbook_2x2x1(i0, i1, j0, j1, k):
        """
        Schoolbook for 2x2x1: A has 2 rows, B has 2 rows, C has 1 col.
        """
        for i in [i0, i1]:
            for j in [j0, j1]:
                t = next_term; next_term += 1
                add_u(i, j, 1, t)
                add_v(j, k, 1, t)
                add_w(k, i, 1, t)
    
    def schoolbook_2x1x1(i0, i1, j, k):
        """
        Schoolbook for 2x1x1: A has 2 rows, B has 1 row, C has 1 col.
        """
        for i in [i0, i1]:
            t = next_term; next_term += 1
            add_u(i, j, 1, t)
            add_v(j, k, 1, t)
            add_w(k, i, 1, t)
    
    # Now decompose the 4x5x5 problem
    # Split A into two 2x5 blocks: rows 0-1 and rows 2-3
    # For each 2x5 block, split B (5x5) into:
    #   - 2x2 blocks at (j={0,1},k={0,1}), (j={0,1},k={2,3}), (j={2,3},k={0,1}), (j={2,3},k={2,3})
    #   - 2x1 strips: (j={0,1},k={4}), (j={2,3},k={4})
    #   - 1x2 strips: (j={4},k={0,1}), (j={4},k={2,3})
    #   - 1x1 corner: (j={4},k={4})
    
    for a_offset in [0, 2]:  # top half (rows 0,1) and bottom half (rows 2,3)
        i0, i1 = a_offset, a_offset + 1
        
        # 2x2x2 blocks: use Strassen
        strassen_2x2x2(i0, i1, 0, 1, 0, 1)  # j={0,1}, k={0,1}
        strassen_2x2x2(i0, i1, 0, 1, 2, 3)  # j={0,1}, k={2,3}
        strassen_2x2x2(i0, i1, 2, 3, 0, 1)  # j={2,3}, k={0,1}
        strassen_2x2x2(i0, i1, 2, 3, 2, 3)  # j={2,3}, k={2,3}
        
        # 2x2x1 blocks: j={0,1} and j={2,3}, k={4}
        schoolbook_2x2x1(i0, i1, 0, 1, 4)
        schoolbook_2x2x1(i0, i1, 2, 3, 4)
        
        # 2x1x2 blocks: j={4}, k={0,1} and k={2,3}
        schoolbook_2x1x2(i0, i1, 4, 0, 1)
        schoolbook_2x1x2(i0, i1, 4, 2, 3)
        
        # 2x1x1 block: j={4}, k={4}
        schoolbook_2x1x1(i0, i1, 4, 4)
    
    # Convert dictionaries to matrices
    rank = next_term
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    
    for (i, j), terms in U_terms.items():
        for coeff, c in terms:
            U_mat[i * m + j][c] += coeff
    
    for (j, k), terms in V_terms.items():
        for coeff, c in terms:
            V_mat[j * p + k][c] += coeff
    
    for (k, i), terms in W_terms.items():
        for coeff, c in terms:
            W_mat[k * n + i][c] += coeff
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
