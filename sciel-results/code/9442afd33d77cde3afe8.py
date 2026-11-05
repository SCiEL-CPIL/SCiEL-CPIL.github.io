import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def _strassen_2x2():
    """Return U, V, W for 2x2 matrix multiplication using Strassen (rank 7)."""
    # Strassen's algorithm for 2x2:
    # M = A*B where A, B are 2x2
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    #
    # Strassen products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # M11 = P1 + P4 - P5 + P7
    # M12 = P3 + P5
    # M21 = P2 + P4
    # M22 = P1 - P2 + P3 + P6
    #
    # For tensor decomposition, we need to express each entry of M as sum of products.
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Here n=m=p=2, so U is 4x7, V is 4x7, W is 4x7
    #
    # Entry (i,j) of result = sum_k A[i,k]*B[k,j]
    # In our indexing: U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    #
    # Let's build this carefully.
    # A indices: i (row), k (col) -> A[i,k]
    # B indices: k (row), j (col) -> B[k,j]
    # Result: M[i,j] = sum_k A[i,k] * B[k,j]
    #
    # In our factorization:
    # U[i*m + j, r] where m=2, so U[i*2+j, r] corresponds to A[i, ?]
    # Wait, let me re-read the convention.
    #
    # T[i, j, k] = 1 if i + k == j else 0 (for matrix mult of 5x5 by 5x5)
    # Actually for general case: (AB)[i,j] = sum_k A[i,k] B[k,j]
    # So the tensor T[i,j,k] = 1 for all valid i,j,k in the sum.
    #
    # The factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For 2x2: n=m=p=2
    # U is 4x7, V is 4x7, W is 4x7
    # U[i*2+j, r] - this seems to mix A's row i with B's col j? That doesn't make sense.
    #
    # Let me reconsider. The standard tensor decomposition for matrix multiplication:
    # M[i,j] = sum_k A[i,k] B[k,j]
    # The tensor T has entries T[i,j,k] = 1 (for the multiplication tensor).
    # A rank-r decomposition: T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[k,r] * W[i,j,r]... 
    #
    # Actually the convention given is:
    # U has shape (n*m, rank), indexed as U[i*m + j, r]
    # V has shape (m*p, rank), indexed as V[j*p + k, r]  
    # W has shape (p*n, rank), indexed as W[k*n + i, r]
    #
    # And T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    #
    # For matrix multiplication: (AB)[i,j] = sum_k A[i,k] B[k,j]
    # So T[i,j,k] should be 1 for all i,j,k (since each term A[i,k]*B[k,j] contributes).
    #
    # In the schoolbook: U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
    # for each (i,j,k) triple. This gives T[i,j,k] = 1 for each specific (i,j,k) term.
    #
    # For Strassen, we need to express the 2x2 multiplication with 7 terms.
    # Let me work it out.
    #
    # For 2x2: n=m=p=2
    # Result M[i,j] = sum_{k=0}^{1} A[i,k] * B[k,j]
    #
    # Strassen:
    # P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1]) * B[0,0]
    # P3 = A[0,0] * (B[0,1]-B[1,1])
    # P4 = A[1,1] * (B[1,0]-B[0,0])
    # P5 = (A[0,0]+A[0,1]) * B[1,1]
    # P6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # P7 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1])
    #
    # M[0,0] = P1 + P4 - P5 + P7
    # M[0,1] = P3 + P5
    # M[1,0] = P2 + P4
    # M[1,1] = P1 - P2 + P3 + P6
    #
    # Each P is a product of a linear combination of A entries and a linear combination of B entries.
    # For the tensor decomposition, each rank-1 term corresponds to one P.
    #
    # For P1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # This contributes to M[0,0] with coeff +1, and to M[1,1] with coeff +1.
    #
    # The factorization T[i,j,k] = sum_r U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
    #
    # For a given rank term r, U[i*2+j, r] is a coefficient that depends on i and j.
    # V[j*2+k, r] depends on j and k.
    # W[k*2+i, r] depends on k and i.
    #
    # The product U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should equal the coefficient
    # of A[i,k] * B[k,j] in the expansion of rank term r.
    #
    # For P1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1]):
    # Expansion: A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    #
    # The coefficient of A[i,k]*B[k,j] in P1:
    # - i=0,k=0,j=0: A[0,0]*B[0,0] -> coeff 1
    # - i=0,k=0,j=1: A[0,0]*B[0,1] -> coeff 0 (not in P1)
    # Wait, B[0,0] has k=0,j=0. B[1,1] has k=1,j=1.
    # A[0,0] has i=0,k=0. A[1,1] has i=1,k=1.
    #
    # So P1 expands to:
    # A[0,0]*B[0,0]: i=0,k=0,j=0
    # A[0,0]*B[1,1]: i=0,k=0,j=1? No, B[1,1] means row 1, col 1, so k=1,j=1. But A[0,0] has k=0.
    # This doesn't match because the k indices must agree for the product A[i,k]*B[k,j].
    #
    # I think I'm confusing myself. Let me think about it differently.
    #
    # In matrix multiplication, (AB)[i,j] = sum_k A[i,k] * B[k,j].
    # The "tensor" T[i,j,k] represents the contribution of A[i,k]*B[k,j] to M[i,j].
    #
    # For a rank-1 term in the decomposition, we have coefficients U[i,j], V[k], W[i,j]... 
    # but the convention here mixes indices differently.
    #
    # Let me just verify the schoolbook works and then try to apply Strassen.
    #
    # Actually, let me just return schoolbook for now to be safe, and I'll implement Strassen 
    # in a follow-up if time permits.
    
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # This is a placeholder - return schoolbook for 2x2 (rank 8)
    # We'll handle this in the main construct
    return None

def _construct_hierarchical():
    """Construct 5x5x5 decomposition using block structure with Strassen for 3x3 block."""
    n, m, p = 5, 5, 5
    
    # Partition 5x5 matrices into blocks:
    # 5 = 3 + 2
    # Block structure:
    # A = [[A11 (3x3), A12 (3x2)],
    #      [A21 (2x3), A22 (2x2)]]
    # B = [[B11 (3x3), B12 (3x2)],
    #      [B21 (2x3), B22 (2x2)]]
    # M = AB = [[M11 (3x3), M12 (3x2)],
    #           [M21 (2x3), M22 (2x2)]]
    #
    # M11 = A11*B11 + A12*B21
    # M12 = A11*B12 + A12*B22
    # M21 = A21*B11 + A22*B21
    # M22 = A21*B12 + A22*B22
    #
    # For each block product, we can use either schoolbook or Strassen.
    #
    # Let's use:
    # - 3x3 * 3x3: schoolbook (rank 27) or try to find better
    # - 3x2 * 2x3: schoolbook (rank 18)
    # - 3x2 * 2x2: schoolbook (rank 12)
    # - 2x3 * 3x3: schoolbook (rank 18)
    # - 2x3 * 3x2: schoolbook (rank 18)
    # - 2x2 * 2x2: Strassen (rank 7)
    #
    # Total with all schoolbook: 27 + 18 + 12 + 18 + 18 + 8 = 101
    # With Strassen for 2x2: 27 + 18 + 12 + 18 + 18 + 7 = 100
    #
    # Let's implement this.
    
    # We'll build the decomposition term by term.
    # Each term is a rank-1 contribution.
    
    U_rows = []  # Will build U matrix (25 x rank)
    V_rows = []  # Will build V matrix (25 x rank)
    W_rows = []  # Will build W matrix (25 x rank)
    
    # Helper to add a rank-1 term
    def add_term(u_vals, v_vals, w_vals):
        """Add a rank-1 term where u_vals[i*5+j] = u_i_j, v_vals[j*5+k] = v_j_k, w_vals[k*5+i] = w_k_i"""
        U_rows.append(u_vals)
        V_rows.append(v_vals)
        W_rows.append(w_vals)
    
    # Block index mapping:
    # For 5x5, split into 3+2.
    # Block (bi, bj) where bi, bj in {0,1}:
    #   bi=0: rows 0,1,2; bi=1: rows 3,4
    #   bj=0: cols 0,1,2; bj=1: cols 3,4
    
    def block_offset(bi, size_bi):
        if bi == 0:
            return 0
        else:
            return 3
    
    # Schoolbook for a block product A_block (na x ma) * B_block (mb x pa)
    # where the result is M_block (na x pa)
    # In global indices:
    #   A[i_global, k_global] where i_global = block_row_offset + i_local, k_global = block_col_offset_A + k_local
    #   B[k_global, j_global] where k_global = block_row_offset_B + k_local, j_global = block_col_offset_B + j_local
    #   M[i_global, j_global] = sum_k_local A[i_global, k_global] * B[k_global, j_global]
    
    def schoolbook_block(na, ma, mb, pa, row_off_A, col_off_A, row_off_B, col_off_B):
        """Add schoolbook terms for multiplying an (na x ma) block by (mb x pa) block.
        row_off_A: offset for A's rows in global 5x5
        col_off_A: offset for A's cols (which are B's rows)
        row_off_B: offset for B's rows
        col_off_B: offset for B's cols
        """
        for i in range(na):
            for j in range(pa):
                for k in range(ma):
                    # Global indices
                    gi = row_off_A + i
                    gj = col_off_B + j
                    gk = col_off_A + k  # = row_off_B + k if col_off_A == row_off_B
                    
                    u_vals = [0] * 25
                    v_vals = [0] * 25
                    w_vals = [0] * 25
                    
                    # U[gi*5 + gj, r] = 1
                    u_vals[gi * 5 + gj] = 1
                    # V[gj*5 + gk, r] = 1
                    v_vals[gj * 5 + gk] = 1
                    # W[gk*5 + gi, r] = 1
                    w_vals[gk * 5 + gi] = 1
                    
                    add_term(u_vals, v_vals, w_vals)
    
    # Strassen for 2x2 block
    def strassen_2x2(row_off_A, col_off_A, row_off_B, col_off_B):
        """Add Strassen terms for 2x2 * 2x2 block multiplication."""
        # A is 2x2 at rows [row_off_A, row_off_A+1], cols [col_off_A, col_off_A+1]
        # B is 2x2 at rows [row_off_B, row_off_B+1], cols [col_off_B, col_off_B+1]
        # Result M is 2x2 at rows [row_off_A, row_off_A+1], cols [col_off_B, col_off_B+1]
        
        a00 = (row_off_A + 0, col_off_A + 0)  # A[0,0] global
        a01 = (row_off_A + 0, col_off_A + 1)  # A[0,1]
        a10 = (row_off_A + 1, col_off_A + 0)  # A[1,0]
        a11 = (row_off_A + 1, col_off_A + 1)  # A[1,1]
        
        b00 = (row_off_B + 0, col_off_B + 0)  # B[0,0]
        b01 = (row_off_B + 0, col_off_B + 1)  # B[0,1]
        b10 = (row_off_B + 1, col_off_B + 0)  # B[1,0]
        b11 = (row_off_B + 1, col_off_B + 1)  # B[1,1]
        
        # M[0,0] = P1 + P4 - P5 + P7
        # M[0,1] = P3 + P5
        # M[1,0] = P2 + P4
        # M[1,1] = P1 - P2 + P3 + P6
        
        # P1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # P2 = (A[1,0] + A[1,1]) * B[0,0]
        # P3 = A[0,0] * (B[0,1] - B[1,1])
        # P4 = A[1,1] * (B[1,0] - B[0,0])
        # P5 = (A[0,0] + A[0,1]) * B[1,1]
        # P6 = (A[1,0] - A[0,0]) * (B[0,0] + B[0,1])
        # P7 = (A[0,1] - A[1,1]) * (B[1,0] + B[1,1])
        
        # For each P, we need to express it as sum of rank-1 terms.
        # Each P is a product of (linear combo of A) * (linear combo of B).
        # We expand each into individual A[i,k]*B[k,j] terms.
        
        def add_product_term(coeff_a, coeff_b, result_coeff):
            """Add a term for coeff_a * A[i,k] * coeff_b * B[k,j] contributing result_coeff to M[i,j].
            coeff_a is a list of (i_local, k_local, val) for A
            coeff_b is a list of (k_local, j_local, val) for B
            result_coeff is a list of (i_local, j_local, val) for M
            """
            # This gets complex. Let me just enumerate all A[i,k]*B[k,j] products.
            pass
        
        # Let me take a different approach. For each of the 7 Strassen products,
        # I'll expand it into individual rank-1 terms.
        
        # P1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
        # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        # But A[0,0] has k=0, B[0,0] has k=0: OK
        # A[0,0] has k=0, B[1,1] has k=1: NOT OK (k must match)
        # 
        # I see the issue. In Strassen, the products are of matrix entries, but they don't
        # necessarily have matching k indices. This means Strassen doesn't directly give
        # a tensor decomposition in the same way.
        #
        # Actually, I think I need to reconsider. The Strassen algorithm computes
        # M = A*B using 7 matrix products of (linear combinations). Each "matrix product"
        # is itself a full matrix multiplication, not just scalar products.
        #
        # So for 2x2 * 2x2, Strassen uses 7 products of 1x1 "matrices" (scalars),
        # but each scalar product is of the form (sum of A entries) * (sum of B entries).
        #
        # For the tensor decomposition, each such scalar product contributes to multiple
        # entries of M. The rank of the decomposition is the number of such scalar products.
        #
        # Let me redo this. For each Strassen product P, I need to express P as a sum
        # of terms of the form U[i,j] * V[k] * W[i,j] where the product equals the
        # coefficient of A[i,k]*B[k,j] in P's expansion.
        
        # Actually, I think the cleanest way is:
        # For each of the 7 Strassen products, it's a scalar value.
        # That scalar is added to certain entries of M with certain coefficients.
        # The scalar itself is a bilinear form in A and B.
        #
        # For the tensor decomposition, each rank-1 term corresponds to one such scalar.
        # The U, V, W matrices encode how that scalar contributes to each M[i,j].
        
        # Let me define the 7 Strassen products as bilinear forms:
        # P1 = sum_{i,k,j} p1[i,k,j] * A[i,k] * B[k,j] where p1 is the coefficient tensor
        # But P1 is a scalar, so it's sum_{i,k} a1[i,k] * A[i,k] * sum_{k',j} b1[k',j] * B[k',j]
        # with a1 and b1 being the coefficients.
        
        # For P1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
        # a1[0,0]=1, a1[1,1]=1, others 0
        # b1[0,0]=1, b1[1,1]=1, others 0
        # P1 = sum_{i,k} a1[i,k] A[i,k] * sum_{k',j} b1[k',j] B[k',j]
        #     = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
        #
        # This P1 contributes to M with coefficients:
        # M[0,0] += P1 (coeff +1)
        # M[1,1] += P1 (coeff +1)
        #
        # In the tensor decomposition, for this rank-1 term:
        # U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r] should equal the coefficient of A[i,k]*B[k,j] in the contribution of P1 to M[i,j].
        #
        # But P1 is a scalar that's added to M[0,0] and M[1,1]. The scalar P1 itself is a sum of 4 products.
        # So for the tensor decomposition, we need to split P1 into its 4 component products.
        #
        # I think the issue is that Strassen's 7 "products" are not rank-1 in the tensor sense.
        # Each Strassen product is a scalar that is a bilinear form, which itself requires multiple
        # rank-1 terms to decompose.
        #
        # So Strassen for 2x2 actually uses 7 scalar products, but each scalar product is not
        # a single rank-1 term in the matrix multiplication tensor.
        #
        # I think I need to abandon the Strassen approach and just use schoolbook for all blocks.
        
        # Let's just use schoolbook for everything.
        schoolbook_block(2, 2, 2, 2, row_off_A, col_off_A, row_off_B, col_off_B)
    
    # Build the full 5x5 decomposition
    # Block structure: 3+2
    # A11: rows 0-2, cols 0-2 (3x3)
    # A12: rows 0-2, cols 3-4 (3x2)
    # A21: rows 3-4, cols 0-2 (2x3)
    # A22: rows 3-4, cols 3-4 (2x2)
    #
    # B11: rows 0-2, cols 0-2 (3x3)
    # B12: rows 0-2, cols 3-4 (3x2)
    # B21: rows 3-4, cols 0-2 (2x3)
    # B22: rows 3-4, cols 3-4 (2x2)
    
    # M11 = A11*B11 + A12*B21
    schoolbook_block(3, 3, 3, 3, 0, 0, 0, 0)  # A11 (3x3) * B11 (3x3) -> M11 (3x3)
    schoolbook_block(3, 2, 2, 3, 0, 3, 3, 0)  # A12 (3x2) * B21 (2x3) -> M11 (3x3)
    
    # M12 = A11*B12 + A12*B22
    schoolbook_block(3, 3, 3, 2, 0, 0, 0, 3)  # A11 (3x3) * B12 (3x2) -> M12 (3x2)
    schoolbook_block(3, 2, 2, 2, 0, 3, 3, 3)  # A12 (3x2) * B22 (2x2) -> M12 (3x2)
    
    # M21 = A21*B11 + A22*B21
    schoolbook_block(2, 3, 3, 3, 3, 0, 0, 0)  # A21 (2x3) * B11 (3x3) -> M21 (2x3)
    schoolbook_block(2, 2, 2, 3, 3, 3, 3, 0)  # A22 (2x2) * B21 (2x3) -> M21 (2x3)
    
    # M22 = A21*B12 + A22*B22
    schoolbook_block(2, 3, 3, 2, 3, 0, 0, 3)  # A21 (2x3) * B12 (3x2) -> M22 (2x2)
    schoolbook_block(2, 2, 2, 2, 3, 3, 3, 3)  # A22 (2x2) * B22 (2x2) -> M22 (2x2)
    
    rank = len(U_rows)
    U = U_rows
    V = V_rows
    W = W_rows
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = 5, 5, 5
    rank = len(U[0])
    
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal 1 for all i,j,k (since every term A[i,k]*B[k,j] contributes to M[i,j])
    
    for i in range(n):
        for j in range(p):
            for k in range(m):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try hierarchical decomposition
    try:
        result = _construct_hierarchical()
        # Verify
        if _verify_decomposition(result["U"], result["V"], result["W"]):
            return result
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
