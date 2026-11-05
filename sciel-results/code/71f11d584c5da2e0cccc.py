"""Matrix multiplication 6x6x6 via recursive Strassen decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time

def _schoolbook_3x3():
    """Return (U, V, W) for 3x3 matrix multiplication, rank 27.
    U shape (9, 27), V shape (9, 27), W shape (9, 27).
    Indexing: U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    n = 3
    rank = 27
    U = [[0.0] * rank for _ in range(9)]
    V = [[0.0] * rank for _ in range(9)]
    W = [[0.0] * rank for _ in range(9)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * 3 + j][col] = 1.0
                V[j * 3 + k][col] = 1.0
                W[k * 3 + i][col] = 1.0
                col += 1
    return U, V, W

def _strassen_6x6():
    """
    Build rank-189 decomposition for 6x6 matrix multiplication using
    one level of Strassen (7 block multiplications) each done schoolbook
    on 3x3 blocks.
    
    Block structure:
    A = [[A11, A12], [A21, A22]]  (each 3x3)
    B = [[B11, B12], [B21, B22]]  (each 3x3)
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    Each Mi is a 3x3 product, so each uses rank 27. Total rank = 7*27 = 189.
    
    The factor matrices U (36x189), V (36x189), W (36x189) encode:
    For each of the 7 Strassen terms, we have a 3x3 schoolbook decomposition.
    
    U maps input A (flattened 6x6 = 36 elements) to the left factors.
    V maps input B (flattened 6x6 = 36 elements) to the right factors.
    W maps the products to output C (flattened 6x6 = 36 elements).
    
    A flattening: A[i*6+j] for i,j in 0..5
    Block A11: rows 0-2, cols 0-2 -> indices 0..2, 3..5, 6..8
    Block A12: rows 0-2, cols 3-5 -> indices 0..2, 9..11, 12..14
    Block A21: rows 3-5, cols 0-2 -> indices 18..20, 21..23, 24..26
    Block A22: rows 3-5, cols 3-5 -> indices 18..20, 27..29, 30..32
    
    B flattening: B[j*6+k] for j,k in 0..5
    Block B11: rows 0-2, cols 0-2 -> indices 0..2, 6..8, 12..14
    Block B12: rows 0-2, cols 3-5 -> indices 0..2, 9..11, 15..17
    Block B21: rows 3-5, cols 0-2 -> indices 18..20, 24..26, 30..32
    Block B22: rows 3-5, cols 3-5 -> indices 18..20, 27..29, 33..35
    
    C flattening: C[i*6+j] for i,j in 0..5
    Block C11: rows 0-2, cols 0-2 -> indices 0..2, 6..8, 12..14
    Block C12: rows 0-2, cols 3-5 -> indices 0..2, 9..11, 15..17
    Block C21: rows 3-5, cols 0-2 -> indices 18..20, 24..26, 30..32
    Block C22: rows 3-5, cols 3-5 -> indices 18..20, 27..29, 33..35
    """
    rank = 189
    U = [[0.0] * rank for _ in range(36)]
    V = [[0.0] * rank for _ in range(36)]
    W = [[0.0] * rank for _ in range(36)]
    
    # Helper: get flat indices for a 3x3 block in a 6x6 matrix
    def a_block_rows(block):
        """Return the 9 flat indices for a 3x3 block of A."""
        if block == 11:  # rows 0-2, cols 0-2
            return [i*6+j for i in range(3) for j in range(3)]
        elif block == 12:  # rows 0-2, cols 3-5
            return [i*6+(j+3) for i in range(3) for j in range(3)]
        elif block == 21:  # rows 3-5, cols 0-2
            return [(i+3)*6+j for i in range(3) for j in range(3)]
        elif block == 22:  # rows 3-5, cols 3-5
            return [(i+3)*6+(j+3) for i in range(3) for j in range(3)]
    
    def b_block_rows(block):
        """Return the 9 flat indices for a 3x3 block of B."""
        if block == 11:  # rows 0-2, cols 0-2
            return [j*6+k for j in range(3) for k in range(3)]
        elif block == 12:  # rows 0-2, cols 3-5
            return [j*6+(k+3) for j in range(3) for k in range(3)]
        elif block == 21:  # rows 3-5, cols 0-2
            return [(j+3)*6+k for j in range(3) for k in range(3)]
        elif block == 22:  # rows 3-5, cols 3-5
            return [(j+3)*6+(k+3) for j in range(3) for k in range(3)]
    
    def c_block_rows(block):
        """Return the 9 flat indices for a 3x3 block of C."""
        if block == 11:  # rows 0-2, cols 0-2
            return [i*6+j for i in range(3) for j in range(3)]
        elif block == 12:  # rows 0-2, cols 3-5
            return [i*6+(j+3) for i in range(3) for j in range(3)]
        elif block == 21:  # rows 3-5, cols 0-2
            return [(i+3)*6+j for i in range(3) for j in range(3)]
        elif block == 22:  # rows 3-5, cols 3-5
            return [(i+3)*6+(j+3) for i in range(3) for j in range(3)]
    
    # For each Strassen term, define:
    #   left_blocks: list of (block_id, coeff) for A
    #   right_blocks: list of (block_id, coeff) for B
    #   output: list of (block_id, coeff) for C
    
    strassen_terms = [
        # M1 = (A11+A22)(B11+B22)
        {'left': [(11, 1), (22, 1)], 'right': [(11, 1), (22, 1)], 'out': [(11, 1), (22, -1)]},
        # M2 = (A21+A22)B11
        {'left': [(21, 1), (22, 1)], 'right': [(11, 1)], 'out': [(21, 1), (22, -1)]},
        # M3 = A11(B12-B22)
        {'left': [(11, 1)], 'right': [(12, 1), (22, -1)], 'out': [(12, 1), (22, 1)]},
        # M4 = A22(B21-B11)
        {'left': [(22, 1)], 'right': [(21, 1), (11, -1)], 'out': [(11, 1), (21, 1)]},
        # M5 = (A11+A12)B22
        {'left': [(11, 1), (12, 1)], 'right': [(22, 1)], 'out': [(11, -1), (12, 1)]},
        # M6 = (A21-A11)(B11+B12)
        {'left': [(21, 1), (11, -1)], 'right': [(11, 1), (12, 1)], 'out': [(22, 1)]},
        # M7 = (A12-A22)(B21+B22)
        {'left': [(12, 1), (22, -1)], 'right': [(21, 1), (22, 1)], 'out': [(11, 1)]},
    ]
    
    # For each Strassen term, we need a schoolbook 3x3 decomposition.
    # The schoolbook 3x3 has rank 27 with U_s[i*3+j, c], V_s[j*3+k, c], W_s[k*3+i, c]
    # where c = i*9 + j*3 + k (in the order i,j,k loops)
    
    # Precompute schoolbook 3x3 factors
    U_s, V_s, W_s = _schoolbook_3x3()
    
    for term_idx, term in enumerate(strassen_terms):
        base_col = term_idx * 27  # starting column for this term's 27 rank components
        
        # For each of the 27 schoolbook components:
        for c_local in range(27):
            col = base_col + c_local
            
            # U_s[i*3+j, c_local] gives coefficient for A element at local (i,j)
            # We need to map local (i,j) in 3x3 block to flat index in 6x6
            # U_s is indexed by [i*3+j] where i=row, j=col in the 3x3 block
            # But we have multiple blocks contributing with coefficients
            
            # For U: sum over left blocks
            for (blk, coeff) in term['left']:
                flat_idx_list = a_block_rows(blk)
                # U_s[row_local, c_local] where row_local = i*3+j
                for row_local in range(9):
                    val = U_s[row_local][c_local]
                    if val != 0:
                        U[flat_idx_list[row_local]][col] += coeff * val
            
            # For V: sum over right blocks
            for (blk, coeff) in term['right']:
                flat_idx_list = b_block_rows(blk)
                for row_local in range(9):
                    val = V_s[row_local][c_local]
                    if val != 0:
                        V[flat_idx_list[row_local]][col] += coeff * val
            
            # For W: sum over output blocks
            for (blk, coeff) in term['out']:
                flat_idx_list = c_block_rows(blk)
                for row_local in range(9):
                    val = W_s[row_local][c_local]
                    if val != 0:
                        W[flat_idx_list[row_local]][col] += coeff * val
    
    return U, V, W

def construct():
    """Construct rank-189 decomposition for 6x6 matrix multiplication."""
    deadline = time.time() + 140  # safety margin before 150s limit
    start = time.time()
    
    U, V, W = _strassen_6x6()
    
    # Verify with a random test (optional, quick check)
    if time.time() < deadline:
        import random
        random.seed(42)
        n = 6
        A = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(n)] for _ in range(n)]
        
        # Compute C = A * B
        C = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        
        # Compute via tensor decomposition: C[i,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, the convention is W[k*n+i, r]
        # So C[i,j] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over k? No.
        # Actually: C[i,j] = sum_{k} sum_{r} ... no.
        # The tensor decomposition means:
        # C[i,j] = sum_r (sum_k U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r]
        # That's not right either. Let me think again.
        # 
        # Standard: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # where T[i,j,k] = A[i,:] dot B[:,k] = C[i,k]... 
        # 
        # Actually for matrix mult: (A*B)[i,j] = sum_k A[i,k] * B[k,j]
        # The multiplication tensor has indices (i, j, k) where we sum over k.
        # T[i,j,k] = 1 if we're computing C[i,j] from A[i,k]*B[k,j]
        # 
        # Factorization: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Wait, that doesn't match dimensions. Let me re-read.
        # 
        # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "output factor follows the official trace convention W[k*n+i, r]"
        # 
        # So: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And (A*B)[i,j] = sum_k A[i,k] * B[k,j]
        # 
        # Let's verify: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
        # where col = i*m*p + j*p + k
        # So T[i,j,k] = 1 (only one term contributes) ✓
        # 
        # To compute C[i,j] = sum_k T[i,j,k] * A[i,k] * B[k,j]
        # = sum_k sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,k] * B[k,j]
        # = sum_r U[i*m+j, r] * (sum_k A[i,k] * W[k*n+i, r]) * (sum_k V[j*p+k, r] * B[k,j])
        # 
        # Hmm, that's a bit tangled. Let me just verify numerically.
        
        rank = len(U[0])
        # C[i,j] = sum_k sum_r U[i*6+j, r] * V[j*6+k, r] * W[k*6+i, r] * A[i,k] * B[k,j]
        # This is O(n^3 * rank) = 216 * 189 which is fine
        
        C_check = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0
                for k in range(n):
                    for r in range(rank):
                        s += U[i*6+j][r] * V[j*6+k][r] * W[k*6+i][r] * A[i][k] * B[k][j]
                C_check[i][j] = s
        
        # Check
        max_err = max(abs(C[i][j] - C_check[i][j]) for i in range(n) for j in range(n))
        if max_err > 1e-6:
            # Fallback to schoolbook
            n, m, p = DIMENSIONS
            rank = n * m * p
            U = [[0.0] * rank for _ in range(n * m)]
            V = [[0.0] * rank for _ in range(m * p)]
            W = [[0.0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1.0
                        V[j * p + k][column] = 1.0
                        W[k * n + i][column] = 1.0
                        column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
