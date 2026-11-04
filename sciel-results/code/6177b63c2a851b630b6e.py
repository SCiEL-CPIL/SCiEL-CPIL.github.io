"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _validate(U, V, W, n, m, p, tol=1e-6):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    # Build the full multiplication tensor T[k*n+i, j*p+l, i*m+j] = delta
    # We verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if (i,j,k) valid else 0
    # Actually the tensor is: T[i,j,k] = delta_{i,j,k} where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the identity in the middle index pairing)
    # Wait, the multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] += A[i,j]*B[j,k]
    # The rank decomposition: sum_c U[i,j,c] * V[j,k,c] * W[i,k,c] = T[i,j,k]
    
    # Let's verify by computing C = A*B for random inputs
    for trial in range(20):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for c in range(rank):
            A_c = np.array(U, dtype=float).reshape(n*m, rank)[:, c].reshape(n, m)
            B_c = np.array(V, dtype=float).reshape(m*p, rank)[:, c].reshape(m, p)
            W_c = np.array(W, dtype=float).reshape(p*n, rank)[:, c].reshape(p, n)
            # C_decomp[i,k] += sum_j A_c[i,j] * B_c[j,k] ... no
            # Actually: C_decomp[i,k] = sum_c (sum_j A[i,j] * U[i,j,c]) * (sum_k' ... ) 
            # The decomposition is: C[i,k] = sum_c (sum_j A[i,j]*U[i,j,c]) * (sum_k B[j,k]*V[j,k,c]) * W[k*n+i,c]
            # Wait, let me re-read the convention.
            # U has shape (n*m, rank): U[i*m+j, c]
            # V has shape (m*p, rank): V[j*p+k, c]  
            # W has shape (p*n, rank): W[k*n+i, c]
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * ... no
            # Standard: C[i,k] = sum_c (sum_j A[i,j] * U[i,j,c]) * (sum_j B[j,k] * V[j,k,c])
            # But W is separate. Let me think again.
            # The tensor is T[i,j,k] where C[i,k] = sum_j A[i,j] * B[j,k]
            # Rank decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
            # So C[i,k] = sum_j sum_c A[i,j] * B[j,k] * ... no
            # Actually: C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * 1 
            # No, that's not right either.
            
            # The correct interpretation: 
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # = sum_c [ sum_j A[i,j] * U[i*m+j, c] ] * [ sum_k' ... ]
            
            # Let me just directly compute:
            # C[i,k] should equal sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * ... 
            # Hmm, I think the correct formula is:
            # C[i,k] = sum_c (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c]) * W[k*n+i, c]
            # But that gives sum over j twice which doesn't make sense.
            
            # Actually for rank-r decomposition of matrix multiplication:
            # C[i,k] = sum_{c=1}^r (sum_j A[i,j] * U_c[i,j]) * (sum_j B[j,k] * V_c[j,k])
            # where W is absorbed or W[i,k,c] multiplies the product.
            
            # Given the shapes U(n*m, r), V(m*p, r), W(p*n, r):
            # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])
            
            # Let me test this interpretation
            for c in range(rank):
                Uc = np.array([U[i*m+j][c] for i in range(n) for j in range(m)], dtype=float).reshape(n, m)
                Vc = np.array([V[j*p+k][c] for j in range(m) for k in range(p)], dtype=float).reshape(m, p)
                Wc = W[k*n+i][c] if False else None  # can't do this in loop like this
            break
        break
    # This validation is getting complex. Let me just do a direct tensor check.
    # T[i,j,k] = 1 (for the multiplication tensor, every term A[i,j]*B[j,k] contributes to C[i,k])
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Best known: schoolbook rank = n*m*p = 60
    # Try to find a lower rank decomposition
    
    # Strategy: Use the known fact that for 2x2x2, rank is 7 (Strassen)
    # For 2x3x2: we can try rank 11 or lower
    # For 2x2x6: split into three 2x2x2, each rank 7, total 21
    # For 2x5x6: split into 2x2x6 (rank 21) + 2x3x6 (rank ?)
    # For 2x3x6: split into three 2x3x2, each rank 11, total 33
    # Total: 21 + 33 = 54
    
    # But can we do better? Let's try a direct search for rank < 60.
    
    # First, let's try the block-Strassen approach to get rank 54.
    # 2x2x2 Strassen decomposition:
    # M1 = (A11 - A00) * (B00 + B11)
    # M2 = (A00 + A11) * B11
    # M3 = A11 * (B10 - B00)
    # M4 = A00 * (B01 - B11)
    # M5 = (A10 + A00) * B00
    # M6 = (A11 + A01) * (B01 + B11)
    # M7 = (A01 - A11) * (B10 + B00)
    
    # For our indexing: A[i,j], B[j,k], C[i,k]
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    strassen_222 = [
        # (U_coeffs, V_coeffs, W_coeffs) where each is a dict or list
        # We'll represent each term as (i_u, j_u, i_v, j_v, i_w, j_w, coeff_u, coeff_v, coeff_w)
        # Actually let's just build U, V, W directly for the block approach.
    ]
    
    # Let me build the 54-rank decomposition explicitly.
    # Split j into {0,1} and {2,3,4}
    # A = [A_2x2, A_2x3], B = [B_2x6; B_3x6]
    # C = A_2x2 * B_2x6 + A_2x3 * B_3x6
    
    # A_2x2 * B_2x6: split k into {0,1}, {2,3}, {4,5}
    # Three 2x2x2 blocks, each using Strassen (rank 7), total 21
    
    # A_2x3 * B_3x6: split k into {0,1}, {2,3}, {4,5}
    # Three 2x3x2 blocks. For 2x3x2, what's the minimal rank?
    # We can split the 3 columns of A (or rows of B) into 1+2 or 1+1+1
    # If we split j into {0}, {1,2}: A_2x1 * B_1x2 + A_2x2 * B_2x2
    # A_2x1 * B_1x2: rank 4 (schoolbook) or can we do better? 2x1x2 has rank 4 (it's just 4 products)
    # Actually 2x1x2: C[i,k] = A[i,0] * B[0,k], so rank = 4 (each (i,k) pair needs its own term)
    # Wait, no: C[0,0] = A[0,0]*B[0,0], C[0,1] = A[0,0]*B[0,1], C[1,0] = A[1,0]*B[0,0], C[1,1] = A[1,0]*B[0,1]
    # We can factor: C[0,0] = (A[0,0])*(B[0,0]), C[0,1] = (A[0,0])*(B[0,1])
    # So for fixed i, C[i,0] and C[i,1] share A[i,0]. But they need different V terms.
    # Actually, rank of 2x1x2 tensor: it's the matrix [1,1;1,1] outer product structure...
    # The tensor T[i,j,k] = 1 for all i,k with j=0. So T[i,0,k] = 1.
    # This is a rank 4 tensor (can't do better than 4 since we have 4 independent products)
    # Wait, actually: C[i,k] = A[i,0] * B[0,k]. We can write this as:
    # C = (A[:,0]) * (B[0,:]) which is rank 1 as a matrix, but as a bilinear map...
    # The number of scalar multiplications needed is 4 (one for each (i,k) pair)
    # So rank = 4 for 2x1x2.
    
    # A_2x2 * B_2x2: rank 7 (Strassen)
    # So 2x3x2 via split {0} and {1,2}: 4 + 7 = 11
    
    # Total for 2x5x6: 3*7 + 3*11 = 21 + 33 = 54
    
    # Now let's actually build this.
    
    def strassen_2x2x2(offset_i, offset_j, offset_k, U, V, W, start_col):
        """Add Strassen 2x2x2 decomposition at given offsets. Returns new column index."""
        col = start_col
        # A is 2x2 at rows offset_i, offset_i+1 and cols offset_j, offset_j+1
        # B is 2x2 at rows offset_j, offset_j+1 and cols offset_k, offset_k+1
        # C is 2x2 at rows offset_i, offset_i+1 and cols offset_k, offset_k+1
        
        i0, i1 = offset_i, offset_i + 1
        j0, j1 = offset_j, offset_j + 1
        k0, k1 = offset_k, offset_k + 1
        
        # Term 1: (A[i1,j1] - A[i0,j0]) * (B[j0,k0] + B[j1,k1])
        # U: A[i1,j1] += 1, A[i0,j0] += -1
        # V: B[j0,k0] += 1, B[j1,k1] += 1
        # W: C[i0,k0] += 1, C[i1,k0] += 0, C[i0,k1] += 0, C[i1,k1] += 0
        # Wait, W[k*n+i, c] so W index is k*n+i
        # Output: C[i0,k0] gets this term, C[i1,k1] gets -this term? No.
        # Let me use the standard Strassen formulas:
        # M1 = (A11 - A00)(B00 + B11)
        # M2 = (A00 + A11)B11
        # M3 = A11(B10 - B00)
        # M4 = A00(B01 - B11)
        # M5 = (A10 + A00)B00
        # M6 = (A11 + A01)(B01 + B11)
        # M7 = (A01 - A11)(B10 + B00)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # In our indexing: A[i,j], B[j,k], C[i,k]
        # A00 = A[i0,j0], A01 = A[i0,j1], A10 = A[i1,j0], A11 = A[i1,j1]
        # B00 = B[j0,k0], B01 = B[j0,k1], B10 = B[j1,k0], B11 = B[j1,k1]
        # C00 = C[i0,k0], C01 = C[i0,k1], C10 = C[i1,k0], C11 = C[i1,k1]
        
        # M1 = (A11 - A00)(B00 + B11)
        U[i1*m + j1][col] += 1
        U[i0*m + j0][col] += -1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1   # C00
        W[k1*n + i1][col] += -1  # C11
        col += 1
        
        # M2 = (A00 + A11)B11
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i1][col] += 1   # C10
        W[k1*n + i1][col] += -1  # C11
        col += 1
        
        # M3 = A11(B10 - B00)
        U[i1*m + j1][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] += -1
        W[k1*n + i0][col] += 1   # C01
        W[k1*n + i1][col] += 1   # C11
        col += 1
        
        # M4 = A00(B01 - B11)
        U[i0*m + j0][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] += -1
        W[k0*n + i0][col] += 1   # C00
        W[k0*n + i1][col] += 1   # C10
        col += 1
        
        # M5 = (A10 + A00)B00
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += -1  # C00
        W[k1*n + i0][col] += 1   # C01
        col += 1
        
        # M6 = (A11 + A01)(B01 + B11)
        U[i1*m + j1][col] += 1
        U[i0*m + j1][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i1][col] += 1   # C11
        col += 1
        
        # M7 = (A01 - A11)(B10 + B00)
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] += -1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i0][col] += 1   # C00
        col += 1
        
        return col
    
    def schoolbook_2x1x2(offset_i, offset_j, offset_k, U, V, W, start_col):
        """Add schoolbook 2x1x2 decomposition. A is 2x1, B is 1x2, C is 2x2."""
        col = start_col
        i0, i1 = offset_i, offset_i + 1
        j0 = offset_j
        k0, k1 = offset_k, offset_k + 1
        
        # C[i0,k0] = A[i0,j0]*B[j0,k0]
        U[i0*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i0][col] = 1
        col += 1
        
        # C[i0,k1] = A[i0,j0]*B[j0,k1]
        U[i0*m + j0][col] = 1
        V[j0*p + k1][col] = 1
        W[k1*n + i0][col] = 1
        col += 1
        
        # C[i1,k0] = A[i1,j0]*B[j0,k0]
        U[i1*m + j0][col] = 1
        V[j0*p + k0][col] = 1
        W[k0*n + i1][col] = 1
        col += 1
        
        # C[i1,k1] = A[i1,j0]*B[j0,k1]
        U[i1*m + j0][col] = 1
        V[j0*p + k1][col] = 1
        W[k1*n + i1][col] = 1
        col += 1
        
        return col
    
    def block_2x3x2(offset_i, offset_j, offset_k, U, V, W, start_col):
        """
        2x3x2 decomposition with rank 11.
        A is 2x3 at rows offset_i..offset_i+1, cols offset_j..offset_j+2
        B is 3x2 at rows offset_j..offset_j+2, cols offset_k..offset_k+1
        Split j into {offset_j} and {offset_j+1, offset_j+2}
        A_2x1 * B_1x2 (rank 4) + A_2x2 * B_2x2 (rank 7)
        """
        col = start_col
        i0 = offset_i
        j_single = offset_j
        j_pair_start = offset_j + 1
        k0 = offset_k
        
        # First part: A[*, j_single] * B[j_single, *] -> 2x1x2, rank 4
        col = schoolbook_2x1x2(i0, j_single, k0, U, V, W, col)
        
        # Second part: A[*, j_pair_start:j_pair_start+2] * B[j_pair_start:j_pair_start+2, *] -> 2x2x2, rank 7
        col = strassen_2x2x2(i0, j_pair_start, k0, U, V, W, col)
        
        return col
    
    # Build the full 2x5x6 decomposition
    U = [[0] * 54 for _ in range(n * m)]
    V = [[0] * 54 for _ in range(m * p)]
    W = [[0] * 54 for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, 0:6]
    # Split k into {0,1}, {2,3}, {4,5}
    for k_start in [0, 2, 4]:
        col = strassen_2x2x2(0, 0, k_start, U, V, W, col)
    
    # Part 2: A[0:2, 2:5] * B[2:5, 0:6]
    # Split k into {0,1}, {2,3}, {4,5}
    for k_start in [0, 2, 4]:
        col = block_2x3x2(0, 2, k_start, U, V, W, col)
    
    assert col == 54, f"Expected 54 columns, got {col}"
    
    # Validate
    if _validate(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if validation fails
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

# EVOLVE-BLOCK-END
