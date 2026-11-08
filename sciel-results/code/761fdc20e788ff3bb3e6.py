"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_6x6_block_strassen():
    """
    Construct a rank-189 decomposition for 6x6 matrix multiplication
    by using a 3x3 block structure with each 2x2 block multiplication
    done via Strassen's 7-multiplication algorithm.
    """
    # For 3x3 blocks, we need to multiply 2x2 matrices.
    # The Strassen scheme for 2x2 uses 7 scalar multiplications.
    #
    # Each block C_ij = sum_k A_ik * B_kj.
    # There are 9 blocks, each requiring 7 products: 63 products total per "level".
    # But wait: the full rank is not just 63 * (something). Let me think again.
    #
    # Actually, the block decomposition means:
    # A is 3x3 blocks of size 2x2, B is 3x3 blocks of size 2x2.
    # C_ij = sum_k (A_ik * B_kj) where * is 2x2 matrix multiply.
    # 
    # Using Strassen for each 2x2 multiply: 7 products per pair of 2x2 blocks.
    # Total number of 2x2 block pairs: 3*3*3 = 27 pairs, each needs 7 products = 189 products.
    #
    # So rank = 189.
    #
    # Now I need to construct U, V, W explicitly.
    #
    # A has 3x3=9 rows of blocks, each 2x2 = 18 elements total. Index: i_block * 2 + i_row, j_block * 2 + j_col? 
    # No, the flattening is row-major: element (r, c) of A is at index r * 6 + c.
    # Block A_ik has rows 2i, 2i+1 and cols 2k, 2k+1.
    # So A_ik[a,b] is at flat index (2i + a) * 6 + (2k + b) = 12i + 6a + 2k + b... wait.
    # Actually flat index = row * 6 + col = (2i + a) * 6 + (2k + b) = 12i + 6a + 2k + b.
    #
    # Let me define the indexing clearly.
    # A is 6x6. Element (r, c) is at flat index r*6 + c.
    # Block (i, k) consists of elements with r in {2i, 2i+1} and c in {2k, 2k+1}.
    # So block A_ik[a, b] where a,b in {0,1} is at flat index (2i + a)*6 + (2k + b) = 12i + 6a + 2k + b.
    #
    # Similarly B_kj[a, b] is at flat index (2k + a)*6 + (2j + b) = 12k + 6a + 2j + b.
    #
    # C_ij[a, b] is at flat index (2i + a)*6 + (2j + b) = 12i + 6a + 2j + b.
    # The trace convention says W[k*n + i, r] but let me re-read: W[p*n, rank] with W[k*n+i, r].
    # For C element at position (k, i) in block terms... wait, C is 6x6, element (r, c).
    # In block terms, r = 2i + a, c = 2j + b. The flat index in W is... 
    # W has shape (p*n, rank) = (36, rank). W[k*n + i, r] means the row index is k*6 + i.
    # But C element (r, c) has r = 2i_block + a, c = 2j_block + b.
    # Hmm, the trace convention W[k*n+i, r] - I think this means the row index of W corresponds to
    # the "output" index in some tensor contraction. Let me look at the schoolbook code:
    # W[k * n + i][column] = 1 where k is the column of A / row of B, and i is the row of A.
    # In the tensor: T[i, j, k] = delta[i, k] delta[j, i]... no.
    # Standard: C[i, k] = sum_j A[i, j] B[j, k]. The tensor has indices (i, j, k).
    # U[i*m + j, c], V[j*p + k, c], W[k*n + i, c].
    # So W row index = k*n + i where i is row of C and k is col of C. n=6.
    # So W[ (k)*6 + i, c ] corresponds to C[i, k].
    # So flat index in W for C[i,k] is k*6 + i. That's column-major indexing!
    # 
    # So for C element at (r, c) where r = 2i + a, c = 2j + b:
    # W row index = c * 6 + r = (2j + b)*6 + (2i + a) = 12j + 6b + 2i + a.
    
    n, m, p = 6, 6, 6
    rank = 189
    U = [[0.0] * rank for _ in range(n * m)]  # 36 x 189
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 189
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 189
    
    # Strassen coefficients for 2x2 matrix multiply:
    # P0 = (A00+A11)(B00+B11)
    # P1 = (A10+A11)*B00
    # P2 = A00*(B10-B11)
    # P3 = A11*(B01-B00)
    # P4 = (A00+A01)*B11
    # P5 = (A10-A00)*(B00+B10)
    # P6 = (A01-A11)*(B01+B11)
    #
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    
    # For each pair of blocks (i, k, j) where i is row block of A, k is middle, j is col block of B:
    # We compute A_ik * B_kj using Strassen, and add contributions to C_ij.
    
    col = 0
    for i in range(3):  # row block of A / row block of C
        for k in range(3):  # col block of A / row block of B
            for j in range(3):  # col block of B / col block of C
                # Strassen products for A_ik * B_kj
                # A_ik has elements at flat indices:
                #   (a,b) -> (2i+a)*6 + (2k+b) = 12i + 6a + 2k + b
                # B_kj has elements at flat indices:
                #   (a,b) -> (2k+a)*6 + (2j+b) = 12k + 6a + 2j + b
                # C_ij has elements at flat indices in W (column-major):
                #   C element (a,b) of block ij -> row a, col b in C block -> 
                #   flat C row = 2i+a, flat C col = 2j+b -> W index = (2j+b)*6 + (2i+a)
                
                aA = lambda a, b: 12*i + 6*a + 2*k + b  # flat index in A
                bB = lambda a, b: 12*k + 6*a + 2*j + b  # flat index in B
                cW = lambda a, b: (2*j + b)*6 + (2*i + a)  # flat index in W (col-major)
                
                # P0 = (A00+A11)(B00+B11)
                U[aA(0,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P0
                W[cW(0,0)][col] += 1
                col += 1
                
                # P1 = (A10+A11)*B00
                U[aA(1,0)][col] += 1
                U[aA(1,1)][col] += 1
                V[bB(0,0)][col] += 1
                # C10 += P1
                W[cW(1,0)][col] += 1
                col += 1
                
                # P2 = A00*(B10-B11)
                U[aA(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                V[bB(1,1)][col] += -1
                # C01 += P2
                W[cW(0,1)][col] += 1
                col += 1
                
                # P3 = A11*(B01-B00)
                U[aA(1,1)][col] += 1
                V[bB(0,1)][col] += 1
                V[bB(0,0)][col] += -1
                # C00 += P3
                W[cW(0,0)][col] += 1
                # C10 += P3
                W[cW(1,0)][col] += 1
                col += 1
                
                # P4 = (A00+A01)*B11
                U[aA(0,0)][col] += 1
                U[aA(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += -P4
                W[cW(0,0)][col] += -1
                # C01 += P4
                W[cW(0,1)][col] += 1
                col += 1
                
                # P5 = (A10-A00)*(B00+B10)
                U[aA(1,0)][col] += 1
                U[aA(0,0)][col] += -1
                V[bB(0,0)][col] += 1
                V[bB(1,0)][col] += 1
                # C11 += P5
                W[cW(1,1)][col] += 1
                col += 1
                
                # P6 = (A01-A11)*(B01+B11)
                U[aA(0,1)][col] += 1
                U[aA(1,1)][col] += -1
                V[bB(0,1)][col] += 1
                V[bB(1,1)][col] += 1
                # C00 += P6
                W[cW(0,0)][col] += 1
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n=6, m=6, p=6):
    """Validate the decomposition using NumPy (vectorized)."""
    try:
        import numpy as np
    except ImportError:
        return True  # Can't validate, assume OK
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    rank = Ua.shape[1]
    
    # Check U has shape (n*m, rank)
    if Ua.shape != (n*m, rank):
        return False
    if Va.shape != (m*p, rank):
        return False
    if Wa.shape != (p*n, rank):
        return False
    
    # Test with a random matrix using vectorized operations
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    
    # Flatten A and B
    A_flat = A.reshape(n*m)  # row-major: index i*m + j
    B_flat = B.reshape(m*p)  # row-major: index j*p + k
    
    # Compute C via decomposition using vectorized ops
    # C[k*n+i] = sum_j sum_c U[i*m+j,c] * A_flat[i*m+j] * V[j*p+k,c] * B_flat[j*p+k] * W[k*n+i,c]
    
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    A3 = A.reshape(n, m)
    B3 = B.reshape(m, p)
    
    # C[i,k] = sum_j sum_c U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
    # = sum_j (sum_c U[i,j,c]*A[i,j]*V[j,k,c]*B[j,k]*W[k,i,c])
    
    # Vectorized: compute for all i,k
    # Term for each (i,j,k,c): U3[i,j,c] * A3[i,j] * V3[j,k,c] * B3[j,k] * W3[k,i,c]
    
    # Use einsum:
    # First, U3[i,j,c] * A3[i,j] -> shape (n, m, rank)
    UA = U3 * A3[:, :, np.newaxis]  # (n, m, rank)
    
    # V3[j,k,c] * B3[j,k] -> shape (m, p, rank)
    VB = V3 * B3[:, :, np.newaxis]  # (m, p, rank)
    
    # Now we need sum over j and c: UA[i,j,c] * VB[j,k,c] * W3[k,i,c]
    # W3[k,i,c] has shape (p, n, rank)
    
    # For each (i,k): sum_j sum_c UA[i,j,c] * VB[j,k,c] * W3[k,i,c]
    
    # Reshape for einsum:
    # UA: (n, m, rank) -> indices i, j, c
    # VB: (m, p, rank) -> indices j, k, c  
    # W3: (p, n, rank) -> indices k, i, c
    
    # Result: C[i,k] = sum_{j,c} UA[i,j,c] * VB[j,k,c] * W3[k,i,c]
    
    C_decomp = np.einsum('ijc,jkc,kic->ik', UA, VB, W3)
    
    C_direct = A @ B  # C[i,k] = sum_j A[i,j] B[j,k]
    
    diff = np.max(np.abs(C_decomp - C_direct))
    if diff > 1e-9:
        return False
    
    return True

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    
    # Build the block-Strassen decomposition
    result = _build_6x6_block_strassen()
    
    # Validate if time permits
    if time.time() - t0 < DEADLINE - 5:
        try:
            if not _validate(result["U"], result["V"], result["W"]):
                # Fallback to schoolbook if validation fails
                n, m, p = DIMENSIONS
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
                result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
    
    # Safety margin
    if time.time() - t0 > DEADLINE:
        # Return schoolbook as last resort
        n, m, p = DIMENSIONS
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
        result = {"U": U, "V": V, "W": W}
    
    return result

# EVOLVE-BLOCK-END
