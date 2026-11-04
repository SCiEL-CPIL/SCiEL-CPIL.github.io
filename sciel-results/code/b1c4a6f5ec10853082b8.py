"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s kill

    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j*p+k), (k'*n+i')] = 1 if (i,j,k) == (i',j',k') and i==i' and k==k'
    # Actually, the matrix multiplication tensor T has:
    # T[(i,j), (j,k), (i,k)] = 1
    # where (i,j) indexes A (size n*m), (j,k) indexes B (size m*p), (i,k) indexes C (size n*p)
    # But W is indexed by k*n+i, so the third index is (k,i) not (i,k).
    # Let's be careful: the decomposition condition is
    # sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 for all i,j,k
    # and 0 for all other combinations.
    
    # Actually, the tensor T is defined such that:
    # For the matrix multiplication C = A * B:
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # The tensor T in U^* ⊗ V^* ⊗ W^* where:
    # U has dimension n*m (index (i,j))
    # V has dimension m*p (index (j,k))  
    # W has dimension p*n (index (k,i))
    # T[(i,j),(j,k),(k,i)] = 1
    # All other entries are 0.
    
    # Let's construct T as a 3D array
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    # Verify decomposition
    def verify(U, V, W, rank):
        # Check that sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = T[(i,j),(j,k),(k,i)]
        # We can do this efficiently using einsum
        # reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        # But this would be huge. Instead, check specific entries.
        # For each (i,j,k), check that the sum equals 1
        # For a sample of other entries, check they are 0
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        # Check a few off-diagonal entries
        import random
        random.seed(42)
        for _ in range(100):
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            # Pick a different (i',j',k')
            i2 = random.randint(0, n - 1)
            j2 = random.randint(0, m - 1)
            k2 = random.randint(0, p - 1)
            if (i2, j2, k2) == (i, j, k):
                continue
            s = 0.0
            for r in range(rank):
                s += U[i * m + j2, r] * V[j * p + k2, r] * W[k * n + i2, r]
            err = abs(s)
            if err > max_err:
                max_err = err
        return max_err < 1e-10
    
    # Try Strassen-based block decomposition
    # Split 3x4x4 into blocks:
    # Rows of A: split into 2+1
    # Cols of A / Rows of B: split into 2+2
    # Cols of B: split into 2+2
    # So we have 1 (row blocks) * 2 (col blocks of A) * 2 (col blocks of B) = 4 blocks of size 2x2x2
    # Plus 1 (row blocks) * 2 * 2 = 4 blocks of size 1x2x2
    
    # Strassen's rank-7 algorithm for 2x2x2:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # In tensor decomposition terms, we need U, V, W for each rank-1 term.
    # For a 2x2x2 block with A indices (i1,i2,j1,j2), B indices (j1,j2,k1,k2), C indices (i1,i2,k1,k2):
    # Each Strassen term M_c corresponds to a rank-1 decomposition where:
    # U[(i,j),c] gives the coefficient of A_{i,j}
    # V[(j,k),c] gives the coefficient of B_{j,k}
    # W[(k,i),c] gives the coefficient of C_{i,k}
    
    # Let's define the Strassen factors for 2x2x2:
    # Using 0-based indexing within the block:
    # A rows: 0,1; A cols: 0,1; B rows: 0,1; B cols: 0,1; C rows: 0,1; C cols: 0,1
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00->1, A11->1; V: B00->1, B11->1; W: C00->1, C11->1 (with signs from C formula)
    # Actually, we need to be more careful. Each M_c is a product of a linear combination of A entries and a linear combination of B entries, and it contributes to C entries with certain coefficients.
    
    # Let me define the Strassen decomposition properly.
    # For 2x2 matrices:
    # C = A * B where A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = [[c00,c01],[c10,c11]]
    
    # Strassen:
    # p1 = (a00+a11)(b00+b11)
    # p2 = (a10+a11)b00
    # p3 = a00(b10-b11)
    # p4 = a11(b01-b00)
    # p5 = (a00+a01)b11
    # p6 = (a10-a00)(b00+b01)
    # p7 = (a01-a11)(b10+b11)
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # For the tensor decomposition, each p_c gives a rank-1 term:
    # U_c[(i,j)] = coefficient of A_{i,j} in the first factor
    # V_c[(j,k)] = coefficient of B_{j,k} in the second factor
    # W_c[(k,i)] = coefficient of C_{i,k} from p_c (from the C formulas above)
    
    # Let's build these:
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition.
        U: (4, 7) for A indices (i,j) where i,j in {0,1}
        V: (4, 7) for B indices (j,k) where j,k in {0,1}
        W: (4, 7) for C indices (k,i) where k,i in {0,1}
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # Indexing: A_{i,j} -> i*2+j, B_{j,k} -> j*2+k, C_{i,k} -> k*2+i (since W is k*n+i)
        
        # p1 = (a00+a11)(b00+b11)
        # U: a00=1, a11=1
        U[0, 0] = 1.0  # a00
        U[3, 0] = 1.0  # a11
        # V: b00=1, b11=1
        V[0, 0] = 1.0  # b00
        V[3, 0] = 1.0  # b11
        # W: c00 += 1, c11 += 1
        W[0, 0] = 1.0  # c00 (k=0,i=0)
        W[3, 0] = 1.0  # c11 (k=1,i=1)
        
        # p2 = (a10+a11)b00
        U[2, 1] = 1.0  # a10
        U[3, 1] = 1.0  # a11
        V[0, 1] = 1.0  # b00
        W[2, 1] = 1.0  # c10 (k=0,i=1)
        
        # p3 = a00(b10-b11)
        U[0, 2] = 1.0  # a00
        V[2, 2] = 1.0  # b10
        V[3, 2] = -1.0 # b11
        W[1, 2] = 1.0  # c01 (k=1,i=0)
        
        # p4 = a11(b01-b00)
        U[3, 3] = 1.0  # a11
        V[1, 3] = 1.0  # b01
        V[0, 3] = -1.0 # b00
        W[2, 3] = 1.0  # c10 (k=0,i=1)
        W[0, 3] = 1.0  # c00 (k=0,i=0)
        
        # p5 = (a00+a01)b11
        U[0, 4] = 1.0  # a00
        U[1, 4] = 1.0  # a01
        V[3, 4] = 1.0  # b11
        W[1, 4] = 1.0  # c01 (k=1,i=0)
        W[0, 4] = -1.0 # c00 (k=0,i=0)
        
        # p6 = (a10-a00)(b00+b01)
        U[2, 5] = 1.0  # a10
        U[0, 5] = -1.0 # a00
        V[0, 5] = 1.0  # b00
        V[1, 5] = 1.0  # b01
        W[3, 5] = 1.0  # c11 (k=1,i=1)
        
        # p7 = (a01-a11)(b10+b11)
        U[1, 6] = 1.0  # a01
        U[3, 6] = -1.0 # a11
        V[2, 6] = 1.0  # b10
        V[3, 6] = 1.0  # b11
        W[3, 6] = 1.0  # c11 (k=1,i=1)
        W[0, 6] = 1.0  # c00 (k=0,i=0)
        
        return U, V, W
    
    # Verify the Strassen decomposition for 2x2x2
    def verify_strassen():
        U_s, V_s, W_s = strassen_2x2x2()
        # Check that sum_r U_s[(i,j),r] * V_s[(j,k),r] * W_s[(k,i),r] = delta_{i,i'}delta_{k,k'}
        # Actually, it should equal 1 when (i,j,k) is a valid triple and 0 otherwise
        # For 2x2x2, the tensor T has T[(i,j),(j,k),(k,i)] = 1
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for r in range(7):
                        s += U_s[i * 2 + j, r] * V_s[j * 2 + k, r] * W_s[k * 2 + i, r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        # Check off-diagonal
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                if (i2, j2, k2) == (i, j, k):
                                    continue
                                s = 0.0
                                for r in range(7):
                                    s += U_s[i * 2 + j2, r] * V_s[j * 2 + k2, r] * W_s[k2 * 2 + i2, r]
                                if abs(s) > 1e-10:
                                    return False
        return True
    
    if not verify_strassen():
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the full 3x4x4 decomposition using blocks
    # Block structure:
    # A rows: split into [0,1] and [2] (sizes 2 and 1)
    # A cols / B rows: split into [0,1] and [2,3] (sizes 2 and 2)
    # B cols: split into [0,1] and [2,3] (sizes 2 and 2)
    
    # Blocks:
    # Block (0,0,0): A rows 0-1, A cols 0-1, B cols 0-1 -> 2x2x2, use Strassen
    # Block (0,0,1): A rows 0-1, A cols 0-1, B cols 2-3 -> 2x2x2, use Strassen
    # Block (0,1,0): A rows 0-1, A cols 2-3, B cols 0-1 -> 2x2x2, use Strassen
    # Block (0,1,1): A rows 0-1, A cols 2-3, B cols 2-3 -> 2x2x2, use Strassen
    # Block (1,0,0): A row 2, A cols 0-1, B cols 0-1 -> 1x2x2, use schoolbook (4 terms)
    # Block (1,0,1): A row 2, A cols 0-1, B cols 2-3 -> 1x2x2, use schoolbook (4 terms)
    # Block (1,1,0): A row 2, A cols 2-3, B cols 0-1 -> 1x2x2, use schoolbook (4 terms)
    # Block (1,1,1): A row 2, A cols 2-3, B cols 2-3 -> 1x2x2, use schoolbook (4 terms)
    
    # Total rank: 4*7 + 4*4 = 28 + 16 = 44
    
    U_full = np.zeros((n * m, 44), dtype=float)
    V_full = np.zeros((m * p, 44), dtype=float)
    W_full = np.zeros((p * n, 44), dtype=float)
    
    col = 0
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Add Strassen blocks
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        for r in range(7):
            for ii in range(2):
                for jj in range(2):
                    u_val = U_s[ii * 2 + jj, r]
                    if u_val != 0:
                        U_full[(i0 + ii) * m + (j0 + jj), col] += u_val
            for jj in range(2):
                for kk in range(2):
                    v_val = V_s[jj * 2 + kk, r]
                    if v_val != 0:
                        V_full[(j0 + jj) * p + (k0 + kk), col] += v_val
            for kk in range(2):
                for ii in range(2):
                    w_val = W_s[kk * 2 + ii, r]
                    if w_val != 0:
                        W_full[(k0 + kk) * n + (i0 + ii), col] += w_val
            col += 1
    
    def add_schoolbook_1x2x2(i, j0, k0):
        nonlocal col
        for jj in range(2):
            for kk in range(2):
                U_full[i * m + (j0 + jj), col] = 1.0
                V_full[(j0 + jj) * p + (k0 + kk), col] = 1.0
                W_full[(k0 + kk) * n + i, col] = 1.0
                col += 1
    
    # Add the four Strassen blocks
    add_strassen_block(0, 0, 0)   # A rows 0-1, A cols 0-1, B cols 0-1
    add_strassen_block(0, 0, 2)   # A rows 0-1, A cols 0-1, B cols 2-3
    add_strassen_block(0, 2, 0)   # A rows 0-1, A cols 2-3, B cols 0-1
    add_strassen_block(0, 2, 2)   # A rows 0-1, A cols 2-3, B cols 2-3
    
    # Add the four schoolbook 1x2x2 blocks
    add_schoolbook_1x2x2(2, 0, 0)  # A row 2, A cols 0-1, B cols 0-1
    add_schoolbook_1x2x2(2, 0, 2)  # A row 2, A cols 0-1, B cols 2-3
    add_schoolbook_1x2x2(2, 2, 0)  # A row 2, A cols 2-3, B cols 0-1
    add_schoolbook_1x2x2(2, 2, 2)  # A row 2, A cols 2-3, B cols 2-3
    
    if col != 44:
        # Fallback
        U, V, W, rank = schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the full decomposition
    if verify(U_full, V_full, W_full, 44):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    # If verification failed, try to find a lower rank using numerical optimization
    # But given time constraints, fallback to schoolbook
    U, V, W, rank = schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
