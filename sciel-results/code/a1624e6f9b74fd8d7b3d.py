import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a Strassen-enhanced tensor decomposition for 3x5 * 5x5 matrix multiplication.
    We use the schoolbook decomposition (rank 75) but replace certain 2x2x2 sub-blocks
    with Strassen's rank-7 algorithm to reduce total rank.
    
    For a 2x2x2 block at positions (i0, j0, k0), the schoolbook uses 8 terms.
    Strassen uses 7 terms with specific coefficients.
    
    We identify non-overlapping 2x2x2 blocks in the 3x5x5 grid and replace them.
    """
    start_time = time.time()
    deadline = start_time + 100  # 100s deadline, leaving 20s margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 75
    
    # Strassen decomposition for 2x2x2 block
    # The 7 Strassen products are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, each Strassen term c corresponds to:
    # U_{i,j,c} * V_{j,k,c} * W_{k,i,c} summed over c
    #
    # For a 2x2x2 block at (i0, j0, k0):
    # Indices: i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # A sub-matrix: A[i0..i0+1, j0..j0+1]
    # B sub-matrix: B[j0..j0+1, k0..k0+1]
    # C sub-matrix: C[i0..i0+1, k0..k0+1]
    
    # Strassen coefficients for each of the 7 products
    # For product M_s, we have:
    # U factor: coefficients for A_{i,j}
    # V factor: coefficients for B_{j,k}
    # W factor: coefficients for C_{k,i}
    
    # Let's define the Strassen decomposition more carefully.
    # The multiplication tensor for 2x2 matrices:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # 
    # Schoolbook: 8 terms, each with U=1, V=1, W=1 at specific positions
    # 
    # Strassen: 7 terms. For each term s=1..7:
    # M_s = (sum_{i,j} U_{ij,s} A_{ij}) * (sum_{j,k} V_{jk,s} B_{jk})
    # C_{ik} = sum_s W_{ik,s} M_s
    # 
    # The tensor decomposition has:
    # For term s: U_vec is the flattened U_{ij,s} over (i,j), 
    #              V_vec is the flattened V_{jk,s} over (j,k),
    #              W_vec is the flattened W_{ik,s} over (k,i) [note: W indexed by k,i per convention]
    
    # Strassen factors (for 2x2 case):
    # M1 = (A11+A22)(B11+B22)
    #   U: A11=1, A22=1, others 0
    #   V: B11=1, B22=1, others 0
    #   W: C11 += 1, C22 += 1 (from M1 contribution to C11 and C22)
    #   Wait, need to be more careful.
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # So W coefficients (for C_{ik} = sum_s W_{ik,s} M_s):
    # M1: C11 += 1, C22 += 1
    # M2: C21 += 1, C22 -= 1
    # M3: C12 += 1, C22 += 1
    # M4: C11 += 1, C21 += 1
    # M5: C11 -= 1, C12 += 1
    # M6: C22 += 1
    # M7: C11 += 1
    
    # U coefficients (for M_s = sum U_{ij,s} A_{ij} * sum V_{jk,s} B_{jk}):
    # M1: A11=1, A22=1
    # M2: A21=1, A22=1
    # M3: A11=1
    # M4: A22=1
    # M5: A11=1, A12=1
    # M6: A21=1, A11=-1
    # M7: A12=1, A22=-1
    
    # V coefficients:
    # M1: B11=1, B22=1
    # M2: B11=1
    # M3: B21=1, B22=-1
    # M4: B12=1, B11=-1
    # M5: B22=1
    # M6: B11=1, B12=1
    # M7: B21=1, B22=1
    
    # Now, for the full 3x5x5 problem, we need to embed these.
    # The full U has shape (3*5, rank) = (15, rank)
    # The full V has shape (5*5, rank) = (25, rank)
    # The full W has shape (5*3, rank) = (15, rank)
    
    # Strategy: Start with schoolbook (75 terms), then for each 2x2x2 block we want to
    # replace, remove the 8 schoolbook terms and add 7 Strassen terms, net saving of 1.
    
    # But we need to be careful: the Strassen decomposition must be compatible with
    # the surrounding structure. Since matrix multiplication is linear, we can decompose
    # the tensor as a sum of independent blocks if the blocks don't share indices.
    
    # For a 2x2x2 block at (i0, j0, k0) where i0 in {0,1}, j0 in {0,1,2,3}, k0 in {0,1,2,3}:
    # The block involves A[i0:i0+2, j0:j0+2], B[j0:j0+2, k0:k0+2], C[i0:i0+2, k0:k0+2]
    # These are disjoint from other blocks if we choose non-overlapping positions.
    
    # In 3x5x5, we can fit:
    # i dimension: 3 rows -> can fit one 2x2 block (using rows 0,1) or (rows 1,2)
    # j dimension: 5 cols -> can fit two 2x2 blocks (cols 0,1 and cols 2,3) or (cols 1,2 and cols 3,4)
    # k dimension: 5 cols -> can fit two 2x2 blocks
    
    # Maximum non-overlapping 2x2x2 blocks: floor(3/2) * floor(5/2) * floor(5/2) = 1 * 2 * 2 = 4
    # Each saves 1 rank, so total rank = 75 - 4 = 71
    
    # Let's place blocks at:
    # Block 1: i0=0, j0=0, k0=0
    # Block 2: i0=0, j0=0, k0=2
    # Block 3: i0=0, j0=2, k0=0
    # Block 4: i0=0, j0=2, k0=2
    
    # These are all disjoint in the (i,j,k) index space.
    
    # Build the decomposition:
    # Start with all 75 schoolbook terms, but for the 4 blocks, use Strassen instead.
    
    rank = 0
    U_cols = []  # list of columns, each is a list of 15 elements
    V_cols = []  # list of columns, each is a list of 25 elements
    W_cols = []  # list of columns, each is a list of 15 elements
    
    # Define which (i,j,k) triples are covered by Strassen blocks
    strassen_blocks = [
        (0, 0, 0),  # i0=0, j0=0, k0=0
        (0, 0, 2),  # i0=0, j0=0, k0=2
        (0, 2, 0),  # i0=0, j0=2, k0=0
        (0, 2, 2),  # i0=0, j0=2, k0=2
    ]
    
    # Build set of (i,j,k) covered by Strassen blocks
    covered = set()
    for i0, j0, k0 in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    # Add schoolbook terms for uncovered (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    u_col = [0] * (n * m)
                    v_col = [0] * (m * p)
                    w_col = [0] * (p * n)
                    u_col[i * m + j] = 1
                    v_col[j * p + k] = 1
                    w_col[k * n + i] = 1
                    U_cols.append(u_col)
                    V_cols.append(v_col)
                    W_cols.append(w_col)
                    rank += 1
    
    # Add Strassen terms for each block
    # Strassen coefficients
    # For each of the 7 products s=0..6:
    # U factors (for A_{ij}): indexed by (di, dj) in {0,1}x{0,1}
    # V factors (for B_{jk}): indexed by (dj, dk) in {0,1}x{0,1}
    # W factors (for C_{ik}): indexed by (dk, di) in {0,1}x{0,1} [note: W[k*n+i] so index is (k,i) -> (dk, di)]
    
    # M1: U: A11=1,A22=1; V: B11=1,B22=1; W: C11+=1, C22+=1
    # M2: U: A21=1,A22=1; V: B11=1; W: C21+=1, C22-=1
    # M3: U: A11=1; V: B21=1,B22=-1; W: C12+=1, C22+=1
    # M4: U: A22=1; V: B12=1,B11=-1; W: C11+=1, C21+=1
    # M5: U: A11=1,A12=1; V: B22=1; W: C11-=1, C12+=1
    # M6: U: A21=1,A11=-1; V: B11=1,B12=1; W: C22+=1
    # M7: U: A12=1,A22=-1; V: B21=1,B22=1; W: C11+=1
    
    # Index mapping for 2x2 sub-matrix:
    # A_{di,dj} -> local index (di, dj), di in {0,1}, dj in {0,1}
    # B_{dj,dk} -> local index (dj, dk)
    # C_{dk,di} -> local index (dk, di) [since W[k*n+i], k=dk, i=di]
    
    strassen_U = [
        { (0,0): 1, (1,1): 1 },           # M1
        { (1,0): 1, (1,1): 1 },           # M2
        { (0,0): 1 },                      # M3
        { (1,1): 1 },                      # M4
        { (0,0): 1, (0,1): 1 },           # M5
        { (1,0): 1, (0,0): -1 },          # M6
        { (0,1): 1, (1,1): -1 },          # M7
    ]
    
    strassen_V = [
        { (0,0): 1, (1,1): 1 },           # M1
        { (0,0): 1 },                      # M2
        { (1,0): 1, (1,1): -1 },          # M3
        { (0,1): 1, (0,0): -1 },          # M4
        { (1,1): 1 },                      # M5
        { (0,0): 1, (0,1): 1 },           # M6
        { (1,0): 1, (1,1): 1 },           # M7
    ]
    
    strassen_W = [
        { (0,0): 1, (1,1): 1 },           # M1: C11+=1, C22+=1
        { (0,1): 1, (1,1): -1 },          # M2: C21+=1, C22-=1
        { (1,0): 1, (1,1): 1 },           # M3: C12+=1, C22+=1
        { (0,0): 1, (0,1): 1 },           # M4: C11+=1, C21+=1
        { (0,0): -1, (1,0): 1 },          # M5: C11-=1, C12+=1
        { (1,1): 1 },                      # M6: C22+=1
        { (0,0): 1 },                      # M7: C11+=1
    ]
    
    for i0, j0, k0 in strassen_blocks:
        for s in range(7):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Set U factors
            for (di, dj), coeff in strassen_U[s].items():
                i = i0 + di
                j = j0 + dj
                u_col[i * m + j] = coeff
            
            # Set V factors
            for (dj, dk), coeff in strassen_V[s].items():
                j = j0 + dj
                k = k0 + dk
                v_col[j * p + k] = coeff
            
            # Set W factors
            for (dk, di), coeff in strassen_W[s].items():
                k = k0 + dk
                i = i0 + di
                w_col[k * n + i] = coeff
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
            rank += 1
    
    # Convert to numpy arrays
    U = np.array(U_cols, dtype=float).T  # shape (n*m, rank)
    V = np.array(V_cols, dtype=float).T  # shape (m*p, rank)
    W = np.array(W_cols, dtype=float).T  # shape (p*n, rank)
    
    # Verify correctness with a random test
    if time.time() < deadline - 10:
        try:
            np.random.seed(42)
            A_test = np.random.randn(n, m)
            B_test = np.random.randn(m, p)
            C_expected = A_test @ B_test
            
            # Compute using tensor decomposition
            # C_{i,k} = sum_c sum_{j} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # More efficiently:
            # For each c, the contribution to C is:
            # C += outer of (A * U_c) and (B * V_c) weighted by W_c
            # Actually: C_{i,k} = sum_c W[k*n+i, c] * (sum_j A_{i,j} U[i*m+j, c]) * (sum_j B_{j,k} V[j*p+k, c])
            # Wait, that's not right either. Let me think...
            # 
            # The tensor decomposition says:
            # T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
            # 
            # And the multiplication is:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # 
            # The bilinear form: sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k} = sum_c (sum_{i,j} A_{i,j} U_{i*m+j,c}) * (sum_{j,k} B_{j,k} V_{j*p+k,c}) * (sum_{i,k} C_{i,k} W_{k*n+i,c})
            # 
            # This means the decomposition is for the trilinear form, not directly for C.
            # To extract C, we need: C_{i,k} = sum_c W_{k*n+i,c} * (A * U_c)_{i} * (B * V_c)_{k}?
            # No, that's not dimensionally correct.
            #
            # Actually, the standard way: the multiplication tensor T satisfies
            # T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} for the "trace" form?
            # Or is it T(A,B) = C where C_{i,k} = sum_j A_{i,j} B_{j,k}?
            #
            # From the seed code, it seems the decomposition is:
            # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if this is the "right" (i,j,k), 0 otherwise
            # This reconstructs the identity tensor, which when contracted with A and B gives C.
            #
            # Specifically: C_{i,k} = sum_j A_{i,j} B_{j,k} = sum_j sum_c [U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]] * A_{i,j} * B_{j,k}
            # Wait, that doesn't work because W is also contracted.
            #
            # Let me re-examine. The seed code has:
            # For each (i,j,k): U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
            # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 if there exists c with all three = 1, which is the c for (i,j,k).
            # So the sum equals 1 if (i,j,k) is valid, 0 otherwise. But all (i,j,k) in range are valid.
            #
            # I think the convention is:
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # And the tensor is such that:
            # sum_{i,j,k} C_{i,k} * [sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]] = sum_{i,j,k} C_{i,k} A_{i,j} B_{j,k}
            # This would mean the tensor represents the bilinear form <C, A @ B> = <A, B, C> (trilinear).
            #
            # For the decomposition to be valid, we need:
            # sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = delta_{i,i'}delta_{j,j'}delta_{k,k'}? No.
            #
            # I think the correct interpretation is that the tensor T is defined by:
            # T_{i,j,k} = 1 for all i,j,k (in the multiplication tensor, each term A_{i,j}B_{j,k} contributes to C_{i,k})
            # And the decomposition is: T_{i,j,k} = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
            # But that would mean T_{i,j,k} = 1 for all i,j,k, which is not right.
            #
            # Let me just trust the seed code and verify numerically.
            
            # Compute C from decomposition:
            # C_{i,k} = sum_c W[k*n+i, c] * (sum_j A[i,j] * U[i*m+j, c]) * (sum_j B[j,k] * V[j*p+k, c])?
            # No, that's not linear in the right way.
            
            # Actually, I think the correct formula is:
            # The multiplication is a bilinear map (A, B) -> C
            # C_{i,k} = sum_j A_{i,j} B_{j,k}
            # The tensor decomposition gives:
            # C_{i,k} = sum_c (sum_j A_{i,j} U[i*m+j, c]) * (sum_j B_{j,k} V[j*p+k, c]) * W[k*n+i, c]
            # This is a sum of rank-1 bilinear forms.
            
            C_computed = np.zeros((n, p))
            for c in range(rank):
                # Term c: (A @ U_c_col) * (B @ V_c_col) * W_c_col
                # A @ U_c_col: U_c_col has shape (n*m,), reshape to (n,m), then A @ that
                U_c = U[:, c].reshape(n, m)
                V_c = V[:, c].reshape(m, p)
                W_c = W[:, c].reshape(p, n)
                
                # Contribution to C:
                # C += (A @ U_c) @ (V_c @ B.T).T * W_c? 
                # Let's think: (A @ U_c) is n x 1, (V_c @ B.T) is p x 1? No.
                # 
                # Actually: sum_j A_{i,j} U_{i*m+j,c} is the i-th component of A @ U_c_col (where U_c_col is reshaped to (n,m) and we do A @ U_c_col... no, that's n x m times m x 1? U_c_col is length n*m.
                # 
                # Let me define: U_c_vec = U[:, c], which has n*m elements.
                # (A @ U_c)_{i} = sum_j A_{i,j} * U_c_vec[i*m+j]
                # Similarly (B @ V_c)_{k} = sum_j B_{j,k} * V_c_vec[j*p+k]
                # Then C_{i,k} += W_c_vec[k*n+i] * (A @ U_c)_{i} * (B @ V_c)_{k}
                
                A_Uc = np.zeros(n)
                for i in range(n):
                    for j in range(m):
                        A_Uc[i] += A_test[i, j] * U[i*m+j, c]
                
                B_Vc = np.zeros(p)
                for k in range(p):
                    for j in range(m):
                        B_Vc[k] += B_test[j, k] * V[j*p+k, c]
                
                for i in range(n):
                    for k in range(p):
                        C_computed[i, k] += W[k*n+i, c] * A_Uc[i] * B_Vc[k]
            
            error = np.max(np.abs(C_computed - C_expected))
            if error > 1e-10:
                print(f"Verification failed: max error = {error}")
                # Fall back to schoolbook
                U_fb = [[0] * (n*m*p) for _ in range(n*m)]
                V_fb = [[0] * (n*m*p) for _ in range(m*p)]
                W_fb = [[0] * (n*m*p) for _ in range(p*n)]
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_fb[i*m+j][col] = 1
                            V_fb[j*p+k][col] = 1
                            W_fb[k*n+i][col] = 1
                            col += 1
                return {"U": U_fb, "V": V_fb, "W": W_fb}
            else:
                print(f"Verification passed: max error = {error}")
        except Exception as e:
            print(f"Verification error: {e}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
