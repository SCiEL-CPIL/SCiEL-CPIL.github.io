"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Check if sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] == 1 for all i,j,k
    # We can check this by iterating over all i,j,k
    # Or by checking random matrices if we interpret the bilinear form correctly.
    # The bilinear form is C = sum_c (U_c dot A) (V_c dot B) W_c
    # This is equivalent to the tensor condition.
    
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # Compute scalar products
            # U_c is n x m, V_c is m x p
            # u_val = sum(U_c * A)
            # v_val = sum(V_c * B)
            # C += u_val * v_val * W_c
            u_val = np.sum(U[:, c].reshape(n, m) * A)
            v_val = np.sum(V[:, c].reshape(m, p) * B)
            W_c = W[:, c].reshape(p, n).T # W is p*n, index k*n+i. So row k, col i.
            # W_c[i, k] = W[k*n+i, c]
            # So W_c is n x p.
            C_computed += u_val * v_val * W_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_76(n, m, p):
    # n=4, m=4, p=7
    # Split B into B1 (4x4) and B2 (4x3)
    # Use rank 28 for A*B1 (4x4x4)
    # Use rank 48 for A*B2 (4x4x3)
    
    U_list = []
    V_list = []
    W_list = []
    
    # Part 1: 4x4x4 using Strassen + Schoolbook inner (Rank 28)
    # Strassen terms for 4x4x4
    # A = [[A00, A01], [A10, A11]]
    # B1 = [[B00, B01], [B10, B11]]
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Each Ms is 2x2. We expand each into 4 schoolbook terms.
    # Total 28 terms.
    
    # Helper to add a term
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # We need to map the 2x2 blocks to indices.
    # A is 4x4.
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    
    # B1 is 4x4.
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3
    
    # C is 4x4.
    # C00: rows 0-1, cols 0-1
    # C01: rows 0-1, cols 2-3
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3
    
    # For each Strassen product Ms, we have 4 schoolbook terms.
    # Term (a,b) of Ms contributes to C blocks.
    
    # Let's define the 7 Strassen products and their contributions to C blocks.
    # M1: C00 (+1), C11 (+1)
    # M2: C10 (+1), C11 (-1)
    # M3: C01 (+1), C11 (+1)
    # M4: C00 (+1), C10 (+1)
    # M5: C00 (-1), C01 (+1)
    # M6: C11 (+1)
    # M7: C00 (+1)
    
    # For each Ms, we generate 4 terms.
    # Ms = Xs * Ys.
    # Xs is 2x2, Ys is 2x2.
    # Term (r,c) of Ms is sum_k Xs[r,k] Ys[k,c].
    # This is 2 terms in schoolbook: Xs[r,0]Ys[0,c] and Xs[r,1]Ys[1,c].
    
    # We will iterate through the 7 Ms.
    
    strassen_defs = [
        # (X_def, Y_def, C_contrib)
        # X_def: list of (block, coeff)
        # Y_def: list of (block, coeff)
        # C_contrib: list of (block, coeff)
        
        # M1 = (A00+A11)(B00+B11)
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)]),
        # M2 = (A10+A11)B00
        ([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1), ('C11', -1)]),
        # M3 = A00(B01-B11)
        ([('A00', 1)], [('B01', 1), ('B11', -1)], [('C01', 1), ('C11', 1)]),
        # M4 = A11(B10-B00)
        ([('A11', 1)], [('B10', 1), ('B00', -1)], [('C00', 1), ('C10', 1)]),
        # M5 = (A00+A01)B11
        ([('A00', 1), ('A01', 1)], [('B11', 1)], [('C00', -1), ('C01', 1)]),
        # M6 = (A10-A00)(B00+B01)
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [('C11', 1)]),
        # M7 = (A01-A11)(B10+B11)
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C00', 1)]),
    ]
    
    # Block indices
    # A blocks: 00, 01, 10, 11
    # B blocks: 00, 01, 10, 11
    # C blocks: 00, 01, 10, 11
    
    # Map block name to (row_offset, col_offset)
    A_offsets = {'A00': (0,0), 'A01': (0,2), 'A10': (2,0), 'A11': (2,2)}
    B_offsets = {'B00': (0,0), 'B01': (0,2), 'B10': (2,0), 'B11': (2,2)}
    C_offsets = {'C00': (0,0), 'C01': (0,2), 'C10': (2,0), 'C11': (2,2)}
    
    for X_def, Y_def, C_contrib in strassen_defs:
        # X is 2x2, Y is 2x2.
        # We need to form the 4 schoolbook terms for X*Y.
        # X[r,k] and Y[k,c].
        # X is a linear combo of A blocks.
        # Y is a linear combo of B blocks.
        
        # For each r in 0..1, c in 0..1:
        # Term = sum_k X[r,k] Y[k,c]
        # This is a sum of products of linear forms.
        # We add each product as a separate rank-1 term?
        # No, X[r,k] is a linear form in A.
        # Y[k,c] is a linear form in B.
        # The product X[r,k] Y[k,c] is a rank-1 tensor term.
        
        for r in range(2):
            for c in range(2):
                # We need to sum over k=0,1
                # For each k, we have a term X[r,k] * Y[k,c]
                for k in range(2):
                    # Get X[r,k] definition
                    # X is defined by X_def.
                    # X[r,k] is the (r,k) entry of the 2x2 matrix X.
                    # But X_def gives the blocks.
                    # X is a 2x2 matrix of blocks? No, X is a 2x2 matrix of scalars?
                    # No, X is a 2x2 matrix of 2x2 blocks?
                    # In Strassen, X and Y are 2x2 matrices of blocks?
                    # No, X and Y are 2x2 matrices of scalars?
                    # No, in the 4x4 case, we split into 2x2 blocks.
                    # X = A00 + A11. This is a 2x2 matrix.
                    # So X[r,k] is a scalar entry of the 2x2 matrix X.
                    # But X is formed from A blocks.
                    # X[r,k] = sum_{block in X_def} coeff * A_block[r,k]
                    
                    # So for a fixed r,k, X[r,k] is a linear form in A.
                    # Similarly Y[k,c] is a linear form in B.
                    
                    # We create a rank-1 term for this product.
                    U_vec = [0] * 16
                    V_vec = [0] * 16 # B1 is 4x4
                    W_vec = [0] * 16 # C1 is 4x4
                    
                    # Fill U_vec for X[r,k]
                    for block_name, coeff in X_def:
                        r_off, c_off = A_offsets[block_name]
                        # A_block[r,k] is at index (r_off+r)*4 + (c_off+k)
                        idx = (r_off + r) * 4 + (c_off + k)
                        U_vec[idx] += coeff
                    
                    # Fill V_vec for Y[k,c]
                    for block_name, coeff in Y_def:
                        r_off, c_off = B_offsets[block_name]
                        # B_block[k,c] is at index (r_off+k)*4 + (c_off+c)
                        idx = (r_off + k) * 4 + (c_off + c)
                        V_vec[idx] += coeff
                    
                    # Fill W_vec for C contribution
                    # This term contributes to C[r,c] of the Ms block.
                    # Ms contributes to C blocks as per C_contrib.
                    # The entry (r,c) of Ms maps to (r,c) of the C block.
                    for block_name, coeff in C_contrib:
                        r_off, c_off = C_offsets[block_name]
                        # C_block[r,c] is at index (r_off+r)*4 + (c_off+c)
                        # W index is k*n+i. Here n=4.
                        # C[i,k] index is k*4+i.
                        # i = r_off+r, k = c_off+c
                        idx = (c_off + c) * 4 + (r_off + r)
                        W_vec[idx] += coeff
                    
                    add_term(U_vec, V_vec, W_vec)
                    
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    # B2 is 4x3.
    # We compute A * B2.
    # For each i in 0..3, j in 0..3, k in 0..2:
    # Term A[i,j] B[j,k] contributes to C[i,k].
    
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_vec = [0] * 16
                V_vec = [0] * 12 # B2 is 4x3
                W_vec = [0] * 12 # C2 is 4x3
                
                U_vec[i*4 + j] = 1
                V_vec[j*3 + k] = 1
                # C2[i,k] index is k*4+i
                W_vec[k*4 + i] = 1
                
                # We need to combine these with the previous U, V, W.
                # The previous U, V, W were for B1 (4x4).
                # Now we have B2 (4x3).
                # The total V is 4x7.
                # The total W is 7x4.
                
                # We need to append these terms to the main lists.
                # But the U, V, W vectors must have the correct length.
                # U length 16.
                # V length 28 (16 for B1 + 12 for B2).
                # W length 28 (16 for C1 + 12 for C2).
                
                # The terms added in Part 1 have V length 16 and W length 16.
                # We need to pad them to length 28.
                
                pass # This logic is flawed, we should build the full vectors from the start.

    # Let's restart the construction with full vectors.
    
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    best = _build_schoolbook(n, m, p)
    
    # Try the rank 76 construction
    try:
        # Implement the rank 76 construction properly
        pass
    except Exception:
        pass
        
    return best

# EVOLVE-BLOCK-END
