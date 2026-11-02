# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    # 2x2 * 2x7 matrix multiplication
    # A: 2x2, B: 2x7, C: 2x7
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # Tensor decomposition: C = U * (A .* V) * W^T where .* is element-wise
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # A is flattened as A_flat[i*m + j]
    # B is flattened as B_flat[j*p + k]
    # C is flattened as C_flat[k*n + i] (trace convention W[k*n+i, r])
    
    n, m, p = 2, 2, 7
    rank = 25
    
    # We'll build the decomposition using Strassen for 2x2 blocks
    # B has 7 columns, so we can do 3 full 2x2 blocks + 1 single column
    # Each 2x2 block uses rank 7 (Strassen), single column uses rank 2
    # Total: 3*7 + 2 = 23? Let me think more carefully.
    
    # Actually for the last single column (column 6 of B), we need:
    # C[0][6] = A[0][0]*B[0][6] + A[0][1]*B[1][6]
    # C[1][6] = A[1][0]*B[0][6] + A[1][1]*B[1][6]
    # This is a 2x1 * 1x1 multiplication, which has rank 2 (schoolbook)
    
    # So total rank = 3*7 + 2 = 23
    
    # Let me use rank 23 and see if it works
    rank = 23
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients for A (flattened as [A00, A01, A10, A11])
    strassen_U = np.array([
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ], dtype=float)
    
    # V coefficients for B (flattened as [B00, B01, B10, B11] for a 2x2 block)
    strassen_V = np.array([
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ], dtype=float)
    
    # W coefficients for C (flattened as [C00, C01, C10, C11] for a 2x2 block)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    strassen_W = np.array([
        [1, 0, 0, 0],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [0, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ], dtype=float)
    
    # Process 3 Strassen blocks (columns 0-5 of B)
    for b in range(3):
        for r in range(7):
            # U: A indices are global (0:A00, 1:A01, 2:A10, 3:A11)
            U[:, col] += strassen_U[r]
            
            # V: B indices for this block
            # B[0][2b], B[0][2b+1], B[1][2b], B[1][2b+1]
            # Flattened: B_flat[j*p + k]
            # B[0][2b] -> V[0*p + 2b]
            # B[0][2b+1] -> V[0*p + 2b+1]
            # B[1][2b] -> V[1*p + 2b]
            # B[1][2b+1] -> V[1*p + 2b+1]
            v_idx = [0 * p + 2 * b, 0 * p + 2 * b + 1, 1 * p + 2 * b, 1 * p + 2 * b + 1]
            V[v_idx[0], col] += strassen_V[r, 0]
            V[v_idx[1], col] += strassen_V[r, 1]
            V[v_idx[2], col] += strassen_V[r, 2]
            V[v_idx[3], col] += strassen_V[r, 3]
            
            # W: C indices for this block
            # C[0][2b], C[0][2b+1], C[1][2b], C[1][2b+1]
            # Flattened: C_flat[k*n + i]
            # C[0][2b] -> W[2b*n + 0]
            # C[0][2b+1] -> W[(2b+1)*n + 0]
            # C[1][2b] -> W[2b*n + 1]
            # C[1][2b+1] -> W[(2b+1)*n + 1]
            w_idx = [2 * b * n + 0, (2 * b + 1) * n + 0, 2 * b * n + 1, (2 * b + 1) * n + 1]
            W[w_idx[0], col] += strassen_W[r, 0]
            W[w_idx[1], col] += strassen_W[r, 1]
            W[w_idx[2], col] += strassen_W[r, 2]
            W[w_idx[3], col] += strassen_W[r, 3]
            
            col += 1
    
    # Schoolbook for last column (column 6 of B)
    # C[0][6] = A[0][0]*B[0][6] + A[0][1]*B[1][6]
    # C[1][6] = A[1][0]*B[0][6] + A[1][1]*B[1][6]
    
    # Term 1: A00 * B06 -> C06
    U[0, col] = 1  # A00
    V[0 * p + 6, col] = 1  # B[0][6]
    W[6 * n + 0, col] = 1  # C[0][6]
    col += 1
    
    # Term 2: A01 * B16 -> C06
    U[1, col] = 1  # A01
    V[1 * p + 6, col] = 1  # B[1][6]
    W[6 * n + 0, col] = 1  # C[0][6]
    col += 1
    
    # Term 3: A10 * B06 -> C16
    U[2, col] = 1  # A10
    V[0 * p + 6, col] = 1  # B[0][6]
    W[6 * n + 1, col] = 1  # C[1][6]
    col += 1
    
    # Term 4: A11 * B16 -> C16
    U[3, col] = 1  # A11
    V[1 * p + 6, col] = 1  # B[1][6]
    W[6 * n + 1, col] = 1  # C[1][6]
    col += 1
    
    # Verify the decomposition
    # Check that U * (A .* V) * W^T = C for random inputs
    # We'll do a quick sanity check with small values
    np.random.seed(42)
    A_test = np.random.randint(-2, 3, size=(n, m)).astype(float)
    B_test = np.random.randint(-2, 3, size=(m, p)).astype(float)
    C_expected = A_test @ B_test
    
    # Compute C from decomposition
    A_flat = A_test.flatten()  # [A00, A01, A10, A11]
    B_flat = B_test.flatten()  # [B00, B01, ..., B16]
    
    # C = sum_r U[:,r] * (A_flat .* V[:,r]) * W[:,r]^T
    # More precisely: C_flat[k*n+i] = sum_r U[i*m+j? No...
    # Let me re-derive: C[i][k] = sum_r U[i*m + ?]... 
    # Actually the decomposition is:
    # C_flat = U^T * (A_flat .* V^T) * W ... no
    # Standard form: C = sum_r U_r * (A .* V_r) * W_r^T where U_r is a vector
    # In flattened form: C_flat = sum_r (U[:,r] * (A_flat .* V[:,r])) * W[:,r]^T
    # Wait, that's not right either. Let me think again.
    # 
    # The tensor decomposition says: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # where T is the multiplication tensor, i indexes A, j indexes B, k indexes C
    # 
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # 
    # In flattened form:
    # A_flat index: i*m + j
    # B_flat index: j*p + k
    # C_flat index: k*n + i
    # 
    # So C_flat[k*n+i] = sum_r U[i*m + j? No, U is indexed by A_flat index]
    # Actually: C_flat[c_idx] = sum_r sum_a sum_b U[a, r] * A_flat[a] * V[b, r] * B_flat[b] * W[c_idx, r]
    # This doesn't look right. Let me reconsider.
    # 
    # The correct form is: C_flat = (U * (A_flat .* V^T) * W^T) ... no
    # 
    # Actually, the standard tensor decomposition for matrix multiplication is:
    # C = sum_r (U_r^T * A) * (V_r^T * B) * W_r^T ... no
    # 
    # Let me just verify numerically:
    # C[i,k] should equal sum_r U[i*m + j?] ... I'm confusing myself.
    # 
    # Let me just check: for each r, we compute:
    # temp = sum_a U[a,r] * A_flat[a] * V[b,r] * B_flat[b] ... no
    # 
    # Actually, the way it works is:
    # C_flat[c_idx] = sum_r W[c_idx, r] * (sum_a U[a,r] * A_flat[a]) * (sum_b V[b,r] * B_flat[b])
    # 
    # Let me verify this:
    C_computed = np.zeros(p * n)
    for r in range(rank):
        u_val = np.dot(U[:, r], A_flat)
        v_val = np.dot(V[:, r], B_flat)
        C_computed += W[:, r] * (u_val * v_val)
    
    C_expected_flat = C_expected.flatten()  # C[i,k] -> index i*p + k? Or k*n + i?
    # The trace convention says W[k*n+i, r], so C_flat index is k*n + i
    # C_expected is C[i,k], so C_expected_flat[k*n+i] = C_expected[i,k]
    C_expected_flat_correct = np.zeros(p * n)
    for i in range(n):
        for k in range(p):
            C_expected_flat_correct[k * n + i] = C_expected[i, k]
    
    if not np.allclose(C_computed, C_expected_flat_correct, atol=1e-10):
        # Decomposition is incorrect, fall back to schoolbook rank 28
        rank = 28
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        
        # Verify schoolbook
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        C_computed = np.zeros(p * n)
        for r in range(rank):
            u_val = np.dot(U[:, r], A_flat)
            v_val = np.dot(V[:, r], B_flat)
            C_computed += W[:, r] * (u_val * v_val)
        
        if not np.allclose(C_computed, C_expected_flat_correct, atol=1e-10):
            # This should never happen
            pass
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
