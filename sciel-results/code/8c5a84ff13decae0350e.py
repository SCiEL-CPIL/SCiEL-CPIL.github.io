import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    DEADLINE = 115  # Safety margin before 150s kill

    def schoolbook():
        n, m, p = 4, 4, 4
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2x2 Strassen decomposition
    # Convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # T[i,j,k] = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    
    # Standard Strassen 2x2
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # U maps A indices (i,j) to Strassen components
    # V maps B indices (j,k) to Strassen components
    # W maps C indices (k,i) to Strassen components
    
    # Let's define the matrices based on the standard Strassen formulas.
    # A is 2x2, B is 2x2.
    # A[i,j], B[j,k].
    # C[i,k] = sum_j A[i,j] B[j,k].
    
    # M1 = (A[0,0] + A[1,1])(B[0,0] + B[1,1])
    # M2 = (A[1,0] + A[1,1])B[0,0]
    # M3 = A[0,0](B[1,0] - B[1,1])
    # M4 = A[1,1](B[0,1] - B[0,0])
    # M5 = (A[0,0] + A[0,1])B[1,1]
    # M6 = (A[1,0] - A[0,0])(B[0,0] + B[0,1])
    # M7 = (A[0,1] - A[1,1])(B[1,0] + B[1,1])
    
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # We need U, V, W such that:
    # U[i,j] is the coefficient of A[i,j] in the linear combination for component c.
    # V[j,k] is the coefficient of B[j,k] in the linear combination for component c.
    # W[k,i] is the coefficient of C[k,i] (note index order) in the linear combination for component c.
    
    # Indices:
    # U row: i*2 + j
    # V row: j*2 + k
    # W row: k*2 + i
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Component 1: M1
    # A: A[0,0]+A[1,1] -> U[0,0]=1, U[1,1]=1
    # B: B[0,0]+B[1,1] -> V[0,0]=1, V[1,1]=1
    # C: C[0,0]+C[1,1] -> W[0,0]=1, W[1,1]=1 (W index k*2+i: k=0,i=0 -> 0; k=1,i=1 -> 3)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # Component 2: M2
    # A: A[1,0]+A[1,1] -> U[2,0]=1, U[3,0]=1
    # B: B[0,0] -> V[0,0]=1
    # C: C[1,0] - C[1,1] -> W[2,0]=1 (k=1,i=0 -> 2), W[3,0]=-1 (k=1,i=1 -> 3)
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # Component 3: M3
    # A: A[0,0] -> U[0,0]=1
    # B: B[1,0]-B[1,1] -> V[2,0]=1, V[3,0]=-1
    # C: C[0,1] + C[1,1] -> W[1,0]=1 (k=0,i=1 -> 1), W[3,0]=1
    U2[0, 2] = 1
    V2[2, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # Component 4: M4
    # A: A[1,1] -> U[3,0]=1
    # B: B[0,1]-B[0,0] -> V[1,0]=1, V[0,0]=-1
    # C: C[0,0] + C[1,0] -> W[0,0]=1, W[2,0]=1
    U2[3, 3] = 1
    V2[1, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # Component 5: M5
    # A: A[0,0]+A[0,1] -> U[0,0]=1, U[1,0]=1
    # B: B[1,1] -> V[3,0]=1
    # C: -C[0,0] + C[0,1] -> W[0,0]=-1, W[1,0]=1
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # Component 6: M6
    # A: A[1,0]-A[0,0] -> U[2,0]=1, U[0,0]=-1
    # B: B[0,0]+B[0,1] -> V[0,0]=1, V[1,0]=1
    # C: C[1,1] -> W[3,0]=1
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # Component 7: M7
    # A: A[0,1]-A[1,1] -> U[1,0]=1, U[3,0]=-1
    # B: B[1,0]+B[1,1] -> V[2,0]=1, V[3,0]=1
    # C: C[0,0] -> W[0,0]=1
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2
    valid2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = np.dot(U2[i*2+j], V2[j*2+k] * W2[k*2+i])
                if abs(s - 1) > 1e-9:
                    valid2 = False
                    break
            if not valid2:
                break
        if not valid2:
            break
    
    if not valid2:
        return schoolbook()

    # Build 4x4 via Kronecker
    # The Kronecker product of two valid decompositions is a valid decomposition for the larger size.
    # However, the indices must be permuted correctly to match the "row-major" flattening of the 4x4 matrices.
    
    # For 4x4:
    # U index: i*4 + j, where i,j in 0..3
    # V index: j*4 + k
    # W index: k*4 + i
    
    # We split i, j, k into high and low bits (base 2).
    # i = i1*2 + i0
    # j = j1*2 + j0
    # k = k1*2 + k0
    
    # The Kronecker product U_kron has shape (16, 49).
    # Row index in U_kron is (i1*2 + j1)*4 + (i0*2 + j0) ? No.
    # np.kron(U2, U2) creates a block matrix.
    # Row r in kron is (r_high * 4) + r_low, where r_high is row in first U2, r_low in second.
    # So row (i1*2+j1, i0*2+j0) in kron corresponds to U2[i1*2+j1, :] * U2[i0*2+j0, :]?
    # Actually, np.kron(A, B) where A is 4x7 and B is 4x7 results in 16x49.
    # The element at (r, c) in kron is A[r1, c1] * B[r0, c0] where r = 4*r1 + r0, c = 7*c1 + c0.
    # This matches the structure we want if we map the 4x4 indices correctly.
    
    # Let's verify the mapping.
    # We want U4[i*4+j, c].
    # Let i = 2*i1 + i0, j = 2*j1 + j0.
    # Then i*4+j = (2*i1+i0)*4 + 2*j1+j0 = 8*i1 + 4*i0 + 2*j1 + j0.
    # This is not simply (i1*2+j1)*4 + (i0*2+j0).
    # (i1*2+j1)*4 + (i0*2+j0) = 8*i1 + 4*j1 + 2*i0 + j0.
    # These are different. We need a permutation.
    
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permutation for U:
    # We want to access U4_kron at the row corresponding to the "block" index and "within-block" index.
    # The Kronecker product row index R = 4*R1 + R0.
    # We want the logical row index L = i*4 + j.
    # We define R1 = i1*2 + j1 and R0 = i0*2 + j0.
    # Then R = 4*(i1*2 + j1) + (i0*2 + j0) = 8*i1 + 4*j1 + 2*i0 + j0.
    # We need to map L -> R.
    # L = (2*i1+i0)*4 + (2*j1+j0) = 8*i1 + 4*i0 + 2*j1 + j0.
    # So for each L in 0..15, we find i1,i0,j1,j0 and compute R.
    
    perm_U = np.zeros(16, dtype=int)
    perm_V = np.zeros(16, dtype=int)
    perm_W = np.zeros(16, dtype=int)
    
    for L in range(16):
        i = L // 4
        j = L % 4
        i1, i0 = i // 2, i % 2
        j1, j0 = j // 2, j % 2
        R1 = i1 * 2 + j1
        R0 = i0 * 2 + j0
        perm_U[L] = 4 * R1 + R0
        
    for L in range(16):
        j = L // 4
        k = L % 4
        j1, j0 = j // 2, j % 2
        k1, k0 = k // 2, k % 2
        R1 = j1 * 2 + k1
        R0 = j0 * 2 + k0
        perm_V[L] = 4 * R1 + R0
        
    for L in range(16):
        k = L // 4
        i = L % 4
        k1, k0 = k // 2, k % 2
        i1, i0 = i // 2, i % 2
        R1 = k1 * 2 + i1
        R0 = k0 * 2 + i0
        perm_W[L] = 4 * R1 + R0
        
    U4 = U4_kron[perm_U, :]
    V4 = V4_kron[perm_V, :]
    W4 = W4_kron[perm_W, :]
    
    # Verify 4x4
    i_idx = np.repeat(np.arange(4), 16)
    j_idx = np.tile(np.repeat(np.arange(4), 4), 4)
    k_idx = np.tile(np.arange(4), 16)
    
    u_rows = i_idx * 4 + j_idx
    v_rows = j_idx * 4 + k_idx
    w_rows = k_idx * 4 + i_idx
    
    U_sel = U4[u_rows, :]
    V_sel = V4[v_rows, :]
    W_sel = W4[w_rows, :]
    
    T_flat = np.einsum('ij,ij->i', np.einsum('ij,ij->ij', U_sel, V_sel), W_sel)
    
    if not np.allclose(T_flat, 1.0, atol=1e-9):
        return schoolbook()
    
    # Attempt rank reduction (heuristic)
    # Check if any columns are linearly dependent or zero.
    # In a minimal rank decomposition, columns should be independent.
    # However, due to floating point, we might find near-dependencies.
    # We will try to remove columns that are linear combinations of others.
    # This is generally not possible for minimal rank, but we try.
    
    def try_reduce_rank(U, V, W, max_iter=10):
        rank = U.shape[1]
        # We can't easily remove a column without re-solving the other coefficients.
        # If we remove column c, we need to express it as a combo of others? No, that changes the tensor.
        # If the tensor decomposition is not unique, we might be able to "rotate" the basis to make some coefficients zero.
        # This is a hard problem. We'll just check for exact zeros in columns.
        for c in range(rank):
            if np.allclose(U[:, c], 0) and np.allclose(V[:, c], 0) and np.allclose(W[:, c], 0):
                U = np.delete(U, c, axis=1)
                V = np.delete(V, c, axis=1)
                W = np.delete(W, c, axis=1)
        return U, V, W

    U4, V4, W4 = try_reduce_rank(U4, V4, W4)
    
    def to_list(mat):
        result = []
        for row in mat:
            r = []
            for val in row:
                if abs(val - round(val)) < 1e-9:
                    r.append(int(round(val)))
                elif abs(val - (round(val*2)/2)) < 1e-9:
                    r.append(round(val*2)/2)
                else:
                    r.append(float(val))
            result.append(r)
        return result

    if time.time() - start_time > DEADLINE:
        return schoolbook()
        
    return {
        "U": to_list(U4),
        "V": to_list(V4),
        "W": to_list(W4)
    }
