import numpy as np
import time

def construct():
    """
    Rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses Kronecker product of Strassen (2x2x2 rank-7) with proper index permutation.
    """
    start_time = time.time()
    deadline = start_time + 140.0 

    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 rank-7 decomposition
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1,  0, 1],
        [0, 1, 0, 0, 0,  1, 0],
        [1, 1, 0, 1, 0,  0,-1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1,  0, 0, 1, 0],
        [0, 0, 0,  1, 0, 0, 1],
        [1, 0,-1,  0, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1,  0, 0, 0],
        [0, 0, 1, 0,  1, 0, 0],
        [1,-1, 1, 0,  0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4_kron = np.kron(U2, U2)
    V4_kron = np.kron(V2, V2)
    W4_kron = np.kron(W2, W2)
    
    # Permute rows to match standard i*4+j, j*4+k, k*4+i ordering
    # U: kron index r_k = 8*i1 + 4*j1 + 2*i0 + j0
    #    target index r_t = 8*i1 + 4*i0 + 2*j1 + j0
    # We want U4_perm[r_t] = U4_kron[r_k]
    # So U4_perm[r_t, :] = U4_kron[r_k, :]
    # Let's create a mapping array map_U such that map_U[r_t] = r_k
    
    def get_kron_index(i1, j1, i0, j0):
        # U2 row index is i*2+j
        # First U2 (high bits i1, j1): index I = i1*2 + j1
        # Second U2 (low bits i0, j0): index i = i0*2 + j0
        # Kron row = I*4 + i
        return (i1*2 + j1)*4 + (i0*2 + j0)
        
    def get_target_U_index(i1, i0, j1, j0):
        # i = 2*i1 + i0, j = 2*j1 + j0
        # index = i*4 + j
        i = 2*i1 + i0
        j = 2*j1 + j0
        return i*4 + j

    def get_target_V_index(j1, j0, k1, k0):
        # j = 2*j1 + j0, k = 2*k1 + k0
        # index = j*4 + k
        j = 2*j1 + j0
        k = 2*k1 + k0
        return j*4 + k
        
    def get_kron_V_index(j1, k1, j0, k0):
        # V2 row index is j*2+k
        # First V2 (high bits j1, k1): index J = j1*2 + k1
        # Second V2 (low bits j0, k0): index j = j0*2 + k0
        # Kron row = J*4 + j
        return (j1*2 + k1)*4 + (j0*2 + k0)
        
    def get_target_W_index(k1, k0, i1, i0):
        # k = 2*k1 + k0, i = 2*i1 + i0
        # index = k*4 + i
        k = 2*k1 + k0
        i = 2*i1 + i0
        return k*4 + i
        
    def get_kron_W_index(k1, i1, k0, i0):
        # W2 row index is k*2+i
        # First W2 (high bits k1, i1): index K = k1*2 + i1
        # Second W2 (low bits k0, i0): index k = k0*2 + i0
        # Kron row = K*4 + k
        return (k1*2 + i1)*4 + (k0*2 + i0)

    # Construct permutations
    U4_perm = np.zeros_like(U4_kron)
    V4_perm = np.zeros_like(V4_kron)
    W4_perm = np.zeros_like(W4_kron)
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    r_t = get_target_U_index(i1, i0, j1, j0)
                    r_k = get_kron_index(i1, j1, i0, j0)
                    U4_perm[r_t, :] = U4_kron[r_k, :]
                    
                    r_t_v = get_target_V_index(j1, j0, i1, i0) # Wait, V depends on j, k. U depends on i, j.
                    # The Strassen decomposition is for A(i,j) * B(j,k) -> C(k,i).
                    # The "middle" index is j.
                    # In U, indices are i, j.
                    # In V, indices are j, k.
                    # In W, indices are k, i.
                    # My loop variables i1, j1, i0, j0 are for U.
                    # For V, we need to iterate over j1, k1, j0, k0.
                    # But the Kronecker product structure is tied to the tensor indices.
                    # The tensor is T[i, j, k].
                    # U[i, j, r], V[j, k, r], W[k, i, r].
                    # The Kronecker product decomposes T_4[i,j,k] = sum_r U4[i,j,r] V4[j,k,r] W4[k,i,r].
                    # The Kronecker product of T_2[i,j,k] = sum_r U2[i,j,r] V2[j,k,r] W2[k,i,r]
                    # gives T_4[i,j,k] where i=(i1,i0), j=(j1,j0), k=(k1,k0).
                    # The rank is sum of products.
                    # The term is U2[i1,j1,r1] U2[i0,j0,r2] * V2[j1,k1,r1] V2[j0,k0,r2] * W2[k1,i1,r1] W2[k0,i0,r2].
                    # This sums over r1, r2.
                    # So the "combined" index r in 4x4x4 is (r1, r2) -> r1*7 + r2?
                    # np.kron does exactly this: it creates a matrix of size (4*7) columns? No, 7*7=49 columns.
                    # The column index in kron(A,B) is c_A * n_B + c_B.
                    # So col = r1 * 7 + r2.
                    # This matches the summation.
                    
                    # So we just need to permute rows of U, V, W independently based on their index definitions.
                    # U rows: i, j.
                    # V rows: j, k.
                    # W rows: k, i.
                    
                    # I already did U. Now V and W in separate loops or combined?
                    # The loop above iterates i1, j1, i0, j0. This covers all (i,j) pairs.
                    # But V needs (j,k) pairs.
                    # Let's do separate loops for clarity.
                    
    # Redo V permutation
    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    r_t = get_target_V_index(j1, j0, k1, k0)
                    r_k = get_kron_V_index(j1, k1, j0, k0)
                    V4_perm[r_t, :] = V4_kron[r_k, :]
                    
    # Redo W permutation
    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    r_t = get_target_W_index(k1, k0, i1, i0)
                    r_k = get_kron_W_index(k1, i1, k0, i0)
                    W4_perm[r_t, :] = W4_kron[r_k, :]
                    
    # Verify 4x4x4 identity
    # T[i,j,k] = sum_r U4_perm[i*4+j, r] V4_perm[j*4+k, r] W4_perm[k*4+i, r]
    # We can spot check or trust the math.
    
    # Build 4x4x8 rank-98 decomposition
    rank = 98
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # First block (output cols 0-3, i.e., k=0..3)
    U[:, 0:49] = U4_perm
    
    # V indices: j*8+k. For k in 0..3, V[j*8+k] = V4_perm[j*4+k]
    for j in range(4):
        for k_local in range(4):
            V[j*8 + k_local, 0:49] = V4_perm[j*4 + k_local, :]
            
    # W indices: k*4+i. For k in 0..3, W[k*4+i] = W4_perm[k*4+i]
    W[0:16, 0:49] = W4_perm
    
    # Second block (output cols 4-7, i.e., k=4..7)
    U[:, 49:98] = U4_perm
    
    for j in range(4):
        for k_local in range(4):
            V[j*8 + 4 + k_local, 49:98] = V4_perm[j*4 + k_local, :]
            
    W[16:32, 49:98] = W4_perm
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct_naive(n, m, p):
    U = np.zeros((n * m, n * m * p), dtype=float)
    V = np.zeros((m * p, n * m * p), dtype=float)
    W = np.zeros((p * n, n * m * p), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
