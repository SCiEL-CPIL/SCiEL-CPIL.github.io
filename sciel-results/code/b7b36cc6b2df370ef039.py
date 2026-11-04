import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    start_time = time.time()
    
    # Pre-computed Strassen 4x4x4 decomposition (Rank 49)
    # This is the Kronecker product of Strassen 2x2x2 with itself, permuted appropriately.
    
    # U2, V2, W2 for 2x2 (standard Strassen)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Permutations for 4x4
    # U index: i*4+j -> (i1, i2, j1, j2) -> r_out=2*i1+j1, r_in=2*i2+j2 -> K = r_out*4+r_in
    perm_U = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            r_out = 2 * i1 + j1
            r_in = 2 * i2 + j2
            K = r_out * 4 + r_in
            perm_U[S] = K
            
    perm_V = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            S = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            r_out = 2 * j1 + k1
            r_in = 2 * j2 + k2
            K = r_out * 4 + r_in
            perm_V[S] = K
            
    perm_W = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            S = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            r_out = 2 * k1 + i1
            r_in = 2 * k2 + i2
            K = r_out * 4 + r_in
            perm_W[S] = K
            
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify the decomposition: check that sum_r U[:,r] * V[:,r] * W[:,r] == multiplication tensor
    # The tensor T[i,j,k] = 1 if i==k (for matrix mult C[i,k] = sum_j A[i,j]*B[j,k])
    # Wait, the convention is W[k*n+i, r] so W index is k*4+i
    # U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    # Product: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] should equal delta(i,k) for all i,j,k
    # Actually no: for matrix multiplication C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T[i,j,k] = delta(i,k)? No, that's not right either.
    # The multiplication tensor has entries T[i,j,k] where the term A[i,j]*B[j,k] contributes to C[i,k]
    # So T[i,j,k] = 1 if and only if the indices are consistent, i.e., for all i,j,k, T[i,j,k]=1
    # Wait, I need to think about this more carefully.
    # The tensor for matrix multiplication M_{n,m,p} has entries T[i,j,k] for i in [n], j in [m], k in [p]
    # where T[i,j,k] = 1 (the scalar 1) for all valid i,j,k, because A[i,j]*B[j,k] always contributes to C[i,k]
    # So the tensor is all-ones? No, that can't be right for a non-trivial decomposition.
    # 
    # Actually, the multiplication tensor T has T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition means: T[i,j,k] = sum_{r=1}^{r_max} U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, the indices need to match. Let me re-read the convention.
    # U has shape (n*m, rank) = (16, rank), indexed by (i,j) -> i*4+j
    # V has shape (m*p, rank) = (16, rank), indexed by (j,k) -> j*4+k  
    # W has shape (p*n, rank) = (16, rank), indexed by (k,i) -> k*4+i
    # The decomposition: for all i,j,k: delta_{i,i} * delta_{j,j} * delta_{k,k} = 1 = sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
    # So we need: sum_r U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r] = 1 for all i,j,k in {0,1,2,3}
    
    # Quick verification with random test
    if time.time() - start_time < 10:
        U4f = U4.astype(float)
        V4f = V4.astype(float)
        W4f = W4.astype(float)
        # Test a few random (i,j,k) tuples
        np.random.seed(42)
        max_err = 0.0
        for _ in range(100):
            i, j, k = np.random.randint(0, 4, size=3)
            val = np.sum(U4f[i*4+j, :] * V4f[j*4+k, :] * W4f[k*4+i, :])
            max_err = max(max_err, abs(val - 1.0))
        if max_err > 1e-6:
            # Fallback: use a simpler but correct construction
            # Use schoolbook rank 64 as fallback (should never happen)
            U4_fallback = np.zeros((16, 64), dtype=int)
            V4_fallback = np.zeros((16, 64), dtype=int)
            W4_fallback = np.zeros((16, 64), dtype=int)
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        idx = i*16 + j*4 + k
                        U4_fallback[i*4+j, idx] = 1
                        V4_fallback[j*4+k, idx] = 1
                        W4_fallback[k*4+i, idx] = 1
            return {"U": U4_fallback.tolist(), "V": V4_fallback.tolist(), "W": W4_fallback.tolist()}
    
    # Return lists as required
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
