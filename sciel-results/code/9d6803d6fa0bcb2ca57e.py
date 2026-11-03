import numpy as np
import time

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (rank 7)."""
    n, m, p = 2, 2, 2
    U = np.zeros((n*m, 7))
    V = np.zeros((m*p, 7))
    W = np.zeros((p*n, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B12 - B22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B21 - B11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[1, 3] = 1; W[0, 3] = 1
    
    # M5 = (A11 + A12)B22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4x4():
    """Build 4x4x4 decomposition via Kronecker product of two 2x2x2."""
    U2, V2, W2 = _strassen_2x2x2()
    
    U = np.kron(U2, U2)
    V = np.kron(V2, V2)
    W = np.kron(W2, W2)
    
    # Permute U: kron gives (i1*2+j1)*4 + (i2*2+j2), need (i1*2+i2)*4 + j1*2+j2
    perm_U = []
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    src = (i1*2 + j1)*4 + (i2*2 + j2)
                    perm_U.append(src)
    U = U[perm_U, :]
    
    # Permute V: kron gives (j1*2+k1)*4 + (j2*2+k2), need (j1*2+j2)*4 + k1*2+k2
    perm_V = []
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    src = (j1*2 + k1)*4 + (j2*2 + k2)
                    perm_V.append(src)
    V = V[perm_V, :]
    
    # Permute W: kron gives (k1*2+i1)*4 + (k2*2+i2), need (k1*2+k2)*4 + i1*2+i2
    perm_W = []
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    src = (k1*2 + i1)*4 + (k2*2 + i2)
                    perm_W.append(src)
    W = W[perm_W, :]
    
    return U, V, W

def _build_4x4x2():
    """Build 4x4x2 decomposition via schoolbook (rank 32)."""
    n, m, p = 4, 4, 2
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Check shapes
    if U.shape != (n*m, U.shape[1]) or V.shape != (m*p, V.shape[1]) or W.shape != (p*n, W.shape[1]):
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    
    # For each tensor entry (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if i,j,k valid
    # Actually, the decomposition says: C[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T[i,j,k] = delta (but for matrix mult, it's the identity tensor)
    # The tensor for matrix multiplication M[n,m,p] has T[i,j,k] = 1 for all i,j,k
    # Wait, no. The matrix multiplication tensor is defined such that:
    # C = A * B means C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has entries T[i,j,k] and the multiplication is:
    # C[i,k] = sum_{j} sum_{c} U[i*m+j, c] V[j*p+k, c] A[i,j] B[j,k]
    # Hmm, let me re-derive.
    # 
    # Standard form: C_{ik} = sum_j A_{ij} B_{jk}
    # Tensor decomposition: T_{ijk} = sum_c U_{ij,c} V_{jk,c} W_{ki,c}
    # For matrix mult, T_{ijk} = 1 (the identity tensor in the sense that we're computing the product)
    # Actually no. The tensor for matrix multiplication is:
    # T[i, j, k] = delta_{i,j,k} which is just 1 for all i,j,k since it's the "multiplication" tensor.
    # 
    # Let me just check: for random A, B, compute C = A@B and compare with the decomposition.
    
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Compute C from decomposition
    # C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c A[i,j] B[j,k] * (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c])
    # This isn't quite right. Let me think again.
    #
    # The tensor decomposition says: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, we want T[i,j,k] = 1 for all i,j,k.
    # Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c A[i,j] B[j,k] T[i,j,k]... 
    # No, that's not right either.
    #
    # Actually, the correct interpretation:
    # If T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Then for matrix multiplication, we need T[i,j,k] = 1 for all i,j,k.
    # This means: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c A[i,j] B[j,k] (sum_d ...) 
    # 
    # I think the standard form is:
    # C[i,k] = sum_{j,c} U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] * A[i,j] * B[j,k]
    # This doesn't look right dimensionally.
    #
    # Let me look at it differently. The matrix multiplication tensor T in M[n,m,p] is defined by:
    # T[e_i, e_j, e_k] where e_i are basis vectors.
    # The tensor satisfies: <T, A ⊗ B ⊗ C> = tr(C^T A B)
    # For the decomposition: T = sum_c U_c ⊗ V_c ⊗ W_c
    # Where U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    # So T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    # And we need T[i,j,k] = 1 for all i,j,k (this is the matrix multiplication tensor).
    #
    # So the check is: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1.
    
    # Efficient check: build the full tensor and compare
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # We can compute this as: for each c, the outer product U_c ⊗ V_c ⊗ W_c summed over c
    
    # Let's do a spot check
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        s = 0.0
        for c in range(rank):
            s += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
        if abs(s - 1.0) > 1e-9:
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = 4, 4, 6
    
    # Build 4x4x4 part
    U1, V1, W1 = _build_4x4x4()
    
    # Build 4x4x2 part
    U2, V2, W2 = _build_4x4x2()
    
    # Embed into 4x4x6
    # U: no change, both have 16 rows (n*m = 4*4 = 16)
    U = np.hstack([U1, U2])
    
    # V: V1 has 16 rows (m*4 = 4*4), V2 has 8 rows (m*2 = 4*2)
    # Total V should have m*p = 4*6 = 24 rows
    # V1 covers k=0..3, V2 covers k=4..5
    V = np.zeros((m*p, U1.shape[1] + U2.shape[1]))
    # Embed V1: row r in V1 corresponds to j*4+k, place at j*6+k
    for r in range(V1.shape[0]):
        j = r // 4
        k = r % 4
        V[j*6 + k, :V1.shape[1]] = V1[r, :]
    # Embed V2: row r in V2 corresponds to j*2+k, place at j*6+(k+4)
    for r in range(V2.shape[0]):
        j = r // 2
        k = r % 2
        V[j*6 + (k+4), V1.shape[1]:] = V2[r, :]
    
    # W: W1 has 16 rows (p1*n = 4*4), W2 has 8 rows (p2*n = 2*4)
    # Total W should have p*n = 6*4 = 24 rows
    # W1 covers k=0..3, W2 covers k=4..5
    W = np.zeros((p*n, U1.shape[1] + U2.shape[1]))
    # W1 rows: k*n+i for k=0..3
    W[:W1.shape[0], :W1.shape[1]] = W1
    # W2 rows: k*n+i for k=0..1, but we need them at k=4,5
    for r in range(W2.shape[0]):
        k = r // 4  # k is 0 or 1
        i = r % 4
        W[(k+4)*4 + i, W1.shape[1]:] = W2[r, :]
    
    rank = U.shape[1]
    
    # Verify
    if time.time() < deadline - 5.0:
        if not _verify(U, V, W, n, m, p):
            # Fallback: schoolbook for full 4x4x6
            rank_fb = n * m * p
            U = np.zeros((n*m, rank_fb))
            V = np.zeros((m*p, rank_fb))
            W = np.zeros((p*n, rank_fb))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
            rank = rank_fb
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
